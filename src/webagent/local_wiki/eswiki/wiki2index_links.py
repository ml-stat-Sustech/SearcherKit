# index_wiki.py

import bz2
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import argparse
import mwparserfromhell
from urllib.parse import quote
import re
import gc
import torch

from elasticsearch import Elasticsearch, ConnectionError
from elasticsearch.helpers import bulk
from elasticsearch.helpers import BulkIndexError
from sentence_transformers import SentenceTransformer

from .prompt import PROMPT_STRATEGIES
from .model import load_model
from .utils import TqdmFileReader


# --- 全局配置 ---
# 建议在运行脚本前设置此环境变量，以帮助PyTorch管理内存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# 强制截断送入模型的最大文本长度（按字符），防止单个超长文档导致OOM
MAX_TEXT_LENGTH_FOR_EMBEDDING = 32768

# --- 核心 Elasticsearch 和解析函数 ---

def parse_wikitext_for_links(wikitext: str):
    """
    解析维基文本，提取内部链接，并生成一个干净、可读的纯文本版本。
    """
    try:
        parsed_code = mwparserfromhell.parse(wikitext, skip_style_tags=False)
    except Exception:
        return [], ""

    links = []
    markdown_parts = []

    for node in parsed_code.nodes:
        if isinstance(node, mwparserfromhell.nodes.wikilink.Wikilink):
            display_text = str(node.text or node.title).strip()
            target_title = str(node.title).strip()
            if display_text and target_title:
                links.append({"text": display_text, "target": target_title})
                markdown_parts.append(f"[{display_text}]({target_title})")
        elif isinstance(node, mwparserfromhell.nodes.heading.Heading):
            level = node.level
            title = node.title.strip_code().strip()
            markdown_parts.append(f"\n\n{'#' * level} {title}\n")
        elif isinstance(node, mwparserfromhell.nodes.tag.Tag):
            tag_name = str(node.tag).lower()
            contents = node.contents.strip_code().strip()
            if tag_name == "'''": markdown_parts.append(f"**{contents}**")
            elif tag_name == "''": markdown_parts.append(f"*{contents}*")
            elif tag_name in ('b', 'strong'): markdown_parts.append(f"**{contents}**")
            elif tag_name in ('i', 'em'): markdown_parts.append(f"*{contents}*")
            elif tag_name == 'ref': pass
            else: markdown_parts.append(contents)
        elif isinstance(node, mwparserfromhell.nodes.text.Text):
            markdown_parts.append(str(node.value))
        elif isinstance(node, mwparserfromhell.nodes.comment.Comment):
            pass
        else:
            node_as_wikicode = mwparserfromhell.parse(str(node))
            stripped_text = node_as_wikicode.strip_code().strip()
            if stripped_text:
                markdown_parts.append(stripped_text + " ")
    
    # 后处理逻辑
    full_markdown_text = "".join(markdown_parts)
    section_markers = [
        "\n## See also", "\n## References", "\n## Further reading",
        "\n## External links", "\n[Category:"
    ]
    split_index = -1
    for marker in section_markers:
        found_pos = full_markdown_text.find(marker)
        if found_pos != -1 and (split_index == -1 or found_pos < split_index):
            split_index = found_pos
    processed_text = full_markdown_text[:split_index] if split_index != -1 else full_markdown_text
    processed_text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', processed_text)
    final_text = re.sub(r'\n{3,}', '\n\n', re.sub(r' +', ' ', processed_text)).strip()
    
    return links, final_text

def create_index(es_client: Elasticsearch, index_name: str, embedding_dim: int, include_vector: bool = True):
    if es_client.indices.exists(index=index_name):
        print(f"索引 '{index_name}' 已存在。正在删除...")
        es_client.indices.delete(index=index_name)

    settings = {"index": {"number_of_shards": 10, "number_of_replicas": 1}}
    mappings = {
        "properties": {
            "title": {"type": "text", "analyzer": "standard"},
            "text": {"type": "text", "analyzer": "standard"},
            "url": {"type": "keyword"},
            "links": {
                "type": "nested",
                "properties": {
                    "text": {"type": "text", "index": False},
                    "target": {"type": "keyword", "index": False}
                }
            }
        }
    }

    if include_vector:
        print(f"正在创建带有向量字段的索引 '{index_name}' (维度: {embedding_dim})...")
        mappings["properties"]["text_vector"] = {
            "type": "dense_vector", "dims": embedding_dim,
            "index": "true", "similarity": "cosine"
        }
    else:
        print(f"正在为仅 BM25 创建索引 '{index_name}' (无向量)...")

    es_client.indices.create(index=index_name, settings=settings, mappings=mappings)
    print(f"索引 '{index_name}' 创建成功。")

def _parse_and_index_bm25(es_client: Elasticsearch, index_name: str, file_path: str, batch_size: int):
    """
    [内部辅助函数] 专为 BM25 优化的解析和索引。
    此函数解析链接并将其与纯文本分开存储。
    """
    actions = []
    doc_count = 0
    redirect_count = 0
    ns = '{http://www.mediawiki.org/xml/export-0.11/}'
    
    total_size = os.path.getsize(file_path)
    
    print("--- 正在以高性能 BM25 模式运行 (带链接解析) ---")
    with bz2.BZ2File(file_path, 'rb') as bz2f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"BM25 处理中 {os.path.basename(file_path)}") as pbar:
            reader = TqdmFileReader(bz2f, pbar)
            context = ET.iterparse(reader, events=('end',))
            
            for event, elem in context:
                if elem.tag == f'{ns}page':
                    try:
                        title_elem = elem.find(f'./{ns}title')
                        text_elem = elem.find(f'./{ns}revision/{ns}text')
                        redirect_elem = elem.find(f'./{ns}redirect')
                        ns_elem = elem.find(f'./{ns}ns')

                        if (ns_elem is None or ns_elem.text != '0') or redirect_elem is not None:
                            if redirect_elem is not None: redirect_count += 1
                            continue

                        title = title_elem.text.strip() if title_elem is not None else ''
                        if not title or title.startswith('Wikipedia:'):
                            continue

                        if text_elem is not None and text_elem.text:
                            links_list, plain_text = parse_wikitext_for_links(text_elem.text)
                            if not plain_text: continue

                            action = {
                                "_index": index_name,
                                "_source": {
                                    "title": title,
                                    "text": plain_text,
                                    "url": f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                                    "links": links_list
                                }
                            }
                            actions.append(action)

                            if len(actions) >= batch_size:
                                try:
                                    bulk(es_client.options(request_timeout=100), actions, raise_on_error=False)
                                except BulkIndexError as e:
                                    print(f"Bulk 索引时发生错误: {len(e.errors)} 个文档失败。")
                                doc_count += len(actions)
                                pbar.set_postfix(docs=f'{doc_count:,}', redirects=f'{redirect_count:,}')
                                actions = []

                    except Exception as e:
                        print(f"\n处理页面时出错: {e}")
                    finally:
                        elem.clear()

    if actions:
        try:
            bulk(es_client.options(request_timeout=100), actions, raise_on_error=False)
            doc_count += len(actions)
            print(f"\n已索引最后的 {len(actions)} 个文档。")
        except BulkIndexError as e:
            print(f"最后的 Bulk 索引时发生错误: {len(e.errors)} 个文档失败。")

    print(f"\n索引的总文档数: {doc_count:,}。跳过的重定向总数: {redirect_count:,}")

def _parse_and_index_hybrid(es_client: Elasticsearch, index_name: str, file_path: str, model: SentenceTransformer, prompt_function, cpu_batch_size: int, gpu_batch_size: int):
    """
    [内部辅助函数] 用于混合搜索（向量）的解析和索引，支持独立的CPU和GPU批处理大小。
    """
    actions = []
    articles_in_batch = []
    doc_count = 0
    redirect_count = 0
    ns = '{http://www.mediawiki.org/xml/export-0.11/}'
    total_size = os.path.getsize(file_path)
    
    print("--- 正在以混合搜索（向量）模式运行 (内存及GPU优化版) ---")
    print(f"文件处理批次大小 (CPU batch size): {cpu_batch_size}")
    print(f"模型编码批次大小 (GPU batch size): {gpu_batch_size}")

    with bz2.BZ2File(file_path, 'rb') as bz2f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"混合模式处理中 {os.path.basename(file_path)}") as pbar:
            reader = TqdmFileReader(bz2f, pbar)
            context = ET.iterparse(reader, events=('end',))

            for event, elem in context:
                if elem.tag == f'{ns}page':
                    try:
                        title_elem = elem.find(f'./{ns}title')
                        text_elem = elem.find(f'./{ns}revision/{ns}text')
                        redirect_elem = elem.find(f'./{ns}redirect')
                        ns_elem = elem.find(f'./{ns}ns')

                        if (ns_elem is None or ns_elem.text != '0') or redirect_elem is not None:
                            if redirect_elem is not None: redirect_count += 1
                            continue
                        
                        title = title_elem.text.strip() if title_elem is not None else ''
                        if not title or title.startswith('Wikipedia:'):
                            continue

                        if text_elem is not None and text_elem.text:
                            links_list, plain_text = parse_wikitext_for_links(text_elem.text)
                            if not plain_text: continue

                            articles_in_batch.append({
                                "title": title, "text": plain_text,
                                "url": f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                                "links": links_list
                            })
                            
                            if len(articles_in_batch) >= cpu_batch_size:
                                passages = [prompt_function(art["text"][:MAX_TEXT_LENGTH_FOR_EMBEDDING]) for art in articles_in_batch]
                                
                                print(f"\n[INFO] 开始编码 {len(passages)} 篇文章，使用GPU批大小 {gpu_batch_size}...")
                                vectors = model.encode(
                                    passages, normalize_embeddings=True,
                                    batch_size=gpu_batch_size, show_progress_bar=True
                                )
                                
                                for i, article in enumerate(articles_in_batch):
                                    source = article.copy()
                                    source["text_vector"] = vectors[i].tolist()
                                    actions.append({"_index": index_name, "_source": source})

                                bulk(es_client.options(request_timeout=100), actions, raise_on_error=False)
                                
                                doc_count += len(actions)
                                pbar.set_postfix(docs=f'{doc_count:,}', redirects=f'{redirect_count:,}')
                                
                                del passages, vectors
                                actions, articles_in_batch = [], []
                                gc.collect()
                                if torch.cuda.is_available(): torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"\n处理页面时出错: {e}")
                    finally:
                        elem.clear()

    # 处理最后一批剩余的文档
    if articles_in_batch:
        passages = [prompt_function(art["text"][:MAX_TEXT_LENGTH_FOR_EMBEDDING]) for art in articles_in_batch]
        print(f"\n[INFO] 开始编码最后一批 {len(passages)} 篇文章，使用GPU批大小 {gpu_batch_size}...")
        vectors = model.encode(
            passages, normalize_embeddings=True,
            batch_size=gpu_batch_size, show_progress_bar=True
        )
        for i, article in enumerate(articles_in_batch):
            source = article.copy()
            source["text_vector"] = vectors[i].tolist()
            actions.append({"_index": index_name, "_source": source})
        
        bulk(es_client.options(request_timeout=100), actions, raise_on_error=False)
        print(f"\n已索引最后的 {len(actions)} 个文档。")

    print(f"\n索引的总文档数: {doc_count:,}。跳过的重定向总数: {redirect_count:,}")

def parse_and_index_dump(es_client: Elasticsearch, index_name: str, file_path: str, model: SentenceTransformer, prompt_function, cpu_batch_size: int, gpu_batch_size: int, include_vector: bool = True):
    if not os.path.exists(file_path):
        print(f"错误: 文件未找到于 {file_path}")
        return

    if include_vector:
        _parse_and_index_hybrid(es_client, index_name, file_path, model, prompt_function, cpu_batch_size, gpu_batch_size)
    else:
        _parse_and_index_bm25(es_client, index_name, file_path, cpu_batch_size)
        print("BM25 模式索引功能当前已注释掉，请在代码中取消注释以启用。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="解析、嵌入维基百科转储并将其索引到 Elasticsearch。")
    
    # --- 文件和连接参数 ---
    parser.add_argument('--wiki_dump_path', type=str, default="/mnt/sharedata/hdd/users/wsy/project/agent/wiki/data/dumps/enwiki-20251001-pages-articles-multistream.xml.bz2",help="维基百科 XML.bz2 转储文件的路径。")
    parser.add_argument('--es_host', type=str, default='http://192.168.77.12:9200', help="Elasticsearch 主机 URL。")
    parser.add_argument('--index_name', type=str, default='wikipedia_hybrid', help="Elasticsearch 索引的名称。")
    
    # --- 模型相关参数 ---
    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help="Sentence Transformer 模型的名称或路径。")
    parser.add_argument('--embedding_dim', type=int, default=384, help="模型嵌入的维度。")
    parser.add_argument('--prompt_strategy', type=str, default='none', choices=PROMPT_STRATEGIES.keys(), help="嵌入模型的提示策略。")
    
    # --- 批处理参数 ---
    parser.add_argument('--cpu_batch_size', type=int, default=200, help="从文件中一次读取和处理的文档数量 (CPU批处理)。")
    parser.add_argument('--gpu_batch_size', type=int, default=16, help="在 model.encode() 中实际送入GPU的批次大小 (GPU批处理)。")
    
    # --- 功能开关 ---
    parser.add_argument('--dense-vector', action='store_true', help="如果设置，则启用密集向量生成和索引以进行混合搜索。")
    
    args = parser.parse_args()

    embedding_model = None
    if args.dense_vector:
        embedding_model = load_model(args.model_name)
    else:
        print("--- 未设置 --dense-vector 标志，跳过模型加载 ---")

    try:
        es = Elasticsearch(args.es_host, request_timeout=100, retry_on_timeout=True, max_retries=3)
        info = es.info()
        print(f"成功连接到 Elasticsearch 版本 {info['version']['number']}")
    except ConnectionError as e:
        print(f"无法连接到 Elasticsearch。错误: {e}")
        exit(1)

    create_index(es, args.index_name, args.embedding_dim, include_vector=args.dense_vector)
    
    prompt_func = PROMPT_STRATEGIES.get(args.prompt_strategy)

    parse_and_index_dump(
        es_client=es,
        index_name=args.index_name,
        file_path=args.wiki_dump_path,
        model=embedding_model,
        prompt_function=prompt_func,
        cpu_batch_size=args.cpu_batch_size,
        gpu_batch_size=args.gpu_batch_size,
        include_vector=args.dense_vector
    )

    print("\n--- 全部完成！索引构建完毕。 ---")