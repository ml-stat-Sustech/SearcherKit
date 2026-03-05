# index_wiki.py

import bz2
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import argparse
import mwparserfromhell
from urllib.parse import quote

from elasticsearch import Elasticsearch, ConnectionError
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer

# 从本地模块导入
from prompt import PROMPT_STRATEGIES
from model import load_model
from utils import TqdmFileReader

# --- 核心 Elasticsearch 和解析函数 ---

# --- 请将此函数添加到 index_wiki.py 文件的顶部 ---

def parse_wikitext_structured(wikitext: str):
    """
    解析维基文本，将其转换为一个包含文本和链接的结构化列表。
    同时，它也会返回一个纯文本版本，用于全文搜索和向量化。

    返回:
        tuple: (structured_content, plain_text)
        - structured_content (list): 一个对象列表，例如 [{'type': 'text', 'content': '...'}, {'type': 'link', 'text': '...', 'target': '...'}]
        - plain_text (str): 用于搜索的纯文本内容。
    """
    try:
        parsed_code = mwparserfromhell.parse(wikitext)
    except Exception:
        # mwparserfromhell 可能会在处理极其复杂的模板时出错
        return [], "" # 返回空内容，避免索引失败

    structured_content = []
    plain_text_parts = []

    for node in parsed_code.nodes:
        # Case 1: 节点是维基链接 (例如 [[Albert Einstein]] 或 [[ theoretical physicist|physicist]])
        if isinstance(node, mwparserfromhell.nodes.wikilink.Wikilink):
            # 链接的显示文本。如果没有指定（如[[Albert Einstein]]），则显示文本就是标题本身
            display_text = node.text if node.text is not None else node.title
            # 链接指向的目标页面标题
            target_title = node.title
            
            # 清理文本和标题中的多余空格
            clean_display_text = str(display_text).strip()
            clean_target_title = str(target_title).strip()

            if clean_display_text: # 确保有内容才添加
                structured_content.append({
                    "type": "link",
                    "text": clean_display_text,
                    "target": clean_target_title
                })
                plain_text_parts.append(clean_display_text)

        # Case 2: 节点是纯文本
        elif isinstance(node, mwparserfromhell.nodes.text.Text):
            text_content = str(node.value)
            structured_content.append({
                "type": "text",
                "content": text_content
            })
            plain_text_parts.append(text_content)
            
        # 其他类型的节点（如模板、HTML标签等）在此被忽略，但不会被 stripping 掉，
        # 而是被 mwparserfromhell 智能地处理或跳过。

    # 将所有文本部分连接成一个完整的字符串，用于搜索
    full_plain_text = "".join(plain_text_parts).strip()
    
    return structured_content, full_plain_text

def create_index(es_client: Elasticsearch, index_name: str, embedding_dim: int, include_vector: bool = True):
    """
    创建一个 Elasticsearch 索引，并根据需要动态添加向量字段。
    """
    if es_client.indices.exists(index=index_name):
        print(f"索引 '{index_name}' 已存在。正在删除...")
        es_client.indices.delete(index=index_name)

    settings = { "number_of_shards": 5, "number_of_replicas": 1 }
    mappings = {
        "properties": {
            "title": {"type": "text", "analyzer": "standard"},
            "text": {"type": "text", "analyzer": "standard"},
            "url": {"type": "keyword"}
        }
    }

    if include_vector:
        print(f"正在创建带有向量字段的索引 '{index_name}' (维度: {embedding_dim})...")
        mappings["properties"]["text_vector"] = {
            "type": "dense_vector",
            "dims": embedding_dim,
            "index": "true",
            "similarity": "cosine"
        }
    else:
        print(f"正在为仅 BM25 创建索引 '{index_name}' (无向量)...")

    es_client.indices.create(index=index_name, settings=settings, mappings=mappings)
    print(f"索引 '{index_name}' 创建成功。")


def _parse_and_index_bm25(es_client: Elasticsearch, index_name: str, file_path: str, batch_size: int):
    """
    [内部辅助函数] 专为 BM25 优化的解析和索引。
    此函数不包含任何与向量相关的逻辑，以实现最高性能。
    """
    actions = []
    doc_count = 0
    redirect_count = 0
    ns = '{http://www.mediawiki.org/xml/export-0.11/}'
    
    total_size = os.path.getsize(file_path)
    
    print("--- 正在以高性能 BM25 模式运行 ---")
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
                            clean_text = mwparserfromhell.parse(text_elem.text).strip_code().strip()
                            if not clean_text: continue

                            # 直接、高效地构建 action
                            action = {
                                "_index": index_name,
                                "_source": {
                                    "title": title,
                                    "text": clean_text,
                                    "url": f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                                }
                            }
                            actions.append(action)

                            # 简单、快速的批处理检查
                            if len(actions) >= batch_size:
                                bulk(es_client, actions, raise_on_error=False, request_timeout=100)
                                doc_count += len(actions)
                                pbar.set_postfix(docs=f'{doc_count:,}', redirects=f'{redirect_count:,}')
                                actions = []

                    except Exception as e:
                        print(f"\n处理页面时出错: {e}")
                    finally:
                        elem.clear()

    if actions:
        bulk(es_client, actions, raise_on_error=False, request_timeout=100)
        doc_count += len(actions)
        print(f"\n已索引最后的 {len(actions)} 个文档。")

    print(f"\n索引的总文档数: {doc_count:,}。跳过的重定向总数: {redirect_count:,}")

def _parse_and_index_hybrid(es_client: Elasticsearch, index_name: str, file_path: str, model: SentenceTransformer, prompt_function, batch_size: int):
    """
    [内部辅助函数] 用于混合搜索（BM25 + 向量）的解析和索引。
    """
    actions = []
    articles_in_batch = []
    doc_count = 0
    redirect_count = 0
    ns = '{http://www.mediawiki.org/xml/export-0.11/}'

    total_size = os.path.getsize(file_path)
    
    print("--- 正在以混合搜索（向量）模式运行 ---")
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
                            clean_text = mwparserfromhell.parse(text_elem.text).strip_code().strip()
                            if not clean_text: continue

                            articles_in_batch.append({
                                "title": title,
                                "text": clean_text,
                                "url": f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                            })

                            if len(articles_in_batch) >= batch_size:
                                passages = [prompt_function(art["text"]) for art in articles_in_batch]
                                vectors = model.encode(passages, normalize_embeddings=True)
                                for i, article in enumerate(articles_in_batch):
                                    source = article.copy()
                                    source["text_vector"] = vectors[i].tolist()
                                    actions.append({"_index": index_name, "_source": source})
                                
                                bulk(es_client, actions, raise_on_error=False, request_timeout=100)
                                doc_count += len(actions)
                                pbar.set_postfix(docs=f'{doc_count:,}', redirects=f'{redirect_count:,}')
                                actions, articles_in_batch = [], []

                    except Exception as e:
                        print(f"\n处理页面时出错: {e}")
                    finally:
                        elem.clear()

    if articles_in_batch:
        passages = [prompt_function(art["text"]) for art in articles_in_batch]
        vectors = model.encode(passages, normalize_embeddings=True)
        for i, article in enumerate(articles_in_batch):
            source = article.copy()
            source["text_vector"] = vectors[i].tolist()
            actions.append({"_index": index_name, "_source": source})
        
        bulk(es_client, actions, raise_on_error=False, request_timeout=100)
        doc_count += len(actions)
        print(f"\n已索引最后的 {len(actions)} 个文档。")

    print(f"\n索引的总文档数: {doc_count:,}。跳过的重定向总数: {redirect_count:,}")


def parse_and_index_dump(
    es_client: Elasticsearch,
    index_name: str,
    file_path: str,
    model: SentenceTransformer,
    prompt_function,
    batch_size: int = 500,
    include_vector: bool = True
):
    """
    解析维基百科转储文件并进行批量索引。
    这是一个调度函数，它根据 'include_vector' 的值调用专门优化的内部函数。
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件未找到于 {file_path}")
        return

    if include_vector:
        # 调用混合模式的专用函数
        _parse_and_index_hybrid(es_client, index_name, file_path, model, prompt_function, batch_size)
    else:
        # 调用 BM25 模式的专用函数
        _parse_and_index_bm25(es_client, index_name, file_path, batch_size)


# --- 主执行块 ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="解析、嵌入维基百科转储并将其索引到 Elasticsearch。")
    
    parser.add_argument('--wiki_dump_path', type=str, default="/mnt/sharedata/hdd/users/wsy/project/agent/wiki/data/dumps/enwiki-20251001-pages-articles-multistream.xml.bz2",help="维基百科 XML.bz2 转储文件的路径。")
    parser.add_argument('--es_host', type=str, default='http://localhost:9200', help="Elasticsearch 主机 URL。")
    parser.add_argument('--index_name', type=str, default='wikipedia_hybrid', help="Elasticsearch 索引的名称。")
    
    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help="Sentence Transformer 模型的名称或路径。")
    parser.add_argument('--embedding_dim', type=int, default=384, help="模型嵌入的维度。")
    
    parser.add_argument('--prompt_strategy', type=str, default='none', choices=PROMPT_STRATEGIES.keys(), help="嵌入模型的提示策略。")
    
    parser.add_argument('--batch_size', type=int, default=1000, help="一个批次中处理的文档数量。")
    
    parser.add_argument('--dense-vector', action='store_true', help="如果设置，则启用密集向量生成和索引以进行混合搜索。")
    
    args = parser.parse_args()

    should_include_vector = args.dense_vector
    embedding_model = None

    if should_include_vector:
        embedding_model = load_model(args.model_name)
    else:
        print("--- 未设置 --dense-vector 标志，跳过模型加载 ---")

    try:
        es = Elasticsearch(args.es_host, request_timeout=100)
        info = es.info()
        print(f"成功连接到 Elasticsearch 版本 {info['version']['number']}")
    except ConnectionError as e:
        print(f"无法连接到 Elasticsearch。错误: {e}")
        exit(1)

    create_index(es, args.index_name, args.embedding_dim, include_vector=should_include_vector)
    
    prompt_func = PROMPT_STRATEGIES.get(args.prompt_strategy)

    # 调用主调度函数
    parse_and_index_dump(
        es_client=es,
        index_name=args.index_name,
        file_path=args.wiki_dump_path,
        model=embedding_model,
        prompt_function=prompt_func,
        batch_size=args.batch_size,
        include_vector=should_include_vector
    )

    print("\n--- 全部完成！索引构建完毕。 ---")