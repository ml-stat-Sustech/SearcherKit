# index_wiki_multicpu.py

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
import traceback

from elasticsearch import Elasticsearch, ConnectionError
from elasticsearch.helpers import bulk
from elasticsearch.helpers import BulkIndexError
from sentence_transformers import SentenceTransformer

from multiprocessing import Pool, cpu_count # 新增导入

from .prompt import PROMPT_STRATEGIES
from .model import load_model
from .utils import TqdmFileReader


# --- 全局配置 ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
MAX_TEXT_LENGTH_FOR_EMBEDDING = 32768
# 定义一个更大的块大小，用于在进程间分发任务，以减少通信开销
PROCESSING_CHUNK_SIZE = 4096 # 可以根据内存调整

# --- 核心 Elasticsearch 和解析函�?---

def parse_wikitext_for_links(wikitext: str):
    """
    解析维基文本，提取内部链接，并生成一个干净、可读的纯文本版本�?
    (此函数保持不�?
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

# 新增：用于多进程的工作函�?
def process_page_worker(page_data: tuple):
    """
    在子进程中运行的函数，负责解析单个页面的 wikitext�?
    """
    title, wikitext = page_data
    if not wikitext:
        return None
    
    try:
        links_list, plain_text = parse_wikitext_for_links(wikitext)
        if not plain_text:
            return None
        
        return {
            "title": title,
            "text": plain_text,
            "url": f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
            "links": links_list
        }
    except Exception as e:
        # 在处理特定页面时可能会出现意外错�?
        print(f"Worker failed on page '{title}': {e}")
        return None

def create_index(es_client: Elasticsearch, index_name: str, embedding_dim: int, include_vector: bool = True):
    # (此函数保持不�?
    if es_client.indices.exists(index=index_name):
        print(f"索引 '{index_name}' 已存在。正在删�?..")
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
        print(f"正在创建带有向量字段的索�?'{index_name}' (维度: {embedding_dim})...")
        mappings["properties"]["text_vector"] = {
            "type": "dense_vector", "dims": embedding_dim,
            "index": "true", "similarity": "cosine"
        }
    else:
        print(f"正在为仅 BM25 创建索引 '{index_name}' (无向�?...")

    es_client.indices.create(index=index_name, settings=settings, mappings=mappings)
    print(f"索引 '{index_name}' 创建成功�?)

# --- 重构后的核心处理逻辑 ---

def parse_and_index_dump(es_client: Elasticsearch, index_name: str, file_path: str, model: SentenceTransformer, prompt_function, num_workers: int, cpu_batch_size: int, gpu_seq_len: int, include_vector: bool = True, gpu_pool=None):
    if not os.path.exists(file_path):
        print(f"错误: 文件未找到于 {file_path}")
        return

    doc_count = 0
    redirect_count = 0
    ns = '{http://www.mediawiki.org/xml/export-0.11/}'
    total_size = os.path.getsize(file_path)

    print(f"--- 开始使�?{num_workers} 个CPU核心进行并行处理 ---")
    if include_vector:
        print("--- 模式: 混合搜索 (BM25 + 向量) ---")
        print(f"模型编码最大长�?(GPU sequence length): {gpu_seq_len}")
    else:
        print("--- 模式: �?BM25 ---")
    
    # 存储从子进程返回的已处理文章
    processed_articles_buffer = [] 

    with Pool(processes=num_workers) as pool, bz2.BZ2File(file_path, 'rb') as bz2f, \
         tqdm(total=total_size, unit='B', unit_scale=True, desc=f"处理�?{os.path.basename(file_path)}") as pbar:
        
        reader = TqdmFileReader(bz2f, pbar)
        context = ET.iterparse(reader, events=('end',))
        
        # 存储从XML中读取的原始页面，以便批量发送给工作进程
        pages_to_process = []

        for event, elem in context:
            if elem.tag == f'{ns}page':
                try:
                    title_elem = elem.find(f'./{ns}title')
                    ns_elem = elem.find(f'./{ns}ns')
                    redirect_elem = elem.find(f'./{ns}redirect')

                    # 过滤掉非主命名空间页面和重定向页
                    if (ns_elem is None or ns_elem.text != '0') or redirect_elem is not None:
                        if redirect_elem is not None: redirect_count += 1
                        continue
                    
                    title = title_elem.text.strip() if title_elem is not None else ''
                    if not title or title.startswith('Wikipedia:'):
                        continue

                    text_elem = elem.find(f'./{ns}revision/{ns}text')
                    wikitext = text_elem.text if text_elem is not None else ""
                    pages_to_process.append((title, wikitext))

                    # 当累积到足够多的页面时，将其分发给进程池处理
                    if len(pages_to_process) >= PROCESSING_CHUNK_SIZE:
                        # imap_unordered 会在任务完成时立即返回结果，效率更高
                        for result in pool.imap_unordered(process_page_worker, pages_to_process):
                            if result:
                                processed_articles_buffer.append(result)
                        
                        pages_to_process = [] # 清空待处理列�?

                        # 检查缓冲区是否足够大，可以提交一批给ES或GPU
                        if len(processed_articles_buffer) >= cpu_batch_size:
                            doc_count += _process_and_submit_batch(
                                es_client, index_name, processed_articles_buffer, 
                                model, prompt_function, gpu_seq_len, include_vector, gpu_pool
                            )
                            processed_articles_buffer = [] # 清空缓冲�?
                            pbar.set_postfix(docs=f'{doc_count:,}', redirects=f'{redirect_count:,}')

                except Exception as e:
                    print(f"\n主进程解析XML时出�? {e}")
                    traceback.print_exc()
                finally:
                    elem.clear() # 及时释放内存

        # --- 主循环结束后 ---
        # 处理剩余�?pages_to_process 列表中的页面
        if pages_to_process:
            for result in pool.imap_unordered(process_page_worker, pages_to_process):
                if result:
                    processed_articles_buffer.append(result)
        
        # 处理并提交所有剩余在缓冲区中的文�?
        while processed_articles_buffer:
            batch_to_submit = processed_articles_buffer[:cpu_batch_size]
            doc_count += _process_and_submit_batch(
                es_client, index_name, batch_to_submit, 
                model, prompt_function, gpu_seq_len, include_vector, gpu_pool
            )
            processed_articles_buffer = processed_articles_buffer[cpu_batch_size:]
        
    print(f"\n索引的总文档数: {doc_count:,}。跳过的重定向总数: {redirect_count:,}")


def _process_and_submit_batch(es_client, index_name, articles_batch, model, prompt_function, gpu_seq_len, include_vector, gpu_pool=None):
    """
    [内部辅助函数] 负责处理一个批次的已解析文章，
    可选地进行向量编码，并批量提交�?Elasticsearch�?
    """
    actions = []
    
    if not include_vector:
        # BM25 模式：直接准�?bulk actions
        for article in articles_batch:
            actions.append({"_index": index_name, "_source": article})
    else:
        # 混合搜索模式：先编码，再准备 bulk actions
        passages = [prompt_function(art["text"][:MAX_TEXT_LENGTH_FOR_EMBEDDING]) for art in articles_batch]
        
        # 按照最大序列长度组织批�?
        batch_len = 0
        batch = []
        vectors = []
        for passage in passages:
            if (batch_len + len(passage)) > gpu_seq_len:
                new_vectors = model.encode(
                    batch, normalize_embeddings=True,
                    batch_size=len(batch), 
                    show_progress_bar=False, # 在主进度条下运行时关闭此进度�?
                    pool=gpu_pool
                )
                vectors.extend(vec.tolist() for vec in new_vectors)
                
                batch = [passage]
                batch_len = len(passage)
            else:
                batch_len += len(passage)
                batch.append(passage)

        # 处理最后一�?
        if batch:
            new_vectors = model.encode(
                batch, normalize_embeddings=True,
                batch_size=len(batch), 
                show_progress_bar=False, # 在主进度条下运行时关闭此进度�?
                pool=gpu_pool
            )
            vectors.extend(vec.tolist() for vec in new_vectors)
            
        for i, article in enumerate(articles_batch):
            source = article.copy()
            source["text_vector"] = vectors[i]
            actions.append({"_index": index_name, "_source": source})
        
        del passages, vectors
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    try:
        bulk(es_client.options(request_timeout=100), actions, raise_on_error=False, stats_only=True)
    except BulkIndexError as e:
        print(f"Bulk 索引时发生错�? {len(e.errors)} 个文档失败�?)
    
    return len(actions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用多CPU并行解析、嵌入维基百科转储并将其索引�?Elasticsearch�?)
    
    # --- 文件和连接参�?---
    parser.add_argument('--wiki_dump_path', type=str, default="/mnt/sharedata/hdd/users/wsy/project/agent/wiki/data/dumps/enwiki-20251001-pages-articles-multistream.xml.bz2",help="维基百科 XML.bz2 转储文件的路径�?)
    parser.add_argument('--es_host', type=str, default='http://192.168.77.12:9200', help="Elasticsearch 主机 URL�?)
    parser.add_argument('--index_name', type=str, default='wikipedia_hybrid_multi', help="Elasticsearch 索引的名称�?)
    
    # --- 模型相关参数 ---
    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help="Sentence Transformer 模型的名称或路径�?)
    parser.add_argument('--embedding_dim', type=int, default=384, help="模型嵌入的维度�?)
    parser.add_argument('--prompt_strategy', type=str, default='none', choices=PROMPT_STRATEGIES.keys(), help="嵌入模型的提示策略�?)
    
    # --- 批处理和并行参数 ---
    parser.add_argument('--cpu_batch_size', type=int, default=256, help="在提交到ES或GPU前，主进程中累积的已处理文档数量�?)
    # parser.add_argument('--gpu_batch_size', type=int, default=32, help="�?model.encode() 中实际送入GPU的批次大�?(GPU批处�?�?)
    parser.add_argument('--gpu_seq_len', type=int, default=MAX_TEXT_LENGTH_FOR_EMBEDDING*32, help="送入模型的文本最大长度�?)
    parser.add_argument('--num_workers', type=int, default=120, help="用于文本解析的工作进程数量。默认使�?(总核心数 - 2)�?)
    
    # --- 功能开�?---
    parser.add_argument('--dense-vector', action='store_true', help="如果设置，则启用密集向量生成和索引以进行混合搜索�?)
    
    args = parser.parse_args()

    embedding_model = None
    if args.dense_vector:
        embedding_model = load_model(args.model_name)
        pool = embedding_model.start_multi_process_pool(target_devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"])
    else:
        print("--- 未设�?--dense-vector 标志，跳过模型加�?---")

    try:
        es = Elasticsearch(args.es_host, request_timeout=100, retry_on_timeout=True, max_retries=3)
        info = es.info()
        print(f"成功连接�?Elasticsearch 版本 {info['version']['number']}")
    except ConnectionError as e:
        print(f"无法连接�?Elasticsearch。错�? {e}")
        exit(1)

    create_index(es, args.index_name, args.embedding_dim, include_vector=args.dense_vector)
    
    prompt_func = PROMPT_STRATEGIES.get(args.prompt_strategy)

    parse_and_index_dump(
        es_client=es,
        index_name=args.index_name,
        file_path=args.wiki_dump_path,
        model=embedding_model,
        prompt_function=prompt_func,
        num_workers=args.num_workers,
        cpu_batch_size=args.cpu_batch_size,
        # gpu_batch_size=args.gpu_batch_size,
        gpu_seq_len=args.gpu_seq_len,
        include_vector=args.dense_vector,
        gpu_pool=pool if args.dense_vector else None
    )
    pool.close() if args.dense_vector else None

    print("\n--- 全部完成！索引构建完毕�?---")