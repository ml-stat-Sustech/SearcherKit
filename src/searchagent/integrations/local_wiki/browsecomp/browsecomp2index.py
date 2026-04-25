import os
import re
import gc
import argparse

import torch
from datasets import load_dataset
from elasticsearch import Elasticsearch, ConnectionError
from elasticsearch.helpers import bulk, BulkIndexError
from sentence_transformers import SentenceTransformer

from model import load_model
from prompt import PROMPT_STRATEGIES


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
MAX_TEXT_LENGTH_FOR_EMBEDDING = 32768


def extract_title_from_text(raw_text: str) -> str:
    match = re.search(r'^title:\s*(.+)$', raw_text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def strip_frontmatter(raw_text: str) -> str:
    cleaned = re.sub(r'^---\n.*?\n---\n', '', raw_text, count=1, flags=re.DOTALL)
    cleaned = re.sub(r'\n{3,}', '\n\n', re.sub(r' +', ' ', cleaned)).strip()
    return cleaned


def create_index(es_client: Elasticsearch, index_name: str, embedding_dim: int, include_vector: bool = True):
    if es_client.indices.exists(index=index_name):
        print(f"索引 '{index_name}' 已存在。正在删�?..")
        es_client.indices.delete(index=index_name)

    settings = {"index": {"number_of_shards": 10, "number_of_replicas": 1}}
    mappings = {
        "properties": {
            "title": {"type": "text", "analyzer": "standard"},
            "text": {"type": "text", "analyzer": "standard"},
            "url": {
                "type": "keyword",
                "fields": {
                    "text": {"type": "text", "analyzer": "standard"}
                }
            },
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


def index_bm25(es_client: Elasticsearch, index_name: str, dataset, batch_size: int):
    actions = []
    doc_count = 0

    print("--- 正在�?BM25 模式运行 ---")
    for article in dataset:
        try:
            docid = article["docid"]
            title = extract_title_from_text(article["text"])
            plain_text = strip_frontmatter(article["text"])
            if not plain_text:
                continue

            action = {
                "_index": index_name,
                "_id": docid,
                "_source": {
                    "title": title,
                    "text": plain_text,
                    "url": article["url"],
                    "links": []
                }
            }
            actions.append(action)

            if len(actions) >= batch_size:
                try:
                    bulk(es_client.options(request_timeout=100), actions, raise_on_error=False)
                except BulkIndexError as e:
                    print(f"Bulk 索引时发生错�? {len(e.errors)} 个文档失败�?)
                doc_count += len(actions)
                print(f"\r已索�?{doc_count:,} 篇文�?, end="", flush=True)
                actions = []

        except (ValueError, KeyError) as e:
            print(f"\n处理文档时出�? {e}")

    if actions:
        try:
            bulk(es_client.options(request_timeout=100), actions, raise_on_error=False)
            doc_count += len(actions)
        except BulkIndexError as e:
            print(f"最后的 Bulk 索引时发生错�? {len(e.errors)} 个文档失败�?)

    print(f"\n索引的总文档数: {doc_count:,}")


def index_hybrid(es_client: Elasticsearch, index_name: str, dataset, model: SentenceTransformer, prompt_function, cpu_batch_size: int, gpu_batch_size: int, pool=None):
    actions = []
    articles_in_batch = []
    doc_count = 0

    print("--- 正在以混合搜索（向量）模式运�?---")
    print(f"文件处理批次大小 (CPU batch size): {cpu_batch_size}")
    print(f"模型编码批次大小 (GPU batch size): {gpu_batch_size}")
    if pool is not None:
        print("多GPU模式已启用，使用 encode_multi_process")

    def _encode(passages):
        if pool is not None:
            return model.encode(passages, pool=pool, batch_size=gpu_batch_size, normalize_embeddings=True)
        return model.encode(passages, normalize_embeddings=True, batch_size=gpu_batch_size, show_progress_bar=True)

    for article in dataset:
        try:
            docid = article["docid"]
            title = extract_title_from_text(article["text"])
            plain_text = strip_frontmatter(article["text"])
            if not plain_text:
                continue

            articles_in_batch.append({
                "_id": docid,
                "title": title,
                "text": plain_text,
                "url": article["url"],
                "links": []
            })

            if len(articles_in_batch) >= cpu_batch_size:
                passages = [prompt_function(art["text"][:MAX_TEXT_LENGTH_FOR_EMBEDDING]) for art in articles_in_batch]

                print(f"\n[INFO] 开始编�?{len(passages)} 篇文章，使用GPU批大�?{gpu_batch_size}...")
                vectors = _encode(passages)

                for i, art in enumerate(articles_in_batch):
                    source = {"title": art["title"], "text": art["text"], "url": art["url"], "links": art["links"]}
                    source["text_vector"] = vectors[i].tolist()
                    actions.append({"_index": index_name, "_id": art["_id"], "_source": source})

                try:
                    bulk(es_client.options(request_timeout=100), actions, raise_on_error=False)
                except BulkIndexError as e:
                    print(f"Bulk 索引时发生错�? {len(e.errors)} 个文档失败�?)

                doc_count += len(actions)
                print(f"已索�?{doc_count:,} 篇文�?)

                del passages, vectors
                actions, articles_in_batch = [], []
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except (ValueError, KeyError) as e:
            print(f"\n处理文档时出�? {e}")

    if articles_in_batch:
        passages = [prompt_function(art["text"][:MAX_TEXT_LENGTH_FOR_EMBEDDING]) for art in articles_in_batch]
        print(f"\n[INFO] 开始编码最后一�?{len(passages)} 篇文章，使用GPU批大�?{gpu_batch_size}...")
        vectors = _encode(passages)
        for i, art in enumerate(articles_in_batch):
            source = {"title": art["title"], "text": art["text"], "url": art["url"], "links": art["links"]}
            source["text_vector"] = vectors[i].tolist()
            actions.append({"_index": index_name, "_id": art["_id"], "_source": source})

        try:
            bulk(es_client.options(request_timeout=100), actions, raise_on_error=False)
        except BulkIndexError as e:
            print(f"最后的 Bulk 索引时发生错�? {len(e.errors)} 个文档失败�?)
        doc_count += len(actions)

    print(f"\n索引的总文档数: {doc_count:,}")


def index_dataset(es_client: Elasticsearch, index_name: str, dataset, model: SentenceTransformer, prompt_function, cpu_batch_size: int, gpu_batch_size: int, include_vector: bool = True, pool=None):
    if include_vector:
        index_hybrid(es_client, index_name, dataset, model, prompt_function, cpu_batch_size, gpu_batch_size, pool=pool)
    else:
        index_bm25(es_client, index_name, dataset, cpu_batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="解析 BrowseComp-Plus 语料库并将其索引�?Elasticsearch�?)

    parser.add_argument('--dataset_path', type=str, default="Tevatron/browsecomp-plus-corpus", help="HuggingFace 数据集名称或本地 parquet 路径�?)
    parser.add_argument('--es_host', type=str, default='http://192.168.77.12:9200', help="Elasticsearch 主机 URL�?)
    parser.add_argument('--index_name', type=str, default='browsecomp_hybrid', help="Elasticsearch 索引的名称�?)

    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-Embedding-8B', help="Sentence Transformer 模型的名称或路径�?)
    parser.add_argument('--embedding_dim', type=int, default=4096, help="模型嵌入的维度�?)
    parser.add_argument('--prompt_strategy', type=str, default='none', choices=PROMPT_STRATEGIES.keys(), help="嵌入模型的提示策略�?)

    parser.add_argument('--cpu_batch_size', type=int, default=200, help="从数据集中一次读取和处理的文档数�?(CPU批处�?�?)
    parser.add_argument('--gpu_batch_size', type=int, default=16, help="�?model.encode() 中实际送入GPU的批次大�?(GPU批处�?�?)

    parser.add_argument('--dense-vector', action='store_true', help="如果设置，则启用密集向量生成和索引以进行混合搜索�?)
    parser.add_argument('--no_multi_gpu', action='store_true', help="如果设置，则禁用多GPU编码，仅在单GPU上运行�?)

    args = parser.parse_args()

    print(f"正在加载数据�? {args.dataset_path}...")
    ds = load_dataset(args.dataset_path, split="train")
    print(f"数据集加载完成，�?{len(ds):,} 条文档�?)

    embedding_model = None
    pool = None
    if args.dense_vector:
        embedding_model = load_model(args.model_name)
        if not args.no_multi_gpu:
            print("正在启动多GPU编码进程�?..")
            pool = embedding_model.start_multi_process_pool()
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

    try:
        index_dataset(
            es_client=es,
            index_name=args.index_name,
            dataset=ds,
            model=embedding_model,
            prompt_function=prompt_func,
            cpu_batch_size=args.cpu_batch_size,
            gpu_batch_size=args.gpu_batch_size,
            include_vector=args.dense_vector,
            pool=pool
        )
    finally:
        if pool is not None:
            print("正在关闭多GPU编码进程�?..")
            embedding_model.stop_multi_process_pool(pool)

    print("\n--- 全部完成！索引构建完毕�?---")
