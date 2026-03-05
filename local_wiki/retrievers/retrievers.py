# retrievers.py

import abc
import collections
from elasticsearch import Elasticsearch, ConnectionError
from typing import List, Dict, Any, Optional, Protocol, Union
from abc import abstractmethod
import numpy as np

class EncoderProtocol(Protocol):
    @abstractmethod
    async def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        ...

from .encoders import BaseEncoder


class BaseRetriever(abc.ABC):
    """
    Retriever 的抽象基类。
    """
    def __init__(self, es_client: Elasticsearch, index_name: str):
        if not es_client.ping():
            raise ConnectionError("无法连接到 Elasticsearch 客户端。")
        self.es = es_client
        self.index_name = index_name

    @abc.abstractmethod
    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    async def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        pass

    # --- 改动点: 恢复为更好、更明确的版本 ---
    def _format_results(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """一个辅助函数，用于将 Elasticsearch 的原始命中结果格式化为更简洁的列表。"""
        formatted_results = []
        for hit in hits:
            source = hit.get("_source", {})
            result = {
                "id": hit["_id"],
                "score": hit["_score"],
                "title": source.get("title"),
                "text": source.get("text"),
                "url": source.get("url"),
                "links": source.get("links", []) # 保证 links 字段总是存在
            }
            formatted_results.append(result)
        return formatted_results

class BM25Retriever(BaseRetriever):
    """
    使用 BM25 进行稀疏检索。
    
    Args:
        es_client: Elasticsearch 客户端实例。
        index_name: 要搜索的索引名称。
        search_fields: 一个字符串列表，指定要在哪些字段中进行搜索。
                       默认为 ["title", "text"]。
    """
    def __init__(self, es_client: Elasticsearch, index_name: str, search_fields: List[str] = None):
        super().__init__(es_client, index_name)
        # if search_fields is None:
        #     self.search_fields = ["title", "text"]
        # else:
        #     self.search_fields = search_fields
        self.search_fields = ["title"] 
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # 使用 self.search_fields 来动态指定搜索字段
        es_query = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": self.search_fields, # <-- 修改点
                    "type": "best_fields"
                }
            }
        }
        try:
            response = self.es.search(index=self.index_name, body=es_query)
            return self._format_results(response['hits']['hits'])
        except Exception as e:
            return []

    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        body = []
        for query in queries:
            body.append({"index": self.index_name})
            body.append({
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": self.search_fields # <-- 修改点
                    }
                }
            })
        try:
            response = self.es.msearch(body=body)
            return [self._format_results(res['hits']['hits']) if 'error' not in res else [] for res in response['responses']]
        except Exception as e:
            return [[] for _ in queries]

class DenseRetriever(BaseRetriever):
    """使用密集向量进行检索。"""
    def __init__(self, es_client: Elasticsearch, index_name: str, encoder: BaseEncoder):
        super().__init__(es_client, index_name)
        self.encoder = encoder

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        vector = self.encoder.encode(query).tolist()
        es_query = {"size": top_k, "knn": {"field": "text_vector", "query_vector": vector, "k": top_k, "num_candidates": 100}}
        try:
            response = self.es.search(index=self.index_name, body=es_query)
            return self._format_results(response['hits']['hits'])
        except Exception as e: return []

    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        vectors = self.encoder.encode(queries).tolist()
        body = []
        for vector in vectors:
            body.append({"index": self.index_name})
            body.append({"size": top_k, "knn": {"field": "text_vector", "query_vector": vector, "k": top_k, "num_candidates": 100}})
        try:
            response = self.es.msearch(body=body)
            return [self._format_results(res['hits']['hits']) if 'error' not in res else [] for res in response['responses']]
        except Exception as e: return [[] for _ in queries]

# class HybridRetriever(BaseRetriever):
#     """使用混合方法进行检索。"""
#     def __init__(self, es_client: Elasticsearch, index_name: str, encoder: BaseEncoder):
#         super().__init__(es_client, index_name)
#         self.encoder = encoder

#     def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#         vector = self.encoder.encode(query).tolist()
#         # 对于混合搜索，k 的值需要比 top_k 大，以便为 RRF 提供足够的候选文档
#         # num_candidates 可以根据需求调整，一般是 top_k 的几倍
#         num_candidates = top_k * 10  
        
#         es_query = {
#             "size": top_k,
#             "query": {"multi_match": {"query": query, "fields": ["title", "text"]}},
#             "knn": {
#                 "field": "text_vector",
#                 "query_vector": vector,
#                 "k": num_candidates,
#                 "num_candidates": num_candidates * 2  # num_candidates 至少是 k 的值
#             },
#             "rank": {
#                 "rrf": {}  # 使用 Reciprocal Rank Fusion (RRF) 来融合 BM25 和 k-NN 的结果
#             }
#         }
#         try:
#             response = self.es.search(index=self.index_name, body=es_query)
#             return self._format_results(response['hits']['hits'])
#         except Exception as e:
#             # 增加一个打印，方便调试
#             print(f"Hybrid search failed for query '{query}': {e}")
#             return []

#     # --- 这里是修改点 ---
#     def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
#         """
#         由于 _msearch API 不支持 RRF 混合查询，
#         我们回退到循环调用单个 search 方法来实现批量处理。
#         """
#         # 注意：向量编码可以保持批量处理以提高效率，但这里为了逻辑简单，
#         # 让 search 方法自己处理编码。
#         # 如果追求极致性能，可以先批量编码，然后传入 search 方法。
#         all_results = []
#         for query in queries:
#             # 直接复用已经可以工作的 single search 方法
#             results_for_query = self.search(query, top_k=top_k)
#             all_results.append(results_for_query)
#         return all_results


class HybridRetriever(BaseRetriever):
    """
    使用混合方法进行检索。
    由于 Elasticsearch 的 RRF 是付费功能，此类在客户端手动实现 RRF 融合。
    """
    def __init__(self, es_client: Elasticsearch, index_name: str, encoder: BaseEncoder):
        super().__init__(es_client, index_name)
        self.encoder = encoder

    def search(self, query: str, top_k: int = 5, k_rrf: int = 60) -> List[Dict[str, Any]]:
        # 1. 执行 BM25 稀疏检索
        # 为了给 RRF 提供足够的候选文档，我们检索比 top_k 更多的结果
        num_candidates = top_k * 10
        bm25_query = {
            "size": num_candidates,
            "query": {"multi_match": {"query": query, "fields": ["title", "text"]}}
        }
        try:
            bm25_response = self.es.search(index=self.index_name, body=bm25_query)
            bm25_hits = bm25_response['hits']['hits']
        except Exception as e:
            print(f"BM25 search failed for query '{query}': {e}")
            bm25_hits = []

        # 2. 执行 Dense 向量检索
        vector = self.encoder.encode(query).tolist()
        dense_query = {
            "size": num_candidates,
            "knn": {
                "field": "text_vector",
                "query_vector": vector,
                "k": num_candidates,
                "num_candidates": num_candidates * 2
            }
        }
        try:
            dense_response = self.es.search(index=self.index_name, body=dense_query)
            dense_hits = dense_response['hits']['hits']
        except Exception as e:
            print(f"Dense search failed for query '{query}': {e}")
            dense_hits = []

        # 3. 在客户端手动执行 RRF
        # 使用文档 ID 作为键，存储原始的 hit 对象和它们的排名
        doc_ranks = collections.defaultdict(lambda: {'bm25_rank': float('inf'), 'dense_rank': float('inf'), 'hit': None})

        # 记录 BM25 排名
        for rank, hit in enumerate(bm25_hits):
            doc_id = hit['_id']
            doc_ranks[doc_id]['bm25_rank'] = rank + 1
            doc_ranks[doc_id]['hit'] = hit

        # 记录 Dense 排名并更新 hit 对象（dense 结果通常更相关）
        for rank, hit in enumerate(dense_hits):
            doc_id = hit['_id']
            doc_ranks[doc_id]['dense_rank'] = rank + 1
            doc_ranks[doc_id]['hit'] = hit
        
        # 如果没有任何结果，直接返回空列表
        if not doc_ranks:
            return []

        # 计算 RRF 分数
        fused_results = []
        for doc_id, data in doc_ranks.items():
            bm25_score_part = 1.0 / (k_rrf + data['bm25_rank']) if data['bm25_rank'] != float('inf') else 0
            dense_score_part = 1.0 / (k_rrf + data['dense_rank']) if data['dense_rank'] != float('inf') else 0
            rrf_score = bm25_score_part + dense_score_part
            
            # 更新 hit 对象的 _score 字段为 RRF 分数
            data['hit']['_score'] = rrf_score
            fused_results.append(data['hit'])

        # 4. 按 RRF 分数排序并格式化输出
        fused_results.sort(key=lambda x: x['_score'], reverse=True)
        
        return self._format_results(fused_results[:top_k])


    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """
        循环调用单个 search 方法来实现批量处理。
        """
        all_results = []
        for query in queries:
            results_for_query = self.search(query, top_k=top_k)
            all_results.append(results_for_query)
        return all_results

class ApiRetriever(BaseRetriever):
    def __init__(self, es_client: Elasticsearch, index_name: str, encoder: EncoderProtocol):
        super().__init__(es_client, index_name)
        self.encoder = encoder

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        vector = (await self.encoder.encode(query)).tolist()
        es_query = {"size": top_k, "knn": {"field": "text_vector", "query_vector": vector, "k": top_k, "num_candidates": 100}}
        try:
            response = self.es.search(index=self.index_name, body=es_query)
            return self._format_results(response['hits']['hits'])
        except Exception as e:
            return []

    async def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        vectors = (await self.encoder.encode(queries)).tolist()
        body = []
        for vector in vectors:
            body.append({"index": self.index_name})
            body.append({"size": top_k, "knn": {"field": "text_vector", "query_vector": vector, "k": top_k, "num_candidates": 100}})
        try:
            response = self.es.msearch(body=body)
            return [self._format_results(res['hits']['hits']) if 'error' not in res else [] for res in response['responses']]
        except Exception as e:
            return [[] for _ in queries]

def build_retriever(
    retriever_type: str,
    es_client: Elasticsearch,
    index_name: str,
    encoder: Optional[Union[BaseEncoder, EncoderProtocol]] = None
) -> BaseRetriever:
    """根据指定的类型创建并返回一个 Retriever 实例。"""
    retriever_map = {'bm25': BM25Retriever,
                     'dense': DenseRetriever,
                     'hybrid': HybridRetriever,
                     'api': ApiRetriever}
    
    retriever_class = retriever_map.get(retriever_type.lower())

    if retriever_class is None:
        raise ValueError(f"未知的 retriever 类型: '{retriever_type}'. 可选: 'bm25', 'dense', 'hybrid'.")

    if retriever_class == BM25Retriever:
        return BM25Retriever(es_client=es_client, index_name=index_name)
    else:
        if not encoder:
            raise ValueError(f"{retriever_class.__name__} 需要一个 encoder 实例。")
        return retriever_class(es_client=es_client, index_name=index_name, encoder=encoder)

# --- 主测试块 ---
if __name__ == '__main__':

    from .encoders import build_encoder, load_model

    ES_HOST = 'http://localhost:9200'
    INDEX_NAME = 'wiki20251001_e5-base-v2'
    MODEL_NAME = 'intfloat/e5-base-v2' 

    try:
        es_client = Elasticsearch(ES_HOST, request_timeout=30)
        model = load_model(MODEL_NAME)
        encoder_instance = build_encoder(MODEL_NAME, model)
        print("\n成功连接到 Elasticsearch 并初始化编码器。")
    except Exception as e:
        print(f"初始化失败: {e}")
        exit(1)

    queries_to_search = [
        "Capital of China",
        "中国的首都",
        "President of the United States",
        "Donald Trump",
        'List of presidential trips made by Donald Trump (2017)'
    ]
    print(f"\n將使用 {len(queries_to_search)} 个查询进行批量搜索...\n")
    
    retrievers_to_test = ['dense']

    for r_type in retrievers_to_test:
        try:
            print("\n" + "="*30)
            print(f"测试 {r_type.upper()} Retriever (Batch Search)")
            print("="*30)
            
            retriever = build_retriever(
                retriever_type=r_type,
                es_client=es_client, 
                index_name=INDEX_NAME, 
                encoder=encoder_instance
            )
            
            batch_results = retriever.batch_search(queries_to_search, top_k=2)
            
            for i, query in enumerate(queries_to_search):
                print(f"\n--- 结果 for query: '{query}' ---")
                results_for_query = batch_results[i]
                if not results_for_query:
                    print("  未找到结果。")
                else:
                    for j, res in enumerate(results_for_query):
                        print(f"  {j+1}. {res['title']} (Score: {res['score']:.4f})")
                        # 我们可以安全地访问 links，因为它总会存在
                        # print(f"     Links count: {len(res['links'])}") 
        
        except Exception as e:
            print(f"执行 {r_type.upper()} Retriever 时出错: {e}")
