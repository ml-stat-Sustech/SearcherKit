# WebAgent 本地 Wiki 检索管线

WebAgent 提供了一套从维基百科原始 dump 构建本地知识库、再到为 Agent 暴露检索/访问工具的完整流水线。核心组件包括：
- `eswiki/`：解析 `enwiki-*.xml.bz2`，按需生成文本与超链，写入 Elasticsearch（支持纯 BM25、稠密向量以及 Hybrid 检索）。
- `retrievers/`：封装 BM25、Dense、Hybrid 三类检索器，支持 `SentenceTransformer` 模型（E5、Qwen 等）的自动适配。
- `tools/`：面向 LLM Agent 的工具接口，提供 `SearchLocalWiki`（检索标题）和 `VisitForLocalWiki`（按标题拉取正文+可行动链接）。

## 功能亮点
- **可定制的索引构建**：`wiki2index_links.py` 支持 CPU/GPU 双批次、可选 Prompt、按需截断文本、自动创建索引映射。
- **多种检索模式**：BM25 稀疏检索、SentenceTransformer 稠密检索、客户端 RRF 融合的 Hybrid 检索随需切换。
- **Agent 友好的工具协议**：输出格式强调「下一步行动」，便于连接到自主 Agent。
- **多模型适配**：针对 E5、Qwen/BGE 等模型内置不同前缀/参数处理；支持 `flash-attention`、多 GPU。

## 目录概览
- `docs/`：安装说明（`install.md`）、模型下载脚本示例。
- `eswiki/`：索引构建脚本、模型与 Prompt 工具、解析缓存脚本。
- `retrievers/`：编码器适配层与三种检索器。
- `scripts/`：常用脚本（启动 ES Docker、批量索引、工具自测）。
- `tools/`：Agent 工具实现。

## 快速上手

### 1. 环境准备
```bash
conda create -n webagent python=3.10
conda activate webagent
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 如需 Qwen 大模型或 flash-attn，请参考 docs/install.md
```

可选的 Hugging Face 配置（脚本默认读取以下变量，请根据实际环境替换，不要提交明文令牌）：
```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=<your-hf-token>
export HF_HOME=/path/to/hf/models
export HF_DATASETS_CACHE=/path/to/hf/datasets
```

### 2. 启动 Elasticsearch
使用 Docker 快速启动单节点并挂载数据卷（可参考 `scripts/docker_luncher.sh`）：
```bash
local_path=/path/to/es/data
docker run \
  --name es-wiki \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -v ${local_path}:/usr/share/elasticsearch/data \
  -d elasticsearch:8.19.5

curl http://127.0.0.1:9200
```

### 3. 构建 Wiki 索引
1. 下载 `enwiki-*-pages-articles-multistream.xml.bz2`（官方 dump 或镜像站）。
2. 按需选择模型，并确认嵌入维度、Prompt 策略（详见 `eswiki/prompt.py`）。
3. 运行索引脚本（以下示例启用向量索引）：
```bash
python eswiki/wiki2index_links.py \
  --wiki_dump_path /data/enwiki-20251001-pages-articles-multistream.xml.bz2 \
  --es_host http://127.0.0.1:9200 \
  --index_name wiki20251001_qwen3-embedding-0.6b \
  --model_name ${HF_HOME}/Qwen3-Embedding-0.6B \
  --embedding_dim 1024 \
  --prompt_strategy none \
  --cpu_batch_size 200 \
  --gpu_batch_size 16 \
  --dense-vector
```

脚本会自动：
- 创建索引（包含文本字段、可选 `dense_vector`）。
- 解析维基文本为正文 + 链接列表（过滤冗余板块）。
- 分批送入 `SentenceTransformer`，可通过 `--dense-vector` 控制是否生成向量。

如果需要在 CPU 侧加速解析，可改用 `eswiki/wiki2index_links_multicpu.py`。`scripts/wiki2index.sh` 提供了多种模型配置范例，可作为模板。

### 4. 验证工具链
确认 Elasticsearch 已建索引后，可运行：
```bash
bash scripts/tool_test.sh
```
该脚本会初始化检索器并触发 `SearchLocalWiki`/`VisitForLocalWiki` 的示例调用，检查模型加载和管道连通性。

## Agent 检索工具集成
- `tools/search.py` 暴露 `SearchTool`，封装检索器，并返回「下一步访问标题」的说明。可指定 `RETRIEVER_TYPE`（`bm25`/`dense`/`hybrid`）与模型路径。
- `tools/visit.py` 提供 `VisitTool`，按标题拉取正文、URL 以及过滤后的可访问链接列表（默认排除 `File:`、`Category:` 前缀）。
- 两个工具均实现了统一的 `BaseTool` 协议，可直接挂载到外部 Agent 框架。

若希望在自定义 Python 代码中调用底层检索器，可参考：
```python
from elasticsearch import Elasticsearch
from retrievers.encoders import load_model, build_encoder
from retrievers.retrievers import build_retriever

es = Elasticsearch("http://127.0.0.1:9200", request_timeout=30)
model = load_model("intfloat/e5-base-v2")
encoder = build_encoder("intfloat/e5-base-v2", model)
retriever = build_retriever("dense", es, "wiki20251001_e5-base-v2", encoder)

print(retriever.search("Capital of China", top_k=3))
```

## 其他资料
- `docs/install.md`：更详细的环境搭建流程。
- `docs/model_download.sh`：批量下载 Hugging Face/ModelScope 模型的脚本模板。
- `eswiki/cache/`：历史版本或纯文本索引脚本，可用于参考或备份方案。

完成索引与工具验证后，即可将 `SearchLocalWiki` / `VisitForLocalWiki` 挂载到任意 Agent 系统中，为长程任务提供本地维基知识库支撑。
