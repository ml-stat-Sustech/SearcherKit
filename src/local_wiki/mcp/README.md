# LocalWiki MCP Server

基于 FastMCP 的本地 Wikipedia 检索和访问服务。

## 架构

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   MCP Client    │────▶│   MCP Server    │────▶│  Elasticsearch  │
│  (如 Claude)    │     │  (uvicorn)      │     │  (192.168.77.12)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  vLLM Embedding │
                       │  (localhost:8200)│
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  vLLM Summary   │
                       │  (localhost:8300)│
                       └─────────────────┘
```

## 端口配置

| 服务 | 默认端口 | 配置变量 |
|------|---------|---------|
| vLLM Embedding | 8200 | `VLLM_PORT` |
| vLLM Summary | 8300 | `SUMMARY_MODEL_PORT` |
| MCP Server | 8100 | 无（固定） |

## 环境变量

### 公共
```bash
export CUDA_VISIBLE_DEVICES=0  # 使用 GPU 0
```

### Embedding 模型
```bash
export VLLM_MODEL_PATH=/mnt/sharedata/ssd_large/common/LLMs/Qwen3-Embedding-0.6B
export VLLM_PORT=8200
export VLLM_MODEL_NAME=Qwen3-Embedding-0.6B
```

### Summary 模型
```bash
export SUMMARY_MODEL_PATH=/mnt/sharedata/ssd_large/common/LLMs/Qwen3-8B
export SUMMARY_MODEL_PORT=8300
export SUMMARY_MODEL_NAME=Qwen3-8B
```

### Elasticsearch
- 默认: `http://192.168.77.12:9200`
- 可通过 `LOCAL_WIKI_ES_HOST` 环境变量覆盖

## 启动流程

### 1. 启动 vLLM Embedding 服务

```bash
cd src/local_wiki/mcp
bash run_embedding_model.sh
```

验证服务是否启动成功：
```bash
curl http://localhost:8200/v1/models
```

### 2. 启动 vLLM Summary 服务（新窗口）

```bash
cd src/local_wiki/mcp
bash run_summary_model.sh
```

验证服务是否启动成功：
```bash
curl http://localhost:8300/v1/models
```

### 3. 启动 MCP Server

```bash
cd src/local_wiki/mcp
bash run_wiki_mcp.sh
```

验证服务是否启动成功：
```bash
curl http://localhost:8100/mcp/
```

## 测试

### 工具自测

```bash
cd src/local_wiki
bash scripts/tool_test.sh
```

### MCP 客户端连接

MCP Server 启动后，通过以下端点连接：

```
http://localhost:8100/mcp/
```

## 常见问题

### 端口被占用

```bash
# 查看端口占用
lsof -i :8200
lsof -i :8300
lsof -i :8100

# 或使用 netstat
netstat -tlnp | grep 8200
```

## 目录结构

```
mcp/
├── run_embedding_model.sh          # 启动 embedding vLLM
├── run_summary_model.sh # 启动 summary vLLM
├── run_wiki_mcp.sh      # 启动 MCP Server
└── local_wiki_mcp.py    # MCP 服务实现
```
