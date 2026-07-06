#!/bin/bash
# start_tei_lb.sh - 多 GPU TEI 负载均衡启动脚本（非容器化）

set -e

# 默认参数
FRONTEND_PORT=8004
BACKEND_BASE_PORT=8020
GPUS="0,1"            # 默认只用 GPU 0，格式如 "0,1,2"
MODEL_PATH="$HOME/Qwen3-Embedding-8B"
MAX_BATCH_TOKENS=40960
POOLING="last-token"
OTHER_ARGS=""

# 显示帮助
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --frontend-port PORT    Load balancer listen port (default: 8004)"
    echo "  --backend-start PORT    First backend TEI port (default: 8100)"
    echo "  --gpus LIST             Comma-separated GPU IDs (default: 0) e.g., 0,1,2,3"
    echo "  --model PATH            Path to model (default: ~/Qwen3-Embedding-8B)"
    echo "  --max-batch-tokens N    --max-batch-tokens value (default: 40960)"
    echo "  --pooling STR           Pooling strategy (default: last-token)"
    echo "  --other-args STR        Extra arguments to text-embeddings-router"
    echo "  -h, --help              Show this help"
    exit 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --frontend-port) FRONTEND_PORT="$2"; shift 2 ;;
        --backend-start) BACKEND_BASE_PORT="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --model) MODEL_PATH="$2"; shift 2 ;;
        --max-batch-tokens) MAX_BATCH_TOKENS="$2"; shift 2 ;;
        --pooling) POOLING="$2"; shift 2 ;;
        --other-args) OTHER_ARGS="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# 检查 text-embeddings-router 是否可用
if ! command -v text-embeddings-router &> /dev/null; then
    echo "错误: 未找到 text-embeddings-router，请确认已安装并加入 PATH"
    exit 1
fi

# 将 GPUS 字符串转为数组
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}
echo "将使用 GPU: ${GPU_ARRAY[@]} (共 $NUM_GPUS 个)"

# 创建日志目录
LOG_DIR="./tei_logs"
mkdir -p "$LOG_DIR"

# 停止已有的 TEI 进程（避免端口冲突）
PID_FILE="tei_pids.txt"
if [ -f "$PID_FILE" ]; then
    echo "发现之前的 PID 文件，正在停止旧进程..."
    while read -r pid; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  终止进程 $pid"
            kill "$pid"
        else
            echo "  进程 $pid 已不存在"
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
    sleep 2
else
    echo "未找到 PID 文件，不会清理任何 TEI 进程"
fi

# 单 GPU 时不需要 nginx 负载均衡，直接把 TEI 暴露在 frontend port。
if [ "$NUM_GPUS" -eq 1 ]; then
    BACKEND_BASE_PORT="$FRONTEND_PORT"
fi

# 为每个 GPU 启动 TEI 实例
echo "启动 $NUM_GPUS 个 TEI 实例..."
PIDS=()
for idx in "${!GPU_ARRAY[@]}"; do
    GPU_ID="${GPU_ARRAY[$idx]}"
    PORT=$((BACKEND_BASE_PORT + idx))
    LOG_FILE="$LOG_DIR/tei_gpu${GPU_ID}_port${PORT}.log"
    
    echo "启动 GPU $GPU_ID (端口 $PORT)，日志: $LOG_FILE"
    
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    nohup text-embeddings-router \
        --model-id "$MODEL_PATH" \
        --port "$PORT" \
        --pooling "$POOLING" \
        --max-batch-tokens "$MAX_BATCH_TOKENS" \
        $OTHER_ARGS \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    PIDS+=($PID)
    echo "  PID: $PID"
    
    sleep 1  # 避免端口抢占
done

# 保存 PID 列表
printf "%s\n" "${PIDS[@]}" > tei_pids.txt
echo "TEI PID 已保存到 tei_pids.txt"

# 等待 TEI 服务就绪
echo "等待 TEI 服务就绪（最多 60 秒）..."
TIMEOUT=60
ELAPSED=0
READY=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    READY=0
    for idx in "${!GPU_ARRAY[@]}"; do
        PORT=$((BACKEND_BASE_PORT + idx))
        if curl -s -f "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            READY=$((READY + 1))
        fi
    done
    if [ $READY -eq $NUM_GPUS ]; then
        echo "所有 TEI 实例均已就绪"
        break
    fi
    echo "就绪: $READY / $NUM_GPUS，等待中..."
    sleep 3
    ELAPSED=$((ELAPSED + 3))
done

if [ $READY -ne $NUM_GPUS ]; then
    echo "警告: 部分 TEI 未就绪，请检查日志 $LOG_DIR"
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "========== 启动完成 =========="
    echo "单 GPU 模式：未启动 Nginx，TEI 直接监听 http://0.0.0.0:$FRONTEND_PORT"
    echo "测试命令: curl -X POST http://localhost:$FRONTEND_PORT/v1/embeddings -H 'Content-Type: application/json' -d '{\"model\":\"$MODEL_PATH\",\"input\":\"Hello world\"}'"
    exit 0
fi

# 动态生成 Nginx 配置
echo "生成 Nginx 配置..."
UPSTREAM_SERVERS=""
for idx in "${!GPU_ARRAY[@]}"; do
    PORT=$((BACKEND_BASE_PORT + idx))
    UPSTREAM_SERVERS+="        server 127.0.0.1:$PORT;\n"
done

# 基于模板创建临时配置
TMP_NGINX_CONF="/tmp/tei_nginx_$$.conf"
sed -e "s|__UPSTREAM_SERVERS__|$UPSTREAM_SERVERS|g" \
    -e "s|__FRONTEND_PORT__|$FRONTEND_PORT|g" \
    nginx.conf.template > "$TMP_NGINX_CONF"

echo "生成的 upstream 配置:"
echo "$UPSTREAM_SERVERS"

# 启动 Nginx（尝试两种方式：系统 nginx 或 docker）
NGINX_RUNNING=0
if command -v nginx &> /dev/null; then
    echo "使用系统 nginx 启动..."
    # 备份原配置并替换
    sudo cp "$TMP_NGINX_CONF" /etc/nginx/nginx.conf
    sudo nginx -t || { echo "Nginx 配置测试失败"; exit 1; }
    sudo systemctl start nginx 2>/dev/null || sudo nginx
    NGINX_RUNNING=1
    echo "系统 nginx 已启动"
else
    echo "错误: 没有系统 nginx，无法启动负载均衡器"
    exit 1
fi

echo "========== 启动完成 =========="
echo "负载均衡器地址: http://0.0.0.0:$FRONTEND_PORT"
echo "后端 TEI 端口范围: $BACKEND_BASE_PORT ~ $((BACKEND_BASE_PORT + NUM_GPUS - 1))"
echo "测试命令: curl -X POST http://localhost:$FRONTEND_PORT/embed -H 'Content-Type: application/json' -d '{\"inputs\":\"Hello world\"}'"
