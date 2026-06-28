#!/bin/bash
# stop_tei_lb.sh - 停止所有 TEI 和 Nginx

echo "停止 TEI 进程..."
if [ -f tei_pids.txt ]; then
    while read pid; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "已终止 PID: $pid"
        fi
    done < tei_pids.txt
    rm tei_pids.txt
else
    pkill -f "text-embeddings-router" && echo "已通过 pkill 停止所有 TEI"
fi

echo "停止 Nginx..."
if command -v nginx &> /dev/null; then
    sudo nginx -s stop 2>/dev/null || sudo systemctl stop nginx 2>/dev/null
    echo "系统 nginx 已停止"
fi

echo "所有服务已停止"