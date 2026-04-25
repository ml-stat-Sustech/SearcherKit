#!/bin/bash

# ====== Proxy设置 ======
# export HF_ENDPOINT="https://hf-mirror.com"  # 如果需要换镜像，可以取消注�?
# export PATH=~/.conda/envs/searchagent/bin:$PATH

# # ====== Huggingface相关环境 ======
# export HF_TOKEN="REMOVED_REVOKED_SECRET"
# export HF_HOME=/mnt/sharedata/ssd_large/common/LLMs/
# export HF_DATASETS_CACHE=/mnt/sharedata/ssd_large/common/datasets/

# # ====== 定义模型列表 ======
# models=(
#     "Qwen/Qwen3-Embedding-8B"
# )


# # ====== 定义保存根目�?======
# base_dir="/mnt/sharedata/ssd_large/common/LLMs"

# # ====== 循环下载每一个模�?======
# for model in "${models[@]}"; do
#     # 自动提取模型短名作为文件夹名
#     model_name=$(basename "$model")
#     local_dir="${base_dir}/${model_name}"

#     echo "🚀 开始下�? $model"
#     echo "💾 保存�? $local_dir"

#     huggingface-cli download "$model" \
#         --local-dir "$local_dir" \
#         --local-dir-use-symlinks False \
#         --resume-download \
#         --token "$HF_TOKEN"

#     echo "�?下载完成: $model"
#     echo "-------------------------------------------"
# done

modelscope download --model Qwen/Qwen3-Embedding-0.6B --local_dir /mnt/sharedata/ssd_large/common/LLMs/Qwen3-Embedding-0.6B