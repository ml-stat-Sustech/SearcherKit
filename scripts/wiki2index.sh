export CUDA_VISIBLE_DEVICES=$1
export PATH=~/.conda/envs/webagent/bin:$PATH
#''
# Huggingface settings
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="REMOVED_REVOKED_SECRET"
export HF_HOME=/mnt/sharedata/ssd_large/common/LLMs
export HF_DATASETS_CACHE=/mnt/sharedata/ssd_large/common/datasets/

cd eswiki

# vanilla
# python wiki2index_links.py \
#     --index_name wiki20251001

# MODEL_NAME=e5-base-v2
# python wiki2index_links.py \
#     --model_name ${HF_HOME}/${MODEL_NAME} \
#     --embedding_dim 768 \
#     --prompt_strategy e5 \
#     --index_name wiki20251001_${MODEL_NAME,,} \
#     --cpu_batch_size 1000 \
#     --gpu_batch_size 1000 \
#     --dense-vector

# MODEL_NAME=Qwen3-Embedding-0.6B
# python wiki2index_links.py \
#     --model_name ${HF_HOME}/${MODEL_NAME} \
#     --embedding_dim 1024 \
#     --prompt_strategy none \
#     --index_name wiki20251001_${MODEL_NAME,,} \
#     --cpu_batch_size 200 \
#     --gpu_batch_size 20 \
#     --dense-vector

MODEL_NAME=Qwen3-Embedding-4B
python wiki2index_links.py \
    --model_name ${HF_HOME}/${MODEL_NAME} \
    --embedding_dim 2560 \
    --prompt_strategy none \
    --index_name wiki20251001_${MODEL_NAME,,} \
    --cpu_batch_size 500 \
    --gpu_batch_size 4 \
    --dense-vector

# MODEL_NAME=Qwen3-Embedding-8B
# python wiki2index_links.py \
#     --model_name ${HF_HOME}/${MODEL_NAME} \
#     --embedding_dim 4096 \
#     --prompt_strategy none \
#     --index_name wiki20251001_${MODEL_NAME,,} \
#     --cpu_batch_size 1000 \
#     --dense-vector

# multi-cpu version
# MODEL_NAME=Qwen3-Embedding-0.6B
# python wiki2index_links_multicpu.py \
#     --model_name ${HF_HOME}/${MODEL_NAME} \
#     --embedding_dim 1024 \
#     --prompt_strategy none \
#     --index_name wiki20251001_${MODEL_NAME,,} \
#     --cpu_batch_size 500 \
#     --gpu_batch_size 8 \
#     --dense-vector
cd ..