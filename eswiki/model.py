import torch
from sentence_transformers import SentenceTransformer


# 在 model.py 文件中
import torch
from sentence_transformers import SentenceTransformer

def load_model(model_name: str) -> SentenceTransformer:
    """
    加载 SentenceTransformer 模型，并处理特定模型的参数。
    【已修改】为 Qwen3 模型增加了 torch_dtype 参数。
    """
    print(f"Loading SentenceTransformer model: {model_name}...")

    # 检查是否有可用的 CUDA 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Will use device: {device}")

    # 为特定模型设置特殊的加载参数
    if "Qwen3" in model_name:
        print("Detected Qwen3 model, enabling special loading parameters for https://huggingface.co/Qwen/Qwen3-Embedding-8B.")
        
        # 定义模型加载参数
        model_kwargs = {
            "attn_implementation": "flash_attention_2", 
            "device_map": "auto",
            "torch_dtype": torch.float16
        }
        
        # 定义分词器参数
        tokenizer_kwargs = {"padding_side": "left"}

        # 加载模型
        model = SentenceTransformer(
            model_name, 
            model_kwargs=model_kwargs, 
            tokenizer_kwargs=tokenizer_kwargs
        )
    else:
        model = SentenceTransformer(model_name, device=device)

    print("Model loaded successfully.")
    return model