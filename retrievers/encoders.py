import abc
from typing import List, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def load_model(model_name: str) -> SentenceTransformer:
    """
    Loads a SentenceTransformer model based on the model name and handles specific model parameters.
    """
    print(f"Loading SentenceTransformer model: {model_name}...")

    # Check if a CUDA device is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Will use device: {device}")

    # Set special loading parameters for specific models
    model_kwargs = {}
    if "Qwen3" in model_name:
        # model_kwargs = {"device_map": "auto"}
        # model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
        print("Detected Qwen3 model, enabling special loading parameters for https://huggingface.co/Qwen/Qwen3-Embedding-8B.")

    model = SentenceTransformer(model_name, device=device, **model_kwargs)
    print("Model loaded successfully.")
    return model

class BaseEncoder(abc.ABC):
    """
    Encapsulates model-specific encoding logic, such as adding prompts or passing special parameters.
    """
    def __init__(self, model: SentenceTransformer, **kwargs):
        self.model = model
        self.encode_kwargs = kwargs

    @abc.abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encodes one or more texts into vectors.

        Args:
            texts (Union[str, List[str]]): The text or list of texts to encode.

        Returns:
            np.ndarray: The generated vector(s).
        """
        pass

class DefaultEncoder(BaseEncoder):
    """
    Default encoder that directly calls model.encode without any special processing.
    Suitable for simple models like all-MiniLM-L6-v2.
    """
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, **self.encode_kwargs)

class E5Encoder(BaseEncoder):
    """
    Encoder suitable for the E5 series of models.
    Automatically adds the "query: " prefix to each query.
    """
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            prefixed_texts = f"query: {texts}"
        else:
            prefixed_texts = [f"query: {q}" for q in texts]
        
        return self.model.encode(prefixed_texts, normalize_embeddings=True, **self.encode_kwargs)

class QwenEncoder(BaseEncoder):
    """
    Encoder suitable for Qwen, BGE, or other models that use the prompt_name parameter.
    """
    def __init__(self, model: SentenceTransformer, **kwargs):
        # Force set prompt_name if the user has not provided it
        kwargs.setdefault('prompt_name', 'query')
        super().__init__(model, **kwargs)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        # self.encode_kwargs already contains prompt_name='query'
        return self.model.encode(texts, normalize_embeddings=True, **self.encode_kwargs)


def build_encoder(model_name: str, model: SentenceTransformer, **kwargs) -> BaseEncoder:
    """
    Returns an appropriate Encoder instance based on the model name.
    """
    model_name_lower = model_name.lower()
    if 'e5' in model_name_lower:
        print(f"Detected E5 model ('{model_name}'), using E5Encoder.")
        return E5Encoder(model, **kwargs)
    elif 'qwen' in model_name_lower or 'bge' in model_name_lower:
        print(f"Detected Qwen/BGE model ('{model_name}'), using QwenEncoder.")
        return QwenEncoder(model, **kwargs)
    else:
        print(f"No specific model matched ('{model_name}'), using DefaultEncoder.")
        return DefaultEncoder(model, **kwargs)