from sentence_transformers import SentenceTransformer
import torch
from typing import List, Union, Dict, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    model_id: str
    embedding_dim: int
    max_seq_length: int
    description: str



class ModelRegistry:

    MODELS: Dict[str, ModelConfig] = {
        "multilingual-e5-large": ModelConfig(
            name="Multilingual E5 Large",
            model_id="intfloat/multilingual-e5-large",
            embedding_dim=1024,
            max_seq_length=512,
            description="Best overall multilingual performance (560M params)"
        ),
        "bge-m3": ModelConfig(
            name="BGE-M3",
            model_id="BAAI/bge-m3",
            embedding_dim=1024,
            max_seq_length=8192,
            description="State-of-the-art multilingual, supports 100+ languages (568M params)"
        ),
        "gte-multilingual-base": ModelConfig(
            name="GTE Multilingual Base",
            model_id="Alibaba-NLP/gte-multilingual-base",
            embedding_dim=768,
            max_seq_length=8192,
            description="Efficient multilingual retrieval (305M params)"
        ),
        "jina-embeddings-v3": ModelConfig(
            name="Jina Embeddings v3",
            model_id="jinaai/jina-embeddings-v3",
            embedding_dim=1024,
            max_seq_length=8192,
            description="Task-specific embeddings with LoRA adapters (570M params)"
        ),
        "snowflake-arctic-embed-l-v2.0": ModelConfig(
            name="Arctic Embed 2.0 Large",
            model_id="Snowflake/snowflake-arctic-embed-l-v2.0",
            embedding_dim=1024,
            max_seq_length=2048,
            description="Maintains strong English + multilingual performance (568M params)"
        ),
        "labse": ModelConfig(
            name="LaBSE",
            model_id="sentence-transformers/LaBSE",
            embedding_dim=768,
            max_seq_length=512,
            description="Language-agnostic BERT, 109 languages"
        ),
        "use-multilingual": ModelConfig(
            name="Universal Sentence Encoder Multilingual",
            model_id="sentence-transformers/use-cmlm-multilingual",
            embedding_dim=768,
            max_seq_length=512,
            description="Lightweight multilingual model, 16 languages"
        ),
        "xlm-roberta-large": ModelConfig(
            name="XLM-RoBERTa model",
            model_id="FacebookAI/xlm-roberta-large",
            embedding_dim=1024,
            max_seq_length=512,
            description="Pre-trained on 2.5TB CommonCrawl data, 100 languages"
        )
    }

    @classmethod
    def list_models(cls) -> Dict[str, ModelConfig]:
        return cls.MODELS

    @classmethod
    def get_model_info(cls, model_key: str) -> Optional[ModelConfig]:
        return cls.MODELS.get(model_key)

    @classmethod
    def print_available_models(cls):
        print("\n" + "=" * 80)
        print("Available Multilingual Embedding Models")
        print("=" * 80)
        for key, config in cls.MODELS.items():
            print(f"\n[{key}]")
            print(f"  Name: {config.name}")
            print(f"  Model ID: {config.model_id}")
            print(f"  Embedding Dimension: {config.embedding_dim}")
            print(f"  Max Sequence Length: {config.max_seq_length}")
            print(f"  Description: {config.description}")
        print("\n" + "=" * 80 + "\n")


class EmbeddingPipeline:
    def __init__(self,
                 model_key: str = "multilingual-e5-large",
                 device: Optional[str] = None,
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the advanced embedding pipeline.

        Args:
            model_key: Key from ModelRegistry
            device: Device to use ('cuda', 'cpu', or None for auto)
            dtype: Data type (torch.float32 or torch.float16)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.dtype = dtype
        self.model_key = model_key

        # Get model configuration
        model_config = ModelRegistry.get_model_info(model_key)
        if model_config is None:
            raise ValueError(f"Model '{model_key}' not found in registry")

        self.model_config = model_config

        # Load model
        print(f"Loading {model_config.name} ({model_config.model_id})...")
        self.model = SentenceTransformer(model_config.model_id, device=device, trust_remote_code=True)

        # Optional: Convert to half precision for faster inference
        if dtype == torch.float16 and device == "cuda":
            self.model = self.model.half()
            print("Model converted to float16 for faster inference")

        self._log_model_info()

    def _log_model_info(self):
        """Log model information."""
        print(f"Model: {self.model_config.name}")
        print(f"Device: {self.device}")
        print(f"Embedding Dimension: {self.model_config.embedding_dim}")
        print(f"Max Sequence Length: {self.model_config.max_seq_length}")

    def encode(self,
               texts: Union[str, List[str]],
               batch_size: int = 32,
               normalize_embeddings: bool = True,
               show_progress_bar: bool = False,
               convert_to_numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode texts to embeddings.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            normalize_embeddings: Normalize to unit length
            show_progress_bar: Show progress bar
            convert_to_numpy: Return as numpy array (True) or tensor (False)

        Returns:
            Embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=(not convert_to_numpy)
        )

        return embeddings

    def encode_with_task(self,
                         texts: Union[str, List[str]],
                         task: str = "default",
                         batch_size: int = 32) -> np.ndarray:
        """
        Encode texts with task-specific optimization (for models that support it).

        Args:
            texts: Text(s) to encode
            task: Task type ('retrieval_query', 'retrieval_document', 'default')
            batch_size: Batch size

        Returns:
            Embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_tensor=False
        )

        return embeddings

    def batch_encode(self,
                     texts_list: List[List[str]],
                     batch_size: int = 32) -> List[np.ndarray]:
        """
        Encode multiple batches of texts.

        Args:
            texts_list: List of text lists
            batch_size: Batch size per list

        Returns:
            List of embedding arrays
        """
        results = []
        for texts in texts_list:
            embeddings = self.encode(texts, batch_size=batch_size)
            results.append(embeddings)
        return results

    def switch_model(self, new_model_key: str):
        """
        Dynamically switch to a different model.

        Args:
            new_model_key: Key from ModelRegistry
        """
        print(f"\nSwitching model from '{self.model_key}' to '{new_model_key}'...")

        # Get new model configuration
        new_config = ModelRegistry.get_model_info(new_model_key)
        if new_config is None:
            raise ValueError(f"Model '{new_model_key}' not found in registry")

        # Load new model
        self.model_key = new_model_key
        self.model_config = new_config
        self.model = SentenceTransformer(new_config.model_id, device=self.device, trust_remote_code=True)

        if self.dtype == torch.float16 and self.device == "cuda":
            self.model = self.model.half()

        self._log_model_info()

    def get_model_info(self) -> Dict:
        return {
            "name": self.model_config.name,
            "model_id": self.model_config.model_id,
            "embedding_dim": self.model_config.embedding_dim,
            "max_seq_length": self.model_config.max_seq_length,
            "device": self.device,
            "dtype": str(self.dtype)
        }
