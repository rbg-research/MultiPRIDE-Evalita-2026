from sentence_transformers import SentenceTransformer
import torch
from typing import List, Union, Dict, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Represents the configuration for a machine learning model.

    This class encapsulates parameters and metadata required for configuring
    a machine learning model. It includes information such as the model's
    name, identifier, embedding dimensions, maximum sequence length, and
    a descriptive summary of the model. The attributes of this class are
    used to define the overall configuration and provide context for how
    the model is expected to function.

    Attributes:
        name (str): Name of the model.
        model_id (str): Unique identifier for the model.
        embedding_dim (int): Size of the embedding dimension used by the model.
        max_seq_length (int): Maximum sequence length supported by the model.
        description (str): Detailed description of the model's purpose and behavior.
    """
    name: str
    model_id: str
    embedding_dim: int
    max_seq_length: int
    description: str



class ModelRegistry:
    """
    Manages and provides access to a registry of multilingual embedding models.

    This class serves as a centralized model registry, containing metadata for various
    pre-trained multilingual embedding models. It provides functionality to list available
    models, retrieve specific model information, and display details about the registered
    models.

    Attributes:
        MODELS (Dict[str, ModelConfig]): A dictionary where the keys are model identifiers
            and the values are corresponding ModelConfig instances describing the models.
    """
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
        """
        Lists all available models and their configurations.

        Returns:
            Dict[str, ModelConfig]: A dictionary where keys are model names and
            values are their corresponding configurations.
        """
        return cls.MODELS

    @classmethod
    def get_model_info(cls, model_key: str) -> Optional[ModelConfig]:
        """
        Retrieves model configuration information for a given model key.

        This method accesses the class-level dictionary of available models and
        returns the configuration associated with the specified model key. If the
        model key does not exist in the dictionary, the method returns None.

        Args:
            model_key (str): The key identifying the model whose configuration is
                to be retrieved.

        Returns:
            Optional[ModelConfig]: The configuration of the model corresponding to
                the provided key, or None if the key does not exist.
        """
        return cls.MODELS.get(model_key)

    @classmethod
    def print_available_models(cls):
        """
        Prints and displays the available multilingual embedding models. This method outputs
        a formatted list of models containing their key properties such as name, model
        ID, embedding dimension, maximum sequence length, and description.

        Returns:
            None
        """
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
    """
    Advanced embedding pipeline for generating text embeddings.

    This class provides a flexible and efficient interface for working with text embeddings
    using pretrained models. It supports encoding single or multiple texts, batching,
    task-specific optimizations, and dynamically switching models. Additionally, the class
    handles model loading, device management, and supports half-precision inference for
    GPU acceleration.

    Attributes:
        device (str): The device used for computation ('cuda' or 'cpu').
        dtype (torch.dtype): The data type for computations (e.g., torch.float32 or torch.float16).
        model_key (str): The identifier for the pretrained model in the registry.
        model_config (Dict): The configuration details of the loaded model, including
            name, model ID, embedding dimensions, and maximum sequence length.
        model (SentenceTransformer): The loaded transformer model used for generating embeddings.
    """
    def __init__(self,
                 model_key: str = "multilingual-e5-large",
                 device: Optional[str] = None,
                 dtype: torch.dtype = torch.float32):
        """
        Initializes an instance of the class to load and configure a specified model
        from the model registry.

        Args:
            model_key (str): The key identifying the model to load. Defaults to
                "multilingual-e5-large".
            device (Optional[str]): The device to use for inference ("cpu" or "cuda").
                If None, automatically sets to "cuda" if available, otherwise "cpu".
            dtype (torch.dtype): The data type to use for model computations. Defaults
                to torch.float32.

        Raises:
            ValueError: If the specified model key is not found in the model registry.
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
        """Logs the model's information such as name, device, embedding dimension,
        and maximum sequence length.
        """
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
        Encodes a given text or list of texts into dense vector embeddings using the model, with optional
        normalization and conversion between tensor or numpy array formats.

        The method processes batches of texts to compute dense vector representations, which can
        potentially be normalized depending on the provided parameters. Progress indicators and
        format selection for the embeddings are also customizable.

        Args:
            texts (Union[str, List[str]]): A single string or a list of strings to be encoded into
                dense vector embeddings.
            batch_size (int): The number of texts to process in a single batch.
            normalize_embeddings (bool): Whether to normalize embeddings to unit length. Defaults to True.
            show_progress_bar (bool): Whether to display a progress bar for the encoding process.
                Defaults to False.
            convert_to_numpy (bool): If True, returns embeddings as `np.ndarray`. If False, returns
                embeddings as `torch.Tensor`. Defaults to True.

        Returns:
            Union[np.ndarray, torch.Tensor]: Encoded dense vector embeddings of the input text(s) in the
                specified format (numpy array or torch tensor).
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
        Encodes input text data into embeddings using the specified task.

        This method processes the provided text(s) using the model and generates
        embeddings based on the specified task. It supports batch processing,
        normalizing embeddings, and ensures conversion to a NumPy array format
        for ease of use in downstream tasks.

        Args:
            texts (Union[str, List[str]]): A single string or a list of strings
                to be encoded into embeddings.
            task (str): Specifies the task type for encoding. Defaults to "default".
                Different tasks may use different configurations or fine-tuned
                embeddings.
            batch_size (int): Number of texts to process in a single batch.
                Defaults to 32.

        Returns:
            np.ndarray: The computed embeddings as a NumPy array.
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
        Encodes a batch of text lists into their corresponding numerical embeddings.

        This method processes a batch of lists of text strings and generates numerical
        embeddings for each list of texts using the specified batch size. The resulting
        embeddings are returned as a list of numpy arrays.

        Args:
            texts_list: A list of lists, where each inner list contains strings that
                represent text to be encoded.
            batch_size: An integer specifying the number of texts to process in a
                single batch. Defaults to 32.

        Returns:
            A list of numpy.ndarray objects, where each array contains the numerical
            embeddings for the corresponding list of texts from the input.
        """
        results = []
        for texts in texts_list:
            embeddings = self.encode(texts, batch_size=batch_size)
            results.append(embeddings)
        return results

    def switch_model(self, new_model_key: str):
        """
        Switches the current model to a new one, based on the provided model key.

        This method updates the existing model to a new model corresponding to the
        `new_model_key` by fetching its configuration from the model registry. It
        loads the new model, updates the internal model configuration, and adjusts
        the model datatype and device. If the `new_model_key` is not found in the model
        registry, it raises a `ValueError`.

        Args:
            new_model_key (str): The key identifying the new model to be loaded.
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
        """
        Retrieves and returns information about the model configuration.

        This method compiles key attributes of the model, such as its name, identifier,
        embedding dimension, maximum sequence length, operating device, and data type,
        into a dictionary format.

        Returns:
            Dict: A dictionary containing the following key-value pairs:
                - 'name': Name of the model.
                - 'model_id': Unique identifier of the model.
                - 'embedding_dim': Embedding dimension size of the model.
                - 'max_seq_length': Maximum sequence length supported by the model.
                - 'device': Device on which the model is configured to run.
                - 'dtype': Data type associated with the model.
        """
        return {
            "name": self.model_config.name,
            "model_id": self.model_config.model_id,
            "embedding_dim": self.model_config.embedding_dim,
            "max_seq_length": self.model_config.max_seq_length,
            "device": self.device,
            "dtype": str(self.dtype)
        }
