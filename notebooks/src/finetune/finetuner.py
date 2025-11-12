import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import json
import logging
from pathlib import Path
from typing import List, Tuple
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, get_linear_schedule_with_warmup


from .dataloader import MultilingualDataset, StratifiedMultilingualSplitter, DynamicUndersamplingSampler
from .train import train_epoch, validate
from .utils import calculate_metrics, calculate_weights

warnings.filterwarnings("ignore")


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_model(Config, num_frozen_layers: int = None):
    """
    Configures and initializes a Transformer-based model for sequence classification.

    This function sets up the model with a specified number of frozen layers in the
    transformer architecture, initializes model weights for the classification head, and
    ensures the appropriate layers are trainable or frozen as needed. The tokenizer is
    also instantiated for preprocessing input text. The function logs details about the
    number of trainable parameters and the structure of the model after configuration.

    Args:
        Config: A configuration object containing model settings, including the model
            name, number of labels, device to use, and the default number of frozen layers.
        num_frozen_layers (int, optional): The number of encoder layers in the transformer
            model to freeze during training. Defaults to the value specified in the
            `Config.NUM_FROZEN_LAYERS`.

    Returns:
        tuple: A tuple containing the initialized and configured model
            (`XLMRobertaForSequenceClassification`) and tokenizer (`XLMRobertaTokenizer`).
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    if num_frozen_layers is None:
        num_frozen_layers = Config.NUM_FROZEN_LAYERS

    tokenizer = XLMRobertaTokenizer.from_pretrained(Config.MODEL_NAME)
    model = XLMRobertaForSequenceClassification.from_pretrained(
        Config.MODEL_NAME, num_labels=Config.NUM_LABELS
    )

    model = model.to(Config.DEVICE)

    torch.nn.init.xavier_uniform_(model.classifier.dense.weight)
    torch.nn.init.zeros_(model.classifier.dense.bias)
    torch.nn.init.xavier_uniform_(model.classifier.out_proj.weight)
    torch.nn.init.zeros_(model.classifier.out_proj.bias)

    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False

    num_total_layers = len(model.roberta.encoder.layer)
    for idx in range(min(num_frozen_layers, num_total_layers)):
        for param in model.roberta.encoder.layer[idx].parameters():
            param.requires_grad = False

    if num_frozen_layers >= num_total_layers and model.roberta.pooler is not None:
        for param in model.roberta.pooler.parameters():
            param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    logger.info(f"Froze: Embeddings + First {min(num_frozen_layers, num_total_layers)} Encoder Layers")
    logger.info("Trainable: Classification Head + Remaining Encoder Layers")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )

    return model, tokenizer


class ModelCheckpointManager:
    """
    Manages saving and keeping track of model checkpoints based on their performance scores.

    This class facilitates saving model checkpoints while limiting the number of saved checkpoints
    to a defined maximum. It maintains a list of the best-performing models and their associated
    checkpoints, sorted by their performance scores. Lower-performing models are deleted as new
    higher-performing models are added.

    Attributes:
        max_models (int): Maximum number of checkpoints to retain.
        output_dir (Path): Directory where checkpoints are stored.
    """
    def __init__(self, max_models: int = 2, output_dir: str = "./models"):
        """
        Initializes the instance with specified parameters and prepares the output directory
        for storing model information.

        Args:
            max_models (int): Maximum number of models to keep.
            output_dir (str): Directory path where model data will be stored.
        """
        self.max_models = max_models
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_models = []  # List of (score, path, epoch, fold)

    def save_checkpoint(
            self, model, score: float, epoch: int, fold: int
    ) -> bool:
        """
        Saves a model checkpoint if it meets the criteria for the best models. Maintains a list of
        best checkpoints up to a specified maximum. Automatically deletes the worst-performing model
        checkpoint when the limit is exceeded.

        Args:
            model: The model instance to save the state from.
            score: The evaluation score (float) representing the model's performance.
            epoch: The current epoch (int) during training.
            fold: The current fold (int) in cross-validation.

        Returns:
            bool: True if the checkpoint has been successfully saved.
        """
        checkpoint_name = f"fold_{fold}_epoch_{epoch}_f1_{score:.4f}.pt"
        checkpoint_path = self.output_dir / checkpoint_name

        self.best_models.append((score, checkpoint_path, epoch, fold))
        self.best_models.sort(reverse=True, key=lambda x: x[0])

        if len(self.best_models) > self.max_models:
            worst_score, worst_path, worst_epoch, worst_fold = self.best_models.pop()
            if worst_path.exists():
                worst_path.unlink()
                logger.info(f"Deleted checkpoint: {worst_path}")

        torch.save(model.state_dict(), checkpoint_path)
        logger.info(
            f"Saved checkpoint: {checkpoint_path} (F1: {score:.4f}, Fold: {fold}, Epoch: {epoch})"
        )

        return True

    def get_best_models(self) -> List[Tuple]:
        return [(score, path, epoch, fold) for score, path, epoch, fold in self.best_models]


def main(df: pd.DataFrame, Config):
    """
    Executes the main fine-tuning pipeline for a multilingual model.

    This function orchestrates the process of fine-tuning a model using the provided
    dataset and configuration. The steps include data preparation, stratified splitting
    of the dataset into training and validation sets, model setup, training-loop with
    early stopping, weight calculation, dynamic undersampling (if applicable), optimizer
    and scheduler setup, and metric evaluation. The best models are saved periodically
    during the training process, and the results are logged and stored.

    Args:
        df (pd.DataFrame): The input dataset containing text, labels, and potentially
            language data. It is expected to be a pandas DataFrame with necessary
            columns such as 'text', 'label', and 'lang'.
        Config: A configuration object containing various parameters required for
            the pipeline, such as output directories, training hyperparameters,
            and model saving preferences.

    Raises:
        OSError: If saving checkpoint files or results fails.
        ValueError: If validation set contains no positive samples or training set
            has very few positive samples.
    """
    logger.info("=" * 80)
    logger.info("Starting Fine-tuning Pipeline")
    logger.info("=" * 80)

    Path(Config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(Config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    config_path = Path(Config.RESULTS_DIR) / "config.json"
    with open(config_path, "w") as f:
        json.dump(Config.__dict__, f, indent=2, default=str)

    splits = StratifiedMultilingualSplitter.create_stratified_splits(
        df,
        n_splits=Config.N_SPLITS,
        train_ratio=Config.TRAIN_RATIO,
        seed=SEED,
    )

    all_results = []
    checkpoint_manager = ModelCheckpointManager(
        max_models=Config.MAX_MODELS_TO_SAVE,
        output_dir=str(Path(Config.OUTPUT_DIR) / "checkpoints"),
    )

    for fold_idx, (train_df, val_df) in enumerate(splits):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Fold {fold_idx + 1}/{Config.N_SPLITS}")
        logger.info(f"{'=' * 80}")

        print(f"\nFOLD {fold_idx}:")
        train_pos = (train_df['label'] == 1).sum()
        val_pos = (val_df['label'] == 1).sum()
        print(f"  Train: {train_pos} positive samples")
        print(f"  Val:   {val_pos} positive samples")

        if val_pos == 0:
            print("PROBLEM: No positive samples in validation!")
        if train_pos < 10:
            print("PROBLEM: Very few positive samples in training!")

        model, tokenizer = setup_model(Config, num_frozen_layers=Config.NUM_FROZEN_LAYERS)

        label_weights, language_weights, pos_weight = calculate_weights(train_df)

        train_dataset = MultilingualDataset(
            texts=train_df["text"].tolist(),
            labels=train_df["label"].tolist(),
            languages=train_df["lang"].tolist(),
            tokenizer=tokenizer,
            max_length=Config.MAX_LENGTH,
        )

        val_dataset = MultilingualDataset(
            texts=val_df["text"].tolist(),
            labels=val_df["label"].tolist(),
            languages=val_df["lang"].tolist(),
            tokenizer=tokenizer,
            max_length=Config.MAX_LENGTH,
        )

        # Setup optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # majority class count = 2 * minority class count, hence the total step is 3
        balanced_train_size = 3 * len(train_df[train_df["label"] == 1])
        total_steps = (
                balanced_train_size
                // (Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS)
                * Config.NUM_EPOCHS
        )
        warmup_steps = int(Config.WARMUP_RATIO * total_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training loop
        best_val_f1 = 0
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            logger.info(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")

            if Config.DYNAMIC_UNDERSAMPLE:
                undersampler = DynamicUndersamplingSampler(
                    train_df, minority_class=1, seed=SEED
                )
                balanced_indices = undersampler.get_balanced_indices(epoch)
                train_subset_df = train_df.iloc[balanced_indices].reset_index(
                    drop=True
                )
            else:
                train_subset_df = train_df

            train_dataset_epoch = MultilingualDataset(
                texts=train_subset_df["text"].tolist(),
                labels=train_subset_df["label"].tolist(),
                languages=train_subset_df["lang"].tolist(),
                tokenizer=tokenizer,
                max_length=Config.MAX_LENGTH,
            )

            train_loader = DataLoader(
                train_dataset_epoch,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
            )

            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                label_weights,
                language_weights,
                pos_weight,
                Config
            )

            val_loss, val_preds, val_labels, val_languages = validate(
                model, val_loader, Config
            )

            val_metrics = calculate_metrics(val_preds, val_labels, val_languages)

            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Overall - Precision: {val_metrics['overall']['macro_precision']:.4f}, "
                        f"Recall: {val_metrics['overall']['macro_recall']:.4f}, "
                        f"F1: {val_metrics['overall']['macro_f1']:.4f}")

            for lang in sorted([k for k in val_metrics.keys() if k != "overall"]):
                logger.info(f"{lang} - Precision: {val_metrics[lang]['macro_precision']:.4f}, "
                            f"Recall: {val_metrics[lang]['macro_recall']:.4f}, "
                            f"F1: {val_metrics[lang]['macro_f1']:.4f}")

            epoch_result = {
                "fold": fold_idx,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            epoch_result.update({f"overall_{k}": v for k, v in val_metrics["overall"].items()})
            for lang, metrics in val_metrics.items():
                if lang != "overall":
                    for metric_name, value in metrics.items():
                        epoch_result[f"{lang}_{metric_name}"] = value

            all_results.append(epoch_result)

            # Early stopping and checkpoint saving
            current_f1 = val_metrics["overall"]["macro_f1"]
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                patience_counter = 0
                checkpoint_manager.save_checkpoint(model, current_f1, epoch, fold_idx)
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch} (patience: {Config.PATIENCE})"
                    )
                    break

    results_df = pd.DataFrame(all_results)
    results_path = Path(Config.RESULTS_DIR) / "training_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to: {results_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Best Models Saved:")
    logger.info("=" * 80)
    for score, path, epoch, fold in checkpoint_manager.get_best_models():
        logger.info(f"Fold {fold}, Epoch {epoch}: F1={score:.4f} -> {path}")
