from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def calculate_metrics(preds: List[int], labels: List[int], languages: List[str]) -> Dict:
    """
    Calculates and returns evaluation metrics for predictions and labels.

    This function computes overall metrics as well as metrics specific to languages,
    given a list of predictions, corresponding true labels,
    and the languages associated with each prediction-label pair.
    Metrics include macro-averaged precision, recall, and F1 score.

    Args:
        preds (List[int]): List of predicted labels.
        labels (List[int]): List of true labels.
        languages (List[str]): List of languages corresponding to the input data.

    Returns:
        Dict: A dictionary containing overall and per-language metrics. Keys include:
            - "overall": Dictionary with overall macro-averaged precision, recall, and F1 score.
            - Per-language keys: Each contains macro-averaged precision, recall, and F1 score
              specific to that language.
    """
    metrics = {}

    macro_precision = precision_score(labels, preds, average="macro", zero_division=0)
    macro_recall = recall_score(labels, preds, average="macro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    metrics["overall"] = {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }

    unique_langs = set(languages)
    for lang in unique_langs:
        lang_mask = np.array([l == lang for l in languages])
        lang_preds = np.array(preds)[lang_mask]
        lang_labels = np.array(labels)[lang_mask]

        if len(lang_labels) > 0:
            lang_precision = precision_score(
                lang_labels, lang_preds, average="macro", zero_division=0
            )
            lang_recall = recall_score(
                lang_labels, lang_preds, average="macro", zero_division=0
            )
            lang_f1 = f1_score(lang_labels, lang_preds, average="macro", zero_division=0)

            metrics[lang] = {
                "macro_precision": lang_precision,
                "macro_recall": lang_recall,
                "macro_f1": lang_f1,
            }

    return metrics


def calculate_weights(df: pd.DataFrame) -> Tuple[Dict, Dict, float]:
    """
    Calculates weights based on the distribution of labels and languages in the given
    DataFrame and computes the positive weight for binary cross-entropy.

    This function computes the weights for each unique label and language to
    address class imbalance. Additionally, it calculates a positive weight
    for use in binary cross-entropy loss based on the ratio of negative to
    positive samples.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing at least two columns:
            "label" (int) and "lang". The "label" column represents binary class
            labels (e.g., 0 and 1), while the "lang" column represents categorical
            language identifiers.

    Returns:
        Tuple[Dict, Dict, float]: A tuple containing the following:
            - A dictionary mapping each unique label to its computed weight.
            - A dictionary mapping each unique language to its computed weight.
            - A float representing the positive weight for binary cross-entropy.
    """
    label_counts = df["label"].value_counts()
    total_samples = len(df)
    label_weights = {
        label: total_samples / (len(label_counts) * count)
        for label, count in label_counts.items()
    }

    lang_counts = df["lang"].value_counts()
    language_weights = {
        lang: 1.0 / (count / total_samples) for lang, count in lang_counts.items()
    }

    language_weights_values = list(language_weights.values())
    sum_weights = sum(language_weights_values)
    language_weights = {
        lang: (weight / sum_weights) * len(language_weights)
        for lang, weight in language_weights.items()
    }

    neg_count = (df["label"] == 0).sum()
    pos_count = (df["label"] == 1).sum()
    pos_weight = neg_count / pos_count

    logger.info(f"Label weights: {label_weights}")
    logger.info(f"Language weights: {language_weights}")
    logger.info(f"Pos weight (for BCE): {pos_weight:.4f}")

    return label_weights, language_weights, pos_weight
