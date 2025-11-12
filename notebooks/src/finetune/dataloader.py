from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MultilingualDataset(Dataset):
    """
    A dataset class for handling multilingual text data.

    The `MultilingualDataset` class is designed for preparing multilingual text data
    for machine learning models. It supports text tokenization, management of labels,
    and handling of multiple languages. Each instance of this dataset provides
    preprocessed data compatible with PyTorch models.

    Attributes:
        texts (List[str]): A list of text samples.
        labels (List[int]): A list of integer labels corresponding to the text samples.
        languages (List[str]): A list of languages corresponding to the text samples.
        tokenizer: A tokenizer instance for processing the text samples.
        max_length (int): The maximum length for tokenized sequences. Defaults to 128.
    """
    def __init__(
            self,
            texts: List[str],
            labels: List[int],
            languages: List[str],
            tokenizer,
            max_length: int = 128,
    ):
        """
        Initializes a class instance with text data, associated labels, supported languages,
        a tokenizer for processing the texts, and a maximum length for tokenized sequences.

        This method is responsible for setting up the necessary variables for working with
        textual data, ensuring appropriate configurations for tokenization, and enabling
        language support.

        Args:
            texts (List[str]): A list of textual data to be processed.
            labels (List[int]): A list of integer labels corresponding to the texts.
            languages (List[str]): A list of languages associated with the texts.
            tokenizer: An object or function used to tokenize text inputs.
            max_length (int): An optional parameter specifying the maximum length for
                tokenized sequences. Defaults to 128.
        """
        self.texts = texts
        self.labels = labels
        self.languages = languages
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        language = self.languages[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "language": language,
        }



class StratifiedMultilingualSplitter:
    """
    Provides functionality for creating stratified train-validation splits for
    multilingual datasets while maintaining distribution balance across
    languages and labels.

    This class allows generating train-validation splits of a given dataset
    while preserving the balance in both label and language distributions. It
    is particularly useful for scenarios involving multilingual datasets with
    imbalanced class distributions.

    Methods:
        create_stratified_splits: Static method for generating stratified train
        and validation splits from the input dataframe.
    """
    @staticmethod
    def create_stratified_splits(
            df: pd.DataFrame,
            n_splits: int = 5,
            train_ratio: float = 0.8,
            seed: int = 42,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:

        splits = []

        df["strata"] = df["lang"] + "_" + df["label"].astype(str)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for fold_idx, (train_val_idx, _) in enumerate(skf.split(df, df["strata"])):
            train_val_df = df.iloc[train_val_idx].reset_index(drop=True)

            # Further split train_val into train and val maintaining stratification
            train_size = int(len(train_val_df) * train_ratio)

            skf_inner = StratifiedKFold(
                n_splits=int(1 / (1 - train_ratio)), shuffle=True, random_state=seed
            )
            inner_splits = list(
                skf_inner.split(train_val_df, train_val_df["strata"])
            )
            train_idx, val_idx = inner_splits[0]

            train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
            val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

            # Shuffle heavily
            train_df = train_df.sample(frac=1, random_state=seed + fold_idx).reset_index(
                drop=True
            )
            val_df = val_df.sample(frac=1, random_state=seed + fold_idx).reset_index(
                drop=True
            )

            splits.append((train_df, val_df))

            logger.info(
                f"Fold {fold_idx}: Train={len(train_df)}, Val={len(val_df)}"
            )
            logger.info(
                f"  Train label dist: {train_df['label'].value_counts().to_dict()}"
            )
            logger.info(
                f"  Train lang dist: {train_df['lang'].value_counts().to_dict()}"
            )

        df.drop("strata", axis=1, inplace=True)
        return splits



class DynamicUndersamplingSampler:
    """
    Handles dynamic undersampling of a dataset for creating balanced samples.

    This class is designed to perform dynamic undersampling on an imbalanced dataset.
    Its primary purpose is to construct balanced sub-samples of the dataset during
    training by selecting equal numbers of samples from both minority and majority
    classes. The minority class is taken as is, while the majority class is dynamically
    undersampled with consistent randomness across epochs.

    Attributes:
        df (pd.DataFrame): Input dataset containing features and labels.
        minority_class (int): Label representing the minority class in the dataset.
        seed (int): Seed for random number generation to ensure reproducibility.
    """
    def __init__(
            self,
            df: pd.DataFrame,
            minority_class: int = 1,
            seed: int = 42,
    ):
        self.df = df.reset_index(drop=True)
        self.minority_class = minority_class
        self.seed = seed

        self.minority_indices = self.df[self.df["label"] == minority_class].index.tolist()
        self.majority_indices = self.df[
            self.df["label"] != minority_class
            ].index.tolist()

        self.minority_count = len(self.minority_indices)
        self.majority_count = int(self.minority_count * 2.0)  # len(self.majority_indices)

        logger.info(
            f"Undersampling: Minority={self.minority_count}, Majority={self.majority_count}"
        )

    def get_balanced_indices(self, epoch: int) -> List[int]:
        """
        Generates a list of balanced indices by combining minority and a randomly sampled
        subset of majority indices. The resulting list is shuffled to maintain a uniform
        distribution.

        Args:
            epoch (int): The current training epoch, used to seed the random number
                generator for reproducibility.

        Returns:
            List[int]: A shuffled list of balanced indices combining minority indices and
            randomly sampled majority indices.
        """
        rng = np.random.RandomState(self.seed + epoch)
        sampled_majority = rng.choice(
            self.majority_indices, size=self.minority_count, replace=False
        )

        balanced_indices = list(self.minority_indices) + list(sampled_majority)
        rng.shuffle(balanced_indices)

        return balanced_indices
