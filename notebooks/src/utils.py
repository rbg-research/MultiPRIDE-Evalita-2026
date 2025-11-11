import os
import random
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, f1_score, balanced_accuracy_score, classification_report,
                             confusion_matrix)
from typing import Dict

SEED = 42

def set_all_seeds():
    """
    Sets random seeds for Python, NumPy, and PyTorch to ensure determinism across various operations
    and reproducibility in experiments.

    This function configures the random number generators in Python's random module, NumPy, and PyTorch.
    It also ensures deterministic behavior in PyTorch by disabling cuDNN auto-tuning
    and enabling deterministic algorithms.

    Args:
        None

    Raises:
        None
    """
    # Python's built-in random module

    seed = SEED

    random.seed(seed)
    
    # Environment variable for Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch GPU (all devices)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Disable PyTorch's cuDNN auto-tuning (ensures deterministic behavior)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Can be slower but more reproducible
    
    # Optional: Use deterministic algorithms (PyTorch 1.9+)
    torch.use_deterministic_algorithms(True)
    
    print(f"✓ All random seeds set to {seed}")


def calculate_metrics(y_true, y_pred):
    """
    Calculates evaluation metrics for classification tasks.

    This function computes four different evaluation metrics: accuracy,
    precision, recall, and F1-score, using the provided ground truth labels
    and predicted labels. These metrics are calculated on a macro-average
    basis, making them suitable for evaluating multi-class classification
    scenarios. Zero-division handling is explicitly defined as 0.

    Args:
        y_true: Ground truth (correct) labels.
        y_pred: Predicted labels, as returned by a classifier.

    Returns:
        dict: A dictionary containing the calculated metrics as
        follows:
            - 'accuracy': The balanced accuracy score.
            - 'precision': The macro-averaged precision score.
            - 'recall': The macro-averaged recall score.
            - 'f1': The macro-averaged F1-score.
    """
    return {
        'accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average="macro", zero_division=0),
        'recall': recall_score(y_true, y_pred, average="macro", zero_division=0),
        'f1': f1_score(y_true, y_pred, average="macro", zero_division=0)
    }


def calculate_overall_metrics(df, model_columns, model_names):
    """
    Calculate overall evaluation metrics for given model columns in a dataframe.

    This function computes evaluation metrics for each specified model column
    in the input dataframe by comparing its predictions against the actual labels.
    The computed metrics for each model, along with the model name and the total
    number of samples, are organized into a dataframe.

    Args:
        df (pd.DataFrame): The input dataframe containing the ground truth labels
            and model prediction columns.
        model_columns (list[str]): A list of column names in the dataframe representing
            the predictions from different models.
        model_names (dict): A mapping of model column names to human-readable model
            names.

    Returns:
        pd.DataFrame: A dataframe containing the calculated metrics for each model,
            the model name, and the number of analyzed samples.
    """
    results = []

    for model_col in model_columns:
        metrics = calculate_metrics(df['label'], df[model_col])
        metrics['model'] = model_names.get(model_col, model_col)
        metrics['n_samples'] = len(df)
        results.append(metrics)

    return pd.DataFrame(results)


def calculate_language_wise_metrics(df, model_columns, model_names):
    """
    Calculates language-wise metrics for the given dataset and models.

    This function processes a dataframe, slicing it language by language based
    on the unique language codes in the 'lang' column. For each language, it
    calculates metrics for the specified model columns using the labels and model
    outputs. The results are aggregated into a dataframe for analysis.

    Args:
        df: DataFrame containing the dataset, which must have a 'lang' column for
            language codes and a 'label' column for true labels.
        model_columns: List of column names in the dataframe corresponding to
            model outputs to evaluate.
        model_names: Dictionary mapping model column names to their respective
            human-readable model names.

    Returns:
        DataFrame: A new dataframe containing model performance metrics for each
            language, including model names, language code, sample counts, and
            computed metrics.
    """
    results = []

    languages = df['lang'].unique()

    for lang in sorted(languages):
        df_lang = df[df['lang'] == lang]

        for model_col in model_columns:
            metrics = calculate_metrics(df_lang['label'], df_lang[model_col])
            metrics['model'] = model_names.get(model_col, model_col)
            metrics['language'] = lang
            metrics['n_samples'] = len(df_lang)
            results.append(metrics)

    return pd.DataFrame(results)


def print_results(overall_df, language_df):
    """
    Prints overall metrics for all languages and language-wise metrics provided in
    the dataframes.

    Args:
        overall_df (pd.DataFrame): DataFrame containing overall metrics for all
            languages. It is expected to have no index and contains all relevant
            metrics for overall evaluation.
        language_df (pd.DataFrame): DataFrame containing language-specific metrics.
            It should have a 'language' column and additional data such as 'model',
            'accuracy', 'precision', 'recall', 'f1', and 'n_samples'.
    """
    print("=" * 80)
    print("OVERALL METRICS (All Languages)")
    print("=" * 80)
    print(overall_df.to_string(index=False))
    print("\n")

    print("=" * 80)
    print("LANGUAGE-WISE METRICS")
    print("=" * 80)

    languages = language_df['language'].unique()
    for lang in sorted(languages):
        print(f"\n{lang.upper()} Language:")
        print("-" * 80)
        lang_data = language_df[language_df['language'] == lang]
        print(lang_data[['model', 'accuracy', 'precision', 'recall', 'f1', 'n_samples']].to_string(index=False))

    print("\n")


def create_comparison_table(overall_df):
    """
    Creates a comparison table for models based on performance metrics, ranks the models
    by their F1-score in descending order, and prints a summary ranking.

    Args:
        overall_df: pandas.DataFrame. DataFrame containing performance metrics of models
            including columns for 'f1', 'model', 'accuracy', 'precision', 'recall',
            and 'n_samples'.

    Returns:
        pandas.DataFrame: A sorted DataFrame with an additional column for rank, where
        models are ranked by their F1-score.
    """
    df_sorted = overall_df.sort_values('f1', ascending=False).reset_index(drop=True)
    df_sorted['rank'] = range(1, len(df_sorted) + 1)

    print("=" * 80)
    print("MODEL RANKING (by F1-Score)")
    print("=" * 80)
    print(df_sorted[['rank', 'model', 'f1', 'accuracy', 'precision', 'recall', 'n_samples']].to_string(index=False))
    print("\n")

    return df_sorted


def calculate_class_distribution(df):
    """
    Calculate and print the class distribution of a dataset both overall and per language.

    This function processes a pandas DataFrame to determine the distribution of classes
    defined in the 'label' column. It calculates the count and percentage for each class
    (0 and 1) overall and then breaks this down by unique languages specified in the
    'lang' column of the dataset. The results are displayed as formatted output.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing at least two columns:
            'label' (int): A column representing class labels (e.g., 0 for NOT_RECLAMATORY,
            1 for RECLAMATORY).
            'lang' (str): A column representing the language associated with each entry.
    """
    print("=" * 80)
    print("CLASS DISTRIBUTION")
    print("=" * 80)

    overall_dist = df['label'].value_counts().sort_index()
    print(f"\nOverall:")
    print(f"  Class 0 (NOT_RECLAMATORY): {overall_dist.get(0, 0)} ({overall_dist.get(0, 0) / len(df) * 100:.1f}%)")
    print(f"  Class 1 (RECLAMATORY): {overall_dist.get(1, 0)} ({overall_dist.get(1, 0) / len(df) * 100:.1f}%)")
    print(f"  Total: {len(df)}")

    print(f"\nPer Language:")
    for lang in sorted(df['lang'].unique()):
        df_lang = df[df['lang'] == lang]
        lang_dist = df_lang['label'].value_counts().sort_index()
        print(f"  {lang.upper()}: Class 0={lang_dist.get(0, 0)}, Class 1={lang_dist.get(1, 0)}, Total={len(df_lang)}")
    print("\n")


def generate_detailed_report(df, model_col, model_names):
    """
    Generates a detailed classification report for a specific model with an
    overview of overall performance and per-language breakdown, including confusion
    matrix and classification metrics.

    Args:
        df: DataFrame containing the data to evaluate, including true labels,
            predicted labels, and language information.
        model_col: Column name in `df` representing the model's predictions.
        model_names: Dictionary mapping column names to human-readable model names
            for the report.
    """
    model_name = model_names.get(model_col, model_col)

    print(f"\n{'=' * 80}")
    print(f"DETAILED REPORT: {model_name}")
    print(f"{'=' * 80}")

    print("\nOverall Classification Report:")
    print(classification_report(df['label'], df[model_col],
                                target_names=['NOT_RECLAMATORY', 'RECLAMATORY'],
                                zero_division=0))

    cm = confusion_matrix(df['label'], df[model_col])
    print("\nConfusion Matrix:")
    print(f"                Predicted NOT    Predicted REC")
    print(f"Actual NOT      {cm[0][0]:<15}  {cm[0][1]:<15}")
    print(f"Actual REC      {cm[1][0]:<15}  {cm[1][1]:<15}")

    print("\n" + "-" * 80)
    print("Per-Language Reports:")
    print("-" * 80)

    for lang in sorted(df['lang'].unique()):
        df_lang = df[df['lang'] == lang]
        print(f"\n{lang.upper()} Language:")
        print(classification_report(df_lang['label'], df_lang[model_col],
                                    target_names=['NOT_RECLAMATORY', 'RECLAMATORY'],
                                    zero_division=0))


def get_random_examples(df: pd.DataFrame, n_examples_per_class: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Selects a subset of random examples per class for each language in the given DataFrame.
    The function retrieves random samples from the DataFrame for each language, ensuring
    an equal number of samples for each class per language, limited by the specified number
    of examples per class. The resulting examples for each language are shuffled.

    Args:
        df (pd.DataFrame):
            DataFrame containing the data with columns 'lang' (language) and 'label' (class labels).
        n_examples_per_class (int, optional):
            Maximum number of examples to retrieve per class for each language. Defaults to 10.

    Returns:
        Dict[str, pd.DataFrame]:
            A dictionary where each key is a language code (from 'lang' column) and each value
            is a DataFrame containing a random subset of examples for that language,
            equally distributed between the two classes.
    """
    random_examples = {}

    for lang in sorted(df['lang'].unique()):
        df_lang = df[df['lang'] == lang].copy()

        examples_class_0 = df_lang[df_lang['label'] == 0].sample(
            n=min(n_examples_per_class, len(df_lang[df_lang['label'] == 0])),
            random_state=SEED
        )

        examples_class_1 = df_lang[df_lang['label'] == 1].sample(
            n=min(n_examples_per_class, len(df_lang[df_lang['label'] == 1])),
            random_state=SEED
        )

        lang_examples = pd.concat([examples_class_0, examples_class_1])
        lang_examples = lang_examples.sample(frac=1, random_state=SEED).reset_index(drop=True)

        random_examples[lang] = lang_examples

        print(
            f"{lang.upper()}: {len(examples_class_0)} class 0 + {len(examples_class_1)} class 1 = {len(lang_examples)} total")

    return random_examples


def get_misclassified_examples(df: pd.DataFrame,
                               prediction_col,
                               n_examples_per_class: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Retrieves a specified number of misclassified examples for each class and language from the
    given DataFrame. The examples are sampled randomly, ensuring reproducibility by utilizing
    a fixed random seed. Misclassified examples are those where the predicted label does not match
    the true label.

    Args:
        df (pd.DataFrame): Input dataframe containing at least the columns 'lang', 'label', and
            the prediction column. It is assumed that 'lang' denotes different languages, 'label'
            contains the true labels (0 or 1), and the prediction column contains predicted labels.
        prediction_col: The name of the column in the DataFrame containing the predicted labels.
        n_examples_per_class (int): Maximum number of misclassified instances to retrieve per class
            (0 and 1) for each language. Default is 10.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are language identifiers (from the 'lang'
        column) and values are DataFrames containing randomized misclassified examples for each
        language. The DataFrames for each language include examples misclassified into both
        class 0 and class 1 (up to the provided limit per class).

    Raises:
        None
    """
    misclassified_examples = {}

    for lang in sorted(df['lang'].unique()):
        df_lang = df[df['lang'] == lang].copy()

        df_misclassified = df_lang[df_lang['label'] != df_lang[prediction_col]].copy()

        misclass_0 = df_misclassified[df_misclassified['label'] == 0]

        misclass_1 = df_misclassified[df_misclassified['label'] == 1]

        examples_class_0 = misclass_0.sample(
            n=min(n_examples_per_class, len(misclass_0)),
            random_state=SEED
        ) if len(misclass_0) > 0 else pd.DataFrame()

        examples_class_1 = misclass_1.sample(
            n=min(n_examples_per_class, len(misclass_1)),
            random_state=SEED
        ) if len(misclass_1) > 0 else pd.DataFrame()

        if not examples_class_0.empty or not examples_class_1.empty:
            lang_examples = pd.concat([examples_class_0, examples_class_1])
            lang_examples = lang_examples.sample(frac=1, random_state=SEED).reset_index(drop=True)
            misclassified_examples[lang] = lang_examples

            print(f"{lang.upper()}: {len(examples_class_0)} misclassified class 0 + "
                  f"{len(examples_class_1)} misclassified class 1 = {len(lang_examples)} total")
        else:
            print(f"{lang.upper()}: No misclassified examples found")

    return misclassified_examples
