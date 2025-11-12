import os
import json
import gzip
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from .config import *
from .embeddings import ModelRegistry
from .embeddings import EmbeddingPipeline
from .utils import set_all_seeds, SEED, calculate_class_distribution

set_all_seeds()


os.makedirs(figures_root, exist_ok=True)
os.makedirs(embeddings_root, exist_ok=True)

train_files = [file for file in os.listdir(data_root) if (file.endswith(".csv") and ("train" in file))]
print(f"training files: {train_files}")

train_df = pd.DataFrame()


for file in train_files:
    temp_df = pd.read_csv(os.path.join(data_root, file))
    if "en" in file:
        temp_df["bio"] = [None] * temp_df.shape[0]
    train_df = pd.concat([train_df, temp_df], ignore_index=True)

print(f"Total training samples: {train_df.shape[0]}")

calculate_class_distribution(train_df)

stop_words_dict = {
    'en': stopwords.words('english'),
    'es': stopwords.words('spanish'),
    'it': stopwords.words('italian'),
}


scoring = {
    'accuracy': make_scorer(balanced_accuracy_score),
    'precision': make_scorer(precision_score, average="macro", zero_division=0),
    'recall': make_scorer(recall_score, average="macro", zero_division=0),
    'f1': make_scorer(f1_score, average="macro", zero_division=0)
}


models = {
    "LinearSVC (u)": LinearSVC(class_weight=None, max_iter=1000),
    "LinearSVC (b)": LinearSVC(class_weight='balanced', max_iter=1000),
    "LogisticRegression (u)": LogisticRegression(max_iter=500),
    "LogisticRegression (b)": LogisticRegression(class_weight='balanced', max_iter=500),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def _print_language_specific_result(results_df):
    """
    Prints language-specific results and combined average performance scores.

    This function processes the given results DataFrame to sort and display the best performing
    model for each language based on F1 score. Additionally, it calculates and prints the combined
    average performance metrics (Accuracy, Precision, Recall, F1-score) across all languages.

    Args:
        results_df (pd.DataFrame): A DataFrame containing evaluation results with columns including
            'Language', 'Model', 'Accuracy', 'Precision', 'Recall', and 'F1'.
    """
    results_df = results_df.sort_values(['Language', 'F1'], ascending=[True, False])
    # Print best per language
    for lang in results_df['Language'].unique():
        best = results_df[results_df['Language'] == lang].iloc[0]
        print(f"\n Best for {lang.upper()}: {best['Model']} "
              f"(Acc={best['Accuracy']:.3f}, Prec={best['Precision']:.3f}, "
              f"Rec={best['Recall']:.3f}, F1={best['F1']:.3f})")

    print("\nAll Results Summary:\n", results_df.round(3))

    # --- Combined Average Score (across all languages) ---
    combined_avg = results_df[['Accuracy', 'Precision', 'Recall', 'F1']].mean()

    print("\nCombined Average Performance Across Languages:")
    print(f" Accuracy:  {combined_avg['Accuracy']:.3f}")
    print(f" Precision: {combined_avg['Precision']:.3f}")
    print(f" Recall:    {combined_avg['Recall']:.3f}")
    print(f" F1-score:  {combined_avg['F1']:.3f}")

    print("* b -> balanced, u -> unbalanced")


def get_bow_language_specific_baseline():
    """
    Performs stratified k-fold validation on language-specific datasets using various models.

    This function processes datasets grouped by language, vectorizing text data and evaluating
    the performance of different ML models. It leverages TF-IDF for feature extraction and
    cross-validation to compute metrics such as accuracy, precision, recall, and F1 score.
    The results for each combination of language and model are aggregated.

    Raises:
        Exception: If a particular model fails during processing for a specific language,
        an error will be printed for debugging.

    Returns:
        None: The function displays the results instead of returning any value.
    """
    print("\nPerforming StratifiedKFold 5 fold validation on language-specific datasets...")
    all_results = []

    df = train_df

    for lang in df['lang'].unique():

        lang_df = df[df['lang'] == lang]
        texts = lang_df['text'].astype(str)
        labels = lang_df['label'].astype(int)

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=stop_words_dict.get(lang, None),
            ngram_range=(1, 3),
            max_features=20000
        )
        X = vectorizer.fit_transform(texts)

        for name, model in models.items():
            try:
                cv_results = cross_validate(model, X, labels, cv=cv, scoring=scoring)
                all_results.append({
                    'Language': lang,
                    'Model': name,
                    'Accuracy': np.mean(cv_results['test_accuracy']),
                    'Precision': np.mean(cv_results['test_precision']),
                    'Recall': np.mean(cv_results['test_recall']),
                    'F1': np.mean(cv_results['test_f1'])
                })
            except Exception as e:
                print(f"{name} failed for {lang}: {e}")

    results_df = pd.DataFrame(all_results)
    _print_language_specific_result(results_df)


def get_bow_multilingual_baseline():
    """
    Executes stratified k-fold cross-validation on a multilingual unified dataset using bag-of-words
    (TFIDF) as the feature extraction method and evaluates multiple classification models.

    This function processes the text dataset by merging language-specific stopwords, applies TFIDF
    vectorization to transform the text into feature vectors, and evaluates given models using
    cross-validation metrics. The function outputs model performance metrics, including Accuracy,
    Precision, Recall, and F1 scores.

    Raises:
        KeyError: If the required data keys are missing from the dataset or stop words dictionary.
        ValueError: If the dataset or specified models are invalid.
        TypeError: If the provided data is not in an acceptable format.

    Returns:
        None
    """
    print("\nPerforming StratifiedKFold 5 fold validation on Multilingual Unified Dataset...")
    texts = train_df['text'].astype(str)
    labels = train_df['label'].astype(int)
    langs = train_df['lang']

    # Merge language-specific stopwords
    combined_stopwords = set()
    for lang in langs.unique():
        combined_stopwords.update(stop_words_dict.get(lang, []))

    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words=list(combined_stopwords),
        ngram_range=(1, 3),
        max_features=30000
    )

    results = []

    for name, model in models.items():
        pipeline = Pipeline([('tfidf', tfidf), ('clf', model)])
        scores = cross_validate(pipeline, texts, labels, cv=cv, scoring=scoring, n_jobs=-1)

        results.append({
            'Model': name,
            'Accuracy': np.mean(scores['test_accuracy']),
            'Precision': np.mean(scores['test_precision']),
            'Recall': np.mean(scores['test_recall']),
            'F1': np.mean(scores['test_f1'])
        })

    results_df = pd.DataFrame(results).sort_values(by='F1', ascending=False).reset_index(drop=True)

    print("\n Model Performance (Multilingual Unified Dataset):")
    print(results_df.round(4))
    print("* b -> balanced, u -> unbalanced")


def get_embeddings():
    """
    Processes text embeddings for a variety of models and stores them in compressed JSON format.

    This function computes embeddings for texts using multiple models listed in the
    `ModelRegistry`. If embeddings for a model do not already exist in a specified
    directory, they are generated using an `EmbeddingPipeline`, saved to a JSON
    structure containing embeddings and their associated labels, and then compressed
    with gzip.

    Raises:
        FileNotFoundError: If required files or directories for saving embeddings
            are not present.
        KeyError: If the `ModelRegistry` is not initialized or models are not
            properly registered.
        ValueError: If there are inconsistencies with input text data or labels.
        OSError: If file writing or compression fails due to system-level issues.

    """
    considered_models = list(ModelRegistry.list_models().keys())
    print(f"Embedding computed for the models: {considered_models}")

    texts = list(train_df.text)
    ids = list(train_df.id)
    labels = [int(l) for l in list(train_df.label)]

    pipeline = None
    for i, considered_model in enumerate(considered_models):
        gzip_path = os.path.join(embeddings_root, considered_model + ".json.gz")

        if not os.path.exists(gzip_path):
            if pipeline is None:
                pipeline = EmbeddingPipeline(model_key=considered_model)
            else:
                pipeline.switch_model(considered_model)

            text_embeddings = pipeline.encode(texts, batch_size=32, show_progress_bar=True)
            text_embeddings = text_embeddings.tolist()

            embeddings_dict = {}
            for text_id, emb, label in zip(ids, text_embeddings, labels):
                embeddings_dict[text_id] = {"emb": emb, "label": label}

            with gzip.open(gzip_path, 'wt', encoding='utf-8') as f:
                json.dump(embeddings_dict, f)


def get_embeddings_language_specific_baseline():
    """
    Computes and evaluates embeddings for language-specific models.

    This function processes language-specific embeddings from precomputed files,
    evaluates them using a cross-validation approach, and computes several metrics
    such as accuracy, precision, recall, and F1 score for each language model.
    Results are displayed for each evaluated language and model.

    Raises:
        Exception: If an error occurs during the evaluation of a specific model,
            the function prints the model, language, and error details.

    Args:
        None

    Returns:
        None
    """
    considered_models = list(ModelRegistry.list_models().keys())
    print(f"Embedding computed for the models: {considered_models}")

    for considered_model in considered_models:
        print(f"\n============================")
        print(f"Evaluating embedding model: {considered_model}")
        print(f"\n============================")
        embeddings_path = os.path.join(embeddings_root, considered_model + ".json.gz")
        with gzip.open(embeddings_path, 'rt', encoding='utf-8') as f:
            emb_data = json.load(f)
        feature_matrix = []
        labels = []
        text_ids = []
        langs = []
        for item in emb_data:
            text_ids.append(item)
            langs.append(item.split("_")[0])
            feature_matrix.append(emb_data[item]["emb"])
            labels.append(emb_data[item]["label"])

        train_df = pd.DataFrame.from_dict(
            {
                "ids": text_ids,
                "lang": langs,
                "feat": feature_matrix,
                "label": labels
            }
        )

        all_results = []

        df = train_df

        for lang in df['lang'].unique():

            lang_df = df[df['lang'] == lang]
            X = np.asarray(list(lang_df['feat']))
            labels = lang_df['label'].astype(int)

            for name, model in models.items():
                try:
                    cv_results = cross_validate(model, X, labels, cv=cv, scoring=scoring)
                    all_results.append({
                        'Language': lang,
                        'Model': name,
                        'Accuracy': np.mean(cv_results['test_accuracy']),
                        'Precision': np.mean(cv_results['test_precision']),
                        'Recall': np.mean(cv_results['test_recall']),
                        'F1': np.mean(cv_results['test_f1'])
                    })
                except Exception as e:
                    print(f"{name} failed for {lang}: {e}")

        results_df = pd.DataFrame(all_results)
        _print_language_specific_result(results_df)


def get_embeddings_multilingual_baseline():
    """
    Processes multilingual embeddings, evaluates models across metrics, and outputs model
    performance rankings for a multilingual unified dataset.

    This function reads multilingual embedding models, processes their embeddings alongside
    metadata, and evaluates classification models using cross-validation. It ranks models based
    on their performance metrics such as accuracy, precision, recall, and F1 score. The results
    are displayed in a tabular format.

    Raises:
        FileNotFoundError: If the embeddings file for a model does not exist or cannot be found.
        JSONDecodeError: If the content of the embeddings file does not conform to JSON format.

    Args:
        None

    Returns:
        None
    """
    considered_models = list(ModelRegistry.list_models().keys())
    for considered_model in considered_models:
        print(f"\n============================")
        print(f"Evaluating embedding model: {considered_model}")
        print(f"\n============================")
        embeddings_path = os.path.join(embeddings_root, considered_model + ".json.gz")
        with gzip.open(embeddings_path, 'rt', encoding='utf-8') as f:
            emb_data = json.load(f)
        feature_matrix = []
        labels = []
        text_ids = []
        langs = []
        for item in emb_data:
            text_ids.append(item)
            langs.append(item.split("_")[0])
            feature_matrix.append(emb_data[item]["emb"])
            labels.append(emb_data[item]["label"])

        train_df = pd.DataFrame.from_dict(
            {
                "ids": text_ids,
                "lang": langs,
                "feat": feature_matrix,
                "label": labels
            }
        )

        X = np.asarray(list(train_df['feat']))
        labels = train_df['label'].astype(int)

        results = []

        for name, model in models.items():
            pipeline = Pipeline([('clf', model)])
            scores = cross_validate(pipeline, X, labels, cv=cv, scoring=scoring, n_jobs=-1)

            results.append({
                'Model': name,
                'Accuracy': np.mean(scores['test_accuracy']),
                'Precision': np.mean(scores['test_precision']),
                'Recall': np.mean(scores['test_recall']),
                'F1': np.mean(scores['test_f1'])
            })

        results_df = pd.DataFrame(results).sort_values(by='F1', ascending=False).reset_index(drop=True)

        print("\n Model Performance (Multilingual Unified Dataset):")
        print(results_df.round(4))
        print("* b -> balanced, u -> unbalanced")
