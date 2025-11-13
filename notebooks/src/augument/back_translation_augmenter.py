import pandas as pd
import openai
import time
import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set your OpenAI API key
openai.api_key = None

assert openai.api_key is not None, "OpenAI API key not set. Please set it in the script or in your environment variables."

class BackTranslationAugmenter:
    """
    Facilitates data augmentation using back translation for text data.

    This class is designed to perform back translation for data augmentation. By translating
    text to one or more target languages and then translating it back to the original language,
    it generates augmented variations of the original text while preserving its meaning
    and sentiment. This process can improve the diversity and robustness of text datasets for
    machine learning models.

    Attributes:
        model (str): The language model to use for performing translations.
        temperature (float): Sampling temperature for translation outputs.
        max_retries (int): Maximum number of retries for handling failed translation attempts.
    """
    def __init__(self, model: str = "gpt-4-mini", temperature: float = 0, max_retries: int = 3):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.rate_limit_delay = 0.5  # seconds
        
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translates a given text from the source language to the target language while preserving
        sentiment and meaning.

        This method employs a professional translation model to accurately translate short-form
        text (e.g., tweets). It uses an exponential backoff strategy to handle rate-limiting errors
        and supports automatic retries before ultimately returning the original text in case of failure.

        Args:
            text (str): The text to be translated.
            source_lang (str): The source language of the text.
                Expected to be a two-letter language code (e.g., 'en' for English).
            target_lang (str): The target language to translate the text into.
                Expected to be a two-letter language code (e.g., 'es' for Spanish).

        Returns:
            str: The translated text if the translation is successful, or the original text in
            case of persistent errors.

        Raises:
            openai.RateLimitError: Raised when the translation service's API rate limit
                is exceeded.
            Exception: Raised when an unexpected error occurs during translation.
        """
        lang_names = {
            'en': 'English',
            'es': 'Spanish',
            'it': 'Italian'
        }
        
        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)
        
        prompt = f"""Translate the following tweet from {source_name} to {target_lang.upper()}. 
Keep the sentiment and meaning exactly the same. Respond with ONLY the translated text, nothing else.

Tweet: {text}

Translation:"""
        
        for attempt in range(self.max_retries):
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are a professional translator. Translate tweets accurately while preserving sentiment."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=200,
                    temperature=self.temperature,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                
                translated_text = response.choices[0].message.content.strip()
                time.sleep(self.rate_limit_delay)
                return translated_text
                
            except openai.RateLimitError as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt+1}/{self.max_retries}")
                time.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Translation error (attempt {attempt+1}): {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to translate: {text[:50]}...")
                    return text  # Return original text on failure
                time.sleep(1)
        
        return text
    
    def augment_dataframe(self, df: pd.DataFrame, output_path: str = 'augmented_data.csv') -> pd.DataFrame:
        """
        Augments a pandas DataFrame by performing back-translation on the text data
        to generate new training samples. Supports multiple language pairs for
        translation.

        The function performs back-translation, where the text is translated into
        another language and then back to the original language. This is done to
        augment dataset diversity and reduce potential overfitting during machine
        learning tasks.

        Args:
            df (pd.DataFrame): The input DataFrame containing columns 'id', 'text',
                'label', and 'lang'. Each row represents a single text sample with
                its metadata.
            output_path (str): The file path where the augmented DataFrame will be
                saved. Defaults to 'augmented_data.csv'.

        Returns:
            pd.DataFrame: A new DataFrame containing the original and augmented
                samples.

        Raises:
            ValueError: If the input DataFrame does not contain the required columns:
                'id', 'text', 'label', and 'lang'.
        """
        if not all(col in df.columns for col in ['id', 'text', 'label', 'lang']):
            raise ValueError("DataFrame must have columns: id, text, label, lang")
        
        augmented_rows = []
        augmented_rows.extend(df.to_dict('records'))
        translation_pairs = {
            'en': ['es', 'it'],    # English → Spanish, Italian
            'es': ['en', 'it'],    # Spanish → English, Italian
            'it': ['en', 'es']     # Italian → English, Spanish
        }

        total_samples = len(df)
        for idx, row in df.iterrows():
            original_id = row['id']
            original_text = row['text']
            original_lang = row['lang']
            label = row['label']
            
            if original_lang not in translation_pairs:
                logger.warning(f"Skipping row {idx}: Unknown language '{original_lang}'")
                continue

            target_languages = translation_pairs[original_lang]
            
            for target_lang in target_languages:
                try:
                    translated_text = self.translate(original_text, original_lang, target_lang)
                    new_id = f"{original_id}_bkt_{original_lang}_to_{target_lang}"
                    augmented_rows.append({
                        'id': new_id,
                        'text': translated_text,
                        'label': label,
                        'lang': target_lang,
                        'original_id': original_id,
                        'original_lang': original_lang,
                        'augmentation_type': 'back_translation'
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to augment row {idx}: {str(e)}")
                    continue

        augmented_df = pd.DataFrame(augmented_rows)
        augmented_df.to_csv(output_path, index=False)
        logger.info(f"\n✓ Augmented data saved to: {output_path}")
        self._print_statistics(df, augmented_df)
        return augmented_df
    
    def _print_statistics(self, original_df: pd.DataFrame, augmented_df: pd.DataFrame):
        """
        Prints statistics of the original and augmented datasets including details about total
        samples, positive and negative samples, as well as breakdowns by language and augmentation type.
        This function helps understand the changes and distributions in the datasets before
        and after augmentation.

        Args:
            original_df (pd.DataFrame): Original dataset containing sample entries prior to augmentation.
            augmented_df (pd.DataFrame): Augmented dataset containing original samples along with
                augmented data and associated augmentation information.

        """
        print("\n" + "="*80)
        print("DATA AUGMENTATION STATISTICS")
        print("="*80)
        
        print(f"\nOriginal Dataset:")
        print(f"  Total samples: {len(original_df)}")
        print(f"  Positive samples: {(original_df['label']==1).sum()}")
        print(f"  Negative samples: {(original_df['label']==0).sum()}")
        print(f"\n  By language:")
        for lang in ['en', 'es', 'it']:
            count = (original_df['lang']==lang).sum()
            print(f"    {lang.upper()}: {count}")
        
        print(f"\nAugmented Dataset:")
        print(f"  Total samples: {len(augmented_df)}")
        print(f"  Positive samples: {(augmented_df['label']==1).sum()}")
        print(f"  Negative samples: {(augmented_df['label']==0).sum()}")
        print(f"  Multiplication factor: {len(augmented_df) / len(original_df):.2f}x")
        print(f"\n  By language:")
        for lang in ['en', 'es', 'it']:
            count = (augmented_df['lang']==lang).sum()
            print(f"    {lang.upper()}: {count}")
        
        print(f"\n  By augmentation type:")
        aug_types = augmented_df['augmentation_type'].value_counts()
        for aug_type, count in aug_types.items():
            if aug_type == 'back_translation':
                print(f"    Back-translation: {count}")
        print(f"    Original: {len(original_df)}")
        
        print("\n" + "="*80)
