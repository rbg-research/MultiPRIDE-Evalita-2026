from openai import OpenAI
from typing import Dict, Optional
import pandas as pd

from .config import genai_api_url
from .utils import set_all_seeds

set_all_seeds()


class GenAIClassifier:
    """
    The GenAIClassifier is a tool for analyzing LGBTQ+ discourse in tweets, specifically
    determining whether the use of LGBTQ+ terminology expresses reclamatory intent.

    The class leverages large language models to perform classification via both zero-shot
    and few-shot paradigms. It allows customization of language-specific examples and
    provides deterministic or temperature-driven outputs. This classifier is designed for
    specialized linguistic analysis in the context of social empowerment, identity
    affirmation, and evaluation of derogatory or neutral language use.

    Attributes:
        model_name (str): The name of the model to use for language analysis.
        examples_dict (Optional[Dict[str, pd.DataFrame]]): A dictionary containing
            language-specific DataFrames with examples for few-shot learning.
        client (OpenAI): The client used to interact with the language model API.
    """

    def __init__(self,
                 model_name: str = "mistral-7b-instruct-v0.2",
                 examples_dict: Optional[Dict[str, pd.DataFrame]] = None,
                 api_base: str = genai_api_url):
        """
        Initializes the class with the specified model name, examples dictionary, and
        API base URL. This sets up the configuration for interaction with the external
        API client.

        Args:
            model_name (str): The name of the model to be used. Defaults to
                "mistral-7b-instruct-v0.2".
            examples_dict (Optional[Dict[str, pd.DataFrame]]): A dictionary
                containing example data, where the keys represent example names
                and the values are Pandas DataFrames. Defaults to an empty dictionary
                if none is provided.
            api_base (str): The base URL for the API to be used in the client
                configuration.
        """
        self.model_name = model_name
        self.examples_dict = examples_dict or {}
        self.client = OpenAI(
            api_key="not-needed",
            base_url=api_base
        )

    @staticmethod
    def create_prompt_zero_shot(q_tweet: str, q_language: str) -> str:
        """
        Generates a zero-shot prompt for analyzing and classifying LGBTQ+ terminology based on reclamatory intent.

        This method constructs a linguistic analysis prompt tailored to identify if LGBTQ+ terms in a given tweet
        are used with reclamatory intent, considering tone, markers, context, and intent. The function maps the
        desired language code to its corresponding human-readable name to seamlessly integrate it into the prompt.

        Args:
            q_tweet (str): A string representing the tweet content to be analyzed.
            q_language (str): A string representing the language code of the tweet (e.g., "en", "es", "it").

        Returns:
            str: A formatted prompt string for linguistic analysis of reclamatory intent.
        """
        language_mapper = {
            "en": "English",
            "es": "Spanish",
            "it": "Italian"
        }

        prompt = f"""
            You are a linguistic expert specializing in LGBTQ+ discourse analysis.
            
            TASK: Classify whether LGBTQ+ terminology uses RECLAMATORY INTENT
            
            Strictly give final classification answer only, don't give any explanation.
            
            DEFINITIONS:
            - RECLAMATORY: Term used for identity affirmation, pride, empowerment, community solidarity
            - NOT_RECLAMATORY: Derogatory use, neutral mention, or unclear intent
            
            Tweet (Language: {language_mapper[q_language]}): "{q_tweet}"
            
            ANALYSIS (Chain of Thought):
            
            1. TERM IDENTIFICATION: What LGBTQ+ terms appear?
            2. TONE & MARKERS: What emotional/linguistic indicators are present?
            3. CONTEXT: Is it community affirmation or external discourse?
            4. INTENT: Does it express reclamatory intent?
            
            Classification: [RECLAMATORY / NOT_RECLAMATORY]
            """
        return prompt

    def create_prompt_few_shot(self,
                               q_tweet: str,
                               q_language: str) -> str:
        """
        Creates a few-shot prompt for classifying LGBTQ+ terminology intent in the given tweet.

        This method generates a linguistic expert-style classification prompt for determining whether
        LGBTQ+ terminology in a given tweet has a reclaimable intent (e.g., identity affirmation, pride)
        or not (e.g., derogatory use or neutral mention). It uses pre-defined examples for a specific language
        to aid in the classification. If no examples exist for the given language, the method falls back to a
        zero-shot prompt.

        Args:
            q_tweet (str): The tweet text to be analyzed for LGBTQ+ terminology intent.
            q_language (str): The language of the tweet, used to fetch pre-defined examples. Must be one of the supported
                languages in the predefined language mapping.

        Returns:
            str: The generated prompt containing task definitions, examples, and the input tweet for classification.
        """
        if q_language not in self.examples_dict or self.examples_dict[q_language].empty:
            print(f"Warning: No examples found for language '{q_language}'. Falling back to zero-shot.")
            return self.create_prompt_zero_shot(q_tweet, q_language)

        examples_df = self.examples_dict[q_language]

        # Format examples
        examples_text = self._format_examples(examples_df)

        language_mapper = {
            "en": "English",
            "es": "Spanish",
            "it": "Italian"
        }

        prompt = f"""
            You are a linguistic expert specializing in LGBTQ+ discourse analysis.
            
            TASK: Classify whether LGBTQ+ terminology uses RECLAMATORY INTENT
            
            Strictly give final classification answer only, don't give any explanation.
            
            DEFINITIONS:
            - RECLAMATORY: Term used for identity affirmation, pride, empowerment, community solidarity
            - NOT_RECLAMATORY: Derogatory use, neutral mention, or unclear intent
            
            EXAMPLES:
            The following are some examples of LGBTQ+ tweets in {language_mapper[q_language]}:
            
            {examples_text}
            
            Based on the observations from the DEFINITIONS and the EXAMPLES, classify the following tweet in {language_mapper[q_language]}:
            
            Tweet : "{q_tweet}"
            
            Classification: [RECLAMATORY / NOT_RECLAMATORY]
            """
        return prompt

    @staticmethod
    def _format_examples(examples_df: pd.DataFrame) -> str:
        """
        Formats examples contained in a DataFrame into a string representation for display.

        The method processes each row of the provided DataFrame to generate a formatted
        representation of the examples. Each example will include the tweet text and its
        corresponding label, either as "RECLAMATORY" if the label is 1 or "NOT_RECLAMATORY"
        if the label is 0. The resulting formatted string separates examples with "---".

        Args:
            examples_df (pd.DataFrame): The DataFrame containing examples to be formatted.
                It is expected to have at least two columns: 'text' (str) containing the
                tweet text and 'label' (int) indicating the associated label.

        Returns:
            str: The formatted string including all the examples from the input DataFrame.
        """
        formatted = []

        for idx, row in examples_df.iterrows():
            label_text = "RECLAMATORY" if row['label'] == 1 else "NOT_RECLAMATORY"

            example = f"""Tweet: "{row['text']}" Label: {label_text}"""

            formatted.append(example)

        return "\n---\n".join(formatted)


    def classify(self,
                 q_tweet: str,
                 q_language: str,
                 use_few_shot: bool = False,
                 temperature: float = 0.0) -> dict:
        """
        Classifies a tweet regarding its likelihood of being "reclamatory" or "not-reclamatory"
        using linguistic NLP models. Two classification approaches can be applied: few-shot
        or zero-shot, depending on available examples and user preference.

        Args:
            q_tweet (str): The tweet text to be classified.
            q_language (str): The language of the tweet for appropriate model adaptation.
            use_few_shot (bool): Optional flag indicating whether to use a few-shot approach
                in the classification if language-specific examples are available. Defaults
                to False.
            temperature (float): Sampling temperature for the model's response generation.
                Values closer to 0 enforce deterministic behavior, while higher values
                introduce randomness. Defaults to 0.0.

        Returns:
            dict: A dictionary containing:
                - "tweet" (str): The original tweet text.
                - "language" (str): The language of the tweet.
                - "classification" (str): The classification result, either "RECLAMATORY" or
                    "NOT_RECLAMATORY".
                - "classification_label" (int): Numerical classification label, 1 for
                    "RECLAMATORY" and 0 for "NOT_RECLAMATORY".
                - "raw_response" (str): The raw response text from the model.
                - "model" (str): The name of the model used for this classification.
                - "prompt_type" (str): The type of prompt used, either "few-shot" or
                    "zero-shot".
                - "error" (str, optional): Error message in the event of an exception during
                    classification, otherwise None.
        """
        # Create appropriate prompt
        if use_few_shot and q_language in self.examples_dict:
            prompt = self.create_prompt_few_shot(q_tweet, q_language)
            prompt_type = "few-shot"
        else:
            prompt = self.create_prompt_zero_shot(q_tweet, q_language)
            prompt_type = "zero-shot"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a linguistic expert analyzing LGBTQ+ content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=100,
                top_p=0.95
            )

            result_text = response.choices[0].message.content.strip()

            # Parse result
            is_reclamatory = "RECLAMATORY" in result_text.upper() and "NOT_RECLAMATORY" not in result_text.upper()

            return {
                "tweet": q_tweet,
                "language": q_language,
                "classification": "RECLAMATORY" if is_reclamatory else "NOT_RECLAMATORY",
                "classification_label": 1 if is_reclamatory else 0,
                "raw_response": result_text,
                "model": self.model_name,
                "prompt_type": prompt_type
            }

        except Exception as e:
            return {
                "tweet": q_tweet,
                "language": q_language,
                "error": str(e),
                "classification": None,
                "classification_label": None,
                "prompt_type": prompt_type
            }
