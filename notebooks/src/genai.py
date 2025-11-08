from openai import OpenAI

from .utils import set_all_seeds


set_all_seeds()


class GenAIClassifier:
    """Zero-shot classifier for LGBTQ+ reclamatory intent in tweets."""

    def __init__(self, model_name="mistral-7b-instruct-v0.2"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key="not-needed",
            base_url="http://dhvani.rbg.ai:9999/v1"
        )

    @staticmethod
    def create_prompt(q_tweet: str, q_language: str, use_advanced=True) -> str:

        if use_advanced:
            prompt = f"""
                
                You are a linguistic expert specializing in LGBTQ+ discourse analysis. 
                
                
                TASK: Classify whether LGBTQ+ terminology uses RECLAMATORY INTENT
                
                Strictly give final classification answer only, don't give any explanation. 
            
                DEFINITIONS:
                - RECLAMATORY: Term used for identity affirmation, pride, empowerment, community solidarity
                - NOT_RECLAMATORY: Derogatory use, neutral mention, or unclear intent
                
                Tweet (Language: {q_language}): "{q_tweet}"
    
                ANALYSIS (Chain of Thought):
                
                1. TERM IDENTIFICATION: What LGBTQ+ terms appear?
                2. TONE & MARKERS: What emotional/linguistic indicators are present?
                3. CONTEXT: Is it community affirmation or external discourse?
                4. INTENT: Does it express reclamatory intent?
                
                Classification: [RECLAMATORY / NOT_RECLAMATORY]

            """
        else:
            prompt = f"""Analyze this tweet for LGBTQ+ reclamatory intent.
            Strictly give final classification answer only, don't give any explanation.

            RECLAMATORY = Pride, empowerment, self-affirmation
            NOT_RECLAMATORY = Derogatory, neutral, or unclear
            
            Tweet ({q_language}): "{q_tweet}"
            
            Let's think step by step:
            1. What term(s) are used?
            2. Is the tone positive/affirmative?
            3. Is it reclamatory?
            
            Answer:
            Classification: [RECLAMATORY / NOT_RECLAMATORY]
            """

        return prompt

    def classify(self,
                 q_tweet: str,
                 q_language: str,
                 temperature: float = 0.0,
                 use_advanced: bool = True) -> dict:

        prompt = self.create_prompt(q_tweet, q_language, use_advanced)

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
                "model": self.model_name
            }

        except Exception as e:
            return {
                "tweet": q_tweet,
                "error": str(e),
                "classification": None
            }

    def batch_classify(self,
                       q_tweets: list,
                       q_languages: list = None,
                       batch_size: int = 10) -> list:

        if q_languages is None:
            languages = ["english"] * len(q_tweets)

        results = []
        for i, (q_tweet, q_lang) in enumerate(zip(q_tweets, q_languages)):
            result = self.classify(q_tweet, q_lang)
            results.append(result)

            if (i + 1) % batch_size == 0:
                print(f"Processed {i + 1}/{len(q_tweets)} tweets")
        return results
