import anthropic
import os


class Model:
    def __init__(self, model_name: str = "claude-instant-v1"):
        self.client = anthropic.Client(os.environ['ANTHROPIC_API_KEY'])
        self.model_name = model_name

    def query(self, query, tokens_to_sample=500):
        response = self.client.completion(
            prompt=f"{anthropic.HUMAN_PROMPT} {query}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=self.model_name,
            max_tokens_to_sample=tokens_to_sample,
        )
        return response["completion"]
