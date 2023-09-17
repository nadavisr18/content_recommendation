from typing import List

import torch
from transformers import BertTokenizer, BertModel


class Text2Vec:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        self.vector_size = self.model.db_conf.hidden_size

        # Check if CUDA (GPU support) is available, and if so, move the model to the GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def encode(self, text: str) -> List[float]:
        """
        Get the vector representation (embedding) of the input text.
        """
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

            # Move input tensors to the same device as the model (GPU if available)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            outputs = self.model(**inputs)

            # Taking the mean of the last hidden state (as a simple method) to get embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1)
            recast_vector = [float(value) for value in embeddings[0].cpu()]
            return recast_vector
