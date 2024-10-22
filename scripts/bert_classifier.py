# scripts/bert_classifier.py

import torch
from transformers import BertTokenizer, BertForSequenceClassification

class BERTClassifier:
    def __init__(self, model_path=None):
        if model_path:
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
        else:
            # Load pre-trained ClinicalBERT with num_labels=3
            self.model = BertForSequenceClassification.from_pretrained(
                "emilyalsentzer/Bio_ClinicalBERT",
                num_labels=3  # Specify number of labels
            )
            self.tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model.eval()


    def classify(self, disease, text):
        input_text = disease + " [SEP] " + text
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        return predicted_class  # 0: Negative, 1: Positive, 2: Neutral
