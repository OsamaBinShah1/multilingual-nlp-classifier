from typing import List, Dict, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TextClassifier:
    def __init__(self, model_name="bert-base-multilingual-cased", num_labels=2, max_length=256, device=None):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        print(f"[Classifier] Loading {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(self.device)

    def tokenize(self, texts: List[str]) -> dict:
        return self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

    def predict(self, texts: Union[str, List[str]]) -> List[Dict]:
        if isinstance(texts, str):
            texts = [texts]
        self.model.eval()
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in self.tokenize(texts).items()}
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return [{"text": texts[i], "label": self.id2label.get(preds[i].item(), str(preds[i].item())),
                 "confidence": round(probs[i][preds[i].item()].item(), 4)} for i in range(len(texts))]

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[Classifier] Saved to {path}")

    @classmethod
    def load(cls, path: str, num_labels: int = 2):
        inst = cls.__new__(cls)
        inst.device = "cuda" if torch.cuda.is_available() else "cpu"
        inst.max_length = 256
        inst.num_labels = num_labels
        inst.label2id = {}
        inst.id2label = {}
        inst.tokenizer = AutoTokenizer.from_pretrained(path)
        inst.model = AutoModelForSequenceClassification.from_pretrained(path).to(inst.device)
        return inst
