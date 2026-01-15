import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

df = pd.read_csv("processed_data.csv")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class ScamDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=256
        )
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

dataset = ScamDataset(df["clean_text"], df["is_scam"])

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

args = TrainingArguments(
    output_dir="../model",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=100,
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset
)

trainer.train()

model.save_pretrained("../model/scam_model")
tokenizer.save_pretrained("../model/scam_model")
