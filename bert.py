import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json
import os
import pandas as pd

# === Step 1: Load CSV from Google Sheet ===
print("\n📥 Step 1: Loading data from Google Sheet...")
csv_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTVioHssf9ZN-QvUxEmZ-ZzJ3JRddwaF3mjZV4IpSxl_lJo6ktkO7Y1QsCsB4hhpm2KeaUCJ0gLXPED/pub?gid=0&single=true&output=csv"
df = pd.read_csv(csv_url).dropna()
texts = df['sentence'].tolist()
labels = df['label'].tolist()
print(f"✅ Loaded {len(texts)} rows of data.")

# === Step 2: Prepare prompt examples ===
print("\n📝 Step 2: Preparing prompt examples for few-shot learning...")
prompt_examples = [{"sentence": t, "label": i} for t, i in zip(texts, labels)]
with open("prompt_training_examples.json", "w") as f:
    json.dump(prompt_examples, f, indent=2)
print("✅ Saved prompt examples to 'prompt_training_examples.json'")

# === Step 3: Label Encoding ===
print("\n🔢 Step 3: Encoding labels...")
label_encoder = LabelEncoder()
label_ids = label_encoder.fit_transform(labels)
np.save("label_classes.npy", label_encoder.classes_)
print(f"✅ Labels encoded into {len(label_encoder.classes_)} unique classes.")

# === Step 4: Tokenization ===
print("\n🔤 Step 4: Tokenizing input text...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=64)
        self.labels = labels

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} | {"labels": torch.tensor(self.labels[idx])}

    def __len__(self):
        return len(self.labels)

dataset = IntentDataset(texts, label_ids)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
print("✅ Tokenization complete.")

# === Step 5: Initialize Model ===
print("\n🤖 Step 5: Initializing BERT model for sequence classification...")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ Model loaded and moved to {'GPU' if torch.cuda.is_available() else 'CPU'}.")

# === Step 6: Training ===
print("\n🚀 Step 6: Starting training...")
optimizer = AdamW(model.parameters(), lr=5e-5)
model.train()

total_samples = len(dataset)
total_batches = len(dataloader)

print(f"Total training samples: {total_samples}")
print(f"Total batches per epoch: {total_batches}")

for epoch in range(4):
    print(f"\n📚 Epoch {epoch+1} started...")
    total_loss = 0
    trained_samples = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        batch_size = batch["labels"].size(0)
        trained_samples += batch_size

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        print(f"  ✅ Batch {batch_idx}/{total_batches} | Samples: {trained_samples}/{total_samples} | Loss: {loss.item():.4f}")

    print(f"✅ Epoch {epoch+1} completed | Total Loss: {total_loss:.4f}")

# === Step 7: Save Model ===
print("\n💾 Step 7: Saving trained BERT model and tokenizer...")
model.save_pretrained("intent_model")
tokenizer.save_pretrained("intent_model")
print("✅ BERT model saved to './intent_model'")

