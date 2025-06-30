import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score
from huggingface_hub import login
login("your_token")
import wandb
wandb.login()


df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vTf2Tm2H0Yvqs7-g5n_ysK0QYd0mVhhPKArdd7s-Z06mKd7UV4fjOJjbgUVODhqmXpk4_-OQHdyEnjn/pub?gid=1187926665&single=true&output=csv')
#sheet1
#df = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vTf2Tm2H0Yvqs7-g5n_ysK0QYd0mVhhPKArdd7s-Z06mKd7UV4fjOJjbgUVODhqmXpk4_-OQHdyEnjn/pub?gid=0&single=true&output=csv')
df['sentence'] = df['sentence'].str.lower()

unique_labels = df['label'].unique()
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
df['label_id'] = df['label'].map(label_to_id)
num_labels = len(unique_labels)

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding="max_length", truncation=True, max_length=128)

train_tokenized = train_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)

train_tokenized = train_tokenized.remove_columns(['sentence', 'label', '__index_level_0__'])
train_tokenized = train_tokenized.rename_column("label_id", "labels")
train_tokenized.set_format("torch")

test_tokenized = test_tokenized.remove_columns(['sentence', 'label', '__index_level_0__'])
test_tokenized = test_tokenized.rename_column("label_id", "labels")
test_tokenized.set_format("torch")

from transformers import GPTNeoXForSequenceClassification

model = GPTNeoXForSequenceClassification.from_pretrained(
    "EleutherAI/pythia-70m",
    num_labels=num_labels
)

model.config.pad_token_id = tokenizer.pad_token_id

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1.5e-5,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=3,
    num_train_epochs=5,
    weight_decay=0.02,
    logging_steps=10,
    logging_dir='./logs',
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
)

trainer.train()

eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# predictions = trainer.predict(test_tokenized)
# preds = np.argmax(predictions.predictions, axis=1)
# true_labels = predictions.label_ids
# accuracy = accuracy_score(true_labels, preds)
# print(f"Accuracy: {accuracy:.4f}")
predictions = trainer.predict(test_tokenized)

# Safe extraction of logits
if isinstance(predictions.predictions, (list, tuple)) and isinstance(predictions.predictions[0], np.ndarray):
    logits = predictions.predictions[0]  # Use first item if it's a tuple/list
else:
    logits = predictions.predictions     # Otherwise, use directly

# Now compute predicted classes
preds = np.argmax(logits, axis=1)
true_labels = predictions.label_ids

accuracy = accuracy_score(true_labels, preds)
print(f"Accuracy: {accuracy:.4f}")


model.save_pretrained("./Tifin/trained_model")
tokenizer.save_pretrained("./Tifin/trained_model")
print("Model saved to ./Tifin/trained_model")