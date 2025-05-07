!pip install datasets
!pip install transformers==4.50.3 #la versione più recente (4.51.3) causa un errore
import os
import json
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments

# 1. LOAD DATA
train_data_path = "./data/gz_all.json"
test_data_path = "./data/gz_101.json"
with open(train_data_path, 'r', encoding='utf8') as f:
    data_full = json.load(f)
with open(test_data_path, 'r', encoding='utf8') as f:
    data_test = json.load(f)

data_full = [el for el in data_full if 'difficulty' in el]
data_test = [el for el in data_test if 'difficulty' in el]

# Convert in DataFrame
df_full = pd.DataFrame(data_full)
df_test = pd.DataFrame(data_test)
print("Difficulty distribution (full dataset):")
print(df_full['difficulty'].value_counts())
print(f"Total recipies full dataset: {len(df_full)}\n")
print("Difficulty distribution (dataset test 101):")
print(df_test['difficulty'].value_counts())
print(f"Total recipies test: {len(df_test)}\n")

# 2. TRAIN/VAL SPLIT WITH STRATIICATION
train_frac = 0.8
df_train, df_val = train_test_split(
    df_full,
    test_size=1-train_frac,
    stratify=df_full['difficulty'],
    random_state=42
)
print("Data split in train/val:")
print(f"Train set: {len(df_train)} recipies, Validation set: {len(df_val)} reccipies")
print("Difficulty distriution in train set after split:")
print(df_train['difficulty'].value_counts(), "\n")

# 3. OVERSAMPLING MINORITY CLASSES ON TRAIN SET
class_counts = df_train['difficulty'].value_counts()
max_count = class_counts.max()
df_train_balanced = df_train.copy()
for difficulty_level, count in class_counts.items():
    if count < max_count:
        # Duplication with random repetition until max_count
        df_to_add = df_train[df_train['difficulty'] == difficulty_level]
        df_train_balanced = pd.concat([
            df_train_balanced,
            df_to_add.sample(max_count - count, replace=True, random_state=42)
        ])
# Final shuffle of balanced dataframe
df_train_balanced = df_train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
print("Distribuzione difficoltà dopo oversampling sul train set:")
print(df_train_balanced['difficulty'].value_counts(), "\n")

# 4. PREPARATION FOR BERT (tokenization and encoding)

label_map = {
    'molto_facile': 0,
    'facile':       1,
    'media':        2,
    'difficile':    3,
    'molto_difficile': 4
}
num_labels = len(label_map)

# Initialize BERT and tokenizer 
model_name = "dbmdz/bert-base-italian-cased"  
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Preprocessing: join text fields and tokenize
def preprocess_batch(examples):
    texts = []
    for i in range(len(examples['difficulty'])):
        # Join presentation + ingredients + steps (the most important fields)
        presentation = str(examples.get('presentation', [""])[i])
        ingredients = " ".join(examples.get('ingredients', [""])[i]) if 'ingredients' in examples else ""
        steps = " ".join(examples.get('steps', [""])[i]) if 'steps' in examples else ""
        full_text = f"{presentation} {ingredients} {steps}"
        texts.append(full_text)
    # batch tokenization of texts
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512
    )
    # Labels mapping
    encodings['labels'] = [label_map[label] for label in examples['difficulty']]
    return encodings

# Prepare datasets
from datasets import Dataset, DatasetDict
train_dataset = Dataset.from_pandas(df_train_balanced)
val_dataset   = Dataset.from_pandas(df_val.reset_index(drop=True))
# Preprocess function
train_dataset = train_dataset.map(preprocess_batch, batched=True, batch_size=32)
val_dataset   = val_dataset.map(preprocess_batch, batched=True, batch_size=32)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
dataset = DatasetDict({"train": train_dataset, "val": val_dataset})

# 5. TRAINING SETTINGS
training_args = TrainingArguments(
    output_dir="./results",          
    overwrite_output_dir=True,
    num_train_epochs=3,             # 3 epochs
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,             # slightly lower for more stability
    evaluation_strategy="epoch",    
    save_strategy="epoch",          
    load_best_model_at_end=True,    
    metric_for_best_model="accuracy",
    logging_steps=50,               
    logging_dir="./logs"
)

# Accuracy function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    compute_metrics=compute_metrics
)

# 6. TRAINING
print("Start training...")
trainer.train()
print("End training.")

# 7. EVALUATION VALIDATION SET
val_metrics = trainer.evaluate()  
print(f"Accuracy on validation: {val_metrics['eval_accuracy']:.4f}")

val_preds = np.argmax(trainer.predict(dataset['val']).predictions, axis=-1)
val_true = np.array(dataset['val']['labels'])
print("Classification report on validation:")
print(classification_report(val_true, val_preds, target_names=list(label_map.keys())))

# 8. SAVE BEST MODEL
best_model_dir = "./models/best_model_balanced"
trainer.save_model(best_model_dir)
tokenizer.save_pretrained(best_model_dir)
print(f"Best model saved in {best_model_dir}")
