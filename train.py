import pandas as pd
import numpy as np
import json

from datasets import Dataset, DatasetDict
from transformers import (
   BertForSequenceClassification,
   BertTokenizer,
   Trainer,
   TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

num_epochs = 1
batch_size = 8

json_path_train = './gz_all.json'
json_path_test = './gz_100.json'

with open(json_path_train, 'r', encoding='utf8') as f:
    data_train = json.load(f)
data_train = [el for el in data_train if 'difficulty' in el.keys()]
# data_train = data_train[:10]

with open(json_path_test, 'r', encoding='utf8') as f:
    data_test = json.load(f)
data_test = [el for el in data_test if 'difficulty' in el.keys()]
# data_test = data_test[:10]

df_train = pd.DataFrame(data_train)
print(f'Train value counts: {df_train['difficulty'].value_counts()}')

df_test = pd.DataFrame(data_test)
print(f'Test value counts: {df_test['difficulty'].value_counts()}')

# Label mapping
label_map = {
    'molto_facile': 0,
    'facile': 1,
    'media': 2,
    'difficile': 3,
    'molto_difficile': 4,
}

# Split the dataset into train and test sets
df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
dataset_train = Dataset.from_pandas(df_train)
dataset_val = Dataset.from_pandas(df_val)

# Load Italian BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained(
    'dbmdz/bert-base-italian-cased',
    num_labels=5
)
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-italian-cased')

# Preprocessing function
def preprocess_function(examples):
    input_texts = []
    for i in range(len(examples['difficulty'])):
        input_text = ' '.join([
            str(examples[col][i]) for col in examples.keys()
            if col != 'difficulty'
        ])
        input_texts.append(input_text)

    tokenized = tokenizer(
        input_texts,
        truncation=True,
        padding=True
    )

    tokenized['labels'] = [label_map[examples['difficulty'][i]] for i in range(len(examples['difficulty']))]
    return tokenized

# Apply preprocessing
dataset_train = dataset_train.map(preprocess_function, batched=True, batch_size=batch_size)
dataset_val = dataset_val.map(preprocess_function, batched=True, batch_size=batch_size)

# Combine datasets
dataset = DatasetDict({
    'train': dataset_train,
    'val': dataset_val
})

# Define compute_metrics function
def compute_metrics_acc(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }
def compute_metrics_mse(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    mse = mean_squared_error(labels, preds)
    return {
        'mse': mse,
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    eval_strategy="epoch",
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    compute_metrics=compute_metrics_acc,
)

# Train model
trainer.train()

results = trainer.evaluate()

# Print metrics
print(f"Accuracy: {results['eval_accuracy']}")
print(f"Loss: {results['eval_loss']}")

# Confusion matrix and classification report
from sklearn.metrics import classification_report, confusion_matrix

# Predictions
predictions = trainer.predict(dataset["val"])
preds = predictions.predictions.argmax(-1)
true_labels = dataset["val"]["labels"]

print("\nClassification Report:")
print(classification_report(true_labels, preds))

print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, preds))

"""
Results copied here for safety (2 epochs)
Accuracy: 0.5714285714285714
Loss: 1.1467478275299072

Classification Report:
              precision    recall  f1-score   support

           0       0.00      0.00      0.00         5
           1       0.57      1.00      0.73        12
           2       0.00      0.00      0.00         3
           3       0.00      0.00      0.00         1

    accuracy                           0.57        21
   macro avg       0.14      0.25      0.18        21
weighted avg       0.33      0.57      0.42        21


Confusion Matrix:
[[ 0  5  0  0]
 [ 0 12  0  0]
 [ 0  3  0  0]
 [ 0  1  0  0]]
 """