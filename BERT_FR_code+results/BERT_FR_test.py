# 9. FINAL EVALUATION ON TEST SET (101 ricette)
print("\nFinal evaluation on TEST SET (101 ricette):")
# Prepare dataset
test_dataset = Dataset.from_pandas(df_test.reset_index(drop=True))
test_dataset = test_dataset.map(preprocess_batch, batched=True, batch_size=32)
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
# Predictions
test_pred_output = trainer.predict(test_dataset)
test_preds = np.argmax(test_pred_output.predictions, axis=-1)
test_true = np.array(test_dataset['labels'])
test_acc = accuracy_score(test_true, test_preds)
print(f"Accuracy on test set: {test_acc:.4f}")
print("Classification report on test set:")
# Get the unique labels in the test set predictions
unique_labels_test = np.unique(test_preds)
# Filter the target names to only include labels present in the test set
target_names_test = [label for label, value in label_map.items() if value in unique_labels_test]

print(classification_report(test_true, test_preds, target_names=target_names_test)) # Change here
print("Conusion matrix on test set:")
print(confusion_matrix(test_true, test_preds))