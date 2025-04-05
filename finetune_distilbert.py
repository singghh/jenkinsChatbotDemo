import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertConfig
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import os
import logging
from datetime import datetime

# Set up logging to a file
logging.basicConfig(
    filename=f"training_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load dataset
df = pd.read_csv("jenkins_queries.csv")
queries = df["query"].tolist()
labels = df["intent"].tolist()

# Check dataset size and warn if too small
MIN_DATASET_SIZE = 50
if len(df) < MIN_DATASET_SIZE:
    logging.warning(
        f"Dataset size ({len(df)}) is smaller than recommended ({MIN_DATASET_SIZE}). "
        "Consider expanding the dataset for better model generalization."
    )
    print(
        f"Warning: Dataset size ({len(df)}) is smaller than recommended ({MIN_DATASET_SIZE}). "
        "Consider expanding the dataset for better model generalization."
    )

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Load tokenizer and configure model with dropout
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Create a configuration object with dropout settings
config = DistilBertConfig.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_encoder.classes_),
    hidden_dropout_prob=0.2,  # Set dropout for hidden layers
    attention_probs_dropout_prob=0.2,  # Set dropout for attention layers
)

# Load model with the custom configuration
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    config=config
)

# Tokenize the queries
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

# Prepare dataset
class JenkinsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Save and load train-validation split to prevent data leakage
split_file = "train_val_split.pkl"
if not os.path.exists(split_file):
    train_idx, val_idx = train_test_split(
        range(len(queries)),
        test_size=0.2,
        random_state=42,
        stratify=encoded_labels  # Ensure balanced classes in split
    )
    with open(split_file, "wb") as f:
        pickle.dump({"train_idx": train_idx, "val_idx": val_idx}, f)
    logging.info("Created and saved new train-validation split.")
else:
    logging.info("Loading existing train-validation split to prevent data leakage.")

# Load split indices
with open(split_file, "rb") as f:
    split = pickle.load(f)

# Create datasets using the saved indices
train_queries = [queries[i] for i in split["train_idx"]]
val_queries = [queries[i] for i in split["val_idx"]]
train_labels = [encoded_labels[i] for i in split["train_idx"]]
val_labels = [encoded_labels[i] for i in split["val_idx"]]

# Tokenize datasets
train_encodings = tokenize_function(train_queries)
val_encodings = tokenize_function(val_queries)

train_dataset = JenkinsDataset(train_encodings, train_labels)
val_dataset = JenkinsDataset(val_encodings, val_labels)

# Optional: K-Fold Cross-Validation
USE_CROSS_VALIDATION = False  # Set to True to enable cross-validation
if USE_CROSS_VALIDATION:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracies = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(queries)):
        logging.info(f"Training fold {fold + 1}/5")
        fold_train_queries = [queries[i] for i in train_idx]
        fold_val_queries = [queries[i] for i in val_idx]
        fold_train_labels = [encoded_labels[i] for i in train_idx]
        fold_val_labels = [encoded_labels[i] for i in val_idx]

        fold_train_encodings = tokenize_function(fold_train_queries)
        fold_val_encodings = tokenize_function(fold_val_queries)

        fold_train_dataset = JenkinsDataset(fold_train_encodings, fold_train_labels)
        fold_val_dataset = JenkinsDataset(fold_val_encodings, fold_val_labels)

        # Reinitialize model for each fold with the same config
        fold_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            config=config
        )

        # Define training arguments for cross-validation
        fold_training_args = TrainingArguments(
            output_dir=f"./distilbert_finetuned_fold_{fold}",
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=2,
            weight_decay=0.02,
            logging_dir=f"./logs_fold_{fold}",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        # Initialize trainer for this fold
        fold_trainer = Trainer(
            model=fold_model,
            args=fold_training_args,
            train_dataset=fold_train_dataset,
            eval_dataset=fold_val_dataset,
            compute_metrics=compute_metrics,
        )

        # Train and evaluate
        fold_trainer.train()
        eval_results = fold_trainer.evaluate()
        cv_accuracies.append(eval_results["eval_accuracy"])
        logging.info(f"Fold {fold + 1} accuracy: {eval_results['eval_accuracy']:.4f}")

    avg_cv_accuracy = sum(cv_accuracies) / len(cv_accuracies)
    logging.info(f"Cross-validation average accuracy: {avg_cv_accuracy:.4f}")
    print(f"Cross-validation average accuracy: {avg_cv_accuracy:.4f}")

# Define training arguments for standard training
training_args = TrainingArguments(
    output_dir="./distilbert_finetuned",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=2,
    weight_decay=0.02,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Define compute_metrics function for evaluation
def compute_metrics(pred):
    # labels = pred.label_ids
    # preds = pred.predictions.argmax(-1)
    # accuracy = (preds == labels).mean()
    # return {"accuracy": accuracy}
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Initialize trainer for standard training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Log final evaluation metrics
eval_results = trainer.evaluate()
logging.info(f"Final evaluation metrics: {eval_results}")
print(f"Final evaluation metrics: {eval_results}")

# Save the model and tokenizer
model.save_pretrained("./distilbert_finetuned")
tokenizer.save_pretrained("./distilbert_finetuned")

print("Fine-tuning complete! Model saved to ./distilbert_finetuned")