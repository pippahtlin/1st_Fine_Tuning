x# %%
# Before you start, remember to set your training log path and your model path to .gitignore so
# they won't be pushed to the repo (file too big to push)
# We will save our model on Huggingface
#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset, load_dataset
import numpy as np
import json
import evaluate

# For fine-tuning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#%%
# Load_dataset from hugging face do train test split automatically
complete_df = load_dataset("BertFineTuneProject/arxiv")
df = complete_df['train'].to_pandas()
# df = df.sample(frac=0.5, random_state=100)

#%%
# First 1000 rows
# df = pd.read_parquet("../data/first_1k_rows.parquet") ###### Replace this with our complete data
df = df.dropna(subset=["journal_name", "abstract"])

# Some preprocessing for fine-tuning:
# 1. We usually only have 1 input, so combine title and abstract
df["combined"] = df["title"] + " [SEP] " + df["abstract"]

# 2. Create numerical labels for journal_names
unique_labels = sorted(df["journal_name"].unique())
label2id = {label: idx for idx, label in enumerate(unique_labels)}
df["label"] = df["journal_name"].map(label2id)

# Save the label-journal dictionary in our data folder, so we can use it for prediction
# with open("../data/label2id_1000.json", "w") as f: ###### Change file name
#    json.dump(label2id, f)
with open("../data/label2id_1000.json", "w") as f: ###### Change file name
    json.dump(label2id, f)

# Number of labels
n_labels = len(pd.unique(df["journal_name"]))

#%%
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# Function to tokenize data
def tokenize_function(examples):
    return tokenizer(examples["combined"], padding="max_length", truncation=True, max_length=512)

# Prepare the dataset for tokenization
dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split dataset into train and test
train_test_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=100)
train_dataset = train_test_dataset['train'].add_column("labels", train_test_dataset['train']["label"])
test_dataset = train_test_dataset['test'].add_column("labels", train_test_dataset['test']["label"])

#%%
# Load model
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=n_labels, torch_dtype="auto")

# Set up training arguments
# Set up checkpoints so we can pause and resume from the current checkpoint (more flexible)
training_args = TrainingArguments(output_dir="bert_trainer", ###### Remember to add this folder in .gitignore
                                  eval_strategy="steps",
                                  save_strategy="steps",
                                  save_steps=2000,
                                  save_total_limit=3)  # Saves a checkpoint every 250 steps) 
                                                   ###### Should change it when we have larger dataset!!!

# Here is the basic argument:
# training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

# Evaluation metric:
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions, average='macro')  # Treat all class equally, can change to weighted
    f1 = f1_score(labels, predictions, average='macro')  # Treat all class equally, can change to weighted
    return {
        "accuracy": acc,
        "recall": recall,
        "f1": f1
    }


# Set up trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

#%%
#trainer.train()
# If interrupted, can resume from checkpoint:
trainer.train(resume_from_checkpoint=True)

# %%
trainer.save_model("./pippalin/BERT_cluster") 
tokenizer.save_pretrained("./pippalin/BERT_cluster")
# %%
# Push model to hugging face
model.push_to_hub("pippalin/BERT_cluster")
tokenizer.push_to_hub("pippalin/BERT_cluster")

#%%
# %%
# Compare original and fine-tuned model
# Original BERT:
original_model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-cased", num_labels=n_labels, torch_dtype="auto"
)
trainer_pretrained = Trainer(
    model=original_model,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)
results_pretrained = trainer_pretrained.evaluate()
print("Pre-trained BERT accuracy:", results_pretrained["eval_accuracy"])
print("Pre-trained BERT recall:", results_pretrained["eval_recall"])
print("Pre-trained BERT f1:", results_pretrained["eval_f1"])


# Fine-tuned BERT:
results_finetuned = trainer.evaluate(test_dataset)
print("Fine-tuned BERT accuracy:", results_finetuned["eval_accuracy"])
print("Fine-tuned BERT recall:", results_finetuned["eval_recall"])
print("Fine-tuned BERT f1:", results_finetuned["eval_f1"])