import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

def fine_tune():
    # Load labeled data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base_dir, '../data/labeled_data.csv')
    df = pd.read_csv(data_file)

    # Preprocess data
    texts = df['Evaluated_Disease'] + " [SEP] " + df['Report_Text']
    labels = df['Label'].tolist()
    labels = [int(label) for label in labels]

    # Split data into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )

    # Tokenize data
    tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    train_encodings = tokenizer(
        train_texts.tolist(),
        truncation=True,
        padding=True,
        max_length=512
    )
    val_encodings = tokenizer(
        val_texts.tolist(),
        truncation=True,
        padding=True,
        max_length=512
    )

    # Create Dataset objects
    class ReportDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = ReportDataset(train_encodings, train_labels)
    val_dataset = ReportDataset(val_encodings, val_labels)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load pre-trained model with num_labels=3
    model = BertForSequenceClassification.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT",
        num_labels=3
    ).to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='../models/fine_tuned_bert',
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir='./logs',
        save_total_limit=3,
        warmup_steps=500,  # Helps stabilize initial training
        weight_decay=0.01,  # Adds regularization
        fp16=True  # Enable mixed precision training
    )

    # Define compute_metrics function
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Fine-tune model
    trainer.train()

    # Explicitly save the model and tokenizer outside of checkpoint saving
    model.save_pretrained('../models/fine_tuned_bert')  # Saves model weights and config
    tokenizer.save_pretrained('../models/fine_tuned_bert')  # Saves tokenizer config and vocab

    print("Fine-tuned model and tokenizer saved to ../models/fine_tuned_bert")

if __name__ == '__main__':
    fine_tune()
