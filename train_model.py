from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_from_disk
import torch

# Check for any empty sequences in the dataset
def check_empty_samples(dataset):
    for i, sample in enumerate(dataset):
        if len(sample['input_ids']) == 0 or len(sample['labels']) == 0:
            print(f"Warning: Empty input or label found in sample {i}")
        if len(sample['attention_mask']) == 0:
            print(f"Warning: Empty attention_mask in sample {i}")

# Define a safe data collator that filters out any empty sequences
def safe_data_collator(features, data_collator):
    # Filter out features that have zero-length sequences
    non_empty_features = [f for f in features if len(f['input_ids']) > 0 and len(f['labels']) > 0]

    if len(non_empty_features) == 0:
        raise ValueError("All features in this batch have zero-length sequences.")
    
    # Use the default data collator for the non-empty features
    return data_collator(non_empty_features)

def train_model():
    # Load tokenized dataset
    tokenized_data = load_from_disk("./data/tokenized_data")
    
    # Print dataset size to ensure it's not empty
    print(f"Number of training examples: {len(tokenized_data['train'])}")
    print(f"Number of evaluation examples: {len(tokenized_data['test'])}")

    # Check for empty inputs or labels
    check_empty_samples(tokenized_data['train'])
    check_empty_samples(tokenized_data['test'])

    # Load Flan-T5-Large model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")

    # Check if CUDA (GPU) is available and use it
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir='./models/flan-t5-address-verification',  # Directory to save checkpoints
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        learning_rate=2e-5,  # Learning rate for fine-tuning
        per_device_train_batch_size=4,  # Batch size per GPU/CPU
        per_device_eval_batch_size=4,   # Evaluation batch size
        weight_decay=0.01,  # Weight decay to prevent overfitting
        save_total_limit=3,  # Limit the number of checkpoints
        num_train_epochs=3,  # Number of epochs for training
        predict_with_generate=True,  # Enable generation for evaluation
        fp16=True,  # Enable mixed precision training (for GPUs that support it)
        logging_dir='./logs',  # Directory for logging
        logging_steps=100,  # Log every 100 steps
    )

    # Define the default data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model,
        label_pad_token_id=-100  # Padding tokens in labels should be ignored
    )

    # Create the trainer object and use the safe data collator
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data['train'],  # Training dataset
        eval_dataset=tokenized_data['test'],    # Evaluation dataset
        tokenizer=tokenizer,
        data_collator=lambda features: safe_data_collator(features, data_collator),  # Use the safe data collator
    )

    # Train the model
    trainer.train()

    # Save the trained model
    model.save_pretrained('./models/flan-t5-address-verification')
    tokenizer.save_pretrained('./models/flan-t5-address-verification')

if __name__ == '__main__':
    train_model()
