from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk
import torch

# Function to clean the model's prediction
def clean_output(prediction):
    if "Valid" in prediction:
        return "Valid"
    elif "Invalid" in prediction:
        return "Invalid"
    else:
        return "Unknown"

def evaluate_model():
    # Load the tokenized dataset
    tokenized_data = load_from_disk("./data/tokenized_data")

    # Load the model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained("./models/flan-t5-address-verification")
    tokenizer = T5Tokenizer.from_pretrained("./models/flan-t5-address-verification")

    # Check if CUDA (GPU) is available and use it
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Define training arguments for evaluation
    training_args = Seq2SeqTrainingArguments(
        output_dir='./results',
        per_device_eval_batch_size=4,
        predict_with_generate=True,  # This will ensure predictions are generated
    )

    # Initialize the Seq2SeqTrainer for evaluation
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_data['test'],  # Use the test dataset for evaluation
        tokenizer=tokenizer
    )

    # Evaluate the model
    predictions = trainer.predict(tokenized_data['test']).predictions
    decoded_predictions = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]

    # Clean the output to show only Valid/Invalid results
    cleaned_results = [clean_output(pred) for pred in decoded_predictions]
    
    # Print the cleaned results
    for result in cleaned_results:
        print("Verification result:", result)

if __name__ == '__main__':
    evaluate_model()