from datasets import load_dataset
from transformers import T5Tokenizer
import os

# Function to preprocess the dataset
def preprocess_data(data_files, tokenizer):
    # Load dataset
    dataset = load_dataset('json', data_files=data_files)

    # Preprocessing function to tokenize the data
    def preprocess_fn(examples):
        inputs = []
        outputs = []
        
        # Loop through each example in the batch
        for i in range(len(examples['input'])):
            # Check if any of the fields are empty and skip if they are
            if examples['input'][i].strip() == "" or examples['retrieved_data'][i].strip() == "" or examples['label'][i].strip() == "" or examples['explanation'][i].strip() == "":
                print(f"Skipping empty or invalid record at index {i}:")
                print(f"  input: {examples['input'][i]}")
                print(f"  retrieved_data: {examples['retrieved_data'][i]}")
                print(f"  label: {examples['label'][i]}")
                print(f"  explanation: {examples['explanation'][i]}")
                continue

            # Build input and output text
            input_text = f"Verify if the address: {examples['input'][i]} with result: {examples['retrieved_data'][i]} is valid."
            output_text = f"{examples['label'][i]}. {examples['explanation'][i]}"
            inputs.append(input_text)
            outputs.append(output_text)

        if not inputs:
            raise ValueError("All input records were skipped due to being empty or invalid. Please check the dataset.")

        # Tokenize inputs and outputs (labels)
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
        labels = tokenizer(outputs, max_length=128, truncation=True, padding='max_length').input_ids

        # Ensure that labels are non-empty
        model_inputs['labels'] = labels if len(labels) > 0 else None

        return model_inputs

    # Apply the preprocessing function to the dataset
    tokenized_dataset = dataset.map(preprocess_fn, batched=True)
    return tokenized_dataset

if __name__ == '__main__':
    # Initialize the tokenizer for Flan-T5
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")

    # Define the file paths for the training and test data
    data_files = {
        "train": "./data/train_data.json",
        "test": "./data/test_data.json"
    }

    # Preprocess the dataset
    tokenized_data = preprocess_data(data_files, tokenizer)

    # Create directory if it doesn't exist
    output_dir = "./data/tokenized_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the tokenized dataset
    print("Saving tokenized data...")
    tokenized_data.save_to_disk(output_dir)
    print(f"Tokenized data saved at {output_dir}")