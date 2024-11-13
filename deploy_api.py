# scripts/deploy_api.py

from flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./models/flan-t5-address-verification")
tokenizer = T5Tokenizer.from_pretrained("./models/flan-t5-address-verification")

app = Flask(__name__)

@app.route('/verify-address', methods=['POST'])
def verify_address():
    data = request.json
    address = data.get('address')
    retrieved_data = data.get('retrieved_data')

    # Prepare input for model inference
    input_text = f"Verify address: {address} with result: {retrieved_data}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate output from model
    outputs = model.generate(inputs['input_ids'], max_length=128)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"verification_result": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)