from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
app = Flask(__name__)

# Load the JokerBERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained('./fine_tuned_bert')

label_mapping = {0: "Negative", 1: "Positive"}

@app.route('/')
def home():
    return "Welcome to JokerBERT Backend API! Use /predict for predictions.", 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = data.get('text', '')

    # Tokenize and prepare input
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(dim=-1).item()

    # Return the prediction
    return jsonify({
        "input": user_input,
        "prediction": label_mapping[predicted_class]
    })

if __name__ == '__main__':
    app.run(debug=True)
