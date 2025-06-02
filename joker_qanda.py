from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import logging
import os
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from redis import Redis
from time import time

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend compatibility

# Configure Redis-based rate limiting
redis_client = Redis(host='localhost', port=6379)
limiter = Limiter(key_func=get_remote_address, storage_uri="redis://localhost:6379")
limiter.init_app(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the fine-tuned JokerBERT QA model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.getenv("MODEL_PATH", "fine_tuned_bert")  # Default to fine_tuned_bert

# Verify that model files exist
if not os.path.exists(MODEL_PATH):
    logging.error(f"Model path {MODEL_PATH} does not exist!")
    exit(1)

required_files = ["config.json", "model.safetensors"]
missing_files = [f for f in required_files if not os.path.exists(os.path.join(MODEL_PATH, f))]
if missing_files:
    logging.error(f"Missing files in {MODEL_PATH}: {', '.join(missing_files)}")
    exit(1)

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = BertForQuestionAnswering.from_pretrained(MODEL_PATH, local_files_only=True).to(device)

# Perform model warm-up to reduce first-request latency
with torch.no_grad():
    dummy_inputs = tokenizer.encode_plus(
        "warm-up question",
        "warm-up context",
        add_special_tokens=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )
    model(**dummy_inputs)

logging.info("Model warmed up and ready for inference!")

@app.route('/health', methods=['GET'])
@limiter.limit("10 per minute")
def health():
    """Health check endpoint to verify API status."""
    return jsonify({"status": "healthy", "message": "JokerAI is running smoothly!"}), 200

@app.route('/', methods=['GET'])
@limiter.limit("10 per minute")
def home():
    """Root endpoint with API usage information."""
    return jsonify({
        "message": "Welcome to JokerAI's Question-Answering API!",
        "instructions": "Use the /answer endpoint to ask questions.",
        "example_request": {
            "endpoint": "/answer",
            "method": "POST",
            "body": {
                "question": "What is AI?",
                "context": "AI refers to artificial intelligence, the simulation of human intelligence in machines."
            }
        }
    }), 200

@app.route('/answer', methods=['POST'])
@limiter.limit("5 per minute")
def get_answer():
    """Endpoint to handle question-answering requests."""
    # Validate request content type
    if request.content_type != 'application/json':
        return jsonify({
            "status": "error",
            "message": "Invalid content type. Please send a JSON object."
        }), 415  # Unsupported Media Type

    # Parse JSON data from the request
    data = request.get_json()
    if not data or not isinstance(data, dict):
        return jsonify({
            "status": "error",
            "message": "Invalid input format. Please send a JSON object with 'question' and 'context'."
        }), 400

    # Extract question and context from the request body
    question = data.get('question', '').strip()
    context = data.get('context', '').strip()

    # Check for missing fields or empty inputs
    if not question or not context:
        return jsonify({
            "status": "error",
            "message": "Both 'question' and 'context' fields are required and cannot be empty."
        }), 400

    logging.info("Received question: %s", question)

    # Measure response time
    start_time = time()

    # Tokenize the inputs
    try:
        inputs = tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        ).to(device)
    except Exception as e:
        logging.error("Tokenization error: %s", str(e))
        return jsonify({
            "status": "error",
            "message": "Failed to process inputs. Ensure the context and question are valid strings."
        }), 500

    # Perform inference with the model
    try:
        with torch.no_grad():
            outputs = model(**inputs)

        start_logits, end_logits = outputs.start_logits, outputs.end_logits

        # Get start and end indices of the answer
        start_idx = torch.argmax(start_logits)
        end_idx = torch.argmax(end_logits)

        # Check for valid predictions
        if start_idx >= end_idx or start_idx == 0 or end_idx == 0:
            return jsonify({
                "status": "success",
                "question": question,
                "answer": "No valid answer found.",
                "confidence": "0.00"
            }), 200

        # Decode the predicted answer
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx + 1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

        # Handle empty or nonsensical answers
        if not answer.strip():
            return jsonify({
                "status": "success",
                "question": question,
                "answer": "No valid answer found.",
                "confidence": "0.00"
            }), 200

        # Calculate confidence scores
        start_prob = torch.softmax(start_logits, dim=-1)[0][start_idx].item()
        end_prob = torch.softmax(end_logits, dim=-1)[0][end_idx].item()
        confidence = (start_prob + end_prob) / 2

        # Log response time
        response_time = time() - start_time
        logging.info("Response time: %.2f seconds", response_time)

        # Return success response
        return jsonify({
            "status": "success",
            "question": question,
            "answer": answer,
            "confidence": f"{confidence:.2f}",
            "response_time": f"{response_time:.2f} seconds"
        }), 200

    except Exception as e:
        logging.error("Model inference error: %s", str(e))
        return jsonify({
            "status": "error",
            "message": "An error occurred during model inference. Please try again later."
        }), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
