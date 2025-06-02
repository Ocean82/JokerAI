from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the JokerBERT tokenizer and model
class JokerFamilyPipeline:
    def __init__(self, model_path='./fine_tuned_bert'):
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.label_mapping = {0: "Negative", 1: "Positive"}  # Example classification labels

    def process_input(self, input_text):
        # Tokenize the input text
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        return inputs

    def predict(self, inputs):
        # Run the tokenized input through the model
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(dim=-1).item()
        return predicted_class

    def run_pipeline(self, input_text):
        # Full pipeline from input to output
        inputs = self.process_input(input_text)
        prediction = self.predict(inputs)
        return self.label_mapping[prediction]
        
# Example script to run the pipeline
if __name__ == "__main__":
    print("Welcome to the Joker Family AI!")
    
    # Initialize the pipeline
    joker_pipeline = JokerFamilyPipeline()

    # Test inputs
    test_inputs = [
        "The JokerAI is unstoppable!",
        "Fine-tuning JokerBERT was worth it.",
        "Tokenization tests are fun!"
    ]
    
    # Process each input and display predictions
    for text in test_inputs:
        result = joker_pipeline.run_pipeline(text)
        print(f"Input: {text} | Prediction: {result}")
