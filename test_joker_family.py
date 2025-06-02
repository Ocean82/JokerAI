from transformers import BertTokenizer, BertForSequenceClassification
import tempfile
import torch

# Define the Joker family's tokenizer
class JokerTokenizerTest:
    def __init__(self, vocab_file=None):
        # Load the tokenizer for JokerBERT
        if vocab_file:
            self.tokenizer = BertTokenizer.from_pretrained(vocab_file)
        else:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def test_tokenization(self, input_text):
        # Tokenize the input text and return results
        tokens = self.tokenizer.tokenize(input_text)
        print(f"Tokenized input: {tokens}")
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        print(f"Token IDs: {token_ids}")
        return tokens, token_ids

# Testing with JokerAI-like inputs
def test_joker_family():
    print("Running Joker family tokenization tests...\n")

    tokenizer_test = JokerTokenizerTest()

    # Test example inputs (from JokerAI-like frontend)
    examples = [
        "The JokerAI is unstoppable!",
        "Fine-tuning JokerBERT was so worth it.",
        "Tokenization tests are fun!"
        "It's nice when it rains outside."
        "Sometimes its windy."
        "YouTube is better with no commercials."
        "Crocs are not okay to wear in public... EVER!"
        "Eggs are a wonderful part of breakfast."
        "We had eggs yesterday, but I don't mind eating them everyday!"
        "You should bring an umbrella if you dont want to get wet when it rains!"
        "I really like the way your ass looks in that thong"
        "You suck dick like a pro!"
        "I love how wet you get!"
    ]

    for example in examples:
        print(f"Testing input: {example}")
        tokens, token_ids = tokenizer_test.test_tokenization(example)
        print(f"Tokens: {tokens}\nToken IDs: {token_ids}\n")

if __name__ == "__main__":
    test_joker_family()
