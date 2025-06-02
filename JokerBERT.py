import wikipedia
import torch
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import TokenizationTest

# Class definition for JokerBERT
class JokerBERT:
    def __init__(self):
        print("Loading NLP pipelines...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
        self.qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        self.tokenizer = BertTokenizer.from_pretrained('./fine_tuned_bert')
        self.model = BertForSequenceClassification.from_pretrained('./fine_tuned_bert')
        self.conversation_history = []  # Initialize conversation history

    def sentiment_analysis(self, text):
        sentiment = self.sentiment_analyzer(text)
        return f"Sentiment: {sentiment[0]['label']} (Confidence: {sentiment[0]['score']:.2f})"

    def named_entity_recognition(self, text):
        entities = self.ner_model(text)
        entity_list = [f"{ent['entity_group']} - {ent['word']}" for ent in entities]
        return "Entities: " + ", ".join(entity_list)

    def generate_dynamic_context(self):
        # Create dynamic context from conversation history
        return " ".join([f"User: {entry[0]} Bot: {entry[1]}" for entry in self.conversation_history])

    def question_answering(self, question, context=None):
        # Use dynamic context if no explicit context is provided
        if not context:
            context = self.generate_dynamic_context()
        answer = self.qa_model(question=question, context=context)
        return f"Answer: {answer['answer']} (Confidence: {answer['score']:.2f})"

    def fine_tuned_response(self, text):
        # Tokenize input text and run it through the fine-tuned model
        tokens = self.tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        self.model.eval()
        with torch.no_grad():
            inputs = {key: val.to(self.device) for key, val in tokens.items()}
            outputs = self.model(**inputs).logits
            prediction = torch.argmax(outputs, dim=-1).item()

        return "Positive response!" if prediction == 1 else "Negative response!"

    def detect_intent(self, user_input):
        if any(keyword in user_input.lower() for keyword in ["who", "what", "where", "how", "why"]):
            return "question"
        elif any(keyword in user_input.lower() for keyword in ["happy", "sad", "angry"]):
            return "sentiment"
        elif "tell me about" in user_input.lower():
            return "wikipedia"
        else:
            return "general"

    def fetch_wikipedia_summary(self, query):
        try:
            return wikipedia.summary(query, sentences=5)
        except wikipedia.DisambiguationError as e:
            return f"Ambiguous query. Options: {', '.join(e.options[:5])}."
        except wikipedia.exceptions.PageError:
            return "No Wikipedia article found."

    def add_to_memory(self, user_input, bot_response):
        self.conversation_history.append((user_input, bot_response))
        if len(self.conversation_history) > 25:  # Limit memory size
            self.conversation_history.pop(0)

    def generate_response_with_memory(self, user_input):
        bot_response = self.generate_response(user_input)
        self.add_to_memory(user_input, bot_response)
        return bot_response

    def fallback_response(self, user_input):
        return f"I'm not sure how to respond to '{user_input}'. Could you clarify?"

    def generate_response(self, user_input):
        intent = self.detect_intent(user_input)
        if intent == "question":
            context = self.fetch_wikipedia_summary(user_input)  # Fetch relevant context dynamically
            return self.question_answering(user_input, context)
        elif intent == "sentiment":
            return self.sentiment_analysis(user_input)
        elif intent == "wikipedia":
            query = user_input.lower().replace("tell me about", "").strip()
            return self.fetch_wikipedia_summary(query)
        else:
            return self.fallback_response(user_input)