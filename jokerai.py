from JokerBERT import JokerBERT  # Import JokerBERT class

class JokerAI:
    def __init__(self):
        print("Initializing JokerAI...")
        self.nlp = JokerBERT()  # JokerBERT handles all NLP tasks

    def generate_response(self, user_input):
        return self.nlp.generate_response(user_input)  # Delegate to JokerBERT

    def run(self):
        print("JokerAI is running. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower().strip() in ['exit', 'quit']:
                print("Goodbye!")
                break

            response = self.generate_response(user_input)
            print("JokerAI:", response)

if __name__ == "__main__":
    joker_ai = JokerAI()
    joker_ai.run()
