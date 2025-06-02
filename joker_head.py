import requests

# URL of the JokerBERT API
API_URL = "http://127.0.0.1:5000/predict"

def get_prediction(user_input):
    response = requests.post(API_URL, json={"text": user_input})
    if response.status_code == 200:
        data = response.json()
        return data['prediction']
    else:
        return "Error: Unable to connect to backend"

# Example input from JokerAI
user_input = "The JokerAI is unstoppable!"
prediction = get_prediction(user_input)
print(f"User Input: {user_input} | Prediction: {prediction}")
