import random
import json
import torch
from model import NeuralNet
from nltk_utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

FILE = 'data.pth'
data = torch.load(FILE)
input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

## Tatsätzliche Chatbot Implementierung

bot_name = "Assistant"

print("Willkommen zu Moodle-Chatbot!")
while True:
    sentence = input("Du: ")
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if tag == "goodbye" and prob.item() > 0.58:
        goodbye_intent = next((item for item in intents['intents'] if item["tag"] == tag), None)
        if goodbye_intent:
            print(f"{bot_name}: {random.choice(goodbye_intent['responses'])}")
        break

    elif prob.item() > 0.58:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                break
    else:
        print(f"{bot_name}: Leider habe ich keine geeignete Antwort dafür! Bitte versuchen Sie es noch einmal.")
        print(f"Debug:  intent '{tag}' Konfidenz {prob.item():.4f}")  # Debug output



