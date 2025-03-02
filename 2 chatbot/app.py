import openai
import pandas as pd
import numpy as np
import os
import speech_recognition as sr
import tempfile
import pyttsx3

# Set up OpenAI API
openai.api_type = "open_ai"
openai.api_base = "http://localhost:1234/v1"
openai.api_key = "NULL"

def read_csv_data(file_path):
    df = pd.read_csv(file_path, encoding='latin1')  # Try different encodings if needed
    return df

def preprocess_data(df):
    prompts = df['RecipeName'].tolist()  # Column name
    responses = df['Procedure'].tolist()  # Column name
    training_data = [{'prompt': f"Recipe name: {p}\nInstructions:", 'response': r} for p, r in zip(prompts, responses)]
    return training_data

def dynamic_hyperparameters(training_data):
    # Simulate dynamic adjustment of hyperparameters
    data_size = len(training_data)
    n_epochs = min(10, max(1, data_size // 100))  # Example: adjust epochs based on data size
    return n_epochs

def data_augmentation(training_data):
    # Example of data augmentation by generating slightly varied prompts
    augmented_data = []
    for item in training_data:
        prompt_variations = [item['prompt'], item['prompt'].replace("Recipe name:", "Dish:")]
        for prompt in prompt_variations:
            augmented_data.append({'prompt': prompt, 'response': item['response']})
    return augmented_data

def train_model(training_data):
    # Prepare training data for OpenAI API format
    training_examples = [{"prompt": item['prompt'], "completion": item['response']} for item in training_data]

    # Dynamic hyperparameters
    n_epochs = dynamic_hyperparameters(training_examples)

    # Fine-tune the model using OpenAI API
    try:
        response = openai.FineTune.create(
            training_file=training_examples,
            model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",  # Updated model name
            n_epochs=n_epochs  # Dynamically adjusted epochs
        )
        print("Model trained successfully!")
        return response
    except openai.error.OpenAIError as e:
        print("Error:", e)
        return None

def generate_voice_output(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio)
        print(f"User said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None

def exit_program():
    print("Exiting the program.")
    exit()

# Example usage
file_path = "food2.csv"
data = read_csv_data(file_path)
processed_data = preprocess_data(data)
augmented_data = data_augmentation(processed_data)  # Apply data augmentation
response = train_model(augmented_data)

# Example interaction with the trained model
if response:
    while True:
        user_input = recognize_speech()
        if user_input:
            if user_input.lower() == 'exit':
                exit_program()
            try:
                chat_response = openai.ChatCompletion.create(
                    model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",  # Same model used for training
                    messages=[
                        {"role": "user", "content": user_input}
                    ],
                    timeout=30  # Set a timeout for the API request
                )
                response_text = chat_response['choices'][0]['message']['content']
                print("Chatbot:", response_text)
                generate_voice_output(response_text)  # Generate and play voice output
            except openai.error.OpenAIError as e:
                print("Error during chat interaction:", e)
else:
    print("Error in training the model.")
