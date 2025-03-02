import openai
import pdfplumber
import pandas as pd
import speech_recognition as sr
import pyttsx3
import json

# Set up OpenAI API
openai.api_type = "open_ai"
openai.api_base = "http://localhost:1234/v1"
openai.api_key = "NULL"

def read_pdf_data(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def preprocess_data(text):
    # Assuming the PDF text is structured with questions and answers
    # Adjust this based on the actual content format of the PDF
    data = text.split("\n\n")  # Split by double newline as a rough separator
    questions = []
    answers = []
    
    for i in range(0, len(data) - 1, 2):
        questions.append(data[i].strip())
        answers.append(data[i + 1].strip())
    
    training_data = [{'prompt': f"Legal question: {q}\nAnswer:", 'response': a} for q, a in zip(questions, answers)]
    return training_data

def dynamic_hyperparameters(training_data):
    data_size = len(training_data)
    n_epochs = min(10, max(1, data_size // 100))  # Adjust epochs based on data size
    return n_epochs

def data_augmentation(training_data):
    augmented_data = []
    for item in training_data:
        prompt_variations = [
            item['prompt'],
            item['prompt'].replace("Legal question:", "Law query:"),
            item['prompt'].replace("Legal question:", "Question:")
        ]
        for prompt in prompt_variations:
            augmented_data.append({'prompt': prompt, 'response': item['response']})
    return augmented_data

def train_model(training_data):
    # Save training data to a temporary file with UTF-8 encoding
    with open("training_data.jsonl", "w", encoding='utf-8') as f:
        for item in training_data:
            f.write(f'{{"prompt": "{item["prompt"]}", "completion": "{item["response"]}"}}\n')
    
    # Fine-tune the model using OpenAI API
    try:
        response = openai.FineTune.create(
            training_file="training_data.jsonl",  # Path to the training data file
            model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",  # Use the same model
            n_epochs=dynamic_hyperparameters(training_data)
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
file_path = "law_data.pdf"  # Your legal PDF file path
text = read_pdf_data(file_path)
processed_data = preprocess_data(text)
augmented_data = data_augmentation(processed_data)
response = train_model(augmented_data)

# Example interaction with the trained model
if response:
    print("The model is ready to start listening to the user's input.")
    while True:
        user_input = recognize_speech()
        if user_input:
            if user_input.lower() == 'exit':
                exit_program()
            try:
                chat_response = openai.ChatCompletion.create(
                    model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",  # Same model used for training
                    messages=[
                        {"role": "user", "content": user_input},
                        {"role": "system", "content": "Please provide a natural language explanation."}
                    ],
                    timeout=30
                )
                response_text = chat_response['choices'][0]['message']['content']

                # If the response is JSON, handle it accordingly
                try:
                    json_response = json.loads(response_text)
                    plain_text = json_response.get('description', {}).get('text', "No specific condition was found.")
                    print("Chatbot:", plain_text)
                    generate_voice_output(plain_text)
                except json.JSONDecodeError:
                    # If it's not JSON, assume it's plain text
                    print("Chatbot:", response_text)
                    generate_voice_output(response_text)
            except openai.error.OpenAIError as e:
                print("Error during chat interaction:", e)
else:
    print("Error in training the model.")
