import openai
import pandas as pd

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
    training_data = [{'prompt': p, 'response': r} for p, r in zip(prompts, responses)]
    return training_data

def train_model(training_data):
    # Prepare training data for OpenAI API format
    training_examples = [{"prompt": item['prompt'], "completion": item['response']} for item in training_data]
    
    # Fine-tune the model using OpenAI API
    try:
        response = openai.FineTune.create(
            training_file=training_examples,
            model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",  # Updated model name
            n_epochs=4  # Number of epochs for training, you can adjust this
        )
        print("Model trained successfully!")
        return response
    except openai.error.OpenAIError as e:
        print("Error:", e)
        return None

def exit_program():
    print("Exiting the program.")
    exit()

# Example usage
file_path = "food2.csv"
data = read_csv_data(file_path)
processed_data = preprocess_data(data)
response = train_model(processed_data)

# Example interaction with the trained model
if response:
    while True:
        user_input = input("Ask a question related to a recipe or procedure (type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            exit_program()
        try:
            chat_response = openai.ChatCompletion.create(
                model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",  # Same model used for training
                messages=[
                    {"role": "user", "content": user_input}
                ]
            )
            print("Chatbot:", chat_response['choices'][0]['message']['content'])
        except openai.error.OpenAIError as e:
            print("Error during chat interaction:", e)
else:
    print("Error in training the model.")