# from flask import Flask, render_template, request
# import torch
# from transformers import RobertaTokenizer, pipeline
# import pickle

# # Initialize Flask app
# app = Flask(__name__)

# # Load the tokenizer
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# # Load the RoBERTa model from the pickle file
# def load_model_from_pkl(filename):
#     try:
#         with open(filename, 'rb') as f:
#             return pickle.load(f)
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return None

# model = load_model_from_pkl('roberta_model2.pkl')

# # Load the label encoder
# try:
#     with open('label_encoder.pkl', 'rb') as f:
#         label_encoder = pickle.load(f)
# except Exception as e:
#     print(f"Error loading label encoder: {e}")
#     label_encoder = None

# # Set the device for PyTorch
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# if model:
#     model = model.to(device)
#     model.eval()

# # Initialize sentiment analysis pipeline (optional use of sentiment)
# sentiment_analyzer = pipeline('sentiment-analysis',device=0 if torch.cuda.is_available() else -1)

# # Custom prediction function
# def custom_predict(text, model, tokenizer, device):
#     # First, check sentiment using the sentiment analyzer
#     sentiment_result = sentiment_analyzer(text)[0]
#     if sentiment_result['label'] == 'POSITIVE':
#         return "No disorder - Positive sentiment", 1.0  # Early return for positive sentiment
    
#     # If sentiment is not positive, perform mental health prediction using the RoBERTa model
#     encoding = tokenizer.encode_plus(
#         text,
#         add_special_tokens=True,
#         max_length=128,
#         return_token_type_ids=False,
#         padding='max_length',
#         truncation=True,
#         return_attention_mask=True,
#         return_tensors='pt',
#     )

#     input_ids = encoding['input_ids'].to(device)
#     attention_mask = encoding['attention_mask'].to(device)

#     # Get the model's predictions
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         logits = outputs.logits
#         probabilities = torch.softmax(logits, dim=1)

#         # Get the predicted class and confidence
#         max_prob, predicted_class = torch.max(probabilities, dim=1)
#         predicted_label = label_encoder.inverse_transform([predicted_class.item()])[0]  # Decode the predicted label

#         return predicted_label, max_prob.item()

# # Home route to render the UI
# @app.route('/')
# def index():
#     return render_template('modelresult.html')

# # Predict route to handle form submission and predictions
# @app.route('/p', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         text = request.form['text']

#         # Predict the mental health condition using the custom_predict function
#         predicted_class, confidence = custom_predict(text, model, tokenizer, device)

#         # Render the prediction result back to the template
#         return render_template('modelresult.html', text=text, predicted_class=predicted_class, confidence=confidence)

# if __name__ == '__main__':
#     app.run(port=5001, debug=True)
