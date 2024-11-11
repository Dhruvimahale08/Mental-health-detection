import torch # type: ignore
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup # type: ignore
from torch.utils.data import DataLoader, Dataset # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load dataset
data_csv = pd.read_csv('D:\\Study\\study-sem 5\\SGP\\datasets\\dataset\\dreaddit-train_updated.csv', encoding='latin-1')

# Custom mapping of subreddits to aggregated categories
def map_subreddits(subreddit):
    if subreddit in ['domesticviolence']:
        return 'ptsd'
    elif subreddit in ['homeless', 'almosthomeless']:
        return 'stress'
    elif subreddit in ['survivorsofabuse', 'assistance']:   
        return 'anxiety'
    else:
        return subreddit  # Keep other subreddits unchanged

# Apply the custom mapping
data_csv['subreddit'] = data_csv['subreddit'].apply(map_subreddits)

# Drop rows with missing values
data_csv = data_csv.dropna(subset=['text', 'subreddit'])

# Encode the 'subreddit' column as labels
label_encoder = LabelEncoder()
data_csv['label'] = label_encoder.fit_transform(data_csv['subreddit'])

# Visualize class distribution
data_csv['subreddit'].value_counts().plot(kind='bar', title='Class Distribution')
plt.show()

# Split the data into features (X) and target (y)
X = data_csv['text'].values
y = data_csv['label'].values

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Tokenize the data
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Parameters
BATCH_SIZE = 8
MAX_LEN = 256  # Increased max length to handle longer text

# Create DataLoader
train_dataset = TextDataset(X_train, y_train, tokenizer, MAX_LEN)
test_dataset = TextDataset(X_test, y_test, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load RoBERTa Large model
model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=len(np.unique(y)), hidden_dropout_prob=0.2)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

# Training parameters
EPOCHS = 5  # Fewer epochs because 'roberta-large' is heavier
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Learning rate scheduler
num_training_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0, 
    num_training_steps=num_training_steps
)

# Loss function with class weights
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

# Training function
def train_epoch(model, data_loader, optimizer, loss_fn, device, scheduler):
    model.train()
    total_loss = 0

    for batch in data_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()  # Step the scheduler

    return total_loss / len(data_loader)

# Training loop
for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, scheduler)
    print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss}')

# Evaluation function
def eval_model(model, data_loader, device, label_encoder):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return classification_report(true_labels, predictions, target_names=label_encoder.classes_)

# Evaluate the model
report = eval_model(model, test_loader, device, label_encoder)
print(report)

# Save the model to a PKL file
def save_model_to_pkl(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

# Load the model from a PKL file
def load_model_from_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Saving the model
save_model_to_pkl(model, 'roberta_large_model.pkl')

# Loading the model
model_loaded = load_model_from_pkl('roberta_large_model.pkl')

# Custom prediction function with confidence check
def custom_predict(text, model, tokenizer):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Get the model predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)  # Get probabilities

        # Get the predicted class and confidence
        max_prob, predicted_class = torch.max(probabilities, dim=1)

        # Check if the probability of the predicted class is above a certain threshold
        if max_prob.item() < 0.6:  # threshold can be adjusted
            predicted_label = "uncertain"
        else:
            predicted_label = label_encoder.inverse_transform(predicted_class.cpu().numpy())[0]

        return predicted_label, max_prob.item()

# Test the updated function with a happy sentence
new_text = "I'm so happy today!"
predicted_label, confidence = custom_predict(new_text, model_loaded, tokenizer)
print(f"Predicted label: {predicted_label}, Confidence: {confidence}")
