import torch 
from transformers import RobertaTokenizer, RobertaForSequenceClassification 
from torch.utils.data import DataLoader, Dataset 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
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

# Split the data into features (X) and target (y)
X = data_csv['text'].values
y = data_csv['label'].values

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Tokenize the data
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

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
MAX_LEN = 128

# Create DataLoader
train_dataset = TextDataset(X_train, y_train, tokenizer, MAX_LEN)
test_dataset = TextDataset(X_test, y_test, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load RoBERTa model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(np.unique(y)))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

# Training parameters
EPOCHS = 10
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

# Training function
def train_epoch(model, data_loader, optimizer, loss_fn, device):
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

    return total_loss / len(data_loader)

# Training loop
for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
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

# Save the model's state dictionary to a .pkl file
with open('roberta_model.pkl', 'wb') as file:
    pickle.dump(model.state_dict(), file)

# Save the label encoder to a .pkl file
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

# Custom prediction function
def custom_predict(text, model, tokenizer, label_encoder, device):
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
        _, predicted_class = torch.max(logits, dim=1)

    predicted_label = label_encoder.inverse_transform(predicted_class.cpu().numpy())[0]
    return predicted_label

# Example prediction after loading the model
new_text = "He broke up with me, accused me of lying about my cousin..."
predicted_label = custom_predict(new_text, model, tokenizer, label_encoder, device)
print(f"Predicted class: {predicted_label}")


