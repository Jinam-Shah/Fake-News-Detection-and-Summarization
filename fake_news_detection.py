import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Install necessary packages for Colab
!pip install torch-geometric
!pip install transformers

torch_geometric_installed = False
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    torch_geometric_installed = True
except ModuleNotFoundError:
    print("Torch Geometric is not installed. Please run the pip install command above.")

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
fake_news_df = pd.read_csv('Fake_trim.csv', encoding='ISO-8859-1')  # Placeholder path for fake news data
true_news_df = pd.read_csv('True_trim.csv', encoding='ISO-8859-1')  # Placeholder path for true news data

# Add label column to each dataset
fake_news_df['label'] = 0  # 0 for fake news
true_news_df['label'] = 1  # 1 for true news

# Combine datasets
fake_news_df = fake_news_df.dropna(subset=['text'])
true_news_df = true_news_df.dropna(subset=['text'])
df = pd.concat([fake_news_df, true_news_df], ignore_index=True)
df['text'] = df['text'].astype(str)
df = pd.concat([fake_news_df, true_news_df], ignore_index=True)

# Data Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

df['text'] = df['text'].apply(preprocess_text)

# Display first 5 rows before preprocessing
print("Before Preprocessing:")
print(df['text'].head(5))

# Apply preprocessing
df['text'] = df['text'].apply(preprocess_text)

# Display first 5 rows after preprocessing
print("\nAfter Preprocessing:")
print(df['text'].head(5))

# Splitting the dataset
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32).unsqueeze(1)  # Add sequence length dimension
X_test_tensor = torch.tensor(X_test_tfidf, dtype=torch.float32).unsqueeze(1)    # Add sequence length dimension
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h_0)
        out = self.fc(out[:, -1, :])
        return out

# Initialize the RNN model
input_size = X_train_tfidf.shape[1]
rnn_hidden_size = 128
output_size = 2  # Assuming binary classification
rnn_model = RNNModel(input_size, rnn_hidden_size, output_size)

# GNN Model
if torch_geometric_installed:
    class GNNModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GNNModel, self).__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = F.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

    # Dummy Graph Data for GNN (Placeholder)
    edge_index = torch.tensor([[i, i + 1] for i in range(X_train_tensor.squeeze(1).size(0) - 1)], dtype=torch.long).t().contiguous()  # Creating a chain graph for all nodes
    data_train = Data(x=X_train_tensor.squeeze(1), edge_index=edge_index)
    edge_index_test = torch.tensor([[i, i + 1] for i in range(X_test_tensor.squeeze(1).size(0) - 1)], dtype=torch.long).t().contiguous()  # Creating a chain graph for all nodes in test set
    data_test = Data(x=X_test_tensor.squeeze(1), edge_index=edge_index_test)

    # Initialize the GNN model
    gnn_hidden_dim = 64
    gnn_model = GNNModel(input_size, gnn_hidden_dim, output_size)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
if torch_geometric_installed:
    gnn_optimizer = optim.Adam(gnn_model.parameters(), lr=0.001)

# Training the RNN Model
rnn_train_losses = []
rnn_train_accuracies = []

epochs = 10
for epoch in range(epochs):
    rnn_model.train()
    rnn_optimizer.zero_grad()
    rnn_outputs = rnn_model(X_train_tensor)
    rnn_loss = criterion(rnn_outputs, y_train_tensor)
    rnn_loss.backward()
    rnn_optimizer.step()
    rnn_train_losses.append(rnn_loss.item())

    # Calculate training accuracy
    _, rnn_train_predicted = torch.max(rnn_outputs, 1)
    rnn_train_accuracy = (rnn_train_predicted == y_train_tensor).sum().item() / y_train_tensor.size(0)
    rnn_train_accuracies.append(rnn_train_accuracy)

    print(f'RNN Model - Epoch [{epoch+1}/{epochs}], Loss: {rnn_loss.item():.4f}, Accuracy: {rnn_train_accuracy:.4f}')

# Plot RNN Training Loss and Accuracy
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), rnn_train_losses, label='RNN Training Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('RNN Training Loss Over Epochs')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), rnn_train_accuracies, label='RNN Training Accuracy', marker='o', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('RNN Training Accuracy Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# Training the GNN Model
if torch_geometric_installed:
    gnn_train_losses = []
    gnn_train_accuracies = []
    for epoch in range(epochs):
        gnn_model.train()
        gnn_optimizer.zero_grad()
        gnn_outputs = gnn_model(data_train)
        gnn_loss = criterion(gnn_outputs, y_train_tensor)
        gnn_loss.backward()
        gnn_optimizer.step()
        gnn_train_losses.append(gnn_loss.item())

        # Calculate training accuracy
        _, gnn_train_predicted = torch.max(gnn_outputs, 1)
        gnn_train_accuracy = (gnn_train_predicted == y_train_tensor).sum().item() / y_train_tensor.size(0)
        gnn_train_accuracies.append(gnn_train_accuracy)

        print(f'GNN Model - Epoch [{epoch+1}/{epochs}], Loss: {gnn_loss.item():.4f}, Accuracy: {gnn_train_accuracy:.4f}')

    # Plot GNN Training Loss and Accuracy
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), gnn_train_losses, label='GNN Training Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('GNN Training Loss Over Epochs')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), gnn_train_accuracies, label='GNN Training Accuracy', marker='o', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('GNN Training Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Testing the RNN Model
rnn_model.eval()
rnn_correct = 0
with torch.no_grad():
    rnn_test_outputs = rnn_model(X_test_tensor)
    _, rnn_predicted = torch.max(rnn_test_outputs, 1)
    rnn_correct = (rnn_predicted == y_test_tensor).sum().item()
    rnn_accuracy = rnn_correct / y_test_tensor.size(0)
    print(f'RNN Model Test Accuracy: {rnn_accuracy:.4f}')

# Testing the GNN Model
if torch_geometric_installed:
    gnn_model.eval()
    gnn_correct = 0
    with torch.no_grad():
        gnn_test_outputs = gnn_model(data_test)
        _, gnn_predicted = torch.max(gnn_test_outputs, 1)
        gnn_correct = (gnn_predicted == y_test_tensor).sum().item()
        gnn_accuracy = gnn_correct / y_test_tensor.size(0)
        print(f'GNN Model Test Accuracy: {gnn_accuracy:.4f}')

# User Input for Prediction
user_input = input("Enter news text for prediction: ")
preprocessed_input = preprocess_text(user_input)
input_tfidf = vectorizer.transform([preprocessed_input]).toarray()
input_tensor = torch.tensor(input_tfidf, dtype=torch.float32).unsqueeze(1)

# RNN Prediction
rnn_model.eval()
with torch.no_grad():
    rnn_output = rnn_model(input_tensor)
    _, rnn_prediction = torch.max(rnn_output, 1)

# GNN Prediction
if torch_geometric_installed:
    gnn_model.eval()
    with torch.no_grad():
        data_input = Data(x=input_tensor.squeeze(1), edge_index=torch.tensor([[0], [0]], dtype=torch.long).contiguous())
        gnn_output = gnn_model(data_input)
        _, gnn_prediction = torch.max(gnn_output, 1)

# Summarization using Pegasus
def summarize_text(text):
    model_name = "google/pegasus-xsum"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    summary_ids = model.generate(**tokens)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Final Decision
if rnn_prediction.item() == 1 and (not torch_geometric_installed or gnn_prediction.item() == 1):
    print("This news is True.")
    summary = summarize_text(user_input)
    print(f'Summary: {summary}')
else:
    print("This news is Fake.")
