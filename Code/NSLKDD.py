import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Load and preprocess data
data = pd.read_csv('KDDTrain.csv', header=None)

# Last column is the label
features = data.iloc[:, :-1]
labels = data.iloc[:, -1]

# Check for non-numeric values and handle them
# Identify problematic columns
problematic_columns = []
for col in features.columns:
    try:
        # Try converting to numeric to check for non-numeric values
        pd.to_numeric(features[col], errors='raise')
    except ValueError:
        problematic_columns.append(col)

# Handle problematic columns with label encoding
if problematic_columns:
    label_encoder = LabelEncoder()
    for col in problematic_columns:
        features[col] = label_encoder.fit_transform(features[col])

# Normalize numerical features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Convert to PyTorch dataset
class NetworkTrafficDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

train_dataset = NetworkTrafficDataset(X_train, y_train)
test_dataset = NetworkTrafficDataset(X_test, y_test)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define a basic transformer-based model for numerical data
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, 256)  # Linear layer to match embedding layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8), num_layers=3
        )
        self.classifier = nn.Linear(256, n_classes)  # Output classification layer

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x.unsqueeze(0))  # Unsqueeze for batch processing
        x = self.classifier(x.squeeze(0))
        return x

# Model
input_dim = X_train.shape[1]
n_classes = len(np.unique(labels))
model = SimpleTransformer(input_dim, n_classes)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

criterion = nn.CrossEntropyLoss()

for epoch in range (4):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for batch in train_loader:
        batch_features, batch_labels = batch
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == batch_labels).sum().item()
        total_samples += batch_labels.size(0)
        
        loss.backward()
        optimizer.step()
    
    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_samples
    
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}")
