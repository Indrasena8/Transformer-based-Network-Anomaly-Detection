import torch
import torch.nn as nn
import pandas as pd
import numpy as np  # Add this line
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler



# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=256, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        x = self.classifier(x)
        return x

# Load the CIC dataset
cic_data = pd.read_csv("/home/azureuser/Project_V2/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv")  # Update with the correct path

# Check the column names
print(cic_data.columns)

# Preprocess the data
X = cic_data.drop(columns=["Label"])  # Assuming the label column is named "Label"
y = cic_data["Label"]


print(X[X == np.inf].sum())
print(X[X == -np.inf].sum())
print(X[X > 1e10].sum())

# Replace infinity or large values with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)  # Replace NaN values with 0 or other appropriate values


# Normalize features using MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Encode categorical labels (if needed)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)  # Assuming y is already encoded

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create DataLoader for training and testing sets
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the Transformer model
model = TransformerModel(input_dim=X.shape[1], output_dim=len(pd.Series(y).unique()))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
model.train()
for epoch in range(5):  # Adjust the number of epochs as needed
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss}")

# Evaluate the model
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
precision = precision_score(all_labels, all_preds, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
