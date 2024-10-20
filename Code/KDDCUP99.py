import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


print("Current Working Directory: ", os.getcwd())
# Step 1: Load the dataset
file_path = "kddcup.data_10_percent_corrected"  # Adjust this path to your file
data = pd.read_csv(file_path, header=None)

# Step 2: Data Preprocessing
# - Encode categorical data
label_encoder = LabelEncoder()

# Assumption: last column is the label, others are features
features = data.iloc[:, :-1]
labels = data.iloc[:, -1]

# Convert categorical features to numerical
for col in features.columns:
    if features[col].dtype == 'object':
        features[col] = label_encoder.fit_transform(features[col])

# Encode the labels
labels = label_encoder.fit_transform(labels)

# Normalize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Step 3: Create training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 4: Define a basic transformer-based model for tabular data
class SimpleTransformer(nn.Module):
    def __init__(self, num_features, num_classes, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(num_features, d_model)
        transformer_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=256, dropout=dropout)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x.unsqueeze(1)).squeeze(1)  # Adding seq dimension, then removing
        x = self.classifier(x)
        return x

# Step 5: Initialize model, loss function, and optimizer
model = SimpleTransformer(num_features=X_train.shape[1], num_classes=len(set(y_train)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: Train the model
num_epochs = 10  # Adjust as needed
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# Step 7: Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")

try:
    torch.save(model.state_dict(), '/home/azureuser/transformer_model.pth')
    joblib.dump(scaler, '/home/azureuser/scaler.pkl')
    joblib.dump(label_encoder, '/home/azureuser/label_encoder.pkl')
    print("Model and preprocessors have been saved successfully.")
except Exception as e:
    print("Failed to save files:", e)

