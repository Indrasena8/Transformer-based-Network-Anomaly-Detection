import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import torch
import joblib
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define column names
column_names = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl',
                'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin',
                'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
                'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm',
                'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
                'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 'attack_cat', 'label']

# Read the CSV files with column names
train_df = pd.read_csv('dataset/UNSW_NB15_training-set.csv', names=column_names, low_memory=False)
test_df = pd.read_csv('dataset/UNSW_NB15_testing-set.csv', names=column_names, low_memory=False)
features_df = pd.read_csv('dataset/NUSW-NB15_features.csv', encoding='latin1')
list_events_df = pd.read_csv('dataset/UNSW-NB15_LIST_EVENTS.csv')

columns_to_convert = [
    'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 
    'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 
    'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 
    'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 
    'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 
    'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 
    'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports'
]

train_df[columns_to_convert] = train_df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

print(train_df.dtypes)

additional_file_1 = pd.read_csv('dataset/UNSW-NB15_1.csv', names=column_names, low_memory=False)
additional_file_2 = pd.read_csv('dataset/UNSW-NB15_2.csv', names=column_names, low_memory=False)
additional_file_3 = pd.read_csv('dataset/UNSW-NB15_3.csv', names=column_names, low_memory=False)

# Concatenate additional datasets with training data
train_df = pd.concat([train_df, additional_file_1, additional_file_2, additional_file_3], ignore_index=True)

print("Training Dataset:")
print(train_df.dtypes)

# Check for mixed data types in the testing dataset
print("\nTesting Dataset:")
print(test_df.dtypes)

print("\nAdditional Dataset 1:")
print(additional_file_1.dtypes)
print("\nAdditional Dataset 2:")
print(additional_file_2.dtypes)

# Check for missing values or unexpected values
print("\nMissing Values in Training Dataset:")
print(train_df.isnull().sum())

# Handle missing values in the target label
train_df['attack_cat'].fillna('Unknown', inplace=True)

# Convert the target label to numeric
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_df['attack_cat'])

# Define features to use for training
features = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl',
            'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin',
            'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
            'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm',
            'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
            'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']

# Prepare X (features) and y (target)
X = train_df[features]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=train_df['attack_cat'])

print("Missing Values in Encoded Labels:")
print(pd.Series(y_train).isnull().sum())

# Preprocessing pipelines for numerical and categorical features
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Define the model architecture
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, 64)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64, nhead=8), num_layers=6)
        self.classifier = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Swap batch and sequence dimensions
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Aggregate across sequence dimension
        x = self.classifier(x)
        return x

# Define training parameters
input_dim = len(features)
output_dim = len(np.unique(y_train))

# Initialize model, optimizer, and loss function
model = TransformerModel(input_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(preprocessor.fit_transform(X_train).toarray(), dtype=torch.float32)
X_val_tensor = torch.tensor(preprocessor.transform(X_val).toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Create DataLoader objects
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=64)

# Train the model
for epoch in range(5):
    model.train()
    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            outputs = model(batch_features)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_labels.tolist())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    print(f"Accuracy: {accuracy}")

# Evaluate on the test set
X_test = test_df[features]
y_test = test_df['attack_cat']
X_test_tensor = torch.tensor(preprocessor.transform(X_test).toarray(), dtype=torch.float32)
y_test_tensor = torch.tensor(label_encoder.transform(y_test), dtype=torch.long)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64)

torch.save(model.state_dict(), 'transformer_model.pth')

joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model, preprocessor, and label encoder have been saved.")
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        outputs = model(batch_features)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(batch_labels.tolist())

# Calculate accuracy on test set
test_accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {test_accuracy}")


