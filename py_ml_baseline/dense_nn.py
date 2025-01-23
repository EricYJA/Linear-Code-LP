import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

# 1) Import your function from opt_mheight.py
#    Adjust the path as needed; here we assume it's in ../py_baseline/opt_mheight.py
import sys
sys.path.append(os.path.abspath("../py_baseline"))  # Make sure Python sees that folder
from opt_mheight import solve_m_height


# ---------------------------
# 2) Hyperparameters
# ---------------------------
MAX_SIZE = 15       # Maximum dimension for matrix G (G in R^(d x d), d <= 15)
FLATTEN_LEN = MAX_SIZE * MAX_SIZE  # 15 x 15 = 225
INPUT_DIM = FLATTEN_LEN + 1  # Flattened matrix plus integer m
HIDDEN_DIM = 64
OUTPUT_DIM = 1       # Single float output h
NUM_TRAIN_SAMPLES = 1000
NUM_VALID_SAMPLES = 200
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)


# ---------------------------
# 3) Utility: Pad and Flatten
# ---------------------------
def pad_and_flatten(matrix, m, max_size=MAX_SIZE):
    """
    Flatten a 2D matrix G, zero-pad it up to max_size^2 elements,
    then append integer m as one more float dimension.
    Returns a 1D tensor of length (max_size^2 + 1).
    """
    flattened = matrix.flatten()
    original_len = flattened.shape[0]
    target_len = max_size * max_size  # e.g., 15 * 15 = 225

    # Create a zero vector of length target_len
    padded = torch.zeros(target_len, dtype=torch.float32)
    # Copy the flattened matrix into the padded tensor
    padded[:original_len] = flattened

    # Now append the integer m (as a float). We'll return a 1D tensor of shape (226,)
    final_input = torch.cat([padded, torch.tensor([float(m)], dtype=torch.float32)])
    return final_input


# ---------------------------
# 4) Data Generation
# ---------------------------
def generate_dataset(num_samples, max_size=MAX_SIZE):
    """
    Generate a dataset of random (G, m) pairs with d <= max_size,
    compute h via solve_m_height(G, m),
    and return (X, y) where:
       X -> padded & flattened tensor of shape (num_samples, max_size^2 + 1)
       y -> ground-truth h-values of shape (num_samples, 1)
    """
    X_list, y_list = [], []

    # Fix random seeds for reproducibility (optional)
    # random.seed(42)
    # torch.manual_seed(42)

    for _ in range(num_samples):
        # Random dimension d in [1..max_size]
        d = random.randint(1, max_size)

        # Random matrix G of shape (d, d)
        G = torch.randn(d, d)

        # A random nonzero integer m; adapt the choices as needed
        m_choices = [-10, -5, -2, -1, 1, 2, 5, 10]
        m_val = random.choice(m_choices)

        # Obtain ground-truth h from your custom function
        # Adjust the call signature if your function expects different arguments
        h_value, _, _ = solve_m_height(G, m_val)

        # Convert G, m into a single input vector
        input_vec = pad_and_flatten(G, m_val, max_size=max_size)

        X_list.append(input_vec)
        y_list.append(h_value)

    # Convert to torch tensors
    X_tensor = torch.stack(X_list)  # shape: (num_samples, 226)
    y_tensor = torch.tensor(y_list, dtype=torch.float32).view(-1, 1)  # shape: (num_samples, 1)
    return X_tensor, y_tensor


# Generate training dataset
X_train, y_train = generate_dataset(NUM_TRAIN_SAMPLES, MAX_SIZE)
# Generate validation (or verification) dataset
X_valid, y_valid = generate_dataset(NUM_VALID_SAMPLES, MAX_SIZE)


# Create DataLoaders
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
valid_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ---------------------------
# 5) Define the Model (Dense MLP)
# ---------------------------
class DenseModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

model = DenseModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)


# ---------------------------
# 6) Training Setup
# ---------------------------
criterion = nn.MSELoss()  # For regression
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ---------------------------
# 7) Training Loop
# ---------------------------
for epoch in range(EPOCHS):
    # ---- Train ----
    model.train()
    total_train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * batch_x.size(0)

    avg_train_loss = total_train_loss / len(train_loader.dataset)

    # ---- Validation ----
    model.eval()
    total_valid_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in valid_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_valid_loss += loss.item() * batch_x.size(0)

    avg_valid_loss = total_valid_loss / len(valid_loader.dataset)

    print(f"Epoch [{epoch+1}/{EPOCHS}], "
          f"Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")


# ---------------------------
# 8) Verification / Test Example
# ---------------------------
# Suppose we want to verify with a brand new matrix:
model.eval()
with torch.no_grad():
    # Example: d=12, random G, random m
    G_test = torch.randn(12, 12)
    m_test = 2  # an integer
    # The "true" h from your custom function
    h_true, _, _ = solve_m_height(G_test, m_test)

    # Format the input for the model
    test_input = pad_and_flatten(G_test, m_test, MAX_SIZE).unsqueeze(0).to(DEVICE)

    # Inference
    h_pred = model(test_input)
    h_pred_val = h_pred.item()

print("Verification:")
print(f"  G_test.shape: {G_test.shape},  m_test = {m_test}")
print(f"  Ground-truth h (from solve_m_height): {h_true}")
print(f"  Model prediction: {h_pred_val}")
