import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

# 1) Import your function from opt_mheight.py
import sys
sys.path.append(os.path.abspath("../py_baseline"))  # Make sure Python sees that folder
from opt_mheight import solve_m_height

# ---------------------------
# 2) Hyperparameters
# ---------------------------
MIN_ROWS = 3
MAX_ROWS = 5
MIN_COLS = 3
MAX_COLS = 5
MATRICES_PER_SIZE = 10  # number of matrices for each size pair
FIXED_M = 2            # always use m=2

#  handle unbounded objective value
MAX_VALUE = 1e6

FLATTEN_LEN = MAX_ROWS * MAX_COLS  # 10*10 = 100
INPUT_DIM = FLATTEN_LEN + 1        # Flattened matrix + 1 dimension for m
HIDDEN_DIM = 64
OUTPUT_DIM = 1       # Single float output h
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

# ---------------------------
# 3) Utility: Pad and Flatten
# ---------------------------
def pad_and_flatten(matrix, m, max_rows=MAX_ROWS, max_cols=MAX_COLS):
    """
    Flatten a 2D matrix G, zero-pad it up to (max_rows * max_cols) elements,
    then append integer m as one more float dimension.
    Returns a 1D tensor of length (max_rows * max_cols + 1).
    """
    flattened = matrix.flatten()
    original_len = flattened.shape[0]
    target_len = max_rows * max_cols  # e.g., 10*10 = 100

    # Create a zero vector of length target_len
    padded = torch.zeros(target_len, dtype=torch.float32)
    # Copy the flattened matrix into the padded tensor
    padded[:original_len] = flattened

    # Append the integer m (as float) => final shape (101,)
    final_input = torch.cat([padded, torch.tensor([float(m)], dtype=torch.float32)])
    return final_input

# ---------------------------
# 4) Data Generation
# ---------------------------
def generate_dataset_for_all_sizes(
    min_rows=MIN_ROWS, max_rows=MAX_ROWS,
    min_cols=MIN_COLS, max_cols=MAX_COLS,
    matrices_per_size=MATRICES_PER_SIZE
):
    """
    Generate a dataset containing all matrix sizes in the range:
      rows in [min_rows..max_rows], cols in [min_cols..max_cols].
    For each size, create `matrices_per_size` matrices with elements in {+1, -1}.
    For each G, use m=2, then compute h = solve_m_height(G, 2).
    Finally, pad & flatten G + m into a fixed size = (max_rows*max_cols + 1).
    """
    X_list, y_list = [], []

    for r in range(min_rows, max_rows + 1):
        for c in range(min_cols, max_cols + 1):

            for _ in range(matrices_per_size):
                # Create an (r x c) matrix with each element in {+1, -1}
                G_vals = [random.choice([1, -1]) for __ in range(r * c)]
                G = torch.tensor(G_vals, dtype=torch.float32).view(r, c)

                # Our fixed m=2
                m_val = FIXED_M

                # Compute ground truth h
                # print(f"G: {G.shape}, m: {m_val}")
                # print(G)
                h_value, _u, _p = solve_m_height(G.cpu().numpy(), int(m_val))

                # Pad & flatten
                input_vec = pad_and_flatten(G, m_val, max_rows=max_rows, max_cols=max_cols)
                X_list.append(input_vec)
                y_list.append(h_value)

    # Convert to torch tensors
    X_tensor = torch.stack(X_list)
    y_tensor = torch.tensor(y_list, dtype=torch.float32).view(-1, 1)

    return X_tensor, y_tensor

# Generate dataset for all sizes
X_data, y_data = generate_dataset_for_all_sizes()

# fix the inf problem
y_data[torch.isinf(y_data)] = MAX_VALUE

print("Min of y_data:", torch.min(y_data).item())
print("Max of y_data:", torch.max(y_data).item())
print("Any NaN in y_data?", torch.isnan(y_data).any().item())
print("Any Inf in y_data?", torch.isinf(y_data).any().item())

print("Any NaN in X_data?", torch.isnan(X_data).any().item())
print("Any Inf in X_data?", torch.isinf(X_data).any().item())

# Create DataLoader
dataset = torch.utils.data.TensorDataset(X_data, y_data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Dataset shape: X={X_data.shape}, y={y_data.shape}")
# Should be (#size_pairs * 10, 101) for X, (#size_pairs*10,1) for y
#   #size_pairs = 8*8 = 64 (since we go from 3..10 for rows and 3..10 for cols)
#   so total #samples = 64 * 10 = 640

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
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

    avg_loss = total_loss / len(dataset)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")


# ---------------------------
# 8) Test / Verification
# ---------------------------
model.eval()
with torch.no_grad():
    # Example: pick one shape randomly from [3..10, 3..10]
    r_test = random.randint(3, 5)
    c_test = random.randint(3, 5)

    # Create a random +/-1 matrix
    G_test_vals = [random.choice([0.5, -0.5]) for _ in range(r_test*c_test)]
    G_test = torch.tensor(G_test_vals, dtype=torch.float32).view(r_test, c_test)

    # Our fixed m=2
    m_test = FIXED_M

    # True h from your function
    h_true, _, _ = solve_m_height(G_test.cpu().numpy(), int(m_test))

    # Pad + flatten
    test_input = pad_and_flatten(G_test, m_test, MAX_ROWS, MAX_COLS).unsqueeze(0).to(DEVICE)

    h_pred = model(test_input).item()

print(f"Test matrix shape: ({r_test} x {c_test})")
print(f"m_test = {m_test}")
print(f"Ground truth h (solve_m_height): {h_true}")
print(f"Model prediction: {h_pred}")
