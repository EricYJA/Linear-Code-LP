import numpy as np
import random
import torch
from opt_mheight import solve_m_height

def minimal_test_solve_m_height():
    # Choose random shape
    r, c = 3, 7
    
    # Create random +/-1 matrix
    data = np.random.choice([1, -1], size=(r, c)).astype(np.float32)
    
    # Call solve_m_height
    m_val = 2
    h_value, _, _ = solve_m_height(data, m_val)
    print("h_value = ", h_value)

def test_case_solve_m_height():
    # The specified G (3 rows, 4 cols):
    G_tensor = torch.tensor([
        [ 1.,  1., -1., -1.],
        [-1., -1., -1., -1.],
        [ 1.,  1.,  1., -1.],
        [ 1.,  1., -1., -1.],
        [ 1.,  1., -1., -1.],
        [ 1.,  1., -1., -1.]
    ], dtype=torch.float32)
    
    # Convert to NumPy array if solve_m_height expects NumPy
    G_np = G_tensor.numpy()
    
    # Your fixed m=2
    m_val = 2
    
    # Now call solve_m_height
    h_value, _, _ = solve_m_height(G_np, m_val)
    
    print(f"Test Input G (shape xxx):\n", G_tensor)
    print(f"m = {m_val}")
    print("solve_m_height output (h):", h_value)

if __name__ == "__main__":
    for i in range(100):
        print(f"Test {i}")
    minimal_test_solve_m_height()
    # test_case_solve_m_height()