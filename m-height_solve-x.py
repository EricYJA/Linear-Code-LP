import numpy as np
import pickle
import sys
from scipy.optimize import minimize

def load_data(file_name):
    # Load data from a pickle file
    with open(file_name, "rb") as f:
        return pickle.load(f)


def calculate_m_height(c, m):
    # Get absolute values and sort them in descending order
    sorted_abs_values = np.sort(np.abs(c))[::-1]
    
    # Check if m is valid (if m is greater than available entries, return infinity)
    if m >= len(sorted_abs_values) or sorted_abs_values[m] == 0:
        return float('inf')
    
    # Calculate the m-height as the ratio of the largest absolute value to the m-th absolute value
    m_height = abs(sorted_abs_values[0] / sorted_abs_values[m])
    return m_height

def objective(x, G, m):
    # Calculate codeword c = x * G
    c = np.dot(x, G)
    
    # Return negative m-height to maximize it (since minimize minimizes by default)
    return -calculate_m_height(c, m)

if __name__ == "__main__":
    
  # Specify the m value for which to calculate the m-height
  m = 4  # Example m value; adjust as needed

  G = load_data("G1")  # Load the generator matrix G

  # Define the initial guess for x (random or zeros)
  initial_x = np.random.rand(G.shape[0])

  # Use the 'minimize' function to maximize the m-height
  result = minimize(objective, initial_x, args=(G, m), method='L-BFGS-B')

  # Optimal x found
  optimal_x = result.x

  # Calculate the maximum m-height with the found optimal x
  optimal_m_height = -result.fun  # Negate to get the maximized m-height

  print("Optimal x:", optimal_x)
  print("Maximum m-height:", optimal_m_height)



