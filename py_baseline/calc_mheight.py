import numpy as np

def calculate_m_height(G, u, m):
    """
    Calculate the m-height of a codeword generated by a generator matrix G and vector u.

    Parameters:
    G (numpy.ndarray): Generator matrix (k x n).
    u (numpy.ndarray): Input vector (1 x k).
    m (int): Index for m-height calculation.

    Returns:
    float: The m-height value.
    """

    # Ensure u is a 2D row vector for the dot product
    u = u.reshape(1, -1)  # Convert to shape (1, k)

    # Step 1: Compute the codeword
    c = np.dot(u, G)  # u should be (1, k), G should be (k, n)
    
    # Step 2: Sort the elements by their absolute values in descending order
    sorted_abs_c = sorted(np.abs(c[0]), reverse=True)  # Access the first row of c if it's 2D
    
    # Step 3: Calculate the m-height
    if len(sorted_abs_c) > m:  # Ensure m+1 exists
        m_height = abs(sorted_abs_c[0] / sorted_abs_c[m])
    else:
        m_height = float('inf')  # If m+1 element doesn't exist
    
    return m_height


if __name__ == "__main__":
    # Example usage:
    G = np.array([[0.911, 0.03, 1.481, -0.756, 1.249],
                [-0.049, 0.975, 1.511, -1.303, 0.74]])  # Generator matrix
    u = np.array([1.2, -0.8])  # Example input vector (reshaped as (1, 2))
    m = 3  # Index for m-height calculation

    m_height = calculate_m_height(G, u, m)
    print(f"The m-height is: {m_height}")
