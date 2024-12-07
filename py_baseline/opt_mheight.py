import time
import numpy as np
from itertools import combinations
from itertools import permutations
from itertools import product
from scipy.optimize import linprog
from calc_mheight_params import inverse_permutation
from calc_mheight import calculate_m_height

def solve_m_height(G, m):
    """
    Solve for the optimal m-height and the corresponding vector u.

    Parameters:
    G (numpy.ndarray): Generator matrix (k x n).
    m (int): Index for m-height calculation.

    Returns:
    tuple: Optimal m-height, optimal vector u, and the corresponding (a, b, X, psi).
    """
    k, n = G.shape
    best_height = -float('inf')
    best_u = None
    best_params = None

    # Iterate over all (a, b, X, psi) configurations in Gamma
    for a in range(n):
        for b in range(n):
            if b == a:
                continue
            for X in combinations([j for j in range(n) if j not in {a, b}], m-1):
                X = sorted(X)
                Y = sorted([j for j in range(n) if j not in {a, b} and j not in X])

                # print(f"Checking (a, b, X, Y): ({a}, {b}, {X}, {Y})")
                
                # Generate all possible sign combinations for psi
                for psi in list(product([1, -1], repeat=m)):
                    psi = list(psi)
                    
                    # print(f"Checking (a, b, X, Y, psi): ({a}, {b}, {X}, {Y}, {psi})")

                    # Define the quasi-sorted permutation tau
                    tau = [a] + X + [b] + Y
                    tau_inv = inverse_permutation(tau)
                    
                    # Set up the LP
                    c = np.zeros(k)  # Objective: maximize sum(s_0 * g_{i,a} * u_i)
                    for i in range(k):
                        c[i] = psi[0] * G[i, a]
                    
                    # Constraints
                    A = []  # Inequality constraint matrix
                    b_ineq = []  # Inequality constraint RHS
                    A_eq = []  # Equality constraint matrix
                    b_eq = []  # Equality constraint RHS
                    
                    # Add constraints for j in X
                    for j in X:
                        row_pos = np.zeros(k)
                        row_neg = np.zeros(k)
                        for i in range(k):
                            row_pos[i] = psi[tau_inv[j]] * G[i, j] - psi[0] * G[i, a]
                            row_neg[i] = -psi[tau_inv[j]] * G[i, j]
                        A.append(row_pos)
                        b_ineq.append(0)
                        A.append(row_neg)
                        b_ineq.append(-1)
                    
                    # Add constraints for j in Y
                    for j in Y:
                        row = np.zeros(k)
                        for i in range(k):
                            row[i] = G[i, j]
                        A.append(row)
                        b_ineq.append(1)
                        A.append(-row)
                        b_ineq.append(1)
                    
                    # Add equality constraint for b
                    row_eq = np.zeros(k)
                    for i in range(k):
                        row_eq[i] = G[i, b]
                    A_eq.append(row_eq)
                    b_eq.append(1)
                    
                    # Solve the LP
                    res = linprog(-c, A_ub=A, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, bounds=(None, None), method='highs')
                    
                    # Check if the LP solution is better
                    if res.success and -res.fun > best_height:
                        best_height = -res.fun
                        best_u = res.x
                        best_params = (a, b, X, psi)
    
    return best_height, best_u, best_params

if __name__ == "__main__":
    

    print("Example 1: G(2, 5), m = 2")
    G = np.array([[0.911, 0.03, 1.481, -0.756, 1.249],
                    [-0.049, 0.975, 1.511, -1.303, 0.74]]) 
    m = 2  # Index for m-height calculation

    start = time.perf_counter()

    optimal_height, optimal_u, optimal_params = solve_m_height(G, m)

    end = time.perf_counter()
    print(f"Time taken: {end - start} seconds")

    print(f"Optimal m-height: {optimal_height}")
    print(f"Optimal vector u: {optimal_u}")
    print(f"Optimal (a, b, X, psi): {optimal_params}")

    print(f"Verify m-height calculation: {calculate_m_height(G, optimal_u, m)}")


    print("Example 2: G(7, 10), m = 2")
    G2 = np.array([
        [0.556, 0.96, 0.556, 0.63, 0.556, 0.556, -0.507, -0.507, 0.319, 0.319],
        [0.63, 0.491, -0.58, 0.96, 0.63, -0.507, 0.63, 0.319, 0.556, 0.491],
        [0.319, -0.58, 0.319, -0.507, 0.96, 0.491, 0.319, 0.63, -0.507, 0.96],
        [-0.507, -0.507, -0.507, 0.491, 0.491, 0.319, 0.491, 0.491, 0.491, -0.507],
        [-0.58, 0.556, 0.491, 0.319, 0.319, 0.96, 0.96, -0.58, 0.63, 0.556],
        [0.96, 0.63, 0.96, -0.58, -0.507, 0.63, -0.58, 0.96, -0.58, -0.58],
        [0.491, 0.319, 0.63, 0.556, -0.58, -0.58, 0.556, 0.556, 0.96, 0.63]
    ])

    m = 2

    start = time.perf_counter()

    optimal_height, optimal_u, optimal_params = solve_m_height(G2, m)

    end = time.perf_counter()
    print(f"Time taken: {end - start} seconds")

    print(f"Optimal m-height: {optimal_height}")
    print(f"Optimal vector u: {optimal_u}")
    print(f"Optimal (a, b, X, psi): {optimal_params}")
    
    print(f"Verify m-height calculation: {calculate_m_height(G2, optimal_u, m)}")


    print("Example 3: G(7, 10), m = 3")
    G3 = np.array([
        [1.219, -0.028, 1.206, 0.604, 0.055, -2.649, 1.212, -2.646, -0.373, 0.004],
        [0.04, 1.183, -2.666, -0.387, -0.033, 0.07, -0.863, 0.044, 0.63, -2.537],
        [0.622, 0.039, -0.075, 0.123, 0.62, 0.663, -0.06, -0.825, 0.112, 1.151],
        [-0.815, -0.369, 0.065, -0.045, -0.345, -0.831, -2.596, -0.421, -2.608, -0.106],
        [-0.09, -0.813, -0.38, -0.838, -2.602, 1.133, 0.053, 0.008, 1.203, -0.794],
        [-2.583, 0.672, -0.864, -2.654, -0.834, -0.41, -0.385, 1.166, -0.113, 0.584],
        [-0.411, -2.657, 0.648, 1.183, 1.2, -0.084, 0.638, 0.543, -0.824, -0.364]
    ])

    m = 3

    start = time.perf_counter()
    
    optimal_height, optimal_u, optimal_params = solve_m_height(G3, m)

    end = time.perf_counter()
    print(f"Time taken: {end - start} seconds")

    print(f"Optimal m-height: {optimal_height}")
    print(f"Optimal vector u: {optimal_u}")
    print(f"Optimal (a, b, X, psi): {optimal_params}")
    
    print(f"Verify m-height calculation: {calculate_m_height(G3, optimal_u, m)}")


    print("Example 4: G(14, 20), m = 3, Random G")

    G4 = np.random.uniform(-3, 3, (14, 20))
    
    m = 3

    start = time.perf_counter()
    
    optimal_height, optimal_u, optimal_params = solve_m_height(G4, m)

    end = time.perf_counter()
    print(f"Time taken: {end - start} seconds")
    
    print(f"Optimal m-height: {optimal_height}")
    print(f"Optimal vector u: {optimal_u}")
    print(f"Optimal (a, b, X, psi): {optimal_params}")
    
    print(f"Verify m-height calculation: {calculate_m_height(G4, optimal_u, m)}")
    
