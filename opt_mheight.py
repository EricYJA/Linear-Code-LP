import numpy as np
from itertools import combinations
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
                
                # Generate all possible sign combinations for psi
                for psi in combinations([-1, 1], m):
                    psi = list(psi)
                    
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

# Example usage:
G = np.array([[0.911, 0.03, 1.481, -0.756, 1.249],
              [-0.049, 0.975, 1.511, -1.303, 0.74]])  # Generator matrix
m = 2  # Index for m-height calculation

optimal_height, optimal_u, optimal_params = solve_m_height(G, m)
print(f"Optimal m-height: {optimal_height}")
print(f"Optimal vector u: {optimal_u}")
print(f"Optimal (a, b, X, psi): {optimal_params}")

print(f"Verify m-height calculation: {calculate_m_height(G, optimal_u, m)}")