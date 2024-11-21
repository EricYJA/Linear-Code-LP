#include "opt_mheight.hh"

// // Function to compute the inverse permutation
// std::vector<int> inversePermutation(const std::vector<int>& tau) {
//     std::vector<int> tauInv(tau.size());
//     for (size_t i = 0; i < tau.size(); ++i) {
//         tauInv[tau[i]] = i;  // Map each value in tau to its position
//     }
//     return tauInv;
// }

// Function to solve the m-height optimization problem
std::tuple<double, Eigen::VectorXd, std::tuple<int, int, std::vector<int>, std::vector<int>>>
solveMHeight(const Eigen::MatrixXd& G, int m) {
    int k = G.rows();
    int n = G.cols();

    printf("k: %d, n: %d, m: %d\n", k, n, m);

    double bestHeight = -std::numeric_limits<double>::infinity();
    Eigen::VectorXd bestU;
    std::tuple<int, int, std::vector<int>, std::vector<int>> bestParams;

    // Iterate over all configurations of (a, b, X, psi)
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {
            if (b == a) continue;

            std::vector<int> allIndices(n);
            std::iota(allIndices.begin(), allIndices.end(), 0);

            // Generate all combinations of X
            std::vector<int> remainingIndices = allIndices;
            remainingIndices.erase(std::remove(remainingIndices.begin(), remainingIndices.end(), a), remainingIndices.end());
            remainingIndices.erase(std::remove(remainingIndices.begin(), remainingIndices.end(), b), remainingIndices.end());

            do {
                std::vector<int> X(remainingIndices.begin(), remainingIndices.begin() + m - 1);
                std::sort(X.begin(), X.end());

                std::vector<int> Y;
                for (int j : remainingIndices) {
                    if (std::find(X.begin(), X.end(), j) == X.end()) {
                        Y.push_back(j);
                    }
                }
                std::sort(Y.begin(), Y.end());

                // Generate all sign combinations for psi
                for (int psiConfig = 0; psiConfig < (1 << m); ++psiConfig) {
                    std::vector<int> psi(m);
                    for (int i = 0; i < m; ++i) {
                        psi[i] = (psiConfig & (1 << i)) ? 1 : -1;
                    }

                    printf("a: %d, b: %d, X: ", a, b);
                    for (int x : X) printf("%d ", x);
                    printf(", Y: ");
                    for (int y : Y) printf("%d ", y);
                    printf(", psi: ");
                    for (int p : psi) printf("%d ", p);
                    printf("\n");

                    // Quasi-sorted permutation tau
                    std::vector<int> tau = {a};
                    tau.insert(tau.end(), X.begin(), X.end());
                    tau.push_back(b);
                    tau.insert(tau.end(), Y.begin(), Y.end());

                    std::vector<int> tauInv = inversePermutation(tau);

                    // Linear program setup
                    Eigen::VectorXd c(k);
                    for (int i = 0; i < k; ++i) {
                        c[i] = psi[0] * G(i, a);
                    }

                    Eigen::MatrixXd A;
                    Eigen::VectorXd bIneq;
                    Eigen::MatrixXd AEq;
                    Eigen::VectorXd bEq;

                    // Add constraints for X
                    for (int j : X) {
                        Eigen::RowVectorXd rowPos(k);
                        Eigen::RowVectorXd rowNeg(k);
                        for (int i = 0; i < k; ++i) {
                            rowPos[i] = psi[tauInv[j]] * G(i, j) - psi[0] * G(i, a);
                            rowNeg[i] = -psi[tauInv[j]] * G(i, j);
                        }
                        A.conservativeResize(A.rows() + 1, k);
                        A.row(A.rows() - 1) = rowPos;
                        bIneq.conservativeResize(bIneq.size() + 1);
                        bIneq[bIneq.size() - 1] = 0;

                        A.conservativeResize(A.rows() + 1, k);
                        A.row(A.rows() - 1) = rowNeg;
                        bIneq.conservativeResize(bIneq.size() + 1);
                        bIneq[bIneq.size() - 1] = -1;
                    }

                    // Add constraints for Y
                    for (int j : Y) {
                        Eigen::RowVectorXd row(k);
                        for (int i = 0; i < k; ++i) {
                            row[i] = G(i, j);
                        }
                        A.conservativeResize(A.rows() + 1, k);
                        A.row(A.rows() - 1) = row;
                        bIneq.conservativeResize(bIneq.size() + 1);
                        bIneq[bIneq.size() - 1] = 1;

                        A.conservativeResize(A.rows() + 1, k);
                        A.row(A.rows() - 1) = -row;
                        bIneq.conservativeResize(bIneq.size() + 1);
                        bIneq[bIneq.size() - 1] = 1;
                    }

                    // Add equality constraint for b
                    Eigen::RowVectorXd rowEq(k);
                    for (int i = 0; i < k; ++i) {
                        rowEq[i] = G(i, b);
                    }
                    AEq.conservativeResize(AEq.rows() + 1, k);
                    AEq.row(AEq.rows() - 1) = rowEq;
                    bEq.conservativeResize(bEq.size() + 1);
                    bEq[bEq.size() - 1] = 1;

                    // Solve the LP
                    // (Replace this with a suitable LP solver, e.g., CPLEX, Gurobi, or OSQP)
                    Eigen::VectorXd u;  // Solution vector
                    double objectiveValue = -1; // Placeholder for the objective value

                    // Check if the solution improves the best height
                    if (objectiveValue > bestHeight) {
                        bestHeight = objectiveValue;
                        bestU = u;
                        bestParams = {a, b, X, psi};
                    }
                }
            } while (std::next_permutation(remainingIndices.begin(), remainingIndices.end()));
        }
    }

    return {bestHeight, bestU, bestParams};
}
