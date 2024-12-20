#include "opt_mheight.hh"

static const double INF = 1e9;

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
solveMHeight(const Eigen::MatrixXd &G, int m)
{
    int k = G.rows();
    int n = G.cols();

    std::cout << "k: " << k << ", n: " << n << ", m: " << m << std::endl;

    double bestHeight = -std::numeric_limits<double>::infinity();
    Eigen::VectorXd bestU(k);
    std::tuple<int, int, std::vector<int>, std::vector<int>> bestParams;

    std::vector<int> allIndices(n);
    std::iota(allIndices.begin(), allIndices.end(), 0);

    // Generate (a, b), X, Y combinations
    for (int a = 0; a < n; ++a)
    {
        for (int b = 0; b < n; ++b)
        {
            if (b == a)
                continue;

            std::vector<int> remainingIndices = allIndices;
            remainingIndices.erase(std::remove(remainingIndices.begin(), remainingIndices.end(), a), remainingIndices.end());
            remainingIndices.erase(std::remove(remainingIndices.begin(), remainingIndices.end(), b), remainingIndices.end());

            // Generate all combinations of size (m-1) for X
            std::vector<std::vector<int>> combinations;
            {
                std::function<void(int, int, std::vector<int> &)> combGen = [&](int offset, int r, std::vector<int> &temp)
                {
                    if (r == 0)
                    {
                        combinations.push_back(temp);
                        return;
                    }
                    for (int i = offset; i <= (int)remainingIndices.size() - r; ++i)
                    {
                        temp.push_back(remainingIndices[i]);
                        combGen(i + 1, r - 1, temp);
                        temp.pop_back();
                    }
                };
                std::vector<int> temp;
                combGen(0, m - 1, temp);
            }

            // For each combination of X
            for (const auto &X : combinations)
            {
                std::vector<int> Y;
                for (int idx : remainingIndices)
                {
                    if (std::find(X.begin(), X.end(), idx) == X.end())
                    {
                        Y.push_back(idx);
                    }
                }

                // Try all psi combinations
                for (int psiConfig = 0; psiConfig < (1 << m); ++psiConfig)
                {
                    std::vector<int> psi(m);
                    for (int i = 0; i < m; ++i)
                    {
                        psi[i] = (psiConfig & (1 << i)) ? 1 : -1;
                    }

                    // Construct tau
                    std::vector<int> tau;
                    tau.push_back(a);
                    tau.insert(tau.end(), X.begin(), X.end());
                    tau.push_back(b);
                    tau.insert(tau.end(), Y.begin(), Y.end());

                    std::vector<int> tauInv = inversePermutation(tau);

                    // Total constraints:
                    // For each j in X: 2 rows
                    // For each j in Y: 2 rows
                    // For b: 1 row
                    int totalRows = 2 * (int)X.size() + 2 * (int)Y.size() + 1;

                    glp_prob *lp = glp_create_prob();
                    glp_set_obj_dir(lp, GLP_MAX);
                    glp_add_rows(lp, totalRows);
                    glp_add_cols(lp, k);

                    // Set variables to be free: u_i ∈ (-∞, ∞)
                    for (int col = 1; col <= k; ++col)
                    {
                        glp_set_col_bnds(lp, col, GLP_FR, -INF, INF);
                    }

                    // Set objective: maximize sum(psi[0]*G(i,a)*u_i)
                    for (int i = 0; i < k; ++i)
                    {
                        glp_set_obj_coef(lp, i + 1, psi[0] * G(i, a));
                    }

                    int rowIndex = 1;

                    // Constraints for X:
                    // rowPos: (psi[tauInv[j]] * G(:,j) - psi[0]*G(:,a)) * u ≤ 0
                    // rowNeg: (-psi[tauInv[j]] * G(:,j)) * u ≤ -1
                    for (int jx : X)
                    {
                        {
                            // rowPos ≤ 0
                            std::vector<int> idx(k + 1);
                            std::vector<double> val(k + 1);
                            for (int col = 0; col < k; ++col)
                            {
                                idx[col + 1] = col + 1;
                                val[col + 1] = psi[tauInv[jx]] * G(col, jx) - psi[0] * G(col, a);
                            }
                            glp_set_row_bnds(lp, rowIndex, GLP_UP, -INF, 0.0);
                            glp_set_mat_row(lp, rowIndex, k, idx.data(), val.data());
                            rowIndex++;
                        }
                        {
                            // rowNeg ≤ -1
                            // -psi[tauInv[jx]]*G(:,jx)*u ≤ -1
                            // upper bound = -1, lower bound = -∞
                            std::vector<int> idx(k + 1);
                            std::vector<double> val(k + 1);
                            for (int col = 0; col < k; ++col)
                            {
                                idx[col + 1] = col + 1;
                                val[col + 1] = -psi[tauInv[jx]] * G(col, jx);
                            }
                            glp_set_row_bnds(lp, rowIndex, GLP_UP, -INF, -1.0);
                            glp_set_mat_row(lp, rowIndex, k, idx.data(), val.data());
                            rowIndex++;
                        }
                    }

                    // Constraints for Y:
                    // G(:,j)*u ≤ 1 and -G(:,j)*u ≤ 1
                    for (int jy : Y)
                    {
                        {
                            // G(:,jy)*u ≤ 1
                            std::vector<int> idx(k + 1);
                            std::vector<double> val(k + 1);
                            for (int col = 0; col < k; ++col)
                            {
                                idx[col + 1] = col + 1;
                                val[col + 1] = G(col, jy);
                            }
                            glp_set_row_bnds(lp, rowIndex, GLP_UP, -INF, 1.0);
                            glp_set_mat_row(lp, rowIndex, k, idx.data(), val.data());
                            rowIndex++;
                        }
                        {
                            // -G(:,jy)*u ≤ 1
                            std::vector<int> idx(k + 1);
                            std::vector<double> val(k + 1);
                            for (int col = 0; col < k; ++col)
                            {
                                idx[col + 1] = col + 1;
                                val[col + 1] = -G(col, jy);
                            }
                            glp_set_row_bnds(lp, rowIndex, GLP_UP, -INF, 1.0);
                            glp_set_mat_row(lp, rowIndex, k, idx.data(), val.data());
                            rowIndex++;
                        }
                    }

                    // Equality for b: G(:,b)*u = 1
                    {
                        std::vector<int> idx(k + 1);
                        std::vector<double> val(k + 1);
                        for (int col = 0; col < k; ++col)
                        {
                            idx[col + 1] = col + 1;
                            val[col + 1] = G(col, b);
                        }
                        glp_set_row_bnds(lp, rowIndex, GLP_FX, 1.0, 1.0);
                        glp_set_mat_row(lp, rowIndex, k, idx.data(), val.data());
                        rowIndex++;
                    }

                    // Solve LP
                    glp_simplex(lp, NULL);
                    int status = glp_get_status(lp);

                    // If feasible or optimal, check objective
                    if (status == GLP_OPT || status == GLP_FEAS)
                    {
                        double objectiveValue = glp_get_obj_val(lp);
                        if (objectiveValue > bestHeight)
                        {
                            bestHeight = objectiveValue;
                            for (int i = 0; i < k; ++i)
                            {
                                bestU[i] = glp_get_col_prim(lp, i + 1);
                            }
                            bestParams = {a, b, X, psi};
                        }
                    }

                    glp_delete_prob(lp);
                }
            }
        }
    }

    return {bestHeight, bestU, bestParams};
}

std::tuple<double, Eigen::VectorXd, std::tuple<int, int, std::vector<int>, std::vector<int>>>
solveMHeight_openmp(const Eigen::MatrixXd &G, int m)
{
    int k = G.rows();
    int n = G.cols();

    std::cout << "k: " << k << ", n: " << n << ", m: " << m << std::endl;

    double bestHeight = -std::numeric_limits<double>::infinity();
    Eigen::VectorXd bestU(k);
    std::tuple<int, int, std::vector<int>, std::vector<int>> bestParams;

    std::vector<int> allIndices(n);
    std::iota(allIndices.begin(), allIndices.end(), 0);

    // We'll parallelize the loops over a and b.
    // The inner computations are independent as long as we don't share state incorrectly.
    // We'll keep track of thread-local best solutions and then update globally.

    // To avoid data races on bestHeight, bestU, bestParams,
    // updates will be done in a critical section.
#pragma omp parallel
    {
        double localBestHeight = -std::numeric_limits<double>::infinity();
        Eigen::VectorXd localBestU(k);
        std::tuple<int, int, std::vector<int>, std::vector<int>> localBestParams;

#pragma omp for collapse(2) schedule(dynamic)
        for (int a = 0; a < n; ++a)
        {
            for (int b = 0; b < n; ++b)
            {
                if (b == a)
                    continue;

                std::vector<int> remainingIndices = allIndices;
                remainingIndices.erase(std::remove(remainingIndices.begin(), remainingIndices.end(), a), remainingIndices.end());
                remainingIndices.erase(std::remove(remainingIndices.begin(), remainingIndices.end(), b), remainingIndices.end());

                // Generate all combinations of size (m-1) for X
                std::vector<std::vector<int>> combinations;
                {
                    std::function<void(int, int, std::vector<int> &)> combGen = [&](int offset, int r, std::vector<int> &temp)
                    {
                        if (r == 0)
                        {
                            combinations.push_back(temp);
                            return;
                        }
                        for (int i = offset; i <= (int)remainingIndices.size() - r; ++i)
                        {
                            temp.push_back(remainingIndices[i]);
                            combGen(i + 1, r - 1, temp);
                            temp.pop_back();
                        }
                    };
                    std::vector<int> temp;
                    combGen(0, m - 1, temp);
                }

                // For each combination of X
                for (const auto &X : combinations)
                {
                    std::vector<int> Y;
                    for (int idx : remainingIndices)
                    {
                        if (std::find(X.begin(), X.end(), idx) == X.end())
                        {
                            Y.push_back(idx);
                        }
                    }

                    // Try all psi combinations
                    for (int psiConfig = 0; psiConfig < (1 << m); ++psiConfig)
                    {
                        std::vector<int> psi(m);
                        for (int i = 0; i < m; ++i)
                        {
                            psi[i] = (psiConfig & (1 << i)) ? 1 : -1;
                        }

                        // Construct tau
                        std::vector<int> tau;
                        tau.push_back(a);
                        tau.insert(tau.end(), X.begin(), X.end());
                        tau.push_back(b);
                        tau.insert(tau.end(), Y.begin(), Y.end());

                        std::vector<int> tauInv = inversePermutation(tau);

                        int INF = 1000000000; // large number
                        // Total constraints:
                        // For each j in X: 2 rows
                        // For each j in Y: 2 rows
                        // For b: 1 row
                        int totalRows = 2 * (int)X.size() + 2 * (int)Y.size() + 1;

                        glp_prob *lp = glp_create_prob();
                        glp_set_obj_dir(lp, GLP_MAX);
                        glp_add_rows(lp, totalRows);
                        glp_add_cols(lp, k);

                        // Set variables to be free: u_i ∈ (-∞, ∞)
                        for (int col = 1; col <= k; ++col)
                        {
                            glp_set_col_bnds(lp, col, GLP_FR, -INF, INF);
                        }

                        // Set objective: maximize sum(psi[0]*G(i,a)*u_i)
                        for (int i = 0; i < k; ++i)
                        {
                            glp_set_obj_coef(lp, i + 1, psi[0] * G(i, a));
                        }

                        int rowIndex = 1;

                        // Constraints for X:
                        for (int jx : X)
                        {
                            {
                                // rowPos ≤ 0
                                std::vector<int> idx(k + 1);
                                std::vector<double> val(k + 1);
                                for (int col = 0; col < k; ++col)
                                {
                                    idx[col + 1] = col + 1;
                                    val[col + 1] = psi[tauInv[jx]] * G(col, jx) - psi[0] * G(col, a);
                                }
                                glp_set_row_bnds(lp, rowIndex, GLP_UP, -INF, 0.0);
                                glp_set_mat_row(lp, rowIndex, k, idx.data(), val.data());
                                rowIndex++;
                            }
                            {
                                // rowNeg ≤ -1
                                std::vector<int> idx(k + 1);
                                std::vector<double> val(k + 1);
                                for (int col = 0; col < k; ++col)
                                {
                                    idx[col + 1] = col + 1;
                                    val[col + 1] = -psi[tauInv[jx]] * G(col, jx);
                                }
                                glp_set_row_bnds(lp, rowIndex, GLP_UP, -INF, -1.0);
                                glp_set_mat_row(lp, rowIndex, k, idx.data(), val.data());
                                rowIndex++;
                            }
                        }

                        // Constraints for Y:
                        for (int jy : Y)
                        {
                            {
                                // G(:,jy)*u ≤ 1
                                std::vector<int> idx(k + 1);
                                std::vector<double> val(k + 1);
                                for (int col = 0; col < k; ++col)
                                {
                                    idx[col + 1] = col + 1;
                                    val[col + 1] = G(col, jy);
                                }
                                glp_set_row_bnds(lp, rowIndex, GLP_UP, -INF, 1.0);
                                glp_set_mat_row(lp, rowIndex, k, idx.data(), val.data());
                                rowIndex++;
                            }
                            {
                                // -G(:,jy)*u ≤ 1
                                std::vector<int> idx(k + 1);
                                std::vector<double> val(k + 1);
                                for (int col = 0; col < k; ++col)
                                {
                                    idx[col + 1] = col + 1;
                                    val[col + 1] = -G(col, jy);
                                }
                                glp_set_row_bnds(lp, rowIndex, GLP_UP, -INF, 1.0);
                                glp_set_mat_row(lp, rowIndex, k, idx.data(), val.data());
                                rowIndex++;
                            }
                        }

                        // Equality for b: G(:,b)*u = 1
                        {
                            std::vector<int> idx(k + 1);
                            std::vector<double> val(k + 1);
                            for (int col = 0; col < k; ++col)
                            {
                                idx[col + 1] = col + 1;
                                val[col + 1] = G(col, b);
                            }
                            glp_set_row_bnds(lp, rowIndex, GLP_FX, 1.0, 1.0);
                            glp_set_mat_row(lp, rowIndex, k, idx.data(), val.data());
                            rowIndex++;
                        }

                        // Solve LP
                        glp_simplex(lp, NULL);
                        int status = glp_get_status(lp);

                        // If feasible or optimal, check objective
                        if (status == GLP_OPT || status == GLP_FEAS)
                        {
                            double objectiveValue = glp_get_obj_val(lp);
                            if (objectiveValue > localBestHeight)
                            {
                                localBestHeight = objectiveValue;
                                for (int i = 0; i < k; ++i)
                                {
                                    localBestU[i] = glp_get_col_prim(lp, i + 1);
                                }
                                localBestParams = {a, b, X, psi};
                            }
                        }

                        glp_delete_prob(lp);
                    }
                }
            }
        }

        // Update global best with the local best found by this thread
#pragma omp critical
        {
            if (localBestHeight > bestHeight)
            {
                bestHeight = localBestHeight;
                bestU = localBestU;
                bestParams = localBestParams;
            }
        }
    }

    return {bestHeight, bestU, bestParams};
}

struct Task
{
    int a;
    int b;
    std::vector<int> X;
    std::vector<int> Y;
    std::vector<int> psi; // length m
};

std::tuple<double, Eigen::VectorXd, std::tuple<int, int, std::vector<int>, std::vector<int>>>
solveMHeight_openmp_flat(const Eigen::MatrixXd &G, int m)
{
    int k = G.rows();
    int n = G.cols();

    std::cout << "k: " << k << ", n: " << n << ", m: " << m << std::endl;

    double bestHeight = -std::numeric_limits<double>::infinity();
    Eigen::VectorXd bestU(k);
    std::tuple<int, int, std::vector<int>, std::vector<int>> bestParams;

    std::vector<int> allIndices(n);
    std::iota(allIndices.begin(), allIndices.end(), 0);

    // Precompute all tasks
    std::vector<Task> tasks;

// Parallelize (a, b) generation
#pragma omp parallel
    {
        std::vector<Task> localTasks; // Thread-local storage for tasks

#pragma omp for schedule(dynamic)
        for (int a = 0; a < n; ++a)
        {
            for (int b = 0; b < n; ++b)
            {
                if (b == a)
                    continue;

                std::vector<int> remainingIndices = allIndices;
                remainingIndices.erase(std::remove(remainingIndices.begin(), remainingIndices.end(), a), remainingIndices.end());
                remainingIndices.erase(std::remove(remainingIndices.begin(), remainingIndices.end(), b), remainingIndices.end());

                // Generate combinations of size (m-1)
                std::vector<std::vector<int>> combinations;
                {
                    std::function<void(int, int, std::vector<int> &)> combGen = [&](int offset, int r, std::vector<int> &temp)
                    {
                        if (r == 0)
                        {
                            combinations.push_back(temp);
                            return;
                        }
                        for (int i = offset; i <= (int)remainingIndices.size() - r; ++i)
                        {
                            temp.push_back(remainingIndices[i]);
                            combGen(i + 1, r - 1, temp);
                            temp.pop_back();
                        }
                    };
                    std::vector<int> temp;
                    combGen(0, m - 1, temp);
                }

                // For each combination (X)
                for (const auto &X : combinations)
                {
                    std::vector<int> Y;
                    for (int idx : remainingIndices)
                    {
                        if (std::find(X.begin(), X.end(), idx) == X.end())
                        {
                            Y.push_back(idx);
                        }
                    }

                    // For each psi config
                    int psiCount = (1 << m);
                    for (int psiConfig = 0; psiConfig < psiCount; ++psiConfig)
                    {
                        std::vector<int> psi(m);
                        for (int i = 0; i < m; ++i)
                        {
                            psi[i] = (psiConfig & (1 << i)) ? 1 : -1;
                        }

                        // Store the task in the thread-local vector
                        Task t;
                        t.a = a;
                        t.b = b;
                        t.X = X;
                        t.Y = Y;
                        t.psi = psi;
                        localTasks.push_back(t);
                    }
                }
            }
        }

// Merge thread-local tasks into the global tasks vector
#pragma omp critical
        tasks.insert(tasks.end(), localTasks.begin(), localTasks.end());
    }

    int INF = 1000000000; // large "infinity"

    // Parallelize over tasks
#pragma omp parallel
    {
        double localBestHeight = -std::numeric_limits<double>::infinity();
        Eigen::VectorXd localBestU(k);
        std::tuple<int, int, std::vector<int>, std::vector<int>> localBestParams;

#pragma omp for schedule(dynamic)
        for (int i = 0; i < (int)tasks.size(); ++i)
        {
            const auto &T = tasks[i];
            const int a = T.a;
            const int b = T.b;
            const auto &X = T.X;
            const auto &Y = T.Y;
            const auto &psi = T.psi;

            // Construct tau
            std::vector<int> tau;
            tau.push_back(a);
            tau.insert(tau.end(), X.begin(), X.end());
            tau.push_back(b);
            tau.insert(tau.end(), Y.begin(), Y.end());

            std::vector<int> tauInv = inversePermutation(tau);

            int totalRows = 2 * (int)X.size() + 2 * (int)Y.size() + 1;

            glp_prob *lp = glp_create_prob();
            glp_set_obj_dir(lp, GLP_MAX);
            glp_add_rows(lp, totalRows);
            glp_add_cols(lp, k);

            for (int col = 1; col <= k; ++col)
            {
                glp_set_col_bnds(lp, col, GLP_FR, -INF, INF);
            }

            // Objective: psi[0]*G(i,a)*u_i
            for (int ii = 0; ii < k; ++ii)
            {
                glp_set_obj_coef(lp, ii + 1, psi[0] * G(ii, a));
            }

            int rowIndex = 1;

            // Constraints for X
            for (int jx : X)
            {
                {
                    // rowPos ≤ 0
                    std::vector<int> idx(k + 1);
                    std::vector<double> val(k + 1);
                    for (int col = 0; col < k; ++col)
                    {
                        idx[col + 1] = col + 1;
                        val[col + 1] = psi[tauInv[jx]] * G(col, jx) - psi[0] * G(col, a);
                    }
                    glp_set_row_bnds(lp, rowIndex, GLP_UP, -INF, 0.0);
                    glp_set_mat_row(lp, rowIndex, k, idx.data(), val.data());
                    rowIndex++;
                }
                {
                    // rowNeg ≤ -1
                    std::vector<int> idx(k + 1);
                    std::vector<double> val(k + 1);
                    for (int col = 0; col < k; ++col)
                    {
                        idx[col + 1] = col + 1;
                        val[col + 1] = -psi[tauInv[jx]] * G(col, jx);
                    }
                    glp_set_row_bnds(lp, rowIndex, GLP_UP, -INF, -1.0);
                    glp_set_mat_row(lp, rowIndex, k, idx.data(), val.data());
                    rowIndex++;
                }
            }

            // Constraints for Y
            for (int jy : Y)
            {
                {
                    // G(:,jy)*u ≤ 1
                    std::vector<int> idx(k + 1);
                    std::vector<double> val(k + 1);
                    for (int col = 0; col < k; ++col)
                    {
                        idx[col + 1] = col + 1;
                        val[col + 1] = G(col, jy);
                    }
                    glp_set_row_bnds(lp, rowIndex, GLP_UP, -INF, 1.0);
                    glp_set_mat_row(lp, rowIndex, k, idx.data(), val.data());
                    rowIndex++;
                }
                {
                    // -G(:,jy)*u ≤ 1
                    std::vector<int> idx(k + 1);
                    std::vector<double> val(k + 1);
                    for (int col = 0; col < k; ++col)
                    {
                        idx[col + 1] = col + 1;
                        val[col + 1] = -G(col, jy);
                    }
                    glp_set_row_bnds(lp, rowIndex, GLP_UP, -INF, 1.0);
                    glp_set_mat_row(lp, rowIndex, k, idx.data(), val.data());
                    rowIndex++;
                }
            }

            // Equality for b: G(:,b)*u = 1
            {
                std::vector<int> idx(k + 1);
                std::vector<double> val(k + 1);
                for (int col = 0; col < k; ++col)
                {
                    idx[col + 1] = col + 1;
                    val[col + 1] = G(col, b);
                }
                glp_set_row_bnds(lp, rowIndex, GLP_FX, 1.0, 1.0);
                glp_set_mat_row(lp, rowIndex, k, idx.data(), val.data());
                rowIndex++;
            }

            // Solve LP
            glp_simplex(lp, NULL);
            int status = glp_get_status(lp);

            // Check feasibility and update best
            if (status == GLP_OPT || status == GLP_FEAS)
            {
                double objectiveValue = glp_get_obj_val(lp);
                if (objectiveValue > localBestHeight)
                {
                    localBestHeight = objectiveValue;
                    for (int ii = 0; ii < k; ++ii)
                    {
                        localBestU[ii] = glp_get_col_prim(lp, ii + 1);
                    }
                    localBestParams = {a, b, X, psi};
                }
            }

            glp_delete_prob(lp);
        }

        // Update global best
#pragma omp critical
        {
            if (localBestHeight > bestHeight)
            {
                bestHeight = localBestHeight;
                bestU = localBestU;
                bestParams = localBestParams;
            }
        }
    }

    return {bestHeight, bestU, bestParams};
}
