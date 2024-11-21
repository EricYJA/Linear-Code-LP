#include "calc_mheight.hh"

double calculateMHeight(const Eigen::MatrixXd& G, const Eigen::RowVectorXd& u, int m) {
    // Ensure the dimensions of G and u are compatible
    if (G.rows() != u.cols()) {
        throw std::invalid_argument("Dimensions of G and u are not compatible for multiplication");
    }

    // Step 1: Compute the codeword c = u * G
    Eigen::RowVectorXd c = u * G;

    // Step 2: Sort the absolute values of c in descending order
    std::vector<double> sortedAbsC(c.data(), c.data() + c.size());
    std::transform(sortedAbsC.begin(), sortedAbsC.end(), sortedAbsC.begin(), [](double val) {
        return std::abs(val);
    });
    std::sort(sortedAbsC.begin(), sortedAbsC.end(), std::greater<double>());

    // Step 3: Calculate the m-height
    if (sortedAbsC.size() > m) {
        return std::abs(sortedAbsC[0] / sortedAbsC[m]);
    } else {
        return std::numeric_limits<double>::infinity(); // If m+1 element doesn't exist
    }
}

int calculateMinimumDistance(const Eigen::MatrixXd& G) {
    int k = G.rows();
    int n = G.cols();
    int minDistance = std::numeric_limits<int>::max();

    // Iterate through all possible input vectors u (2^k combinations)
    for (int i = 1; i < (1 << k); ++i) {  // Start from 1 to exclude the all-zero vector
        Eigen::RowVectorXd u(k);
        for (int j = 0; j < k; ++j) {
            u(j) = (i & (1 << j)) ? 1 : 0;  // Binary representation
        }

        // Compute the codeword
        Eigen::RowVectorXd codeword = u * G;

        // Calculate the Hamming weight (number of non-zero elements)
        int hammingWeight = 0;
        for (int j = 0; j < codeword.size(); ++j) {
            if (codeword(j) != 0) {
                hammingWeight++;
            }
        }

        // Update the minimum distance
        if (hammingWeight < minDistance) {
            minDistance = hammingWeight;
        }
    }

    return minDistance;
}


std::tuple<int, int, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>> calculateABX(
    const Eigen::MatrixXd& G, const Eigen::RowVectorXd& u, int m) {
    
    // Step 1: Compute the codeword
    Eigen::RowVectorXd c = u * G;

    // Step 2: Sort indices by the absolute values of the elements in descending order
    std::vector<int> indices(c.size());
    std::iota(indices.begin(), indices.end(), 0);  // Initialize indices [0, 1, ..., n-1]
    std::sort(indices.begin(), indices.end(), [&c](int i, int j) {
        return std::abs(c(i)) > std::abs(c(j));
    });

    // Step 3: Determine a, b, X, and Y
    int a = indices[0];             // Index of the largest absolute value
    int b = indices[m];             // Index of the (m+1)-th largest absolute value
    std::vector<int> X(indices.begin() + 1, indices.begin() + m);  // Next m-1 largest
    std::vector<int> Y(indices.begin() + m + 1, indices.end());   // Remaining elements

    // Get ψ values
    std::vector<int> psi;
    for (int i : indices) {
        psi.push_back(c(i) > 0 ? 1 : -1);  // Sign of the element at index i
    }

    // Combine a, X, b, and Y into τ
    std::vector<int> tau = {a};
    tau.insert(tau.end(), X.begin(), X.end());
    tau.push_back(b);
    tau.insert(tau.end(), Y.begin(), Y.end());

    return {a, b, X, Y, psi, tau};
}


std::vector<int> inversePermutation(const std::vector<int>& tau) {
    std::vector<int> tauInv(tau.size());
    for (size_t i = 0; i < tau.size(); ++i) {
        tauInv[tau[i]] = i;  // Map each value in tau to its position
    }
    return tauInv;
}