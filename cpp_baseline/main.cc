#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <tuple>

#include "opt_mheight.hh"

int main() {
    // Define the matrix G_5_2_2
    Eigen::MatrixXd G(2, 5);
    G << 0.911, 0.03, 1.481, -0.756, 1.249,
        -0.049, 0.975, 1.511, -1.303, 0.74;

    // Define m
    int m = 2;

    // Call the solveMHeight function
    try {
        auto [bestHeight, bestU, bestParams] = solveMHeight(G, m);

        // Unpack the bestParams tuple
        auto [a, b, X, psi] = bestParams;

        // Print the results
        std::cout << "Optimal m-height: " << bestHeight << std::endl;

        std::cout << "Optimal vector u: [";
        for (int i = 0; i < bestU.size(); ++i) {
            std::cout << bestU[i] << (i == bestU.size() - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;

        std::cout << "Optimal parameters:" << std::endl;
        std::cout << "a: " << a << std::endl;
        std::cout << "b: " << b << std::endl;
        std::cout << "X: ";
        for (int x : X) std::cout << x << " ";
        std::cout << std::endl;

        std::cout << "psi: ";
        for (int p : psi) std::cout << p << " ";
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}