#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <algorithm>

#include "calc_mheight.hh"

int main() {
    // Example Generator Matrix G
    Eigen::MatrixXd G(3, 5);
    G << 1, 2, 3, 4, 5,
         2, 3, 4, 5, 6,
         3, 4, 5, 6, 7;

    // Example Input Vector u
    Eigen::RowVectorXd u(3);
    u << 1, -1, 0;

    // Test calculateMHeight
    try {
        int m = 2;
        double mHeight = calculateMHeight(G, u, m);
        std::cout << "M-Height: " << mHeight << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error in calculateMHeight: " << e.what() << std::endl;
    }

    // Test calculateMinimumDistance
    try {
        int minDistance = calculateMinimumDistance(G);
        std::cout << "Minimum Distance: " << minDistance << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error in calculateMinimumDistance: " << e.what() << std::endl;
    }

    // Test calculateABX
    try {
        int m = 2;
        auto [a, b, X, Y, psi, tau] = calculateABX(G, u, m);

        std::cout << "a (largest abs value index): " << a << std::endl;
        std::cout << "b ((m+1)-th largest abs value index): " << b << std::endl;

        std::cout << "X (next m-1 largest indices): ";
        for (int x : X) std::cout << x << " ";
        std::cout << std::endl;

        std::cout << "Y (remaining indices): ";
        for (int y : Y) std::cout << y << " ";
        std::cout << std::endl;

        std::cout << "psi (signs of elements): ";
        for (int p : psi) std::cout << p << " ";
        std::cout << std::endl;

        std::cout << "tau (combined indices): ";
        for (int t : tau) std::cout << t << " ";
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error in calculateABX: " << e.what() << std::endl;
    }

    // Test inversePermutation
    try {
        std::vector<int> tau = {3, 4, 1, 0, 2};
        std::vector<int> tauInv = inversePermutation(tau);

        std::cout << "tau: ";
        for (int t : tau) std::cout << t << " ";
        std::cout << std::endl;

        std::cout << "tau^-1: ";
        for (int t : tauInv) std::cout << t << " ";
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error in inversePermutation: " << e.what() << std::endl;
    }

    return 0;
}
