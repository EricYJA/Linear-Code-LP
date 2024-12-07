#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <chrono>
#include <omp.h>
#include <random>

#include "opt_mheight.hh"

int main()
{
    std::cout << "First example: G(2, 5)" << "m = 2" << std::endl;

    // Define the matrix G_5_2_2
    Eigen::MatrixXd G1(2, 5);
    G1 << 0.911, 0.03, 1.481, -0.756, 1.249,
        -0.049, 0.975, 1.511, -1.303, 0.74;

    // Define m
    int m = 2;

    // Call the solveMHeight function
    try
    {
        auto start = std::chrono::high_resolution_clock::now();

        auto [bestHeight, bestU, bestParams] = solveMHeight_openmp_flat(G1, m);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

        // Unpack the bestParams tuple
        auto [a, b, X, psi] = bestParams;

        // Print the results
        std::cout << "Optimal m-height: " << bestHeight << std::endl;

        std::cout << "Optimal vector u: [";
        for (int i = 0; i < bestU.size(); ++i)
        {
            std::cout << bestU[i] << (i == bestU.size() - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;

        std::cout << "Optimal parameters:" << std::endl;
        std::cout << "a: " << a << std::endl;
        std::cout << "b: " << b << std::endl;
        std::cout << "X: ";
        for (int x : X)
            std::cout << x << " ";
        std::cout << std::endl;

        std::cout << "psi: ";
        for (int p : psi)
            std::cout << p << " ";
        std::cout << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }


    std::cout<< "Second example: G(7, 10)" <<  "m = 2" << std::endl; 

    Eigen::MatrixXd G2(7, 10);
    G2 << 0.556, 0.96, 0.556, 0.63, 0.556, 0.556, -0.507, -0.507, 0.319, 0.319,
        0.63, 0.491, -0.58, 0.96, 0.63, -0.507, 0.63, 0.319, 0.556, 0.491,
        0.319, -0.58, 0.319, -0.507, 0.96, 0.491, 0.319, 0.63, -0.507, 0.96,
        -0.507, -0.507, -0.507, 0.491, 0.491, 0.319, 0.491, 0.491, 0.491, -0.507,
        -0.58, 0.556, 0.491, 0.319, 0.319, 0.96, 0.96, -0.58, 0.63, 0.556,
        0.96, 0.63, 0.96, -0.58, -0.507, 0.63, -0.58, 0.96, -0.58, -0.58,
        0.491, 0.319, 0.63, 0.556, -0.58, -0.58, 0.556, 0.556, 0.96, 0.63;

    m = 2;

    try
    {
        auto start = std::chrono::high_resolution_clock::now();

        auto [bestHeight, bestU, bestParams] = solveMHeight_openmp_flat(G2, m);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

        auto [a, b, X, psi] = bestParams;

        std::cout << "Optimal m-height: " << bestHeight << std::endl;

        std::cout << "Optimal vector u: [";
        for (int i = 0; i < bestU.size(); ++i)
        {
            std::cout << bestU[i] << (i == bestU.size() - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;

        std::cout << "Optimal parameters:" << std::endl;
        std::cout << "a: " << a << std::endl;
        std::cout << "b: " << b << std::endl;
        std::cout << "X: ";
        for (int x : X)
            std::cout << x << " ";
        std::cout << std::endl;

        std::cout << "psi: ";
        for (int p : psi)
            std::cout << p << " ";
        std::cout << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    std::cout << "Third example: G(7, 10)" << "m = 3" << std::endl;

    Eigen::MatrixXd G3(7, 10);
    G3 << 1.219, -0.028, 1.206, 0.604, 0.055, -2.649, 1.212, -2.646, -0.373, 0.004,
     0.04, 1.183, -2.666, -0.387, -0.033, 0.07, -0.863, 0.044, 0.63, -2.537,
     0.622, 0.039, -0.075, 0.123, 0.62, 0.663, -0.06, -0.825, 0.112, 1.151,
    -0.815, -0.369, 0.065, -0.045, -0.345, -0.831, -2.596, -0.421, -2.608, -0.106,
    -0.09, -0.813, -0.38, -0.838, -2.602, 1.133, 0.053, 0.008, 1.203, -0.794,
    -2.583, 0.672, -0.864, -2.654, -0.834, -0.41, -0.385, 1.166, -0.113, 0.584,
    -0.411, -2.657, 0.648, 1.183, 1.2, -0.084, 0.638, 0.543, -0.824, -0.364;

    m = 3;

    try
    {
        auto start = std::chrono::high_resolution_clock::now();

        auto [bestHeight, bestU, bestParams] = solveMHeight_openmp_flat(G3, m);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

        auto [a, b, X, psi] = bestParams;

        std::cout << "Optimal m-height: " << bestHeight << std::endl;

        std::cout << "Optimal vector u: [";
        for (int i = 0; i < bestU.size(); ++i)
        {
            std::cout << bestU[i] << (i == bestU.size() - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;

        std::cout << "Optimal parameters:" << std::endl;
        std::cout << "a: " << a << std::endl;
        std::cout << "b: " << b << std::endl;
        std::cout << "X: ";
        for (int x : X)
            std::cout << x << " ";
        std::cout << std::endl;

        std::cout << "psi: ";
        for (int p : psi)
            std::cout << p << " ";
        std::cout << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }


    std::cout << "Fourth example: G(14, 20)" << "m = 3" << " Random G" << std::endl;

    // Dimensions of the matrix
    const int rows = 14;
    const int cols = 20;

    // Create an Eigen matrix
    Eigen::MatrixXd G4(rows, cols);

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-3.0, 3.0);

    // Fill the matrix with random values
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            G4(i, j) = dis(gen);
        }
    }

    m = 3;

    try
    {
        auto start = std::chrono::high_resolution_clock::now();

        auto [bestHeight, bestU, bestParams] = solveMHeight_openmp_flat(G4, m);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

        auto [a, b, X, psi] = bestParams;

        std::cout << "Optimal m-height: " << bestHeight << std::endl;

        std::cout << "Optimal vector u: [";
        for (int i = 0; i < bestU.size(); ++i)
        {
            std::cout << bestU[i] << (i == bestU.size() - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;

        std::cout << "Optimal parameters:" << std::endl;
        std::cout << "a: " << a << std::endl;
        std::cout << "b: " << b << std::endl;
        std::cout << "X: ";
        for (int x : X)
            std::cout << x << " ";
        std::cout << std::endl;

        std::cout << "psi: ";
        for (int p : psi)
            std::cout << p << " ";
        std::cout << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}