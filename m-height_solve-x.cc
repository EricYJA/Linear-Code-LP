#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <ceres/ceres.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Load data from a binary file (similar to pickle file in Python)
MatrixXd load_data(const std::string &file_name) {
    std::ifstream file(file_name, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file " + file_name);
    }

    int rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(int));
    file.read(reinterpret_cast<char*>(&cols), sizeof(int));

    MatrixXd G(rows, cols);
    file.read(reinterpret_cast<char*>(G.data()), rows * cols * sizeof(double));
    return G;
}

// Function to calculate m-height
double calculate_m_height(const VectorXd &c, int m) {
    std::vector<double> abs_values(c.size());
    for (int i = 0; i < c.size(); ++i) {
        abs_values[i] = std::abs(c(i));
    }

    // Sort in descending order
    std::sort(abs_values.begin(), abs_values.end(), std::greater<double>());

    // Check if m is valid (if m is out of range or entry at m is zero)
    if (m >= abs_values.size() || abs_values[m] == 0) {
        return std::numeric_limits<double>::infinity();
    }

    // Calculate m-height
    return abs_values[0] / abs_values[m];
}

// Objective function struct for Ceres-Solver
struct ObjectiveFunction {
    ObjectiveFunction(const MatrixXd &G, int m) : G_(G), m_(m) {}

    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> x_map(x, G_.rows());
        Eigen::Matrix<T, Eigen::Dynamic, 1> c = x_map.transpose() * G_.cast<T>();

        std::vector<T> abs_values(c.size());
        for (int i = 0; i < c.size(); ++i) {
            abs_values[i] = ceres::abs(c(i));
        }

        std::sort(abs_values.begin(), abs_values.end(), std::greater<T>());

        if (m_ >= abs_values.size() || abs_values[m_] == T(0)) {
            residual[0] = std::numeric_limits<T>::infinity();
            return true;
        }

        residual[0] = -abs_values[0] / abs_values[m_]; // Negative to maximize
        return true;
    }

private:
    const MatrixXd &G_;
    int m_;
};

int main() {
    try {
        int m = 4; // Example m value; adjust as needed
        MatrixXd G = load_data("G1"); // Load the generator matrix G

        // Initial guess for x
        VectorXd initial_x = VectorXd::Random(G.rows());

        // Set up the optimization problem
        ceres::Problem problem;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ObjectiveFunction, 1, Eigen::Dynamic>(
                new ObjectiveFunction(G, m), 1, G.rows()),
            nullptr,
            initial_x.data());

        // Configure the solver options
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = true;

        // Solve the optimization problem
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << summary.FullReport() << "\n";

        // Optimal x found
        std::cout << "Optimal x: " << initial_x.transpose() << "\n";

        // Calculate the maximum m-height
        VectorXd c = initial_x.transpose() * G;
        double optimal_m_height = calculate_m_height(c, m);
        std::cout << "Maximum m-height: " << optimal_m_height << "\n";
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}
