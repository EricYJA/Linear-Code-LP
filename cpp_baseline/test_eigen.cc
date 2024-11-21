#include <Eigen/Dense>
#include <iostream>

int main() {
    Eigen::MatrixXd G(2, 3);
    G << 1, 2, 3, 4, 5, 6;

    Eigen::VectorXd u(2);
    u << 1, 2;

    Eigen::VectorXd c = u.transpose() * G;  // Matrix-vector multiplication
    std::cout << "Codeword c:\n" << c << std::endl;

    return 0;
}
