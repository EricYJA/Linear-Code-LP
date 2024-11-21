#include <Eigen/Dense>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <numeric>

double calculateMHeight(const Eigen::MatrixXd& G, const Eigen::RowVectorXd& u, int m);
int calculateMinimumDistance(const Eigen::MatrixXd& G);
std::tuple<int, int, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>> calculateABX(
    const Eigen::MatrixXd& G, const Eigen::RowVectorXd& u, int m);
std::vector<int> inversePermutation(const std::vector<int>& tau);
