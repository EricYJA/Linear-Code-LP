#include <iostream>
#include <tuple>
#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include <limits>
#include <atomic>
#include <mutex>

// Include Eigen
#include <Eigen/Dense>

// Include GLPK
#include <glpk.h>

// Include OpenMP
#include <omp.h>

#include "calc_mheight.hh"

std::tuple<double, Eigen::VectorXd, std::tuple<int, int, std::vector<int>, std::vector<int>>> solveMHeight(const Eigen::MatrixXd &G, int m);

std::tuple<double, Eigen::VectorXd, std::tuple<int, int, std::vector<int>, std::vector<int>>> solveMHeight_openmp(const Eigen::MatrixXd &G, int m);

std::tuple<double, Eigen::VectorXd, std::tuple<int, int, std::vector<int>, std::vector<int>>> solveMHeight_openmp_flat(const Eigen::MatrixXd &G, int m);
