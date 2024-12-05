#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <numeric>
#include <glpk.h>

#include "calc_mheight.hh"

std::tuple<double, Eigen::VectorXd, std::tuple<int, int, std::vector<int>, std::vector<int>>> solveMHeight(const Eigen::MatrixXd &G, int m);
