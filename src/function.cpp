//
// Created by 芦yafei  on 14/6/12.
//
#include "function.h"
#include <eigen3/Eigen/Dense>
#include "random.h"

using namespace std;
using namespace Eigen;

namespace lu_net {
    float_t gaussian_random(float_t x) {
        return gaussian_rand(0.0, 1.0);
    }
}

