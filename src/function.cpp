//
// Created by 芦yafei  on 14/6/12.
//
#include "function.h"
#include <eigen3/Eigen/Dense>
#include "random.h"

using namespace std;
using namespace Eigen;

namespace lu_net {
    //sigmoid function
    VectorXf sigmoid(const VectorXf &x){
        VectorXf res = (1.0 + (-x).array().exp()).inverse();
        return res;
    }

    //relu function
    VectorXf relu(const VectorXf &x){
        VectorXf res = x.array().max(0.0).matrix();
        return res;
    }

    //Derivative of the sigmoid function.
    VectorXf sigmoid_prime(const VectorXf &z) {
        return sigmoid(z).array() * (1 - sigmoid(z).array());
    }

    float_t gaussian_random(float_t x) {
        return gaussian_rand(0.0, 1.0);
    }
}

