//
// Created by 芦yafei  on 17/6/12.
//
#include "../include/function.h"
#include <eigen3/Eigen/Dense>

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

    //Objective function
    float calcLoss(const VectorXf &output, const VectorXf &target, VectorXf &output_error){
        //square_error平方误差
        VectorXf output_error = target - output;
        VectorXf square_error = output_error.array().square();

        //loss
        float err_sum = square_error.sum();
        float loss = err_sum / (float)output.rows();

        return loss;
    }

    //Derivative of the sigmoid function.
    VectorXf sigmoid_prime(const VectorXf &z) {
        return sigmoid(z).array() * (1 - sigmoid(z).array());
    }
}

