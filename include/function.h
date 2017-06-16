//
// Created by 芦yafei  on 17/6/12.
//

#ifndef LU_NET_FUNCTION_H
#define LU_NET_FUNCTION_H

#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

namespace lu_net{
    //sigmoid function
    VectorXf sigmoid(const VectorXf &x);

    //relu function
    VectorXf relu(const VectorXf &x);

    //Objective function
    void calcLoss(const VectorXf &output, const VectorXf &target, VectorXf &output_error, float &loss);
}

#endif //LU_NET_FUNCTION_H
