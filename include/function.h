//
// Created by 芦yafei  on 17/6/12.
//

#ifndef LU_NET_FUNCTION_H
#define LU_NET_FUNCTION_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>

using namespace std;
using namespace Eigen;

namespace lu_net{
    //sigmoid function
    VectorXf sigmoid(const VectorXf &x);

    //relu function
    VectorXf relu(const VectorXf &x);

    VectorXf sigmoid_prime(const VectorXf &z);

    //Objective function
    float calcLoss(const VectorXf &output, const VectorXf &target);

    template <typename T>
    int max_index(const vector<T> &vec) {
        auto begin_iterator = begin(vec);
        return max_element(begin_iterator, end(vec)) - begin_iterator;
    }
}

#endif //LU_NET_FUNCTION_H
