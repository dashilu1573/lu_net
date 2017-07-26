//
// Created by 芦yafei  on 14/2/19.
//
#include "dropout_layer.h"
#include "random.h"
#include <eigen3/Eigen/Dense>

namespace lu_net{
    void dropout_layer::forward_propagation(const  Eigen::VectorXf &in_data, Eigen::VectorXf &out_data) {
        if (phase_ == net_phase::train) {
            mask_ = mask_.unaryExpr(ptr_fun(bernoulli(dropout_rate_)));
            out_data = mask_.array() * scale_ * in_data.array();
        }
        else {
            out_data = in_data;
        }
    }

    void dropout_layer::back_propagation(Eigen::VectorXf &out_grad, Eigen::VectorXf &in_grad) {
        out_grad = in_grad.array() * mask_; //????how to memory the mask used by forward_propagation
    }
}

