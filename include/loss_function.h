//
// Created by 芦yafei  on 14/7/5.
//

#ifndef LU_NET_LOSS_FUNCTION_H
#define LU_NET_LOSS_FUNCTION_H

#include "net.h"
#include <cmath>
#include <eigen3/Eigen/Dense>

namespace lu_net {

    // mean-squared-error loss function for regression
    class MSE {
    public:
        // Return the cost associated with an output and desired output target
        static float_t f(const float_t &output, const float_t &target);
        static float_t f(const Eigen::VectorXf &output, const Eigen::VectorXf &target);

        // gradient
        static float_t df(const float_t &output, const float_t &target);
        static Eigen::VectorXf df(const Eigen::VectorXf &output, const Eigen::VectorXf &target);
    };


    // cross-entropy loss function for (multiple independent) binary classifications
    class cross_entropy {
    public:
        // define function to be applied coefficient-wise
        static float_t nan_to_num(float_t x);

        // Return the cost associated with an output and desired output target
        static float_t f(const Eigen::VectorXf &output, const Eigen::VectorXf &target);

        // gradient
        static Eigen::VectorXf df(const Eigen::VectorXf &output, const Eigen::VectorXf &target);
    };


    // cross-entropy loss function for multi-class classification
    class cross_entropy_multiclass {
    public:
        static float_t f(const vec_t &y, const vec_t &t);

        static vec_t df(const vec_t &y, const vec_t &t);
    };


    template <typename E>
    vec_t gradient(const vec_t& y, const vec_t& t);

    template <typename E>
    std::vector<vec_t> gradient(const std::vector<vec_t>& y, const std::vector<vec_t>& t);
}

#endif //LU_NET_LOSS_FUNCTION_H
