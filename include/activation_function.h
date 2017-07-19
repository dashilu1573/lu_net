//
// Created by 芦yafei  on 17/7/11.
//

#ifndef LU_NET_ACTIVATION_FUNCTION_H
#define LU_NET_ACTIVATION_FUNCTION_H

#include <eigen3/Eigen/Dense>

namespace lu_net{
    namespace activation{
        class sigmoid {
        public:
            static Eigen::VectorXf f(const Eigen::VectorXf &x);

            static Eigen::VectorXf df(const Eigen::VectorXf &z);
        };


        class relu {
        public:
            static Eigen::VectorXf f(const Eigen::VectorXf &x);

            static Eigen::VectorXf df(const Eigen::VectorXf &z);
        };
    }
}

#endif //LU_NET_ACTIVATION_FUNCTION_H
