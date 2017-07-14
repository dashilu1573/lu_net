//
// Created by 芦yafei  on 17/7/11.
//

#ifndef LU_NET_ACTIVATION_FUNCTION_H
#define LU_NET_ACTIVATION_FUNCTION_H

#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

namespace lu_net{
    namespace activation{
        class sigmoid {
        public:
            static VectorXf f(const VectorXf &x);

            static VectorXf df(const VectorXf &z);
        };


        class relu {
        public:
            static VectorXf f(const VectorXf &x);

            static VectorXf df(const VectorXf &z);
        };
    }
}

#endif //LU_NET_ACTIVATION_FUNCTION_H
