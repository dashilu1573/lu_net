//
// Created by 芦yafei  on 17/7/14.
//
#include <eigen3/Eigen/Dense>
#include "activation_function.h"

using namespace std;
using namespace Eigen;

namespace lu_net {
    namespace activation {
        VectorXf sigmoid::f(const VectorXf &x) {
            VectorXf res = (1.0 + (-x).array().exp()).inverse();
            return res;
        }

        VectorXf sigmoid::df(const VectorXf &z) {
            return f(z).array() * (1 - f(z).array());
        }


        VectorXf relu::f(const VectorXf &x) {
            VectorXf res = x.array().max(0.0).matrix();
            return res;
        }

        VectorXf relu::df(const VectorXf &z) {
            VectorXf tmp0 = VectorXf::Zero(z.rows());
            VectorXf tmp1 = VectorXf::Ones(z.rows());
            return (z.array() <= 0).select(tmp0, tmp1);
        }

        VectorXf tanh::f(const VectorXf &x) {
            VectorXf res = x.array().tanh();
            return res;
        }

        VectorXf tanh::df(const VectorXf &z) {
            return 1.0 - z.array().square();
        }
    }
}

