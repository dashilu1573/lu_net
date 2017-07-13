//
// Created by 芦yafei  on 17/7/13.
//

#include "net.h"
#include <cmath>
#include <eigen3/Eigen/Dense>
#include "loss_function.h"

namespace lu_net {
    // Return the cost associated with an output and desired output target
    float_t mse::f(const VectorXf &output, const VectorXf &target) {
        assert(output.size() == target.size());
        VectorXf output_error = target - output;
        VectorXf square_error = output_error.array().square();

        float err_sum = square_error.sum();
        float loss = err_sum / (float) output.rows();

        return loss;
    }

    // gradient
    VectorXf mse::df(const VectorXf &output, const VectorXf &target) {
        assert(output.size() == target.size());
        //float_t factor = 2.0 / static_cast<float_t>(target.size()); ????不知为何要
        //return factor * (output - target);
        return output - target;
    }

    // define function to be applied coefficient-wise
    float_t cross_entropy::nan_to_num(float_t x) {
        if (std::isfinite(x))
            return x;
        else
            return 0;
    }

    // Return the cost associated with an output and desired output target
    float_t cross_entropy::f(const VectorXf &output, const VectorXf &target) {
        assert(output.size() == target.size());

        // -( y*ln(a) + (1-y)*ln(1-a) )
        // In particular, if both ``a`` and ``y`` have a 1.0 in the same slot,
        // then the expression (1-y)*np.log(1-a) returns nan.
        // The nan_to_num ensures that that is converted to the correct value (0.0).
        VectorXf tmp = (target.array() * log(output.array()) +
                        (1.0 - target.array()) * log(1.0 - output.array())).matrix();
        float loss = -1.0 * tmp.unaryExpr(ptr_fun(nan_to_num)).sum();

        return loss;
    }

    // gradient
    VectorXf cross_entropy::df(const VectorXf &output, const VectorXf &target) {
        assert(output.size() == target.size());

        // (a-y)/(a*(1-a))
        // delta of cross-entropy is (a-y), because denominator (a*(1-a)) is
        // removed by multiply sigmoid_prime.
        return ((output.array() - target.array())
                / (output.array() * (1.0 - output.array()))).matrix().unaryExpr(ptr_fun(nan_to_num));
    }


    // cross-entropy loss function for multi-class classification
    float_t cross_entropy_multiclass::f(const vec_t &y, const vec_t &t) {
        assert(y.size() == t.size());
        float_t d = 0.0;

        for (int i = 0; i < y.size(); ++i)
            d += -t[i] * std::log(y[i]);

        return d;
    }

    vec_t cross_entropy_multiclass::df(const vec_t &y, const vec_t &t) {
        assert(y.size() == t.size());
        vec_t d(t.size());

        for (int i = 0; i < y.size(); ++i)
            d[i] = -t[i] / y[i];

        return d;
    }

    template<typename E>
    vec_t gradient(const vec_t &y, const vec_t &t) {
        assert(y.size() == t.size());
        return E::df(y, t);
    }

    template<typename E>
    std::vector<vec_t> gradient(const std::vector<vec_t> &y, const std::vector<vec_t> &t) {
        std::vector<vec_t> grads;

        assert(y.size() == t.size());

        for (int i = 0; i < y.size(); i++)
            grads.push_back(gradient<E>(y[i], t[i]));

        return grads;
    }
}

