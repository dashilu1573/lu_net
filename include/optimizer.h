//
// Created by 芦yafei  on 17/7/14.
//
#ifndef LU_NET_OPTIMIZER_H
#define LU_NET_OPTIMIZER_H

#include <eigen3/Eigen/Dense>
#include <unordered_map>

using namespace std;
using namespace Eigen;

namespace lu_net {
    namespace optimizer {

        /**
        * base class of optimizer
        * usesHessian : true if an optimizer uses hessian (2nd order derivative of loss function)
        **/
        class optimizer {
        public:
            // default constructor and destructor must be assigned default.
            optimizer() = default;
            virtual ~optimizer() = default;

            // pure virtual function, must be defined in children class
            virtual void update_w(const MatrixXf &dW, MatrixXf &W, const float alpha) = 0;
            virtual void update_b(const VectorXf &dW, VectorXf &W, const float alpha) = 0;
        };


        /*
         * SGD without momentum
         *
         * slightly faster than tiny_dnn::momentum
         **/
        class gradient_descent : public optimizer {
        public:
            float alpha; // learning rate
            float lambda; // weight decay

            gradient_descent() : lambda(0.0) {}

            void update_w(const MatrixXf &dW, MatrixXf &W, const float alpha) override {
                W = W - alpha * (dW + lambda * W);
            }

            void update_b(const VectorXf &dW, VectorXf &W, const float alpha) override {
                W = W - alpha * (dW + lambda * W);
            }

        };
    }
}

#endif //LU_NET_OPTIMIZER_H
