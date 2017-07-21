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
            virtual void update_w(MatrixXf &W, const MatrixXf &dW, const float alpha) = 0;
            virtual void update_b(VectorXf &W, const VectorXf &dW, const float alpha) = 0;
        };


        /**
         * SGD without momentum
         *
         * slightly faster than tiny_dnn::momentum
         **/
        class gradient_descent : public optimizer {
        public:
            float alpha; // learning rate
            float lambda; // weight decay

            gradient_descent() {}

            void update_w(MatrixXf &W, const MatrixXf &dW, const float alpha) override {
                W = W - alpha * dW;
            }

            void update_b(VectorXf &W, const VectorXf &dW, const float alpha) override {
                W = W - alpha * dW;
            }
        };

        /**
         * SGD with momentum
         *
         * B T Polyak,
         * Some methods of speeding up the convergence of iteration methods
         * USSR Computational Mathematics and Mathematical Physics, 4(5):1-17, 1964.
         **/
        class momentum : public optimizer {
        public:
            float alpha; // learning rate
            float lambda; // weight decay
            float mu; // momentum

            momentum() : mu(0.9) {}

            void update_w(MatrixXf &W, const MatrixXf &dW, const float alpha) override {
                MatrixXf V = - alpha * dW + mu * dWprev_w[0];
                W = W + V;
                dWprev_w[0] = V;
            }

            void update_b(VectorXf &W, const VectorXf &dW, const float alpha) override {
                MatrixXf V = - alpha * dW + mu * dWprev_b[0];
                W = W + V;
                dWprev_b[0] = V;
            }

        private:
            vector<MatrixXf> dWprev_w;
            vector<VectorXf> dWprev_b;
        };

    }
}

#endif //LU_NET_OPTIMIZER_H
