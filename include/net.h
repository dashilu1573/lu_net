//
// Created by 芦yafei  on 17/6/9.
//

#ifndef LU_NET_NET_H
#define LU_NET_NET_H

#include <vector>
#include <string.h>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;


namespace lu_net {

    class Net {
    public:
        Net() {};

        virtual ~Net() {};

        vector<int> layers_neuron_num;
        float learning_rate = 0.0;
        float accuray = 0.0;
        float loss = 0.0;
        VectorXf target;
        VectorXf output_error;


        // initialize net:generate weights matrices、layer matrices and bias matrices
        // bias default all zero
        void initNet(const vector<int> layers_neuron_num);

        // initialize the weights matrices
        void initWeights(const double w = 0);

        //Initial the bias matrices
        void initBias(const double w = 0);

        void train(MatrixXf train_data);
        void test();
        void predict();
        void save();
        void load();

    private:
        vector<VectorXf> layers;
        vector<MatrixXf> weights;
        vector<VectorXf> bias;
        vector<VectorXf> delta_errs;
        vector<VectorXf> derivatives;

        //Activation function
        VectorXf activationFunction(const VectorXf &x, string func_type);

        // Compute delta error
        void deltaError();

        // Compute derivative
        void derivative(string func_type);

        // Update weights
        void updateWeights();

        //Forward
        void farward(string func_type);

        //Forward
        void backward();
    };
}
#endif //LU_NET_NET_H
