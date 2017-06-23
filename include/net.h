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
    typedef std::uint32_t label_t;
    typedef float float_t;
    typedef std::vector<float_t> vec_t;
    typedef std::vector<vec_t> tensor_t;

    class Net {
    public:
        Net() {};

        virtual ~Net() {};

        vector<int> layers_neuron_num;
        int num_layers = 0;
        float learning_rate = 0.0;
        float accuray = 0.0;
        float loss = 0.0;
        VectorXf target;
        VectorXf output_error;
        vector<float> loss_vec;    //save for draw loss curve
        int output_interval = 0;    //训练中间输出loss
        float fine_tune_factor = 0.0; //学习率调节因子

        // initialize net:generate weights matrices、layer matrices and bias matrices
        // bias default all zero
        void initNet(const vector<int> layers_neuron_num);

        // initialize the weights matrices
        void initWeights(const double w = 0);

        //Initial the bias matrices
        void initBias(const double w = 0);

        bool train(const std::vector<vec_t> &inputs, const std::vector<label_t> &class_labels, int batch_size,
                   int epoch);

    private:
        vector<VectorXf> layers;
        vector<MatrixXf> weights;
        vector<VectorXf> bias;
        vector<VectorXf> gradient;
        vector<VectorXf> zs;    //store all the z vectors(weighted input), layer by layer


        void Net::update_batch(vector< pair<label_t, vector<vec_t> > > mini_batch_data, float lr);

        //Backward
        void farward(VectorXf x);

        VectorXf cost_derivative(VectorXf output_activations, VectorXf y);

        //Forward
        VectorXf backward(const VectorXf &y, vector<MatrixXf> &nabla_w, vector<VectorXf> &nabla_b);
    };
}
#endif //LU_NET_NET_H
