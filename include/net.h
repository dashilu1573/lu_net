//
// Created by 芦yafei  on 17/6/9.
//

#ifndef LU_NET_NET_H
#define LU_NET_NET_H

#include <vector>
#include <string.h>
#include <eigen3/Eigen/Dense>
#include <map>

using namespace std;
using namespace Eigen;


namespace lu_net {
    typedef std::uint32_t label_t;
    typedef float float_t;
    typedef std::vector<float_t> vec_t;
    typedef std::vector<vec_t> tensor_t;

    enum class content_type {
        weights,    //save/load the weights
        model,      //save/load the network architecture
        weights_and_model   //save/load both the weights and the architecture
    };

    enum class file_format {
        binary,
        json
    };

    struct result {
        result() : num_success(0), num_total(0) {}

        int num_success;
        int num_total;
        map<label_t, map<label_t, int> > confusion_matrix;  //不用初始化？

        float accuracy() const {
            return float(num_success * 100.0 / num_total);
        }
    };

    class Net {
    public:
        Net() {};

        virtual ~Net() {};

        vector<int> layers_neuron_num;
        int num_layers = 0;
        float learning_rate = 0.0;
        float batch_loss = 0.0; //一批样本的loss
        VectorXf output_error;
        int output_interval = 0;    //训练中间输出loss
        float fine_tune_factor = 0.0; //学习率调节因子

        // initialize net:generate weights matrices、layer matrices and bias matrices
        // bias default all zero
        void initNet(const vector<int> layers_neuron_num);

        // initialize the weights matrices
        void initWeights(const double w = 0);

        //Initial the bias matrices
        void initBias(const double w = 0);

        //Predict just one sample
        int predict_one(const vec_t &input);

        /**
         * trains the network for a fixed number of epochs (for classification task)
         *
         * This method takes label_t argument and convert to target vector automatically.
         * To train correctly, output dimension of last layer must be greater or equal to
         * number of label-ids.
         *
         * @param inputs             array of input data
         * @param class_labels       array of label-id for each input data(0-origin)
         * @param batch_size         number of samples per parameter update
         * @param epoch              number of training epochs
         */
        bool train(const vector<vec_t> &inputs, const vector<label_t> &class_labels, int batch_size, int epoch);

        result test(const std::vector<vec_t> &inputs, const std::vector<label_t> &class_labels);

        bool save(const string &filename,
                  content_type what = content_type::weights_and_model,
                  file_format format = file_format::binary);

    private:
        vector<VectorXf> layers;
        vector<MatrixXf> weights;
        vector<VectorXf> bias;
        vector<VectorXf> gradient;
        vector<VectorXf> zs;    //store all the z vectors(weighted input), layer by layer

        /**
        * train on one minibatch
        *
        * @param size is the number of data points to use in this batch
        */
        void train_once(const tensor_t *in,
                        const tensor_t *t,
                        int size);

        /**
        * trains on one minibatch, i.e. runs forward and backward propagation to calculate
        * the gradient of the loss function with respect to the network parameters (weights),
        * then calls the optimizer algorithm to update the weights
        *
        * @param batch_size the number of data points to use in this batch
        */
        void train_onebatch(const tensor_t *in,
                            const tensor_t *t,
                            int batch_size);

        void update_batch(const vector<tensor_t> &in, const vector<tensor_t> &t, int batch_size);

        //Backward
        void farward(VectorXf x);

        VectorXf cost_derivative(VectorXf output_activations, VectorXf y);

        //Forward
        void backward(const VectorXf &y, vector<MatrixXf> &nabla_w, vector<VectorXf> &nabla_b);

        label_t fprop_max_index(const VectorXf &in);
    };
}
#endif //LU_NET_NET_H
