//
// Created by 芦yafei  on 14/6/9.
//

#include <iostream>
#include <unistd.h>
#include "net.h"
#include <eigen3/Eigen/Dense>
#include <fstream>
#include "function.h"
#include "io.h"
#include "Matrix.h"
#include "loss_function.h"
#include "random.h"
#include <glog/logging.h>
#include "activation_function.h"

using namespace std;
using namespace Eigen;

namespace lu_net {

    void Net::initNet(std::vector<int> layers_neuron_num, float learning_rate, float lmbda) {
        this->layers_neuron_num = layers_neuron_num;
        num_layers = layers_neuron_num.size();
        this->learning_rate = learning_rate;
        fine_tune_factor = 1;    // finetune factor of learning rate.
        this->lmbda = lmbda;
        output_interval = 1;  //设置训练loss输出间隔,epoch为单位

        // resize(int n,element)表示调整容器v的大小为n，调整后的每个元素的值为element，默认为0，
        // resize()会改变容器的容量和当前元素个数
        layers.resize(num_layers);

        //Generate every layer.
        for (int i = 0; i < num_layers; i++) {
            layers[i] = VectorXf::Zero(layers_neuron_num[i]);
        }
        LOG(INFO) << "Genarate layers, sucessfully!";

        //Generate every weights matrix and bias，index 0 is unused, use num_layers size for uniform index
        weights.resize(num_layers);
        bias.resize(num_layers);
        gradient.resize(num_layers);
        zs.resize(num_layers);

        LOG(INFO) << "Generate weights matrices and bias successfuly!";
        LOG(INFO) << "initialize Net, done!";
    }


    /**
     * Initialize each weight using a Gaussian distribution with mean 0 and standard deviation 1 over the square root
     * of the number of weights connecting to the same neuron.  Initialize the biases using a Gaussian distribution with
     * mean 0 and standard deviation 1.
     **/
    void Net::initWeights(double w) {
        for (int i = 1; i < num_layers; i++) {
            weights[i] = MatrixXf::Zero(layers[i].rows(), layers[i - 1].rows())
                    .unaryExpr(ptr_fun(gaussian_random)) / sqrt(float(layers[i - 1].rows()));
        }
    }


    /**
     * Initialize each weight using a Gaussian distribution with mean 0 and standard deviation 1.
     * */
    void Net::initBias(double w) {
        for (int i = 1; i < num_layers; i++) {
            bias[i] = VectorXf::Zero(layers[i].rows()).unaryExpr(ptr_fun(gaussian_random));
        }
    }

    // farward
    void Net::farward(VectorXf x) {
        layers[0] = x;

        for (int i = 1; i < num_layers; i++){
            //weighted input
            VectorXf z = weights[i] * layers[i - 1] + bias[i];
            zs[i] = z;
            layers[i] = activation::sigmoid::f(z);
        }
    }


    /**
     * Compute the w and b gradient of the cost function C_x
     * */
    template <typename E>
    void Net::backward(const VectorXf &y, vector<MatrixXf> &nabla_w, vector<VectorXf> &nabla_b) {
        // error of last layer
        // VectorXf delta = cost_derivative(layers[num_layers - 1], y).array() * sigmoid_prime(zs[num_layers -1]).array();
        VectorXf delta = E::df(layers[num_layers - 1], y).array() * activation::sigmoid::df(zs[num_layers -1]).array();
        nabla_b[num_layers - 1] = delta;
        nabla_w[num_layers - 1] = delta * layers[num_layers -2].transpose();

        for (int i = num_layers - 2; i >= 1; i--) {
            delta = (weights[i + 1].transpose() * delta).array() * activation::sigmoid::df(zs[i]).array();
            nabla_b[i] = delta;
            nabla_w[i] = delta * layers[i - 1].transpose();
        }
    }


    /**
     * Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.
     * The mini_batch is a list of tuples (x, y), and lr is the learning rate.
     * @n is the total size of the training data set.
     * */
    template <typename E>
    void Net::update_batch(const vector<tensor_t>& in, const vector<tensor_t>& t, int batch_size, int n) {
        //累加到一起的改变
        vector<MatrixXf> acum_nabla_w;
        acum_nabla_w.resize(num_layers);

        vector<VectorXf> acum_nabla_b;
        acum_nabla_b.resize(num_layers);

        //初始化全零
        for (int i = 1; i < num_layers; i++) {
            acum_nabla_w[i] = MatrixXf::Zero(layers[i].rows(), layers[i - 1].rows());
            acum_nabla_b[i] = VectorXf::Zero(layers[i].rows());
        }

        vector<MatrixXf> delta_nabla_w;
        delta_nabla_w.resize(num_layers);

        vector<VectorXf> delta_nabla_b;
        delta_nabla_b.resize(num_layers);

        //一个batch里的loss累加
        float batch_sum_loss = 0.0;

        //一个样本一个样本的训练
        for(int i = 0; i < batch_size; i++) {
            //从std::vector转成Eigen形式
            //VectorXf x(&in[i][0], in[i][0].size());
            VectorXf x(in[i][0].size());
            VectorXf y(t[i][0].size());

            for (int k = 0; k < in[i][0].size(); ++k) {
                x[k] = in[i][0][k];
            }

            for (int k = 0; k < t[i][0].size(); ++k) {
                y[k] = t[i][0][k];
            }

            farward(x);
            backward<E>(y, delta_nabla_w, delta_nabla_b);

            float loss = E::f(layers[num_layers - 1], y);
            batch_sum_loss += loss;

            //每个样本的改变累加到一起
            for (int j = 1; j < num_layers; ++j) {
                acum_nabla_w[j] = acum_nabla_w[j] + delta_nabla_w[j];
                acum_nabla_b[j] = acum_nabla_b[j] + delta_nabla_b[j];
            }
        }

        // 一批样本改变的平均值作为最后的改变
        for (int k = 1; k < num_layers; ++k) {
            weights[k] = weights[k] - learning_rate / batch_size * acum_nabla_w[k];
            // weights[k] = ( 1 - learning_rate * (lmbda / n) ) * weights[k] - learning_rate / batch_size * acum_nabla_w[k];
            bias[k] = bias[k] - learning_rate / batch_size * acum_nabla_b[k];
        }

        // 求loss平均值
        batch_loss = batch_sum_loss / batch_size;

        // add regularization term
        float sum_squares_weights = 0;
        for (int k = 1; k < num_layers; k++)
        {
            sum_squares_weights += weights[k].array().square().matrix().sum();
        }
        // batch_loss += 0.5 * (lmbda / batch_size) * sum_squares_weights;
    }


    /**
    * trains on one minibatch, i.e. runs forward and backward propagation to calculate
    * the gradient of the loss function with respect to the network parameters (weights),
    * then calls the optimizer algorithm to update the weights
    *
    * @param batch_size the number of data points to use in this batch
    */
    template <typename E>
    void Net::train_onebatch(const tensor_t* in, const tensor_t* t, int batch_size, int n) {
        vector<tensor_t> in_batch(&in[0], &in[0] + batch_size);
        vector<tensor_t> t_batch(&t[0], &t[0] + batch_size);

        update_batch<E>(in_batch, t_batch, batch_size, n);
    }


    /**
    * train on one minibatch
    *
    * @param size is the number of data points to use in this batch
    */
    template <typename E>
    void Net::train_once(const tensor_t *in,
                    const tensor_t *t,
                    int size,
                    int n) {
        if (size == 1) {

        } else {
            train_onebatch<E>(in, t, size, n);
        }
    }


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
    template <typename E>
    bool Net::train(const vector<vec_t> &inputs, const vector<label_t> &class_labels, int batch_size, int epoch) {
        if (inputs.size() != class_labels.size()) {
            return false;
        }
        if (inputs.size() < batch_size || class_labels.size() < batch_size) {
            return false;
        }

        // size of training set.
        int n = inputs.size();

        //转化成tensor_t类型
        vector<tensor_t> input_tensor, output_tensor, t_cost_tensor;
        normalize_tensor(inputs, input_tensor);
        normalize_tensor(class_labels, output_tensor);

        //整个样本集训练epoch次
        for (int iter = 0; iter < epoch; iter++) {
            LOG(INFO) << "epoch:" << iter;
            LOG(INFO) << "learning rate:" << learning_rate;

            for (int i = 0; i < inputs.size(); i += batch_size) {
                // train on one minibatch
                train_once<E>(&input_tensor[i],
                           &output_tensor[i],
                           static_cast<int>(min<int>(batch_size, inputs.size() - i)),
                           n);
            }

            LOG(INFO) << "last batch_loss:" << batch_loss;

            //change learning rate
            if (iter % output_interval == 0)
            {
                learning_rate *= fine_tune_factor;
            }
        }

        LOG(INFO) << "End training.";

        return true;
    }

    // instance for template function
    template bool Net::train<cross_entropy>(const vector<vec_t> &inputs, const vector<label_t> &class_labels, int batch_size,
                                       int epoch);


    /**
    * test and generate confusion-matrix for classification task
    **/
    result Net::test(const std::vector<vec_t> &inputs, const std::vector<label_t> &class_labels) {
        result test_result;

        if (inputs.empty())
        {
            cout << "Test inputs is empty!" << endl;
            return test_result;
        }

        for (int i = 0; i < inputs.size(); i++) {
            //change inputs format from vector toVectorXf
            VectorXf x(inputs[i].size());
            for (int k = 0; k < inputs[i].size(); ++k) {
                x[k] = inputs[i][k];
            }

            label_t predicted = fprop_max_index(x);
            label_t actual = class_labels[i];

            if (predicted == actual) {
                test_result.num_success += 1;
            }

            test_result.num_total += 1;
            test_result.confusion_matrix[predicted][actual]++;
        }

        return test_result;
    }


    /**
     * farward prop and retun the index of last layer
     */
    label_t Net::fprop_max_index(const VectorXf &in) {
        farward(in);
        int max_index = 0;
        layers[num_layers - 1].maxCoeff(&max_index);
        return label_t(max_index);
    }

    /**
     * sava model
     */
    bool Net::save(const string &filename,
                   content_type what,
                   file_format format) {
        // Verify that the version of the library that we linked against is
        // compatible with the version of the headers we compiled against.
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        /***************save model param*************/
        ModelMsg *modelMsg = new ModelMsg();
        //内嵌对象必须用new的方式，否则后面set_allocated引起的double Free Core Dump
        NetParameterMsg* netParameterMsg = new NetParameterMsg();
        for (int i = 0; i < layers_neuron_num.size(); ++i) {
            //repeated tag use add operation repeatedly
            //repeated的基本数据类型使用add直接添加
            netParameterMsg->add_layers_neuron_num(layers_neuron_num[i]);
        }
        modelMsg->set_learning_rate(learning_rate);
        modelMsg->set_allocated_net_param(netParameterMsg);

        /****************save weights*************/
        WeightsMsg *weightsMsg = new WeightsMsg();
        for (int i = 1; i < num_layers; ++i) {
            //repeated tag use add operation repeatedly
            //repeated的内嵌对象类型使用add传指针出来添加
            MatrixMsg *weightsData = weightsMsg->add_weights();
            VectorMsg *biasData = weightsMsg->add_bias();
            WriteMatrix(weights[i], weightsData);
            WriteVector(bias[i], biasData);
        }

        /***************save model and weights*****************/
        ModelWeightsMsg modelWeightsMsg;
        modelWeightsMsg.set_allocated_model(modelMsg);
        modelWeightsMsg.set_allocated_weights(weightsMsg);

        if (format == file_format::json) {
            switch (what) {
                case content_type::weights_and_model :
                    WriteProtoToTextFile(modelWeightsMsg, filename.c_str());
                case content_type::weights :
                    WriteProtoToTextFile(*weightsMsg, filename.c_str());
                case content_type::model :
                    WriteProtoToTextFile(*modelMsg, filename.c_str());
            }
        }
        else if (format == file_format::binary) {
            switch (what) {
                case content_type::weights_and_model :
                    WriteProtoToBinaryFile(modelWeightsMsg, filename.c_str());
                case content_type::weights :
                    WriteProtoToBinaryFile(*weightsMsg, filename.c_str());
                case content_type::model :
                    WriteProtoToBinaryFile(*modelMsg, filename.c_str());
            }
        }

        // Optional:  Delete all global objects allocated by libprotobuf.
        google::protobuf::ShutdownProtobufLibrary();

        return 0;
    }
}