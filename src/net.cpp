//
// Created by 芦yafei  on 17/6/9.
//

#include <iostream>
#include "../include/net.h"
#include <eigen3/Eigen/Dense>
#include "../include/function.h"
#include "../include/io.h"


using namespace std;
using namespace Eigen;

namespace lu_net {

    void Net::initNet(std::vector<int> layers_neuron_num) {
        num_layers = layers_neuron_num.size();
        fine_tune_factor = 1.01;    //设置学习率变化因子
        output_interval = 10;  //设置训练loss输出间隔

        // resize(int n,element)表示调整容器v的大小为n，调整后的每个元素的值为element，默认为0，
        // resize()会改变容器的容量和当前元素个数
        layers.resize(num_layers);

        //Generate every layer.
        for (int i = 0; i < num_layers; i++) {
            layers[i] = VectorXf::Zero(layers_neuron_num[i]);
        }
        std::cout << "Genarate layers, sucessfully!" << std::endl;

        //Generate every weights matrix and bias，index 0 is unused, use num_layers size for uniform index
        weights.resize(num_layers);
        bias.resize(num_layers);
        gradient.resize(num_layers);
        zs.resize(num_layers);

        cout << "Generate weights matrices and bias successfuly!" << endl;
        cout << "initialize Net, done!" << endl;
    }

    // initialize weights matrices
    void Net::initWeights(double w) {
        for (int i = 1; i < num_layers; i++) {
            weights[i] = MatrixXf::Random(layers[i].rows(), layers[i - 1].rows());
        }
    }

    // initialize bias vectors
    void Net::initBias(double w) {
        for (int i = 1; i < num_layers; i++) {
            bias[i] = VectorXf::Zero(layers[i].rows());
        }
    }

    //farward
    void Net::farward(VectorXf x) {
        layers[0] = x;

        for (int i = 1; i < num_layers; i++){
            //weighted input
            VectorXf z = weights[i] * layers[i - 1] + bias[i];
            zs[i] = z;
            layers[i] = sigmoid(z);
        }

        //caculate loss on output layer
        calcLoss(layers[num_layers - 1], target, output_error, loss);
    }

    /*compute the partial derivatives for the output activations.
     * */
    VectorXf Net::cost_derivative(VectorXf output_activations, VectorXf y) {
        return (output_activations - y);
    }


    /*compute the w and b gradient of the cost function C_x
     * */
    VectorXf Net::backward(const VectorXf &y, vector<MatrixXf> &nabla_w, vector<VectorXf> &nabla_b) {
        //最后一层的error
        VectorXf delta = cost_derivative(layers[num_layers - 1], y) * sigmoid_prime(zs[num_layers -1]);
        nabla_b[num_layers - 1] = delta;
        nabla_w[num_layers - 1] = delta * layers[num_layers -2].transpose();

        for (int i = num_layers - 2; i >= 1; i--) {
            delta = (weights[i + 1].transpose() * delta).array() * sigmoid_prime(zs[i]).array();
            nabla_b[i] = delta;
            nabla_w[i] = delta * layers[i - 1].transpose();
        }
    }


    /*Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.
     * The mini_batch is a list of tuples (x, y), and lr is the learning rate.
     * */
    void Net::update_batch(vector< pair<label_t, tensor_t > > mini_batch_data, float lr) {

        int batch_size = mini_batch_data.size();

        vector<MatrixXf> acum_nabla_w;
        acum_nabla_w.resize(num_layers);

        vector<VectorXf> acum_nabla_b;
        acum_nabla_b.resize(num_layers);


        vector<MatrixXf> delta_nabla_w;
        vector<VectorXf> delta_nabla_b;

        for(int i = 0; i < mini_batch_data.size(); i++) {
            pair<label_t, tensor_t> one = mini_batch_data[i];
            //farward(one.first);
            //backward(one.second, delta_nabla_w, delta_nabla_b);

            //将一批样本的改变累加到一起
            for (int j = 1; j < num_layers; ++j) {
                acum_nabla_w[j] = acum_nabla_w[j] + delta_nabla_w[j];
                acum_nabla_b[j] = acum_nabla_b[j] + delta_nabla_b[j];
            }
        }

        //一批样本改变的平均值作为最后的改变
        for (int k = 1; k < num_layers; ++k) {
            weights[k] = weights[k] - lr / batch_size * acum_nabla_w[k];
            bias[k] = bias[k] - lr / batch_size * acum_nabla_b[k];
        }
    }


    bool Net::train(const std::vector<vec_t> &inputs, const std::vector<label_t> &class_labels, int batch_size, int epoch) {
        if (inputs.size() != class_labels.size()) {
            return false;
        }
        if (inputs.size() < batch_size || class_labels.size() < batch_size) {
            return false;
        }
        std::vector<tensor_t> input_tensor, output_tensor, t_cost_tensor;
        normalize_tensor(inputs, input_tensor);
        normalize_tensor(class_labels, output_tensor);
    }

}