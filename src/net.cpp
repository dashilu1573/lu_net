//
// Created by 芦yafei  on 17/6/9.
//

#include <iostream>
#include "../include/net.h"
#include <eigen3/Eigen/Dense>
#include "../include/function.h"


using namespace std;
using namespace Eigen;

namespace lu_net {

    void Net::initNet(std::vector<int> layers_neuron_num) {
        // resize(int n,element)表示调整容器v的大小为n，调整后的每个元素的值为element，默认为0，
        // resize()会改变容器的容量和当前元素个数
        layers.resize(layers_neuron_num.size());

        //Generate every layer.
        for (int i = 0; i < layers_neuron_num.size(); i++) {
            layers[i] = VectorXf::Zero(layers_neuron_num[i]);
        }
        std::cout << "Genarate layers, sucessfully!" << std::endl;

        //Generate every weights matrix and bias
        weights.resize(layers.size() - 1);
        bias.resize(layers.size() - 1);
        delta_errs.resize(layers.size() - 1);
        derivatives.resize(layers.size() - 1);

        cout << "Generate weights matrices and bias successfuly!" << endl;
        cout << "initialize Net, done!" << endl;
    }

    void Net::initWeights(double w) {
        //initialize weights matrices and bias
        for (int i = 0; i < weights.size(); i++) {
            weights[i] = MatrixXf::Random(layers[i + 1].rows(), layers[i].rows());
        }
    }

    void Net::initBias(double w) {
        for (int i = 0; i < bias.size(); i++) {
            bias[i] = VectorXf::Zero(layers[i + 1].rows());
        }
    }

    //Activation function
    VectorXf Net::activationFunction(const VectorXf &x, string func_type) {
        VectorXf result;
        if (func_type == "sigmoid")
        {
            result = sigmoid(x);
        }
        if (func_type == "relu")
        {
            result = relu(x);
        }
        return result;
    }

    //Compute delta error 反向计算每层结点的误差
    void Net::deltaError() {
        for (int i = delta_errs.size() - 1; i >= 0; i--) {
            delta_errs[i] = VectorXf::Zero(layers[i + 1].size());

            // output layer delta error
            if (i == delta_errs.size() - 1) {
                delta_errs[i] = output_error;
            }
                // hidden layer delta error
            else {
                delta_errs[i] = weights[i + 1].transpose() * delta_errs[i + 1];
            }
        }
    }

    //Compute derivative 计算节点激活函数的导数
    void Net::derivative(string func_type) {
        for (int i = 0; i < layers_neuron_num.size() - 1; i++) {
            if (func_type == "sigmoid") {
                derivatives[i] = sigmoid(layers[i]).array() * (1 - sigmoid(layers[i]).array());
            } else if (func_type == "relu") {
                derivatives[i] = (layers[i].array() >= 0.0).cast<float>();
            }
        }
    }

    //Upadate weights
    void Net::updateWeights() {
        for (int i = 0; i < weights.size(); i++) {
            MatrixXf delta_weights = learning_rate * (delta_errs[i].array() * derivatives[i].array()).matrix() * layers[i].transpose();
            weights[i] = weights[i] + delta_weights;
        }
    }

    //farward
    void Net::farward(string func_type) {
        for (int i = 0; i < layers_neuron_num.size() - 1; i++){
            VectorXf product = weights[i] * layers[i] + bias[i];
            layers[i + 1] = activationFunction(product, func_type);
        }

        //caculate loss on output layer
        calcLoss(layers.back(), target, output_error, loss);
    }

    //backward refer to http://blog.csdn.net/xingchenbingbuyu/article/details/53677630
    void Net::backward() {
        deltaError();
        updateWeights();
    }

    void Net::train(MatrixXf train_data) {
        if (0 == train_data.rows()){
            cout << "Input is empty!" << endl;
            return;
        }

        cout << "Train begin!" << endl;

        //一个样本
        if (train_data.rows() == layers[0].rows() && train_data.cols() == 1) {

        }
        //一批样本
        else if (train_data.rows() == layers[0].rows() && train_data.cols() > 1){

        }

    }
}