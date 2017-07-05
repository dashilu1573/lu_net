#include <iostream>
#include <opencv2/opencv.hpp>
#include "include/net.h"
#include "include/mnist_parser.h"

using namespace std;
using namespace cv;
using namespace lu_net;

int main(int argc, char** argv) {
    vector<int> layers_neuron_num = {784, 30, 10};

    Net net;
    net.initNet(layers_neuron_num);
    net.initWeights(0);
    net.initBias(0);

    // load MNIST dataset
    string data_dir = "/Users/luyafei/GitHub/lu_net/data";
    vector<label_t> train_labels, test_labels;
    vector<vec_t> train_images, test_images;

    string train_labels_path = data_dir + "/train-labels.idx1-ubyte";
    string train_images_path = data_dir + "/train-images.idx3-ubyte";
    string test_labels_path  = data_dir + "/t10k-labels.idx1-ubyte";
    string test_images_path  = data_dir + "/t10k-images.idx3-ubyte";

    read_Mnist_Label(train_labels_path, train_labels);
    read_Mnist_Images(train_images_path, train_images);
    read_Mnist_Label(test_labels_path, test_labels);
    read_Mnist_Images(test_images_path, test_images);

    cout << "Initial test." << endl;
    result initial_test = net.test(test_images, test_labels);
    cout << "Initial test accuracy:" << initial_test.accuracy() << endl;

    cout << "start learning" << endl;
    int minibatch_size = 10;
    int num_epochs = 30;
    net.train(train_images, train_labels, minibatch_size, num_epochs);
    cout << "End training." << endl;

    cout << "Start test." << endl;
    result test_result = net.test(test_images, test_labels);
    cout << "Test accuracy:" << test_result.accuracy() << endl;

    net.save("lu_net.model", content_type::weights_and_model, file_format::binary);
    return 0;
}