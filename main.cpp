#include <iostream>
#include <opencv2/opencv.hpp>
#include <loss_function.h>
#include "net.h"
#include "mnist_parser.h"
#include "loss_function.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <vector>
#include "optimizer.h"
#include "display.h"
#include "dropout_layer.h"

using namespace std;
using namespace cv;
using namespace lu_net;

DEFINE_string(data_dir, "/Users/luyafei/GitHub/lu_net/data/mnist", "Data directory");

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Print output to stderr (while still logging).
    FLAGS_alsologtostderr = 1;
    google::InitGoogleLogging(argv[0]);
    FLAGS_log_dir = ".";   //set log directory

    vector<int> layers_neuron_num = {784, 100, 10};

    Net net;
    net.initNet(layers_neuron_num, 0.5, 5.0);
    net.initWeights(0);
    net.initBias(0);

    // load MNIST dataset
    string data_dir = FLAGS_data_dir;
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

    timer t;
    cout << t.elapsed() << "s elapsed." << std::endl;

    LOG(INFO) << "Initial val.";

    result initial_test = net.test(test_images, test_labels);
    LOG(INFO) << "Initial val accuracy:" << initial_test.accuracy();

    LOG(INFO) << "start learning";
    int minibatch_size = 10;
    int num_epochs = 30;
    optimizer::gradient_descent op;

    net.train<cross_entropy>(op, train_images, train_labels, minibatch_size, num_epochs);
    LOG(INFO) << "End training.";

    LOG(INFO) << "Start val.";
    result test_result = net.test(test_images, test_labels);
    LOG(INFO) << "Test accuracy:" << test_result.accuracy();

    net.save("lu_net.model", content_type::weights_and_model, file_format::binary);

    gflags::ShutDownCommandLineFlags();
    return 0;
}