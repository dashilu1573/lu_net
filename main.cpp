#include <iostream>
#include <opencv2/opencv.hpp>
#include "include/net.h"

using namespace std;
using namespace cv;
using namespace lu_net;

int main() {
    vector<int> layers_neuron_num = {100, 200, 10};

    Net net;
    net.initNet(layers_neuron_num);
    net.initWeights(0);
    net.initBias(0);
    getchar();
    return 0;
}