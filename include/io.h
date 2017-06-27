//
// Created by 芦yafei  on 17/6/23.
//

#ifndef LU_NET_IO_H
#define LU_NET_IO_H

#include "net.h"

using namespace lu_net;

//label转成向量形式
void label2vec(const label_t *t,
               int num,
               vector<vec_t> &vec) {
    int outdim = 10;
    vec.reserve(num);
    for (int i = 0; i < num; i++) {
        assert(t[i] < outdim);
        vec.emplace_back(outdim, 0);
        vec.back()[t[i]] = 1;
    }
}

void normalize_tensor(const vector<tensor_t> &inputs,
                      vector<tensor_t> &normalized) {
    normalized = inputs;
}

void normalize_tensor(const vector<vec_t> &inputs,
                      vector<tensor_t> &normalized) {
    normalized.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
        //emplace_back和push_back都是向容器内添加数据，emplace_back更高效
        normalized.emplace_back(tensor_t{inputs[i]});
}

void normalize_tensor(const vector<label_t> &inputs,
                      vector<tensor_t> &normalized) {
    vector<vec_t> vec;
    normalized.reserve(inputs.size());
    label2vec(&inputs[0], inputs.size(), vec);
    normalize_tensor(vec, normalized);
}

#endif //LU_NET_IO_H
