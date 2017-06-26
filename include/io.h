//
// Created by 芦yafei  on 17/6/23.
//

#ifndef LU_NET_IO_H
#define LU_NET_IO_H

#include "net.h"

using namespace lu_net;

/*void label2vec(const label_t *t,
               int num,
               std::vector<vec_t> &vec) const {
    serial_size_t outdim = out_data_size();

    vec.reserve(num);
    for (int i = 0; i < num; i++) {
        assert(t[i] < outdim);
        vec.emplace_back(outdim, target_value_min());
        vec.back()[t[i]] = target_value_max();
    }
}*/

void normalize_tensor(const std::vector<tensor_t> &inputs,
                      std::vector<tensor_t> &normalized) {
    normalized = inputs;
}

void normalize_tensor(const std::vector<vec_t> &inputs,
                      std::vector<tensor_t> &normalized) {
    normalized.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
        //emplace_back和push_back都是向容器内添加数据，emplace_back更高效
        normalized.emplace_back(tensor_t{inputs[i]});
}

void normalize_tensor(const std::vector<label_t> &inputs,
                      std::vector<tensor_t> &normalized) {
    std::vector<vec_t> vec;
    normalized.reserve(inputs.size());
    //label2vec(inputs, vec);
    normalize_tensor(vec, normalized);
}

#endif //LU_NET_IO_H
