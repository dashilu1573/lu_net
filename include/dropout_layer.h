//
// Created by 芦yafei  on 14/2/19.
//
#ifndef LU_NET_DROPOUT_H
#define LU_NET_DROPOUT_H

#include <eigen3/Eigen/Dense>

namespace lu_net{
    /**
    * applies dropout to the input
    * With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, otherwise outputs 0.
    * The scaling is so that the expected sum is unchanged.
     * @keep_prob : The probability that each element is kept.
    **/
    class dropout_layer {
    public:
        dropout_layer(int in_dim, float_t keep_prob, net_phase phase = net_phase::train)
                : keep_prob_(keep_prob),
                  phase_(phase),
                  scale_(1.0 / keep_prob),
                  in_size_(in_dim)
        {
            mask_.Zero(in_dim);
        }

        void forward_propagation(const Eigen::VectorXf &in_data, Eigen::VectorXf &out_data);
        void back_propagation(Eigen::VectorXf &out_grad, Eigen::VectorXf &in_grad);
    private:
        net_phase phase_;
        float_t keep_prob_;
        float_t scale_;
        int in_size_;
        Eigen::VectorXi mask_;
    };
}

#endif //LU_NET_DROPOUT_H
