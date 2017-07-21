//
// Created by 芦yafei  on 17/7/20.
//
#ifndef LU_NET_LSTM_H
#define LU_NET_LSTM_H

#include "eigen3/Eigen"
#include "activation_function.h"
#include "loss_function.h"
#include "optimizer.h"
#include <vector>

namespace lu_net{
    class LstmParam {
    public:
        LstmParam(int mem_cell_num, int x_dim)
                : mem_cell_num_(mem_cell_num),
                  x_dim_(x_dim),
                  concat_len_(mem_cell_num + x_dim)
        {
            wg = Eigen::MatrixXf::Zero(mem_cell_num_, concat_len_);
            wi = Eigen::MatrixXf::Zero(mem_cell_num_, concat_len_);
            wf = Eigen::MatrixXf::Zero(mem_cell_num_, concat_len_);
            wo = Eigen::MatrixXf::Zero(mem_cell_num_, concat_len_);

            bg = Eigen::VectorXf::Zero(mem_cell_num_);
            bi = Eigen::VectorXf::Zero(mem_cell_num_);
            bf = Eigen::VectorXf::Zero(mem_cell_num_);
            bo = Eigen::VectorXf::Zero(mem_cell_num_);

            wg_diff = Eigen::MatrixXf::Zero(mem_cell_num_, concat_len_);
            wi_diff = Eigen::MatrixXf::Zero(mem_cell_num_, concat_len_);
            wf_diff = Eigen::MatrixXf::Zero(mem_cell_num_, concat_len_);
            wo_diff = Eigen::MatrixXf::Zero(mem_cell_num_, concat_len_);

            bg_diff = Eigen::VectorXf::Zero(mem_cell_num_);
            bi_diff = Eigen::VectorXf::Zero(mem_cell_num_);
            bf_diff = Eigen::VectorXf::Zero(mem_cell_num_);
            bo_diff = Eigen::VectorXf::Zero(mem_cell_num_);
        }

        void  param_update(float lr) {
            optimizer::gradient_descent::update_w(wg, wg_diff, lr);
            optimizer::gradient_descent::update_w(wi, wi_diff, lr);
            optimizer::gradient_descent::update_w(wf, wf_diff, lr);
            optimizer::gradient_descent::update_w(wo, wo_diff, lr);
            optimizer::gradient_descent::update_b(bg, bg_diff, lr);
            optimizer::gradient_descent::update_b(bi, bi_diff, lr);
            optimizer::gradient_descent::update_b(bf, bf_diff, lr);
            optimizer::gradient_descent::update_b(bo, bo_diff, lr);
        }

        int mem_cell_num_ = 0;   // LSTM cell num
        int x_dim_ = 0; //Dimensions of input x

    private:
        int concat_len_ = 0;    // Input dimensions of LSTM cell(LSTM cell num + dimensions of input x)

        // Weight matrices
        Eigen::MatrixXf wg; // Input node
        Eigen::MatrixXf wi; // Input gate
        Eigen::MatrixXf wf; // Forget gate
        Eigen::MatrixXf wo; // Output gate

        // Bias terms
        Eigen::VectorXf bg; // Input node
        Eigen::VectorXf bi; // Input gate
        Eigen::VectorXf bf; // Forget gate
        Eigen::VectorXf bo; // Output gate

        // Diffs (derivative of loss function w.r.t. all parameters)
        Eigen::MatrixXf wg_diff;
        Eigen::MatrixXf wi_diff;
        Eigen::MatrixXf wf_diff;
        Eigen::MatrixXf wo_diff;

        Eigen::VectorXf bg_diff;
        Eigen::VectorXf bi_diff;
        Eigen::VectorXf bf_diff;
        Eigen::VectorXf bo_diff;
    };


    class LstmState {
    public:
        LstmState(int mem_cell_num, int x_dim) {
            g = Eigen::VectorXf::Zero(mem_cell_num);
            i = Eigen::VectorXf::Zero(mem_cell_num);
            f = Eigen::VectorXf::Zero(mem_cell_num);
            o = Eigen::VectorXf::Zero(mem_cell_num);
            s = Eigen::VectorXf::Zero(mem_cell_num);
            h = Eigen::VectorXf::Zero(mem_cell_num);
        }

    private:
        // State of all LSTM cell.
        Eigen::VectorXf g;
        Eigen::VectorXf i;
        Eigen::VectorXf f;
        Eigen::VectorXf o;
        Eigen::VectorXf s;  // internal state
        Eigen::VectorXf h;  // the values output by each memory cell in the hidden layer
    };


    class LstmNode {
    public:
        LstmNode(LstmParam &lstmParam, LstmState &lstmState)
                :param(lstmParam),
                 state(lstmState)
        {
            // non-recurrent input concatenated with recurrent input
            xc = None;
        }

        void bottom_data_is(Eigen::VectorXf x) {
            Eigen::VectorXf s_prev = Eigen::VectorXf::Zero(state.s.size());
            Eigen::VectorXf h_prev = Eigen::VectorXf::Zero(state.h.size());

            bottom_data_is(x, s_prev, h_prev);
        }

        /**
         *
         * @s_prev: s in t - 1
         * @h_prev: h in t - 1
         * **/
        void bottom_data_is(Eigen::VectorXf x,
                            Eigen::VectorXf s_prev,
                            Eigen::VectorXf h_prev) {
            // save data for use in backprop
            this.s_prev = s_prev;
            this.h_prev = h_prev;

            // concatenate x(t) and h(t-1)
            xc << x, h_prev;
            state.g = activation::tanh(param.wg * xc + param.bg);
            state.i = activation::sigmoid(param.wi * xc + param.bi);
            state.f = activation::sigmoid(param.wf * xc + param.bf);
            state.o = activation::sigmoid(param.wo * xc + param.bo);
            state.s = state.g.array() * state.i.array() + s_prev.array() * state.f.array();
            state.h = state.s.array() * state.o.array();
        }

        void top_diff_is(top_diff_h, top_diff_s) {
        // notice that top_diff_s is carried along the constant error carousel
        }

    private:
        LstmState state;
        LstmParam param;
        Eigen::VectorXf xc;
        Eigen::VectorXf s_prev;
        Eigen::VectorXf h_prev;
    };


    class LstmNetwork {
        LstmNetwork(LstmParam lstmParam)
                :lstmParam_(lstmParam)
        {

        }

        /**
         *Updates diffs by setting target sequence with corresponding loss layer.
         * Will *NOT* update parameters.
         * To update parameters,call self.lstm_param.apply_diff()
         * **/
        void compute_loss(const Eigen::VectorXf &y_list, mse loss_funct) {
            assert(y_list.size() == x_list.size());
            idx = x_list.size() - 1;

            // first node only gets diffs from label ...
            // float loss = loss_funct.f(lstm_node_list[idx]->state.h, y_list[idx]);
            // Eigen::VectorXf diff_h = loss_funct.df()
        }

        void build_x_list(const std::vector<float>& x) {
            x_list.emplace_back(x);

            // Add node with add x;
            if (x_list.size() > lstm_node_list.size()) {
                //need to add new lstm node, create new state mem
                LstmState lstmState = LstmState(lstmParam_.mem_cell_num_, lstmParam_.x_dim_);

                LstmNode* lstmNode = new LstmNode(lstmParam_, lstmState);
                lstm_node_list.emplace_back(lstmNode);
            }

            //Get index of most recent x input
            idx = x_list.size() -1;

            // Forward propagation
            if (0 == idx) {
                // no recurrent inputs yet
                lstm_node_list[idx]->bottom_data_is(x);
            } else {
                Eigen::VectorXf s_prev = lstm_node_list[idx - 1]->states.s;
                Eigen::VectorXf h_prev = lstm_node_list[idx - 1]->states.h;

                lstm_node_list[idx]->bottom_data_is(x, s_prev, h_prev);
            }
        }

    private:
        LstmParam lstmParam_;
        std::vector<LstmNode*> lstm_node_list;
        std::vector<std::vector<float >> x_list;    // input sequence
    };
}
#endif //LU_NET_LSTM_H
