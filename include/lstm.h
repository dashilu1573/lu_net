//
// Created by 芦yafei  on 17/7/20.
//
#ifndef LU_NET_LSTM_H
#define LU_NET_LSTM_H

#include <eigen3/Eigen/Dense>
#include "activation_function.h"
#include "loss_function.h"
#include "optimizer.h"
#include <vector>

namespace lu_net{
    class LstmParam {
    public:
        LstmParam(int mem_cell_num, int x_dim);

        ~LstmParam();

        void param_update(float lr);

        int mem_cell_num_ = 0;      // LSTM cell num
        int x_dim_ = 0;             // Dimensions of input x

        // Weight matrices
        Eigen::MatrixXf wg;         // Input node
        Eigen::MatrixXf wi;         // Input gate
        Eigen::MatrixXf wf;         // Forget gate
        Eigen::MatrixXf wo;         // Output gate

        // Bias terms
        Eigen::VectorXf bg;         // Input node
        Eigen::VectorXf bi;         // Input gate
        Eigen::VectorXf bf;         // Forget gate
        Eigen::VectorXf bo;         // Output gate

        int concat_len_ = 0;        // Input dimensions of LSTM cell(LSTM cell num + dimensions of input x)

        // Diffs (derivative of loss function for all parameters)
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
        LstmState(int mem_cell_num, int x_dim);

        ~LstmState();

        // State of all LSTM cell.
        Eigen::VectorXf g;
        Eigen::VectorXf i;
        Eigen::VectorXf f;
        Eigen::VectorXf o;
        Eigen::VectorXf s;              // internal state
        Eigen::VectorXf h;              // the values output by each memory cell in the hidden layer
        Eigen::VectorXf bottom_diff_h;
        Eigen::VectorXf bottom_diff_s;

    };


    class LstmNode {
    public:
        LstmNode(LstmParam &lstmParam, LstmState &lstmState);

        ~LstmNode();

        /**
        * Forward propagation
        * **/
        void farward_prop(Eigen::VectorXf x);

        /**
         * Forward propagation
         * @s_prev: s in t - 1
         * @h_prev: h in t - 1
         * **/
        void farward_prop(Eigen::VectorXf x, Eigen::VectorXf s_prev, Eigen::VectorXf h_prev);


        void top_diff_is(Eigen::VectorXf top_diff_h, Eigen::VectorXf top_diff_s);

        LstmState state_;

    private:
        LstmParam param_;
        Eigen::VectorXf xc_;
        Eigen::VectorXf s_prev_;
        Eigen::VectorXf h_prev_;
    };


    class LstmNetwork {
        LstmNetwork(LstmParam lstmParam);

        ~LstmNetwork();

        /**
         * Updates diffs by setting target sequence with corresponding loss layer.
         * Will 'NOT' update parameters.
         * To update parameters,call self.lstm_param.apply_diff()
         * **/
        float compute_loss(const Eigen::VectorXf &y_list, MSE mse);

        /**
         * Bulding the X list for time series analysis.
         * **/
        void build_x_list(const Eigen::VectorXf& x);

    private:
        LstmParam lstmParam_;
        std::vector<LstmNode*> lstm_node_list;
        std::vector<Eigen::VectorXf> x_list;    // input sequence
    };
}
#endif //LU_NET_LSTM_H
