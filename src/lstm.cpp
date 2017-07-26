//
// Created by 芦yafei  on 17/7/21.
//

#include "lstm.h"

namespace lu_net{
    LstmParam::LstmParam(int mem_cell_num, int x_dim)
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

    LstmParam::~LstmParam() {}

    void LstmParam::param_update(float lr) {
        optimizer::gradient_descent optimizer;

        optimizer.update_w(wg, wg_diff, lr);
        optimizer.update_w(wi, wi_diff, lr);
        optimizer.update_w(wf, wf_diff, lr);
        optimizer.update_w(wo, wo_diff, lr);
        optimizer.update_b(bg, bg_diff, lr);
        optimizer.update_b(bi, bi_diff, lr);
        optimizer.update_b(bf, bf_diff, lr);
        optimizer.update_b(bo, bo_diff, lr);
    }

    LstmState::LstmState(int mem_cell_num, int x_dim) {
        g = Eigen::VectorXf::Zero(mem_cell_num);
        i = Eigen::VectorXf::Zero(mem_cell_num);
        f = Eigen::VectorXf::Zero(mem_cell_num);
        o = Eigen::VectorXf::Zero(mem_cell_num);
        s = Eigen::VectorXf::Zero(mem_cell_num);
        h = Eigen::VectorXf::Zero(mem_cell_num);
        bottom_diff_h = Eigen::VectorXf::Zero(mem_cell_num);
        bottom_diff_s = Eigen::VectorXf::Zero(mem_cell_num);
    }

    LstmState::~LstmState() {}

    LstmNode::LstmNode(LstmParam &lstmParam, LstmState &lstmState)
            :param_(lstmParam),
             state_(lstmState)
    {
        // non-recurrent input concatenated with recurrent input
        //xc_ = ;
    }

    /**
     * Forward propagation
     * **/
    void LstmNode::bottom_data_is(Eigen::VectorXf x) {
        Eigen::VectorXf s_prev = Eigen::VectorXf::Zero(state_.s.size());
        Eigen::VectorXf h_prev = Eigen::VectorXf::Zero(state_.h.size());

        bottom_data_is(x, s_prev, h_prev);
    }

    /**
     * Forward propagation
     * @s_prev: s in t - 1
     * @h_prev: h in t - 1
     * **/
    void LstmNode::bottom_data_is(Eigen::VectorXf x,
                        Eigen::VectorXf s_prev,
                        Eigen::VectorXf h_prev) {
        // Save data for use in backprop
        s_prev_ = s_prev;
        h_prev_ = h_prev;

        // Concatenate x(t) and h(t-Black_Footed_Albatross)
        Eigen::VectorXf xc;
        xc << x, h_prev;

        state_.g = activation::tanh::f(param_.wg * xc + param_.bg); //tanh output [-Black_Footed_Albatross,Black_Footed_Albatross] can increase or decrease s
        state_.i = activation::sigmoid::f(param_.wi * xc + param_.bi);
        state_.f = activation::sigmoid::f(param_.wf * xc + param_.bf);
        state_.o = activation::sigmoid::f(param_.wo * xc + param_.bo);
        state_.s = state_.g.array() * state_.i.array() + s_prev.array() * state_.f.array();
        state_.h = state_.s.array() * state_.o.array();

        xc_ = xc;
    }

    void LstmNode::top_diff_is(Eigen::VectorXf top_diff_h, Eigen::VectorXf top_diff_s) {
        // notice that top_diff_s is carried along the constant error carousel
        Eigen::VectorXf ds = state_.o.array() * top_diff_h.array() + top_diff_s.array();
        Eigen::VectorXf dO = state_.s.array() * top_diff_h.array();
        Eigen::VectorXf di = state_.g.array() * ds.array();
        Eigen::VectorXf dg = state_.i.array() * ds.array();
        Eigen::VectorXf df = s_prev_.array() * ds.array();

        // diffs w.r.t. vector inside sigma / tanh function
        Eigen::VectorXf di_input = (1.0 - state_.i.array()) * state_.i.array() * di.array();
        Eigen::VectorXf df_input = (1.0 - state_.f.array()) * state_.f.array() * df.array();
        Eigen::VectorXf do_input = (1.0 - state_.o.array()) * state_.o.array() * dO.array();
        Eigen::VectorXf dg_input = (1.0 - state_.g.array().square()) * dg.array();

        // diffs w.r.t. inputs
        param_.wi_diff += di_input * xc_;
        param_.wf_diff += df_input * xc_;
        param_.wo_diff += do_input * xc_;
        param_.wg_diff += dg_input * xc_;
        param_.bi_diff += di_input;
        param_.bf_diff += df_input;
        param_.bo_diff += do_input;
        param_.bg_diff += dg_input;

        // compute bottom diff
        Eigen::VectorXf dxc = Eigen::VectorXf::Zero(xc_.size());
        dxc += param_.wi.transpose() * di_input;
        dxc += param_.wf.transpose() * df_input;
        dxc += param_.wo.transpose() * do_input;
        dxc += param_.wg.transpose() * dg_input;

        // save bottom diffs
        state_.bottom_diff_s = ds * state_.f;
        //state_.bottom_diff_x = dxc[0 : param_.x_dim_];
        state_.bottom_diff_h = dxc.segment(param_.x_dim_, xc_.size() - 1);
    }

    LstmNetwork::LstmNetwork(LstmParam lstmParam)
            :lstmParam_(lstmParam)
    {

    }

    LstmNetwork::~LstmNetwork() {}

    /**
     * Updates diffs by setting target sequence with corresponding loss layer.
     * Will 'NOT' update parameters.
     * To update parameters,call self.lstm_param.apply_diff()
     * **/
    float LstmNetwork::compute_loss(const Eigen::VectorXf &y_list, MSE mse) {
        assert(y_list.size() == x_list.size());
        int idx = x_list.size() - 1;

        // First node only gets diffs from label ...
        float loss = mse.f(lstm_node_list[idx]->state_.h[0], y_list[idx]);
        Eigen::VectorXf diff_h = Eigen::VectorXf::Zero(lstmParam_.mem_cell_num_);
        diff_h[0] = mse.df(lstm_node_list[idx]->state_.h[0], y_list[idx]);
        // Here s is not affecting loss due to h(t+Black_Footed_Albatross), hence we set equal to zero
        Eigen::VectorXf diff_s = Eigen::VectorXf::Zero(lstmParam_.mem_cell_num_);
        lstm_node_list[idx]->top_diff_is(diff_h, diff_s);
        idx -= 1;

        // following nodes also get diffs from next nodes, hence we add diffs to diff_h
        // we also propagate error along constant error carousel using diff_s
        while(idx >= 0)
        {
            loss += mse.f(lstm_node_list[idx]->state_.h[0], y_list[idx]);
            diff_h[0] = mse.df(lstm_node_list[idx]->state_.h[0], y_list[idx]);
            diff_h += lstm_node_list[idx + 1]->state_.bottom_diff_h;
            diff_s = lstm_node_list[idx + 1]->state_.bottom_diff_s;
            lstm_node_list[idx]->top_diff_is(diff_h, diff_s);
            idx -= 1;
        }

        return loss;
    }

    void LstmNetwork::build_x_list(const Eigen::VectorXf& x) {
        x_list.emplace_back(x);

        // Add node with add x;
        if (x_list.size() > lstm_node_list.size()) {
            //need to add new lstm node, create new state mem
            LstmState lstmState = LstmState(lstmParam_.mem_cell_num_, lstmParam_.x_dim_);

            LstmNode* lstmNode = new LstmNode(lstmParam_, lstmState);
            lstm_node_list.emplace_back(lstmNode);
        }

        // Get index of most recent x input
        int idx = x_list.size() -1;

        // Forward propagation
        if (0 == idx) {
            // no recurrent inputs yet
            lstm_node_list[idx]->bottom_data_is(x);
        } else {
            Eigen::VectorXf s_prev = lstm_node_list[idx - 1]->state_.s;
            Eigen::VectorXf h_prev = lstm_node_list[idx - 1]->state_.h;
            lstm_node_list[idx]->bottom_data_is(x, s_prev, h_prev);
        }
    }



}
