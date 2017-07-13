//
// Created by 芦yafei  on 17/7/11.
//

#ifndef LU_NET_ACTIVATION_FUNCTION_H
#define LU_NET_ACTIVATION_FUNCTION_H

namespace lu_net{
    namespace activation{

        class function {
        public:
            function() = default;

            virtual ~function() = default;

            virtual float_t f(const vec_t& v, cnn_size_t index) const = 0;

            // dfi/dyi
            virtual float_t df(float_t y) const = 0;
        };


        class sigmoid : public function {
        public:
            using function::df;
            float_t f(const vec_t& v, cnn_size_t i) const override {
                return float_t(1) / (float_t(1) + std::exp(-v[i]));
            }
            float_t df(float_t y) const override {
                return y * (float_t(1) - y);
            }
        };


        class relu : public function {
        public:
            using function::df;
            float_t f(const vec_t& v, cnn_size_t i) const override {
                return std::max(float_t(0), v[i]);
            }
            float_t df(float_t y) const override {
                return y > float_t(0) ? float_t(1) : float_t(0);
            }
        };


        class softmax : public function {
        public:
            float_t f(const vec_t& v, cnn_size_t i) const override {
                float_t alpha = *std::max_element(v.begin(), v.end());
                float_t numer = std::exp(v[i] - alpha);
                float_t denom = float_t(0);
                for (auto x : v)
                    denom += std::exp(x - alpha);
                return numer / denom;
            }

            float_t df(float_t y) const override {
                return y * (float_t(1) - y);
            }

            virtual vec_t df(const vec_t& y, cnn_size_t index) const override {
                vec_t v(y.size(), 0);
                for (cnn_size_t i = 0; i < y.size(); i++)
                    v[i] = (i == index) ? df(y[index]) : -y[i] * y[index];

                return v;
            }
        };


        class tan_h : public function {
        public:
            using function::df;
            float_t f(const vec_t& v, cnn_size_t i) const override {
                const float_t ep = std::exp(v[i]);
                const float_t em = std::exp(-v[i]);
                return (ep - em) / (ep + em);
            }

            float_t df(float_t y) const override { return float_t(1) - sqr(y); }
        };
    }
}

#endif //LU_NET_ACTIVATION_FUNCTION_H
