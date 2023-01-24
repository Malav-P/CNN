//
// Created by malav on 7/2/2022.
//

#ifndef ANN_SGD_HXX
#define ANN_SGD_HXX

namespace CNN {

// stochastic gradient descent
    class SGD {

    public:

        // constructor
        SGD() = default;

        // constructor
        explicit SGD(double learn_rate)
                : alpha(learn_rate) {}

        // this is daxpy
        void Forward(Array<double> &weights, Array<double> &gradient, size_t normalizer)
        {
            // assert two arrays have equal dimensions TODO

            // do daxpy
            int n = weights.getsize();
            double scalar = -alpha/normalizer;
            cblas_daxpy(n, scalar, gradient.getdata(), 1, weights.getdata(), 1);
        }

        // this is saxpy
        void Forward(Array<float> &weights, Array<float> &gradient, size_t normalizer)
        {
            // assert two arrays have equal dimensions TODO

            // do daxpy
            int n = weights.getsize();
            float scalar = -alpha/normalizer;
            cblas_saxpy(n, scalar, gradient.getdata(), 1, weights.getdata(), 1);
        }

        // reset the optimizer for another pass through the network
        void reset() {/*nothing to do */}

    private:

        // learning rate
        double alpha{0.1};

    };

}
#endif //ANN_SGD_HXX
