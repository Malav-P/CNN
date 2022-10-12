//
// Created by malav on 7/2/2022.
//

#ifndef ANN_SGD_HXX
#define ANN_SGD_HXX

// stochastic gradient descent
class SGD {

    public:

    // constructor
    SGD() = default;

    // constructor
    explicit SGD(double learn_rate)
    : alpha(learn_rate)
    {}

    //! CPU Version, T is Mat<double>, Vector<double>, etc
    template<typename T>
    void Forward(T& weights, T& gradient, size_t normalizer) {weights += gradient * (-alpha/normalizer);}

    //! GPU Version, T is double, int, size_t, etc
    template<typename T>
    void Forward(T* d_weights, T* d_gradient, size_t normalizer, size_t N)
    {
        double multiplier = -alpha/normalizer;
        plus_equals_Kernel<<<1, N>>>(N, d_weights, d_gradient, multiplier);
    }

    // reset the optimizer for another pass through the network
    void reset() {/*nothing to do */}

    private:

    // learning rate
    double alpha {0.1};

};
#endif //ANN_SGD_HXX
