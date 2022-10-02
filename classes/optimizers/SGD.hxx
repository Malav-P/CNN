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

    template<typename T>
    void Forward(T& weights, T& gradient, size_t normalizer) {weights += gradient * (-alpha/normalizer);}

    // reset the optimizer for another pass through the network
    void reset() {/*nothing to do */}

    private:

    // learning rate
    double alpha {0.1};

};
#endif //ANN_SGD_HXX
