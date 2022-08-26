//
// Created by malav on 7/2/2022.
//

#ifndef ANN_SGD_HXX
#define ANN_SGD_HXX

#include "../lin_alg/data_types.hxx"


// stochastic gradient descent
class SGD {

    public:

    // constructor
    SGD() = default;

    // constructor
    explicit SGD(double learn_rate)
    : alpha(learn_rate)
    {}

    // weights = weights - (alpha/normalizer) * gradient
    void Forward(Mat<double>& weights, Mat<double>& gradient, size_t normalizer) { weights += gradient * (-alpha/normalizer);}

    // biases = biases - alpha * gradient
    void Forward(Vector<double>& biases, Vector<double>& gradient, size_t normalizer) {biases += gradient * (-alpha/normalizer);}

    // reset the optimizer for another pass through the network
    void reset() {/*nothing to do */}

    private:

    // learning rate
    double alpha {0.1};

};
#endif //ANN_SGD_HXX
