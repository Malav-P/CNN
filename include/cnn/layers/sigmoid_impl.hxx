//
// Created by malav on 9/25/2022.
//

#ifndef CNN_SIGMOID_IMPL_HXX
#define CNN_SIGMOID_IMPL_HXX

#include "sigmoid.hxx"

namespace CNN {

    Sigmoid::Sigmoid(size_t input_width, size_t input_height, size_t input_depth)
            : Layer(input_width, input_height, input_depth, input_width, input_height, input_depth) {}

    void Sigmoid::Forward(Array<double> &input, Array<double> &output) {
        assert(input.getsize() == output.getsize());

        // might need copy constructor here TODO
        _local_input = input;
        for (size_t i = 0; i < input.getsize(); i++) { output[{0,i}] = func(input[{0,i}]); }
    }

    void Sigmoid::Backward(Array<double> &dLdY, Array<double> &dLdX) {
        assert(dLdY.getsize() == dLdX.getsize());
        for (size_t i = 0; i < dLdY.getsize(); i++) { dLdX[{0,i}] = dLdY[{0,i}] * deriv(_local_input[{0,i}]); }
    }

    double Sigmoid::func(double input) {
        return 1 / (1 + exp(-input));
    }

    double Sigmoid::deriv(double input) {
        return func(input) * (1 - func(input));
    }

}

#endif //CNN_SIGMOID_IMPL_HXX
