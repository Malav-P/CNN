//
// Created by malav on 9/25/2022.
//

#ifndef CNN_TANH_IMPL_HXX
#define CNN_TANH_IMPL_HXX

#include "tanh.hxx"

namespace CNN {

    Tanh::Tanh(size_t input_width, size_t input_height, size_t input_depth) :
            Layer(input_width, input_height, input_depth, input_width, input_height, input_depth) {}

    void Tanh::Forward(Vector<double> &input, Vector<double> &output) {
        assert(input.get_len() == output.get_len());

        _local_input = input;
        for (size_t i = 0; i < input.get_len(); i++) { output[i] = func(input[i]); }
    }

    void Tanh::Backward(Vector<double> &dLdY, Vector<double> &dLdX) {
        assert(dLdY.get_len() == dLdX.get_len());
        for (size_t i = 0; i < dLdY.get_len(); i++) { dLdX[i] = dLdY[i] * deriv(_local_input[i]); }
    }

    double Tanh::func(double input) {
        return (2 / (1 + exp(-2 * input))) - 1;
    }

    double Tanh::deriv(double input) {
        return 1 - func(input) * func(input);
    }

}

#endif //CNN_TANH_IMPL_HXX
