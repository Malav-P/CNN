//
// Created by malav on 9/25/2022.
//

#ifndef CNN_TANH_IMPL_HXX
#define CNN_TANH_IMPL_HXX

#include "tanh.hxx"

namespace CNN {

    Tanh::Tanh(size_t input_width, size_t input_height, size_t input_depth) :
            Layer(input_width, input_height, input_depth, input_width, input_height, input_depth) {}

    void Tanh::Forward(Array<double> &input, Array<double> &output) {
        assert(input.getsize() == output.getsize());

        // might need copy constructor here TODO
        _local_input = input;
        for (size_t i = 0; i < input.getsize(); i++) { output[{0,i}] = func(input[{0,i}]); }
    }

    void Tanh::Backward(Array<double> &dLdY, Array<double> &dLdX) {
        assert(dLdY.getsize() == dLdX.getsize());
        for (size_t i = 0; i < dLdY.getsize(); i++) { dLdX[{0,i}] = dLdY[{0,i}] * deriv(_local_input[{0,i}]); }
    }

    double Tanh::func(double input) {
        return (2 / (1 + exp(-2 * input))) - 1;
    }

    double Tanh::deriv(double input) {
        return 1 - func(input) * func(input);
    }

}

#endif //CNN_TANH_IMPL_HXX
