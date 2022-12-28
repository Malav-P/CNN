//
// Created by malav on 9/25/2022.
//

#ifndef CNN_RELU_IMPL_HXX
#define CNN_RELU_IMPL_HXX


#include "relU.hxx"

namespace CNN {

    RelU::RelU(size_t input_width, size_t input_height, size_t input_depth, double Alpha)
            : alpha(Alpha),
              Layer(input_width, input_height, input_depth, input_width, input_height, input_depth) {
        if (Alpha < 0) { alpha = 0; } //! alpha cannot be negative
    }

    void RelU::Forward(Vector<double> &input, Vector<double> &output) {
        assert(input.get_len() == output.get_len());

        _local_input = input;
        for (size_t i = 0; i < input.get_len(); i++) { output[i] = func(input[i]); }
    }

    void RelU::Backward(Vector<double> &dLdY, Vector<double> &dLdX) {
        assert(dLdY.get_len() == dLdX.get_len());
        for (size_t i = 0; i < dLdY.get_len(); i++) { dLdX[i] = dLdY[i] * deriv(_local_input[i]); }

    }

    double RelU::func(double input) {
        if (input < 0) { return alpha * input; }
        else { return input; }
    }

    double RelU::deriv(double input) {
        if (input < 0) { return alpha; }
        else { return 1; }
    }


}
#endif //CNN_RELU_IMPL_HXX
