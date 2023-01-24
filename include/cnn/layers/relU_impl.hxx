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

    void RelU::Forward(Array<double> &input, Array<double> &output) {
        assert(input.getsize() == output.getsize());

        // might need copy constructor here TODO
        _local_input = input;

        double* indata = input.getdata();
        double* outdata = output.getdata();

        for (size_t i = 0; i < input.getsize(); i++) { *(outdata++) = *(indata) < 0 ? alpha * (*(indata)) : *indata; indata++; }
    }

    void RelU::Backward(Array<double> &dLdY, Array<double> &dLdX) {
        assert(dLdY.getsize() == dLdX.getsize());


        double* dXdata = dLdX.getdata();
        double* dYdata = dLdY.getdata();
        double* localinputdata = _local_input.getdata();

        for (size_t i = 0; i < dLdY.getsize(); i++) { *(dXdata++) = *(dYdata++) * (*(localinputdata++) < 0 ? alpha : 1); }

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
