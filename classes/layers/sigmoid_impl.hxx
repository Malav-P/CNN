//
// Created by malav on 9/25/2022.
//

#ifndef CNN_SIGMOID_IMPL_HXX
#define CNN_SIGMOID_IMPL_HXX

#include "sigmoid.hxx"

Sigmoid::Sigmoid(size_t input_width, size_t input_height):
        _in(input_width, input_height,1),
        _out(input_width, input_height,1)
{}

void Sigmoid::Forward(const Vector<double> &input, Vector<double> &output)
{
    assert(input.get_len() == output.get_len());

    _local_input = input;
    for (size_t i = 0; i<input.get_len() ; i++) {output[i] = func(input[i]);}
}

void Sigmoid::Backward(Vector<double> &dLdY, Vector<double> &dLdX)
{
    assert(dLdY.get_len() == dLdX.get_len());
    for (size_t i = 0; i<dLdY.get_len() ; i++)
    {dLdX[i] = dLdY[i] * deriv(_local_input[i]);}
}

double Sigmoid::func(double input)
{
    return 1 / (1 + exp(-input));
}

double Sigmoid::deriv(double input)
{
    return func(input) * (1 - func(input));
}



#endif //CNN_SIGMOID_IMPL_HXX
