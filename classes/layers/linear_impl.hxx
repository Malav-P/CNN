//
// Created by malav on 5/4/2022.
//

#ifndef ANN_LINEAR_CPP
#define ANN_LINEAR_CPP

#include "linear.hxx"
#include <random>
#include <ctime>

template<typename activ_func>
Linear<activ_func>::Linear(size_t in_size, size_t out_size)
: f()
, _in(1, in_size)
, _out(1, out_size)
, _local_input(in_size)
, _local_output(out_size)
, _weights(out_size, in_size)
, _dLdW(out_size, in_size)
, _biases(out_size)
, _dLdB(out_size)
{

    // create a normal distribution for He initialization
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, 2.0/_in.height);

    // He initialize the weights
    for (size_t i=0; i<_weights.get_rows(); i++) { for (size_t j=0; j<_weights.get_cols(); j++)
        { _weights(i,j) = distribution(generator); }
    }
}

template<typename activ_func>
Linear<activ_func>::Linear(size_t in_size, size_t out_size, double leaky_param)
: f(leaky_param)
, _in(1, in_size)
, _out(1, out_size)
, _local_input(in_size)
, _local_output(out_size)
, _weights(out_size, in_size)
, _dLdW(out_size, in_size)
, _biases(out_size)
, _dLdB(out_size)
{
    // create a normal distribution for He initialization
    std::default_random_engine generator(1);
    std::normal_distribution<double> distribution(0, 2.0/_in.height);

    // He initialize the weights
    for (size_t i=0; i<_weights.get_rows(); i++) { for (size_t j=0; j<_weights.get_cols(); j++)
        { _weights(i,j) = distribution(generator); }
    }
}

template<typename activ_func>
void Linear<activ_func>::Forward(const Vector<double> &input, Vector<double> &output)
{
    //! TODO: ensure that matrix multiplication is compatible with sizes, _weights*input assertion is already done in the operator overload
    assert(output.get_len() == _weights.get_rows());

    // copy input to local variable
    _local_input = input;

    // perform Y = Wx + B
    _local_output = (_weights * input) + _biases;

    // perform f(Y)
    f.func(_local_output, output);
}

template<typename activ_func>
void Linear<activ_func>::Backward(const Vector<double> &dLdZ, Vector<double> &dLdX)
{
    // compute dZdY
    Vector<double> dZdY(_local_output.get_len());
    f.deriv(_local_output, dZdY);


    // compute dLdY
    Vector<double> dLdY = dLdZ.eprod(dZdY);


    // compute dLdX, this vector will be sent to be backpropagated through the previous layer
    dLdX = dLdY*(_weights);


    // compute gradients
    _dLdW += dLdY * _local_input;
    _dLdB += dLdY;
}

template<typename activ_func>
template<typename Optimizer>
void Linear<activ_func>::Update_Params(Optimizer* optimizer, size_t normalizer)
{

    // update the biases and reset dLdB to zeros. MUST UPDATE BIASES FIRST or else member variable k of momentum optmizer
    // is prematurely updated
    (*optimizer).Forward(_biases, _dLdB, normalizer);
    _dLdB.fill(0);

    // update the weights and reset dLdW to zeros
    (*optimizer).Forward(_weights, _dLdW, normalizer);
    _dLdW.fill(0);

}




#endif //ANN_LINEAR_CPP
