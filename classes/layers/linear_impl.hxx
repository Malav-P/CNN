//
// Created by malav on 5/4/2022.
//

#ifndef ANN_LINEAR_CPP
#define ANN_LINEAR_CPP

#include "linear.hxx"


Linear::Linear(size_t in_size, size_t out_size)
: Layer(1, in_size, 1, 1, out_size, 1)
, _local_input(in_size)
, _weights(out_size, in_size)
, _dLdW(out_size, in_size)
, _biases(out_size)
, _dLdB(out_size)
{

    // get the current time to seed the random number generator
    typedef std::chrono::high_resolution_clock myclock;
    myclock::time_point beginning = myclock::now();
    myclock::duration d = myclock::now() - beginning;
    unsigned seed2 = d.count();

    // seed the random number generator
    std::default_random_engine generator(seed2);
    std::normal_distribution<double> distribution(0, sqrt(2.0/_in.height));

    // He initialize the weights
    for (size_t i=0; i<_weights.get_rows(); i++) { for (size_t j=0; j<_weights.get_cols(); j++)
        { _weights(i,j) = distribution(generator); }
    }
}


void Linear::Forward(Vector<double> &input, Vector<double> &output)
{
    //! TODO: ensure that matrix multiplication is compatible with sizes, _weights*input assertion is already done in the operator overload
    assert(output.get_len() == _weights.get_rows());

    // copy input to local variable
    _local_input = input;

    // perform Y = Wx + B
    output = (_weights * input) + _biases;
}

void Linear::Backward(Vector<double> &dLdY, Vector<double> &dLdX)
{

    // compute dLdX, this vector will be sent to be backpropagated through the previous layer
    dLdX = dLdY*(_weights);


    // compute gradients
    _dLdW += dLdY * _local_input;
    _dLdB += dLdY;
}

template<typename Optimizer>
void Linear::Update_Params(Optimizer* optimizer, size_t normalizer)
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
