//
// Created by malav on 5/4/2022.
//

#ifndef ANN_LINEAR_CPP
#define ANN_LINEAR_CPP

#include "linear.hxx"


Linear::Linear(size_t in_size, size_t out_size)
: _in(1, in_size,1)
, _out(1, out_size,1)
, _local_input(in_size)
, _local_output(out_size)
, _weights(out_size, in_size)
, _biases(out_size)
, _dLdW(out_size, in_size)
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

    // Allocate device struct memory
    cudaMalloc( (void**)&d_weights, sizeof(Mat<double>));

    // Allocate device pointer for weight data
    cudaMalloc((void**)&(d_weight_data), out_size*in_size*sizeof(double));

    // copy struct from host to device
    cudaMemcpy(d_weights, &_weights, sizeof(Mat<double>), cudaMemcpyHostToDevice);

    // copy pointer content from host to device
    cudaMemcpy(d_weight_data, _weights._data, out_size*in_size*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(&(d_weights->_data), &(d_weight_data), sizeof(double*), cudaMemcpyHostToDevice);

}

__global__
void Linear_Parent_Kernel(Mat<double>* d_matrix, double* d_in, double* d_out)
{
    size_t N_ROWS = d_matrix->get_rows();
    size_t N_COLS = d_matrix->get_cols();

    int i = threadIdx.x;

    if (i < N_ROWS)
    {
        for (size_t j = 0 ; j < N_COLS ; j++)
        {
            d_out[i] += (*d_matrix)(i,j) * d_in[j];
        }
    }
}

void Linear::Forward(const Vector<double> &input, Vector<double> &output)
{
    // for profiling, can be removed
    cudaProfilerStart();

    //! TODO: ensure that matrix multiplication is compatible with sizes, _weights*input assertion is already done in the operator overload
    assert(output.get_len() == _weights.get_rows());

    // copy input to local variable
    _local_input = input;

    // perform Y = Wx + B

    // copy output to device
    double* d_output;
    cudaMalloc(&d_output, _out.height*_out.width*_out.depth*sizeof(double));
    cudaMemcpy( d_output, output.get_data(), _out.height*_out.width*_out.depth*sizeof(double), cudaMemcpyHostToDevice);

    // copy input to device
    double* d_input;
    cudaMalloc(&d_input, _in.width*_in.height*_in.depth*sizeof(double));
    cudaMemcpy( d_input, input.get_data(), _in.width*_in.height*_in.depth*sizeof(double), cudaMemcpyHostToDevice);

    // do matmul on GPU
    Linear_Parent_Kernel<<<1, _weights.get_rows()>>>(d_weights, d_input, d_output);

    // retrieve data from device and put it into return variable
    cudaMemcpy(output.get_data(), d_output, _out.height*_out.width*_out.depth*sizeof(double), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_output);
    cudaFree(d_input);

    // for profiling, can be removed
    cudaProfilerStop();

    // assign _local_output to output of layer
    _local_output = output;

    //! CPU version
//    _local_output = (_weights * input) + _biases;


//    output = _local_output;
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
