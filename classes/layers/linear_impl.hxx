//
// Created by malav on 5/4/2022.
//

#ifndef ANN_LINEAR_CPP
#define ANN_LINEAR_CPP

#include "linear.hxx"
#include "../lin_alg/lin_alg_kernels.hxx"


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

    //! for weight matrix on GPU

    // Allocate device struct memory
    cudaMalloc( (void**)&d_weights, sizeof(Mat<double>));

    // Allocate device pointer for weight data
    cudaMalloc((void**)&(d_weight_data), out_size*in_size*sizeof(double));

    // copy struct from host to device
    cudaMemcpy(d_weights, &_weights, sizeof(Mat<double>), cudaMemcpyHostToDevice);

    // copy pointer content from host to device
    cudaMemcpy(d_weight_data, _weights._data, out_size*in_size*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(&(d_weights->_data), &(d_weight_data), sizeof(double*), cudaMemcpyHostToDevice);


    //! for bias vector on GPU

    // Allocate device struct memory
    cudaMalloc( (void**)&d_biases, sizeof(Vector<double>));

    // Allocate device pointer for bias data
    cudaMalloc((void**)&(d_biases_data), out_size * sizeof(double ));

    // copy struct from host to device
    cudaMemcpy(d_biases, &_biases, sizeof(Vector<double>), cudaMemcpyHostToDevice);

    // copy pointer content from host to device
    cudaMemcpy(d_biases_data, _biases.get_data(), out_size * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(&(d_biases->get_data()), &(d_biases_data), sizeof(double*), cudaMemcpyHostToDevice);

    //! for gradient matrix on GPU

    // Allocate device struct memory
    cudaMalloc((void**)&d_dLdW, sizeof(Mat<double>));

    // Allocate device pointer for weight data
    cudaMalloc((void**)&(d_dLdW_data), out_size * in_size * sizeof(double));

    // copy struct from host to device
    cudaMemcpy(d_dLdW, &_dLdW, sizeof(Mat<double>), cudaMemcpyHostToDevice);

    // copy pointer content from host to device
    cudaMemcpy(d_dLdW_data, _dLdW._data, out_size * in_size * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(&(d_dLdW->_data), &(d_dLdW_data), sizeof(double*), cudaMemcpyHostToDevice);

    //! for bias vector on GPU

    // Allocate device struct memory
    cudaMalloc((void**)&d_dLdB, sizeof(Vector<double>));

    // Allocate device pointer for bias data
    cudaMalloc((void**)&(d_dLdB_data), out_size * sizeof(double ));

    // copy struct from host to device
    cudaMemcpy(d_dLdB, &_dLdB, sizeof(Vector<double>), cudaMemcpyHostToDevice);

    // copy pointer content from host to device
    cudaMemcpy(d_dLdB_data, _dLdB.get_data(), out_size * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(&(d_dLdB->get_data()), &(d_dLdB_data), sizeof(double*), cudaMemcpyHostToDevice);
}



void Linear::Forward(Vector<double> &input, Vector<double> &output)
{
    // for profiling, can be removed
    cudaProfilerStart();

    //! TODO: ensure that matrix multiplication is compatible with sizes, _weights*input assertion is already done in the operator overload
    assert(output.get_len() == _weights.get_rows());

    // copy input to local variable
    _local_input = input;

    // perform Y = Wx + B

    // copy output to device
    double* d_output = output.port_to_GPU();

    // copy input to device
    double* d_input = input.port_to_GPU();

    // do matmul on GPU
    matVec_Kernel<<<1, _weights.get_rows()>>>(d_weights, d_input, d_output);

    // retrieve data from device and put it into return variable
    cudaMemcpy(output.get_data(), d_output, _out.height*_out.width*_out.depth*sizeof(double), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_output);
    cudaFree(d_input);

    // for profiling, can be removed
    cudaProfilerStop();

    // assign _local_output to output of layer
    _local_output = output;

//    //! CPU version
//    _local_output = (_weights * input) + _biases;
//
//
//    output = _local_output;
}



void Linear::Backward(Vector<double> &dLdY, Vector<double> &dLdX)
{

    // for profiling, can be removed
    cudaProfilerStart();
    //! compute dLdX, this vector will be sent to be backpropagated through the previous layer

    // copy input to GPU
    double* d_dLdY = dLdY.port_to_GPU();

    // copy output to GPU
    double* d_dLdX = dLdX.port_to_GPU();

    // do matmul on GPU
    vecMat_Kernel<<<1, _weights.get_cols()>>>(d_dLdY, d_weights, d_dLdX);

    // retrieve data from device and put it into return variable
    cudaMemcpy(dLdX.get_data(), d_dLdX, dLdX.get_len()*sizeof(double), cudaMemcpyDeviceToHost);



    //! CPU Version
//    dLdX = dLdY*(_weights);



    //! compute gradients GPU Version
    double* d_local_input = _local_input.port_to_GPU();
    dim3 numBlocks(1,1);
    dim3 threadsPerBlock(_dLdW.get_cols(), _dLdW.get_rows());
    vecVecplusequals_Kernel<<<numBlocks, threadsPerBlock>>>(d_dLdW, d_dLdY, d_local_input);

    plus_equals_Kernel<<<1, dLdY.get_len()>>>(dLdY.get_len(), d_dLdB_data, d_dLdY, 1);

    // free device memory
    cudaFree(d_dLdY);
    cudaFree(d_dLdX);
    cudaFree(d_local_input);

    // for profiling, can be removed
    cudaProfilerStop();

    //! compute gradients CPU version
//    _dLdW += dLdY * _local_input;
//    _dLdB += dLdY;

}

template<typename Optimizer>
void Linear::Update_Params(Optimizer* optimizer, size_t normalizer)
{
      //! CPU VERSION
//    // update the biases and reset dLdB to zeros. MUST UPDATE BIASES FIRST or else member variable k of momentum optmizer
//    // is prematurely updated
//    (*optimizer).Forward(_biases, _dLdB, normalizer);
//    _dLdB.fill(0);
//
//    // update the weights and reset dLdW to zeros
//    (*optimizer).Forward(_weights, _dLdW, normalizer);
//    _dLdW.fill(0);


    //! GPU VERSION

    // for profiling, can be removed
    cudaProfilerStart();

    // update the biases and reset dLdB to zeros. MUST UPDATE BIASES FIRST or else member variable k of momentum optmizer
    // is prematurely updated
    (*optimizer).Forward(d_biases_data, d_dLdB_data, normalizer, _dLdB.get_len());
    fill_Kernel<<<1, _dLdB.get_len()>>>(_dLdB.get_len(), d_dLdB_data, 0);

    // update the weights and reset dLdW to zeros
    (*optimizer).Forward(d_weight_data, d_dLdW_data, normalizer, _dLdW.get_cols()*_dLdW.get_rows());
    fill_Kernel<<<1, _dLdW.get_cols()*_dLdW.get_rows()>>>(_dLdW.get_cols()*_dLdW.get_rows(), d_dLdW_data, 0);

    // for profiling, can be removed
    cudaProfilerStop();
}

#endif //ANN_LINEAR_CPP
