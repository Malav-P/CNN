//
// Created by Malav Patel on 9/29/22.
//

#ifndef CNN_MAX_POOLING_IMPL_HXX
#define CNN_MAX_POOLING_IMPL_HXX

#include "max_pooling.hxx"

MaxPooling::MaxPooling(size_t in_maps, size_t in_width, size_t in_height, size_t fld_width, size_t fld_height,
                       size_t h_stride, size_t v_stride):
                       _in(in_width, in_height, in_maps),
                       _field(fld_width, fld_height),
                       _h_str(h_stride),
                       _v_str(v_stride),
                       pool_vector(new MaxPool[in_maps]),
                       _in_maps(in_maps)
{
    for (size_t i = 0; i < in_maps; i++)
    {
        pool_vector[i] = MaxPool(in_width, in_height, fld_width, fld_height, h_stride, v_stride);
    }

    _out = pool_vector[0].out_shape();
    _out.depth = in_maps;
}

__global__
void Parent_Kernel(double *d_in, double *d_out, MaxPool* d_pool, size_t inmaps)
{

    Dims3 _in = d_pool[0].in_shape();
    Dims3 _out = d_pool[0].out_shape();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    Vector<double> in_vec(_in.height*_in.width, d_in + idx *_in.height * _in.width);
    Vector<double> out_vec(_out.height*_out.width);

    if (idx < inmaps)
    {
        d_pool[idx].Forward(in_vec, out_vec);
    }


     //this can be replaced with a memcpy (it is more efficient)
    for (size_t i = 0; i < out_vec.get_len(); i++)
    {
        d_out[idx * _out.height*_out.width + i ] = out_vec[i];
    }



}

// this function is supposed to be called on the CPU
void MaxPooling::Forward(Vector<double> &input, Vector<double> &output)
{

    // copy pool_vector to device
    MaxPool* d_poolvec;
    size_t ** d_winners = (size_t**)malloc(sizeof(size_t*)*_in_maps);
    // Allocate device struct array
    cudaMalloc( (void**)&d_poolvec, _in_maps*sizeof(MaxPool));
    for (size_t i = 0; i< _in_maps; i++)
    {
        // host struct
        MaxPool* elem = &(pool_vector[i]);

        // device struct
        MaxPool* d_elem = &(d_poolvec[i]);

        // copy struct from host to device
        cudaMemcpy(d_elem, elem, sizeof(MaxPool), cudaMemcpyHostToDevice);

        // device array


        // length of device array
        int d_winners_len = _out.width*_out.height;

        // Allocate device pointer
        cudaMalloc((void**)&(d_winners[i]), d_winners_len*sizeof(size_t));

        // copy pointer content from host to device
        cudaMemcpy((d_winners[i]), elem->_winners, d_winners_len*sizeof(size_t), cudaMemcpyHostToDevice);


        cudaMemcpy(&(d_elem->_winners), &(d_winners[i]), sizeof(size_t*), cudaMemcpyHostToDevice);
    }



    // copy output to device
    double* d_output;
    cudaMalloc(&d_output, _out.height*_out.width*_out.depth*sizeof(double));
    cudaMemcpy( d_output, output.get_data(), _out.height*_out.width*_out.depth*sizeof(double), cudaMemcpyHostToDevice);

    // copy input to device
    double* d_input;
    cudaMalloc(&d_input, _in.width*_in.height*_in.depth*sizeof(double));
    cudaMemcpy( d_input, input.get_data(), _in.width*_in.height*_in.depth*sizeof(double), cudaMemcpyHostToDevice);



    // launch pool_vector.size number of kernels, each with a single thread
    Parent_Kernel<<<_in_maps, 1 >>>(d_input, d_output, d_poolvec, _in_maps);
    cudaDeviceSynchronize();

    // retrieve data from device and put it into return variable
    cudaMemcpy(output.get_data(), d_output, _out.height*_out.width*_out.depth*sizeof(double), cudaMemcpyDeviceToHost);

    for(size_t i = 0 ; i < _in_maps ; i++)
    {
        cudaMemcpy(pool_vector[i]._winners, d_winners[i], _out.height*_out.width*sizeof(size_t), cudaMemcpyDeviceToHost);
        //free device memory
        cudaFree(d_winners[i]);
    }

    // free memory
    free(d_winners);
    // free device memory
    cudaFree(d_poolvec);
    cudaFree(d_input);
    cudaFree(d_output);
}



void MaxPooling::Backward(Vector<double> &dLdY, Vector<double> &dLdX)
{
    Vector<double> OUTPUT;
    for (size_t i = 0; i < _in_maps ; i++)
    {

        Vector<double> out(_in.height*_in.width);
        Vector<double> in(_out.height * _out.width, dLdY.get_data() + i *_out.height * _out.width);
        pool_vector[i].Backward(in, out);

        OUTPUT = OUTPUT.merge(out);
    }
    dLdX = OUTPUT;
}



#endif //CNN_MAX_POOLING_IMPL_HXX
