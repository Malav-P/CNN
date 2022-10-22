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


    // Allocate device struct array
    cudaMalloc( (void**)&d_poolvec, _in_maps*sizeof(MaxPool));

    // allocate device winners
    d_winners = (size_t**)malloc(sizeof(size_t*)*_in_maps);

    // copy over data from pool_vector to d_poolvec
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


}

__global__
void Child_Kernel(double *d_in, double *d_out, MaxPool* d_pool)
{
    size_t v_stride = threadIdx.y;
    size_t h_stride = threadIdx.x;

    size_t _v_str = d_pool->_v_str;
    size_t _h_str = d_pool->_h_str;
    Dims3 _in     = d_pool->_in;
    Dims3 _out    = d_pool->_out;
    Dims  _field  = d_pool->_field;
    size_t * winners = d_pool->_winners;

    // initialize array for temporary value storage
    Pair* buffer = new Pair[_field.width*_field.height];

    // calculate offset in data stored in vector
    size_t offset = _v_str * v_stride * (_in.width) + _h_str * h_stride;

    // store values for this window into the buffer
    for (size_t j = 0; j < _field.height; j++)
    {
        for (size_t i = 0; i < _field.width; i++)
        {
            buffer[j * _field.width + i] = {offset + j * _in.width + i, d_in[offset + j * _in.width + i]};
        }
    }

    // calculate max value and index of max value for this window
    Pair winner = d_pool->max_value(buffer, _field.width * _field.height);

    // store max value into output
    d_out[v_stride*_out.width + h_stride] = winner.height;

    // store max value's index
    winners[v_stride * _out.width + h_stride] = winner.width;

    // delete allocated memory
    delete[] buffer;
}

__global__
void Parent_Kernel(double *d_in, double *d_out, MaxPool* d_pool, size_t inmaps)
{

    Dims3 _in = d_pool[0].in_shape();
    Dims3 _out = d_pool[0].out_shape();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double* in = d_in + idx *_in.height * _in.width;
    double* out = d_out + idx * _out.height * _out.width;

    //TODO : POSSIBLE CUDA INVALID LAUNCH CONFIGURATION FOR LARGER SIZES OF ARRAYS, FIX THIS!
    dim3 numBlocks(1,1);
    dim3 threadsPerBlock(_out.width, _out.height);

    if (idx < inmaps)
    {
        Child_Kernel<<<numBlocks, threadsPerBlock>>>(in, out,  &(d_pool[idx]));
    }


}

// this function is supposed to be called on the CPU
void MaxPooling::Forward(Vector<double> &input, Vector<double> &output)
{

    // for profiling, can be removed
    cudaProfilerStart();


    // copy output to device
    double* d_output = output.port_to_GPU();

    // copy input to device
    double* d_input = input.port_to_GPU();


    // launch pool_vector.size number of kernels, each with a single thread
    Parent_Kernel<<<_in_maps, 1 >>>(d_input, d_output, d_poolvec, _in_maps);
//    Parent_Kernel<<<_in_maps, 1 >>>(input.get_data(), output.get_data(), pool_vector, _in_maps);

    // retrieve data from device and put it into return variable
    cudaMemcpy(output.get_data(), d_output, _out.height*_out.width*_out.depth*sizeof(double), cudaMemcpyDeviceToHost);

    // the pool vector is not copied back to the host, unless it is needed! In that case, call get_pool_vector() member function
    // and the data will be copied to the host

    // free device memory

    cudaFree(d_input);
    cudaFree(d_output);

    // for profiling, can be removed
    cudaProfilerStop();
}

__global__
void Backward_Child_Kernel(double *d_in, double *d_out, MaxPool* d_pool_obj, size_t N)
{


    int index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t* winners = d_pool_obj->_winners;

    if (index < N)
    {
        d_out[winners[index]] = d_in[index];
    }


}

__global__
void Backward_Parent_Kernel(double *d_dLdY, double *d_dLdX, MaxPool* d_pool, size_t inmaps)
{

    Dims3 _in = d_pool[0].in_shape();
    Dims3 _out = d_pool[0].out_shape();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double* in = d_dLdY + idx *_out.height * _out.width;
    double* out = d_dLdX + idx * _in.height * _in.width;


    // block size
    size_t block_size = 512;
    // number of threads needed
    size_t N = _out.width*_out.height;
    // number of threads per block
    dim3 threadsPerBlock(block_size);
    // number of blocks
    dim3 numBlocks((N+block_size - 1)/block_size);

    if (idx < inmaps)
    {
        Backward_Child_Kernel<<<numBlocks, threadsPerBlock >>>(in, out, &(d_pool[idx]), N);
    }


}

void MaxPooling::Backward(Vector<double> &dLdY, Vector<double> &dLdX)
{

    // for profiling, can be removed
    cudaProfilerStart();

    // copy output to device
    double* d_dLdX = dLdX.port_to_GPU();

    // copy input to device
    double* d_dLdY = dLdY.port_to_GPU();

    Backward_Parent_Kernel<<<1, _in_maps >>>(d_dLdY, d_dLdX, d_poolvec, _in_maps);

    // retrieve data from device and put it into return variable
    cudaMemcpy(dLdX.get_data(), d_dLdX, _in.height*_in.width*_in.depth*sizeof(double), cudaMemcpyDeviceToHost);

    // free device memory

    cudaFree(d_dLdY);
    cudaFree(d_dLdX);

    // for profiling, can be removed
    cudaProfilerStop();

    //! CPU version
//    Vector<double> OUTPUT;
//    for (size_t i = 0; i < _in_maps ; i++)
//    {
//
//        Vector<double> out(_in.height*_in.width);
//        Vector<double> in(_out.height * _out.width, dLdY.get_data() + i *_out.height * _out.width);
//        pool_vector[i].Backward(in, out);
//
//        OUTPUT = OUTPUT.merge(out);
//    }
//    dLdX = OUTPUT;
}



#endif //CNN_MAX_POOLING_IMPL_HXX
