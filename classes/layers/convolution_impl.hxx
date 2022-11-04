//
// Created by malav on 6/17/2022.
//

#ifndef CNN_CONVOLUTION_IMPL_HXX
#define CNN_CONVOLUTION_IMPL_HXX

#include "convolution.hxx"
#include "../lin_alg/miscellaneous_helpers.hxx"
#include "../lin_alg/lin_alg_kernels.hxx"

Convolution::Convolution(size_t in_maps, size_t out_maps, size_t in_width, size_t in_height, size_t filter_width,
                         size_t filter_height, size_t stride_h, size_t stride_v, size_t padleft, size_t padright,
                         size_t padtop,
                         size_t padbottom)
: _h_str(stride_h),
  _v_str(stride_v),
  _in(in_width + padleft + padright, in_height + padtop + padbottom, in_maps),
  _padleft(padleft),
  _padright(padright),
  _padtop(padtop),
  _padbottom(padbottom),
  _filters(new Cuboid<double>[out_maps]),
  d_filters_data(new double*[out_maps]),
  d_dLdFs_data(new double*[out_maps]),
  _dLdFs(new Cuboid<double>[out_maps]),
  _local_input(new Mat<double>[in_maps]),
  d_local_input_data(new double*[in_maps])
{


    // calculate total number of vertical and horizontal strides
    size_t num_v_strides = std::floor((_in.height - filter_height) / _v_str) + 1;
    size_t num_h_strides = std::floor((_in.width - filter_width) / _h_str) + 1;

    // specify output shape
    _out = {num_h_strides, num_v_strides, out_maps};

    // get the current time to seed the random number generator
    typedef std::chrono::high_resolution_clock myclock;
    myclock::time_point beginning = myclock::now();
    myclock::duration d = myclock::now() - beginning;
    unsigned seed2 = d.count();

    // seed the random number generator
    std::default_random_engine generator(seed2);
    std::uniform_real_distribution<double> distribution(-sqrt(6.0/(_in.height*_in.width + _out.height*_out.width)), sqrt(6.0/(_in.height*_in.width + _out.height*_out.width)));

    // allocate memory for an initialize filters
    for (size_t i = 0; i< out_maps; i++)
    {
        _filters[i] = Cuboid<double>(filter_height, filter_width, in_maps);
        _dLdFs[i] = Cuboid<double>(filter_height, filter_width, in_maps);
    }

    // Glorot initialize the weights
    for (size_t m = 0 ; m < out_maps ; m++)
    {
        for (size_t i=0; i<filter_height; i++) { for (size_t j=0; j<filter_width; j++){ for (size_t k = 0; k < in_maps; k++){
                    _filters[m](i,j,k) = distribution(generator);
                }}}
    }

    // copy over data from filters from host to device

    // this function uses cudamalloc, there must be calls to cudaFree elsewhere (ideally in destructor)
    port_to_GPU(d_filters, _filters, d_filters_data, out_maps);

    //! --------------------------------------------------------
    // this function uses cudamalloc, there must be calls to cudaFree elsewhere (ideally in destructor)
    port_to_GPU(d_dLdFs, _dLdFs, d_dLdFs_data, out_maps);

    //!------------------

    for (size_t i = 0; i < _filters[0].get_depth(); i++)
    {
        _local_input[i] = Mat<double>(_in.height, _in.width);
    }
    port_to_GPU(d_local_input, _local_input, d_local_input_data, in_maps);

}

Convolution::Convolution(size_t in_maps, size_t out_maps, size_t in_width, size_t in_height, size_t filter_width,
                         size_t filter_height, size_t stride_h, size_t stride_v, bool padding)
                         :_h_str(stride_h),
                          _v_str(stride_v),
                          _filters(new Cuboid<double>[out_maps]),
                          d_filters_data(new double*[out_maps]),
                          d_dLdFs_data(new double*[out_maps]),
                          _dLdFs(new Cuboid<double>[out_maps]),
                          _local_input(new Mat<double>[in_maps]),
                          d_local_input_data(new double*[in_maps])
{
    if (!padding)
    {
        _padleft = 0;
        _padright = 0;
        _padtop = 0;
        _padbottom = 0;
        _in = {in_width + _padleft + _padright, in_height + _padtop + _padbottom, in_maps};

    }
    else
    {
        // total number of padded rows
        size_t vert_pad = (in_height - 1) * stride_v - in_height + filter_height;

        // number of padded rows at top of matrix
        _padtop = std::floor(vert_pad/2.0);

        // number of padded rows at bottom of matrix
        _padbottom = std::ceil(vert_pad/2.0);

        // total number of padded columns
        size_t hor_pad = (in_width - 1) * stride_h - in_width + filter_width;

        // number of padded columns to left of matrix
        _padleft = std::floor(hor_pad/2.0);

        // number of padded columns to right of matrix
        _padright = std::ceil(hor_pad/2.0);

        // rest of member variables
        _in = {in_width + _padleft + _padright, in_height + _padtop + _padbottom, in_maps};
    }




    // calculate total number of vertical and horizontal strides
    size_t num_v_strides = std::floor((_in.height - filter_height) / _v_str) + 1;
    size_t num_h_strides = std::floor((_in.width - filter_width) / _h_str) + 1;

    // specify output shape
    _out = {num_h_strides, num_v_strides, out_maps};

    // get the current time to seed the random number generator
    typedef std::chrono::high_resolution_clock myclock;
    myclock::time_point beginning = myclock::now();
    myclock::duration d = myclock::now() - beginning;
    unsigned seed2 = d.count();

    // seed the random number generator
    std::default_random_engine generator(seed2);
    std::uniform_real_distribution<double> distribution(-sqrt(6.0/(_in.height*_in.width + _out.height*_out.width)), sqrt(6.0/(_in.height*_in.width + _out.height*_out.width)));

    // initialize filters
    for (size_t i = 0; i< out_maps; i++)
    {
        _filters[i] = Cuboid<double>(filter_height, filter_width, in_maps);
        _dLdFs[i] = Cuboid<double>(filter_height, filter_width, in_maps);
    }

    // Glorot initialize the weights
    for (size_t m = 0; m < out_maps; m++)
    {
        for (size_t i=0; i<filter_height; i++) { for (size_t j=0; j<filter_width; j++) { for (size_t k = 0; k< in_maps; k++){
             _filters[m](i,j, k) = distribution(generator);
        }}}
    }

    // this function uses cudamalloc, there must be calls to cudaFree elsewhere (ideally in destructor)
    port_to_GPU(d_filters, _filters, d_filters_data, out_maps);

    //! --------------------------------------------------------

    // this function uses cudamalloc, there must be calls to cudaFree elsewhere (ideally in destructor)
    port_to_GPU(d_dLdFs, _dLdFs, d_dLdFs_data, out_maps);

    //!--------

    for (size_t i = 0; i < _filters[0].get_depth(); i++)
    {
        _local_input[i] = Mat<double>(_in.height, _in.width);
    }
    port_to_GPU(d_local_input, _local_input, d_local_input_data, in_maps);

}

__global__
void Conv_Parent_Kernel(double* d_out, Cuboid<double>* d_in, Dims3 _out, size_t _v_str, size_t _h_str, Cuboid<double>* d_filters)
{
    size_t k = blockIdx.z * blockDim.z + threadIdx.z;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < _out.height && j < _out.width && k < _out.depth)
    {d_out[k*_out.height*_out.width + i*_out.width + j] = d_in->partial_dot(d_filters[k], {i * _v_str, j * _h_str, 0});}
}

void Convolution::Forward(Vector<double> &input, Vector<double> &output)
{

    cudaProfilerStart();
    // note that input length matching with _in parameters is indirectly checked in the matrix*vector operator overload

    size_t rows = _in.height-_padbottom-_padtop;
    size_t cols = _in.width - _padleft - _padright;

    for (size_t i = 0; i < _filters[0].get_depth(); i++)
    {
        _local_input[i] = Mat<double>(rows, cols, input.get_data() + i*rows*cols);
        _local_input[i].padding(_padleft, _padright, _padtop, _padbottom);
    }

    copy_to_GPU(d_local_input, _local_input, d_local_input_data, _in.depth);

    // cubify input
    Cuboid<double> input_cube = cubify(_local_input, _in.depth);

    // device struct
    Cuboid<double>* d_input_cube;
    // device pointer
    double* d_arr;

    // deep copy struct to GPU
    input_cube.port_to_GPU(d_input_cube, d_arr);

    double* d_output = output.port_to_GPU();

    //!setup and call kernel

    // block dimensions
    size_t xsize = 16;
    size_t ysize = 16;
    size_t zsize = 4;
    assert(xsize*ysize*zsize <= 1024);

    // number of threads needed in each dimension
    size_t N_x = _out.width;
    size_t N_y = _out.height;
    size_t N_z = _out.depth;

    // number of threads per block
    dim3 threadsPerBlock(xsize,ysize,zsize);
    // number of blocks
    dim3 numBlocks((N_x+xsize - 1)/xsize, (N_y+ysize - 1)/ysize, (N_z+zsize - 1)/zsize);

    // call kernel
    Conv_Parent_Kernel<<<numBlocks, threadsPerBlock>>>(d_output, d_input_cube, _out, _v_str, _h_str, d_filters);

    // retrieve data from device and put it into return variable
    cudaMemcpy(output.get_data(), d_output, output.get_len()*sizeof(double), cudaMemcpyDeviceToHost);


    // free d_arr and d_output
    cudaFree(d_arr);
    cudaFree(d_output);
    cudaFree(d_input_cube);

    cudaProfilerStop();

}


__global__
void Kernel2(Cuboid<double>* A,  Mat<double>* B, Mat<double>* C, size_t idx)
{
    size_t N_COLS = A->get_cols();
    size_t N_ROWS = A->get_rows();
    size_t N_DEPTH = A->get_depth();

    size_t k = blockIdx.z * blockDim.z + threadIdx.z;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N_ROWS && j < N_COLS && k < N_DEPTH)
    {A[idx](i,j,k) += B[k].partial_dot(*C, {i,j});}

}


__global__
void Kernel3Child(size_t n_rows, size_t n_cols, size_t k, double* d_out, Mat<double>* filter_plane, size_t _padtop, size_t _padleft, Mat<double>* d_reformatted_output)
{
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_rows && j < n_cols)
    {
        d_out[k*n_rows*n_cols + (i)*n_cols + (j)] += d_reformatted_output->partial_dot(*filter_plane, {i+_padtop,j + _padleft});
    }
}

__global__
void Kernel3(size_t N_in_maps, size_t idx, size_t n_cols, size_t n_rows, Cuboid<double>* d_filters, double* d_out, size_t _padtop, size_t _padleft, Mat<double>* d_reformatted_output)
{

    size_t k = blockIdx.x * blockDim.x + threadIdx.x;

    size_t filter_height = d_filters[idx].get_rows();
    size_t filter_width = d_filters[idx].get_cols();

    if (k < N_in_maps)
    {
        auto filter_plane = new Mat<double>(filter_height, filter_width, d_filters[idx]._data + k * filter_width * filter_height);

        //rotate filter by 180 degrees
        filter_plane->set_rot(2);

        // number of threads needed
        size_t N_x = n_cols;
        size_t N_y = n_rows;

        // block dimensions (PRODUCT MUST BE 1024 or LESS)
        size_t xsize = 32;
        size_t ysize = 32;

        // number of threads per block
        dim3 threadsPerBlock(xsize,ysize);

        // number of blocks
        dim3 numBlocks((N_x+xsize - 1)/xsize, (N_y+ysize - 1)/ysize);

        Kernel3Child<<<numBlocks, threadsPerBlock>>>(n_rows, n_cols, k, d_out, filter_plane, _padtop, _padleft, d_reformatted_output);

        cudaDeviceSynchronize();

        delete filter_plane;
    }
}


void Convolution::Backward(Vector<double> &dLdYs, Vector<double> &dLdXs)
{
    cudaProfilerStart();

    size_t m = _out.height;
    size_t n = _out.width;

    size_t p = _in.height;
    size_t q = _in.width;

    size_t filter_height = _filters[0].get_rows();
    size_t filter_width = _filters[0].get_cols();

    size_t N_filters = _out.depth;
    size_t N_in_maps = _filters[0].get_depth();

    //!---- for kernel calls
    // block dimensions
    size_t xsize = 16;
    size_t ysize = 16;
    size_t zsize = 4;
    assert(xsize*ysize*zsize <= 1024);
    // number of threads per block
    dim3 threadsPerBlock(xsize,ysize,zsize);

    double* d_out = dLdXs.port_to_GPU();

    for (size_t idx = 0; idx < N_filters ; idx++)
    {

        double* dLdY = dLdYs.get_data() + idx*m*n;

        // reformatted output
        Mat<double> reformatted_output(m + ((p-filter_height)%_v_str) + (m-1)*(_v_str - 1), n+ ((q - filter_width)%_h_str) + (n-1)*(_h_str -1));

        // fill in reformatted output matrix with the correct values
        for (size_t i = 0; i < m; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                reformatted_output(i*(_v_str), j*(_h_str)) = dLdY[i*n + j];
            }
        }

        // convolve the input images with reformatted output with unit strides
        size_t num_v_strides = std::floor((p - reformatted_output.get_rows())) + 1;
        size_t num_h_strides = std::floor((q - reformatted_output.get_cols())) + 1;


        // number of threads needed in each dimension
        size_t N_x = num_h_strides;
        size_t N_y = num_v_strides;
        size_t N_z = N_in_maps;

        // number of blocks
        dim3 numBlocks((N_x+xsize - 1)/xsize, (N_y+ysize - 1)/ysize, (N_z+zsize - 1)/zsize);

        Mat<double>* d_reformatted_output;
        double* d_reformatted_output_data;
        reformatted_output.port_to_GPU(d_reformatted_output, d_reformatted_output_data);

        Kernel2<<<numBlocks, threadsPerBlock>>>(d_dLdFs, d_local_input, d_reformatted_output, idx);

        cudaFree(d_reformatted_output_data);
        cudaFree(d_reformatted_output);

        // this concludes the calculation of _dLdFs

        // we move to calculation of dLdX

        // add padding to reformatted matrix
        reformatted_output.padding(filter_width-1, filter_width-1, filter_height-1, filter_height-1);


        num_v_strides = std::floor((reformatted_output.get_rows() - filter_height)) + 1;
        num_h_strides = std::floor((reformatted_output.get_cols()- filter_width)) + 1;

        // number of strides in each direction should be equal to the dimensions of dLdX_matrix
        assert(num_v_strides == _in.height && num_h_strides == _in.width);

        size_t n_rows = num_v_strides - _padbottom - _padtop;
        size_t n_cols = num_h_strides - _padleft - _padright;

        reformatted_output.port_to_GPU(d_reformatted_output, d_reformatted_output_data);

        Kernel3<<<1, N_in_maps>>>(N_in_maps, idx, n_cols, n_rows, d_filters, d_out, _padtop, _padleft, d_reformatted_output);

        cudaFree(d_reformatted_output_data);
        cudaFree(d_reformatted_output);

    }

    // retrieve data from device and put it into return variable
    cudaMemcpy(dLdXs.get_data(), d_out, dLdXs.get_len()*sizeof(double), cudaMemcpyDeviceToHost);

    // we are averaging the loss gradient over the total number of filters
        dLdXs *= 1.0/N_filters;

    cudaFree(d_out);
    cudaProfilerStop();
}

template<typename Optimizer>
void Convolution::Update_Params(Optimizer* optimizer, size_t normalizer)
{
    // block size
    size_t block_size = 1024;
    // number of threads needed
    size_t N = _dLdFs->get_depth() * _dLdFs->get_cols() * _dLdFs->get_rows();
    // number of threads per block
    dim3 threadsPerBlock(block_size);
    // number of blocks
    dim3 numBlocks((N+block_size - 1)/block_size);

    for (size_t i = 0; i < _out.depth; i++)
    {

        (*optimizer).Forward(d_filters_data[i], d_dLdFs_data[i], normalizer, N);
        fill_Kernel<<<numBlocks, threadsPerBlock>>>(N, d_dLdFs_data[i], 0);
    }

}

void Convolution::print_filters()
{
    for (size_t i = 0; i < _out.depth; i++)
    {
        std::cout << "FILTER " << i << "----------------\n\n";
        _filters[i].print();
    }
}


#endif //CNN_CONVOLUTION_IMPL_HXX
