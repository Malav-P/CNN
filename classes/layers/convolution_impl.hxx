//
// Created by malav on 6/17/2022.
//

#ifndef CNN_CONVOLUTION_IMPL_HXX
#define CNN_CONVOLUTION_IMPL_HXX

#include "convolution.hxx"
#include "../lin_alg/miscellaneous_helpers.hxx"

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
  _local_input(new Mat<double>[in_maps])
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

    // Allocate device struct array
    cudaMalloc( (void**)&d_filters, out_maps*sizeof(Cuboid<double>));

    // copy over data from pool_vector to d_poolvec
    for (size_t i = 0; i< out_maps; i++)
    {
        // host struct
        Cuboid<double>* elem = &(_filters[i]);

        // device struct
        Cuboid<double>* d_elem = &(d_filters[i]);

        // copy struct from host to device
        cudaMemcpy(d_elem, elem, sizeof(Cuboid<double>), cudaMemcpyHostToDevice);

        // device array


        // length of device array
        int d_data_len = filter_height* filter_width* in_maps;

        // Allocate device pointer
        cudaMalloc((void**)&(d_filters_data[i]), d_data_len*sizeof(double));

        // copy pointer content from host to device
        cudaMemcpy((d_filters_data[i]), elem->_data, d_data_len*sizeof(double), cudaMemcpyHostToDevice);


        cudaMemcpy(&(d_elem->_data), &(d_filters_data[i]), sizeof(double*), cudaMemcpyHostToDevice);
    }

    //! --------------------------------------------------------
    // Allocate device struct array
    cudaMalloc( (void**)&d_dLdFs, out_maps*sizeof(Cuboid<double>));

    // copy over data from pool_vector to d_poolvec
    for (size_t i = 0; i< out_maps; i++)
    {
        // host struct
        Cuboid<double>* elem = &(_dLdFs[i]);

        // device struct
        Cuboid<double>* d_elem = &(d_dLdFs[i]);

        // copy struct from host to device
        cudaMemcpy(d_elem, elem, sizeof(Cuboid<double>), cudaMemcpyHostToDevice);

        // device array


        // length of device array
        int d_data_len = filter_height* filter_width* in_maps;

        // Allocate device pointer
        cudaMalloc((void**)&(d_dLdFs_data[i]), d_data_len*sizeof(double));

        // copy pointer content from host to device
        cudaMemcpy((d_dLdFs_data[i]), elem->_data, d_data_len*sizeof(double), cudaMemcpyHostToDevice);


        cudaMemcpy(&(d_elem->_data), &(d_dLdFs_data[i]), sizeof(double*), cudaMemcpyHostToDevice);
    }

}

Convolution::Convolution(size_t in_maps, size_t out_maps, size_t in_width, size_t in_height, size_t filter_width,
                         size_t filter_height, size_t stride_h, size_t stride_v, bool padding)
                         :_h_str(stride_h),
                          _v_str(stride_v),
                          _filters(new Cuboid<double>[out_maps]),
                          d_filters_data(new double*[out_maps]),
                          d_dLdFs_data(new double*[out_maps]),
                          _dLdFs(new Cuboid<double>[out_maps]),
                          _local_input(new Mat<double>[in_maps])
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

    // Allocate device struct array
    cudaMalloc( (void**)&d_filters, out_maps*sizeof(Cuboid<double>));

    // copy over data from pool_vector to d_poolvec
    for (size_t i = 0; i< out_maps; i++)
    {
        // host struct
        Cuboid<double>* elem = &(_filters[i]);

        // device struct
        Cuboid<double>* d_elem = &(d_filters[i]);

        // copy struct from host to device
        cudaMemcpy(d_elem, elem, sizeof(Cuboid<double>), cudaMemcpyHostToDevice);

        // device array


        // length of device array
        int d_data_len = filter_height* filter_width* in_maps;

        // Allocate device pointer
        cudaMalloc((void**)&(d_filters_data[i]), d_data_len*sizeof(double));

        // copy pointer content from host to device
        cudaMemcpy((d_filters_data[i]), elem->_data, d_data_len*sizeof(double), cudaMemcpyHostToDevice);


        cudaMemcpy(&(d_elem->_data), &(d_filters_data[i]), sizeof(double*), cudaMemcpyHostToDevice);
    }

    //! --------------------------------------------------------
    // Allocate device struct array
    cudaMalloc( (void**)&d_dLdFs, out_maps*sizeof(Cuboid<double>));

    // copy over data from pool_vector to d_poolvec
    for (size_t i = 0; i< out_maps; i++)
    {
        // host struct
        Cuboid<double>* elem = &(_dLdFs[i]);

        // device struct
        Cuboid<double>* d_elem = &(d_dLdFs[i]);

        // copy struct from host to device
        cudaMemcpy(d_elem, elem, sizeof(Cuboid<double>), cudaMemcpyHostToDevice);

        // device array


        // length of device array
        int d_data_len = filter_height* filter_width* in_maps;

        // Allocate device pointer
        cudaMalloc((void**)&(d_dLdFs_data[i]), d_data_len*sizeof(double));

        // copy pointer content from host to device
        cudaMemcpy((d_dLdFs_data[i]), elem->_data, d_data_len*sizeof(double), cudaMemcpyHostToDevice);


        cudaMemcpy(&(d_elem->_data), &(d_dLdFs_data[i]), sizeof(double*), cudaMemcpyHostToDevice);
    }

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
    // note that input length matching with _in parameters is indirectly checked in the matrix*vector operator overload

    size_t rows = _in.height-_padbottom-_padtop;
    size_t cols = _in.width - _padleft - _padright;

    for (size_t i = 0; i < _filters[0].get_depth(); i++)
    {
        _local_input[i] = Mat<double>(rows, cols, input.get_data() + i*rows*cols);
        _local_input[i].padding(_padleft, _padright, _padtop, _padbottom);
    }

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

}


void Convolution::Backward(Vector<double> &dLdYs, Vector<double> &dLdXs)
{

    size_t m = _out.height;
    size_t n = _out.width;

    size_t p = _in.height;
    size_t q = _in.width;

    size_t filter_height = _filters[0].get_rows();
    size_t filter_width = _filters[0].get_cols();

    size_t N_filters = _out.depth;
    size_t N_in_maps = _filters[0].get_depth();


    for (size_t idx = 0; idx < N_filters ; idx++)
    {
        // reshape dLdY into a matrix
        Mat<double> dLdY_matrix(m, n, dLdYs.get_data() + idx*m*n);

        // reformatted output
        Mat<double> reformatted_output(m + ((p-filter_height)%_v_str) + (m-1)*(_v_str - 1), n+ ((q - filter_width)%_h_str) + (n-1)*(_h_str -1));

        // fill in reformatted output matrix with the correct values
        for (size_t i = 0; i < dLdY_matrix.get_rows(); i++)
        {
            for (size_t j = 0; j < dLdY_matrix.get_cols(); j++)
            {
                reformatted_output(i*(_v_str), j*(_h_str)) = dLdY_matrix(i,j);
            }
        }

        // convolve the input images with reformatted output with unit strides
        size_t num_v_strides = std::floor((p - reformatted_output.get_rows())) + 1;
        size_t num_h_strides = std::floor((q - reformatted_output.get_cols())) + 1;

        for (size_t k = 0; k < N_in_maps; k++)
        {
            for (size_t i = 0; i < num_v_strides; i++)
            {
                for (size_t j = 0; j < num_h_strides; j++)
                {
                    _dLdFs[idx](i, j, k) = _local_input[k].partial_dot(reformatted_output, {i, j});
                }
            }
        }

        // this concludes the calculation of _dLdFs

        // we move to calculation of dLdX

        // add padding to reformatted matrix
        reformatted_output.padding(filter_width-1, filter_width-1, filter_height-1, filter_height-1);

        // rotate filter by 180 degrees
        _filters[idx].set_rot(2);

        num_v_strides = std::floor((reformatted_output.get_rows() - filter_height)) + 1;
        num_h_strides = std::floor((reformatted_output.get_cols()- filter_width)) + 1;

        // number of strides in each direction should be equal to the dimensions of dLdX_matrix
        assert(num_v_strides == _in.height);
        assert(num_h_strides == _in.width);

        auto filter_as_list = cube_to_matarray(_filters[idx]);

        std::vector<Mat<double>> dLdX_matrices(N_in_maps);

        for (size_t k = 0; k< N_in_maps; k++)
        {
            Mat<double> mat(num_v_strides, num_h_strides);
            for (size_t i = 0; i < num_v_strides; i++)
            {
                for (size_t j = 0; j < num_h_strides; j++)
                {
                    mat(i,j) = reformatted_output.partial_dot(filter_as_list[k], {i,j});
                }
            }
            // crop the matrices (depad the matrix)
            mat.crop(_padleft, _padright, _padtop, _padbottom);
            for (size_t i = 0; i < mat.get_rows() ; i++)
            {
                for (size_t j = 0 ; j< mat.get_cols() ; j++)
                {
                    // unsure about this +=, look into it more. May need to instead average each dLdX[i] at the end of
                    // the global loop
                    dLdXs[k*mat.get_rows()*mat.get_cols() + i*mat.get_cols() + j] += mat(i,j);
                }
            }

        }

        // return filter to original, non-rotated state
        _filters[idx].set_rot(0);
    }
    // we are averaging the loss gradient over the total number of filters
        dLdXs *= 1.0/N_filters;
}

template<typename Optimizer>
void Convolution::Update_Params(Optimizer* optimizer, size_t normalizer)
{
    for (size_t i = 0; i < _out.depth; i++)
    {
        // update the weights according to the optimizer
        (*optimizer).Forward(_filters[i], _dLdFs[i], normalizer);

        // fill the gradient with zeros
        _dLdFs[i].fill(0);
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
