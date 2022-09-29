//
// Created by malav on 6/17/2022.
//

#ifndef ANN_CONVOLUTION_IMPL_HXX
#define ANN_CONVOLUTION_IMPL_HXX

#include "convolution.hxx"
#include "../lin_alg/miscellaneous_helpers.hxx"

Convolution::Convolution(size_t in_maps, size_t in_width, size_t in_height, size_t filter_width, size_t filter_height, size_t stride_h, size_t stride_v, size_t padleft,
                         size_t padright, size_t padtop, size_t padbottom)
: _in(in_width + padleft + padright, in_height + padtop + padbottom),
  _h_str(stride_h),
  _v_str(stride_v),
  _padleft(padleft),
  _padright(padright),
  _padtop(padtop),
  _padbottom(padbottom),
  _filter(filter_height, filter_width, in_maps),
  _dLdF(filter_height, filter_width, in_maps)
{

    // calculate total number of vertical and horizontal strides
    size_t num_v_strides = std::floor((_in.height - filter_height) / _v_str) + 1;
    size_t num_h_strides = std::floor((_in.width - filter_width) / _h_str) + 1;

    // specify output shape
    _out = {num_h_strides, num_v_strides};

    // get the current time to seed the random number generator
    typedef std::chrono::high_resolution_clock myclock;
    myclock::time_point beginning = myclock::now();
    myclock::duration d = myclock::now() - beginning;
    unsigned seed2 = d.count();

    // seed the random number generator
    std::default_random_engine generator(seed2);
    std::uniform_real_distribution<double> distribution(-sqrt(6.0/(_in.height*_in.width + _out.height*_out.width)), sqrt(6.0/(_in.height*_in.width + _out.height*_out.width)));

    // Glorot initialize the weights
    for (size_t i=0; i<filter_height; i++) { for (size_t j=0; j<filter_width; j++){ for (size_t k = 0; k < in_maps; k++){
         _filter(i,j,k) = distribution(generator);
    }}}

    _local_input.resize(in_maps);

}

Convolution::Convolution(size_t in_maps, size_t in_width, size_t in_height, size_t filter_width, size_t filter_height, size_t stride_h, size_t stride_v,
                         bool padding)
                         :_h_str(stride_h),
                          _v_str(stride_v),
                          _filter(filter_height, filter_width, in_maps),
                          _dLdF(filter_height, filter_width, in_maps)
{
    if (!padding)
    {
        _padleft = 0;
        _padright = 0;
        _padtop = 0;
        _padbottom = 0;
        _in = {in_width, in_height};

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
        _in = {in_width + _padleft + _padright, in_height + _padtop + _padbottom};
    }


    // calculate total number of vertical and horizontal strides
    size_t num_v_strides = std::floor((_in.height - filter_height) / _v_str) + 1;
    size_t num_h_strides = std::floor((_in.width - filter_width) / _h_str) + 1;

    // specify output shape
    _out = {num_h_strides, num_v_strides};

    // get the current time to seed the random number generator
    typedef std::chrono::high_resolution_clock myclock;
    myclock::time_point beginning = myclock::now();
    myclock::duration d = myclock::now() - beginning;
    unsigned seed2 = d.count();

    // seed the random number generator
    std::default_random_engine generator(seed2);
    std::uniform_real_distribution<double> distribution(-sqrt(6.0/(_in.height*_in.width + _out.height*_out.width)), sqrt(6.0/(_in.height*_in.width + _out.height*_out.width)));

    // Glorot initialize the weights
    for (size_t i=0; i<filter_height; i++) { for (size_t j=0; j<filter_width; j++) { for (size_t k = 0; k< in_maps; k++){
         _filter(i,j, k) = distribution(generator);
    }}}

    _local_input.resize(in_maps);

}

void Convolution::Forward(std::vector<Vector<double>>& input, Vector<double> &output)
{
    // note that input length matching with _in parameters is indirectly checked in the matrix*vector operator overload

    // this routine can be optimized (we take a vector, turn it into matrix, pad it, then flatten back to vector)
    // find a way to do the padding with the vector itself

    size_t i = 0;
    for (Vector<double>& vec : input)
    {
        _local_input[i] = vec.reshape(_in.height - _padtop - _padbottom, _in.width - _padleft - _padright);
        _local_input[i].padding(_padleft, _padright, _padtop, _padbottom);
        i++;
    }


    Cuboid<double> input_cube = cubify(_local_input);

    // initialize return variable
    Mat<double> output_image(_out.height, _out.width);

    // do convolution
    for (i = 0; i < output_image.get_rows(); i++)
    {
        for (size_t j = 0; j < output_image.get_cols(); j++)
        {
            output_image(i,j) = input_cube.partial_dot(_filter, {i*_v_str,j*_h_str, 0});
        }
    }


    // flatten matrix to vector and return it
    output = output_image.flatten();
}



void Convolution::Backward(Vector<double> &dLdY, std::vector<Vector<double>> &dLdX)
{
    // reshape dLdY into a matrix
    Mat<double> dLdY_matrix = dLdY.reshape(_out.height, _out.width);

    size_t m = _out.height;
    size_t n = _out.width;

    size_t p = _in.height;
    size_t q = _in.width;

    size_t filter_height = _filter.get_rows();
    size_t filter_width = _filter.get_cols();

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
    size_t num_v_strides = std::floor((_in.height - reformatted_output.get_rows())) + 1;
    size_t num_h_strides = std::floor((_in.width - reformatted_output.get_cols())) + 1;
    for (size_t k = 0; k< _filter.get_depth(); k++)
    {
        for (size_t i = 0; i < num_v_strides; i++)
        {
            for (size_t j = 0; j < num_h_strides; j++)
            {
                _dLdF(i, j, k) = _local_input[k].partial_dot(reformatted_output, {i, j});
            }
        }
    }


    // this concludes the calculation of _dLdF

    // we move to calculation of dLdX

    // add padding to reformatted matrix
    reformatted_output.padding(filter_width-1, filter_width-1, filter_height-1, filter_height-1);

    // rotate filter by 180 degrees
    _filter.set_rot(2);

    num_v_strides = std::floor((reformatted_output.get_rows() - _filter.get_rows())) + 1;
    num_h_strides = std::floor((reformatted_output.get_cols()- _filter.get_cols())) + 1;

    // number of strides in each direction should be equal to the dimensions of dLdX_matrix
    assert(num_v_strides == _in.height);
    assert(num_h_strides == _in.width);


    std::vector<Mat<double>> filter_as_list = cube_to_matarray(_filter);

    std::vector<Mat<double>> dLdX_matrices(_filter.get_depth());

    for (size_t k = 0; k< _filter.get_depth(); k++)
    {
        Mat<double> mat(num_v_strides, num_h_strides);
        for (size_t i = 0; i < num_v_strides; i++)
        {
            for (size_t j = 0; j < num_h_strides; j++)
            {
                mat(i,j) = reformatted_output.partial_dot(filter_as_list[k], {i,j});
            }
        }
        dLdX_matrices[k] = mat;
    }


    // return filter to original, non-rotated state
    _filter.set_rot(0);

    // crop the matrices (depad the matrix)
    size_t i = 0;
    for (Mat<double>& mat : dLdX_matrices)
    {
        mat.crop(_padleft, _padright, _padtop, _padbottom);
        dLdX[i] = mat.flatten();
        i++;
    }

}

template<typename Optimizer>
void Convolution::Update_Params(Optimizer* optimizer, size_t normalizer)
{
    // update the weights according to the optimizer
    (*optimizer).Forward(_filter, _dLdF, normalizer);

    // fill the gradient with zeros
    _dLdF.fill(0);
}


#endif //ANN_CONVOLUTION_IMPL_HXX
