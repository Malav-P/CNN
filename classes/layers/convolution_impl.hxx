//
// Created by malav on 6/17/2022.
//

#ifndef ANN_CONVOLUTION_IMPL_HXX
#define ANN_CONVOLUTION_IMPL_HXX

#include "convolution.hxx"

Convolution::Convolution(size_t in_width, size_t in_height, const Mat<double>& filter, size_t stride_h, size_t stride_v, size_t padleft,
                         size_t padright, size_t padtop, size_t padbottom)
: _in(in_width + padleft + padright, in_height + padtop + padbottom),
  _h_str(stride_h),
  _v_str(stride_v),
  _padleft(padleft),
  _padright(padright),
  _padtop(padtop),
  _padbottom(padbottom),
  _local_input((in_width + padleft + padright) * (in_height + padtop + padbottom)),
  _kernel( (std::floor((in_width + padleft + padright - filter.get_cols())/stride_h) + 1) * (std::floor((in_height + padbottom + padtop - filter.get_rows())/stride_v) + 1), (in_width + padright + padleft)*(in_height + padtop + padbottom)),
  _filter(filter),
  _dLdW((std::floor((in_width + padleft + padright - filter.get_cols())/stride_h) + 1) * (std::floor((in_height + padbottom + padtop - filter.get_rows())/stride_v) + 1), (in_width + padright + padleft)*(in_height + padtop + padbottom))
{

    // calculate total number of vertical and horizontal strides
    size_t num_v_strides = std::floor((_in.height - filter.get_rows()) / _v_str) + 1;
    size_t num_h_strides = std::floor((_in.width - filter.get_cols()) / _h_str) + 1;

    // specify output shape
    _out = {num_h_strides, num_v_strides};

}

Convolution::Convolution(size_t in_width, size_t in_height, const Mat<double> &filter, size_t stride_h, size_t stride_v,
                         bool padding)
                         :_h_str(stride_h),
                          _v_str(stride_v)
{
    if (!padding)
    {
        _padleft = 0;
        _padright = 0;
        _padtop = 0;
        _padbottom = 0;
        _in = {in_width + _padleft + _padright, in_height + _padtop + _padbottom};
        _local_input = Vector<double>((in_width + _padleft + _padright) * (in_height + _padtop + _padbottom));
        _kernel = Mat<double>( (std::floor((_in.width - filter.get_cols())/stride_h) + 1) * (std::floor((_in.height - filter.get_rows())/stride_v) + 1), (_in.width)*(_in.height));
        _filter = filter;
        _dLdW = Mat<double>((std::floor((_in.width - filter.get_cols())/stride_h) + 1) * (std::floor((_in.height - filter.get_rows())/stride_v) + 1), (_in.width)*(_in.height));

    }
    else
    {
        // total number of padded rows
        size_t vert_pad = (in_height - 1) * stride_v - in_height + filter.get_rows();

        // number of padded rows at top of matrix
        _padtop = std::floor(vert_pad/2.0);

        // number of padded rows at bottom of matrix
        _padbottom = std::ceil(vert_pad/2.0);

        // total number of padded columns
        size_t hor_pad = (in_width - 1) * stride_h - in_width + filter.get_cols();

        // number of padded columns to left of matrix
        _padleft = std::floor(hor_pad/2.0);

        // number of padded columns to right of matrix
        _padright = std::ceil(hor_pad/2.0);

        // rest of member variables
        _in = {in_width + _padleft + _padright, in_height + _padtop + _padbottom};
        _local_input = Vector<double>((in_width + _padleft + _padright) * (in_height + _padtop + _padbottom));
        _kernel = Mat<double>( (std::floor((in_width + _padleft + _padright - filter.get_cols())/stride_h) + 1) * (std::floor((in_height + _padbottom + _padtop - filter.get_rows())/stride_v) + 1), (in_width + _padright + _padleft)*(in_height + _padtop + _padbottom));
        _filter = filter;
        _dLdW = Mat<double>((std::floor((in_width + _padleft + _padright - filter.get_cols())/stride_h) + 1) * (std::floor((in_height + _padbottom + _padtop - filter.get_rows())/stride_v) + 1), (in_width + _padright + _padleft)*(in_height + _padtop + _padbottom));


    }


    // calculate total number of vertical and horizontal strides
    size_t num_v_strides = std::floor((_in.height - filter.get_rows()) / _v_str) + 1;
    size_t num_h_strides = std::floor((_in.width - filter.get_cols()) / _h_str) + 1;

    // specify output shape
    _out = {num_h_strides, num_v_strides};

    // initialize variables for the for loop
    size_t krnl_row, fltr_col, fltr_row, offset;
}

void Convolution::Forward(Vector<double> &input, Vector<double> &output)
{
    // note that input length matching with _in parameters is indirectly checked in the matrix*vector operator overload

    // this routine can be optimized (we take a vector, turn it into matrix, pad it, then flatten back to vector)
    // find a way to do the padding with the vector itself
    _local_input = input.reshape(_in.height - _padtop - _padbottom, _in.width - _padleft - _padright);
    _local_input.padding(_padleft, _padright, _padtop, _padbottom);


    // initialize return variable
    Mat<double> output_image(_out.height, _out.width);

    // do convolution
    for (size_t i = 0; i < output_image.get_rows(); i++)
    {
        for (size_t j = 0; j < output_image.get_cols(); j++)
        {
            output_image(i,j) = _local_input.partial_dot(_filter, {i,j});
        }
    }


    // flatten matrix to vector and return it
    output = output_image.flatten();
}



void Convolution::Backward(Vector<double> &dLdY, Vector<double> &dLdX)
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
    Mat<double> reformatted_output(m + p%filter_height + (m-1)*(_v_str - 1), n+ q%filter_width + (n-1)*(_h_str -1));

    // fill in reformatted output matrix with the correct values
    for (size_t i = 0; i < dLdY_matrix.get_rows(); i++)
    {
        for (size_t j = 0; j < dLdY_matrix.get_cols(); j++)
        {
            reformatted_output(i*(_v_str-1), j*(_h_str-1)) = dLdY_matrix(i,j);
        }
    }

    // convolve the input image with reformatted output with unit strides
    size_t num_v_strides = std::floor((_in.height - reformatted_output.get_rows())) + 1;
    size_t num_h_strides = std::floor((_in.width - reformatted_output.get_cols())) + 1;

    for (size_t i = 0; i < num_v_strides; i++)
    {
        for (size_t j = 0; j < num_h_strides; j++)
        {
            dLdF(i,j) = _local_input.partial_dot(reformatted_output, {i,j});
        }
    }

    // this concludes the calculation of dLdF

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

    // create a return variable
    Mat<double> dLdX_matrix(num_v_strides, num_h_strides);

    // iterate through return variable
    for (size_t i = 0; i < dLdX_matrix.get_rows(); i++)
    {
        for (size_t j = 0; j < dLdX_matrix.get_cols(); j++)
        {
            dLdX_matrix(i,j) = reformatted_output.partial_dot(_filter, {i,j});
        }
    }

    // return filter to original, non-rotated state
    _filter.set_rot(0);

    // reshape dLdX matrix to remove padded rows and columns
    dLdX_matrix.crop(_padleft, _padright, _padtop, _padbottom);

    // flatten matrix into vector, assign it to dLdX
    dLdX = dLdX_matrix.flatten();
}

// TODO: THIS NEEDS TO BE REDONE
template<typename Optimizer>
void Convolution::Update_Params(Optimizer* optimizer, size_t normalizer)
{
    // positions in the kernel that had zeros are parameters that are not meant to be learned. We must
    // set the same positions in _dLdW to zero before updating the _kernel
    _dLdW.keep(_indices);

    // update the weights according to the optimizer
    (*optimizer).Forward(_kernel, _dLdW, normalizer);

    // fill the gradient with zeros
    _dLdW.fill(0);
}


#endif //ANN_CONVOLUTION_IMPL_HXX
