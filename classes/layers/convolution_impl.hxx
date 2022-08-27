//
// Created by malav on 6/17/2022.
//

#ifndef ANN_CONVOLUTION_IMPL_HXX
#define ANN_CONVOLUTION_IMPL_HXX

#include "convolution.hxx"
#include "../lin_alg/vector.hxx"

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
  _dLdW((std::floor((in_width + padleft + padright - filter.get_cols())/stride_h) + 1) * (std::floor((in_height + padbottom + padtop - filter.get_rows())/stride_v) + 1), (in_width + padright + padleft)*(in_height + padtop + padbottom))
{

    // allocate memory for winning indices
    _indices = new Dims[filter.get_cols() * filter.get_rows() * _kernel.get_rows()];

    // initialize iterator variable for _indices array
    size_t i = 0;

    // calculate total number of vertical and horizontal strides
    size_t num_v_strides = std::floor((_in.height - filter.get_rows()) / _v_str) + 1;
    size_t num_h_strides = std::floor((_in.width - filter.get_cols()) / _h_str) + 1;

    // specify output shape
    _out = {num_h_strides, num_v_strides};

    // initialize variables for the for loop
    size_t krnl_row, fltr_col, fltr_row, offset;

    // fill in kernel matrix with values
    for (size_t v_stride = 0; v_stride < num_v_strides; v_stride++) {for (size_t h_stride = 0; h_stride < num_h_strides; h_stride++)
    {
        krnl_row = num_h_strides * v_stride + h_stride;
        offset = _v_str * v_stride * (_in.width) + _h_str * h_stride;

        for (fltr_row = 0; fltr_row < filter.get_rows(); fltr_row++)
        {
            for (fltr_col = 0; fltr_col < filter.get_cols(); fltr_col++)
            {
                _kernel(krnl_row, fltr_col + offset + fltr_row * (_in.width)) = filter(fltr_row, fltr_col);
                *(_indices + i) = Dims{krnl_row, fltr_col + offset + fltr_row * (_in.width)};
                i++;
            }
        }

    }}

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
        _dLdW = Mat<double>((std::floor((in_width + _padleft + _padright - filter.get_cols())/stride_h) + 1) * (std::floor((in_height + _padbottom + _padtop - filter.get_rows())/stride_v) + 1), (in_width + _padright + _padleft)*(in_height + _padtop + _padbottom));


    }


    // allocate memory for winning indices
    _indices = new Dims[filter.get_cols() * filter.get_rows() * _kernel.get_rows()];

    // initialize iterator variable for _indices array
    size_t i = 0;

    // calculate total number of vertical and horizontal strides
    size_t num_v_strides = std::floor((_in.height - filter.get_rows()) / _v_str) + 1;
    size_t num_h_strides = std::floor((_in.width - filter.get_cols()) / _h_str) + 1;

    // specify output shape
    _out = {num_h_strides, num_v_strides};

    // initialize variables for the for loop
    size_t krnl_row, fltr_col, fltr_row, offset;

    // fill in kernel matrix with values
    for (size_t v_stride = 0; v_stride < num_v_strides; v_stride++) {for (size_t h_stride = 0; h_stride < num_h_strides; h_stride++)
        {
            krnl_row = num_h_strides * v_stride + h_stride;
            offset = _v_str * v_stride * (_in.width) + _h_str * h_stride;

            for (fltr_row = 0; fltr_row < filter.get_rows(); fltr_row++)
            {
                for (fltr_col = 0; fltr_col < filter.get_cols(); fltr_col++)
                {
                    _kernel(krnl_row, fltr_col + offset + fltr_row * (_in.width)) = filter(fltr_row, fltr_col);
                    _indices[i] = Dims{krnl_row, fltr_col + offset + fltr_row * (_in.width)};
                    i++;
                }
            }

        }}
}

void Convolution::Forward(Vector<double> &input, Vector<double> &output)
{
    // note that input length matching with _in parameters is indirectly checked in the matrix*vector operator overload

    // this routine can be optimized (we take a vector, turn it into matrix, pad it, then flatten back to vector)
    // find a way to do the padding with the vector itself
    Mat<double> tmp = input.reshape(_in.height - _padtop - _padbottom, _in.width - _padleft - _padright);
    tmp.padding(_padleft, _padright, _padtop, _padbottom);
    _local_input = tmp.flatten();

    // do convolution as matrix multiplication
    output = _kernel*_local_input;
}

void Convolution::Backward(Vector<double> &dLdY, Vector<double> &dLdX)
{
    // compute dLdX, this vector will be sent to be backpropagated through the previous layer
    dLdX = dLdY*_kernel;

    // compute gradients and add to existing gradient
    _dLdW += dLdY * _local_input;
}

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

//    switch (optimizer)
//    {
//        // gradient descent
//        case 1:
//        {
//            // learning rate
//            double alpha = 0.1;
//
//            // the below routine is terribly inefficient. We only really need to access and update the values
//            // in the important indices and ignore everything else.
//
//            // positions in the kernel that had zeros are parameters that are not meant to be learned. We must
//            // set the same positions in _dLdW to zero before updating the _kernel
//            _dLdW.keep(_indices);
//
//            // update weights and reset dLdW to zeros
//            _kernel += _dLdW * ((-alpha) * (1.0/normalizer));
//            _dLdW.fill(0);
//        }
//    }
}


#endif //ANN_CONVOLUTION_IMPL_HXX
