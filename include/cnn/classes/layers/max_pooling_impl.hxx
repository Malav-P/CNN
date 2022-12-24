//
// Created by Malav Patel on 9/29/22.
//

#ifndef CNN_MAX_POOLING_IMPL_HXX
#define CNN_MAX_POOLING_IMPL_HXX

#include "max_pooling.hxx"

namespace CNN{

MaxPool::MaxPool(size_t in_width, size_t in_height, size_t fld_width, size_t fld_height, size_t h_stride, size_t v_stride)
        : _in(in_width, in_height, 1),
          _field(fld_width, fld_height),
          _h_str(h_stride),
          _v_str(v_stride),
          _winners( (std::floor((_in.width - _field.width) / _h_str) + 1)*(std::floor((_in.height - _field.height) / _v_str) + 1))
{
    // calculate total number of vertical and horizontal strides
    size_t num_v_strides = std::floor((_in.height - _field.height) / _v_str) + 1;
    size_t num_h_strides = std::floor((_in.width - _field.width) / _h_str) + 1;

    // specify the output shape of the max pool layer
    _out = {num_h_strides, num_v_strides, 1};
}

void
MaxPool::Forward(Vector<double> &input, Vector<double> &output)
{
    // check to make sure input length matches the input shape of pooling layer
    assert(input.get_len() == _in.width*_in.height);

    // initialize array for temporary value storage
    Pair buffer[_field.width * _field.height];

    // do max pooling operation
    for (size_t v_stride = 0; v_stride < _out.height; v_stride++) {for (size_t h_stride = 0; h_stride < _out.width; h_stride++)
        {
            // calculate offset in data stored in vector
            size_t offset = _v_str * v_stride * (_in.width) + _h_str * h_stride;

            // store values for this window into the buffer
            for (size_t j = 0; j < _field.height; j++)
            {
                for (size_t i = 0; i < _field.width; i++)
                {
                    buffer[j * _field.width + i] = {offset + j * _in.width + i, input[offset + j * _in.width + i]};
                }
            }

            // calculate max value and index of max value for this window
            Pair winner = max_value(buffer, _field.width * _field.height);

            // store max value into output
            output[v_stride*_out.width + h_stride] = winner.second;

            // store max value's index
            _winners[v_stride * _out.width + h_stride] = winner.first;
        }}
}


void MaxPool::Backward(Vector<double> &dLdY, Vector<double> &dLdX)
{

    // ensure that dLdY has same length as output of pooling layer
    assert(_winners.get_len() == dLdY.get_len());

    // do backpropagation
    for (size_t i=0; i < _winners.get_len(); i++)
    {
        dLdX[_winners[i]] = dLdY[i];
    }


}

Pair MaxPool::max_value(Pair *arr, size_t n)
{
    // assume winner is first element
    Pair answer = arr[0];

    // find the winner (max value's index, max value)
    for (size_t i = 1; i < n; i++)
    {
        if (answer.second < arr[i].second) {answer = arr[i];}
    }

    // return result
    return answer;
}
//!--------------------------

MaxPooling::MaxPooling(size_t in_maps, size_t in_width, size_t in_height, size_t fld_width, size_t fld_height,
                       size_t h_stride, size_t v_stride):
                       pool_vector(in_maps),
                       Layer(in_width, in_height, in_maps, 0,0,0),
                       _field(fld_width, fld_height),
                       _h_str(h_stride),
                       _v_str(v_stride)
{
    for (size_t i = 0; i < in_maps; i++)
    {
        pool_vector[i] = MaxPool(in_width, in_height, fld_width, fld_height, h_stride, v_stride);
    }

    _out = pool_vector[0].out_shape();
    _out.depth = in_maps;
}

void MaxPooling::Forward(Vector<double> &input, Vector<double> &output)
{

    Vector<double> out(_out.width*_out.height);
    Vector<double> in(_in.height * _in.width);
    for (size_t i = 0; i < pool_vector.size() ; i++)
    {

        in.reset_data(input.get_data() + i *_in.height * _in.width);
        pool_vector[i].Forward(in, out);
        output.write(out, i*out.get_len());
    }

}

void MaxPooling::Backward(Vector<double> &dLdY, Vector<double> &dLdX)
{
    Vector<double> out(_in.height*_in.width);
    Vector<double> in(_out.height * _out.width);
    for (size_t i = 0; i < pool_vector.size() ; i++)
    {
        in.reset_data(dLdY.get_data() + i *_out.height * _out.width);
        pool_vector[i].Backward(in, out);

        dLdX.write(out, i*out.get_len());
    }
}
}
#endif //CNN_MAX_POOLING_IMPL_HXX
