//
// Created by malav on 6/20/2022.
//

#ifndef ANN_MEAN_POOL_IMPL_HXX
#define ANN_MEAN_POOL_IMPL_HXX

#include "mean_pool.hxx"

MeanPool::MeanPool(size_t in_width, size_t in_height, size_t fld_width, size_t fld_height, size_t h_stride, size_t v_stride)
: _in(in_width, in_height),
  _field(fld_width, fld_height),
  _h_str(h_stride),
  _v_str(v_stride)
{
    // calculate total number of vertical and horizontal strides
    size_t num_v_strides = std::floor((_in.height - _field.height) / _v_str) + 1;
    size_t num_h_strides = std::floor((_in.width - _field.width) / _h_str) + 1;

    // specify output shape of mean pool layer
    _out = {num_h_strides, num_v_strides} ;
}

void
MeanPool::Forward(Vector<double> &input, Vector<double> &output)
{
    // ensure input length matches input shape of mean pool layer
    assert(input.get_len() == _in.width*_in.height);

    // create a buffer to hold values from the sliding window
    double buffer[_field.width * _field.height];

    // iterate over the vertical and horizontal strides of the sliding window
    for (size_t v_stride = 0; v_stride < _out.height; v_stride++) {for (size_t h_stride = 0; h_stride < _out.width; h_stride++)
        {
            // this offset defines the start position in the array _data as a function of vertical and horizontal strides
            size_t offset = _v_str * v_stride * (_in.width) + _h_str * h_stride;

            // add the elements of the array to the buffer
            for (size_t j = 0; j < _field.height; j++) {for (size_t i = 0; i < _field.width; i++)
                {
                    buffer[j * _field.width + i] = input[offset + j * _in.width + i];
                }
            }

            // compute the average value of the buffer
            output[v_stride*_out.width + h_stride] = avg_value(buffer, _field.width * _field.height);
        }}
}

void MeanPool::Backward(Vector<double> &dLdY, Vector<double> &dLdX)
{
    // compute total number of vertical and horizontal strides
    size_t num_v_strides = std::floor((_in.height - _field.height) / _v_str) + 1;
    size_t num_h_strides = std::floor((_in.width - _field.width) / _h_str) + 1;

    // iterate over vertical and horizontal strides of window
    for (size_t v_stride = 0; v_stride < num_v_strides; v_stride++) {for (size_t h_stride = 0; h_stride < num_h_strides; h_stride++)
        {
            // this offset defines the start position in the array _data as a function of vertical and horizontal strides
            size_t offset = _v_str * v_stride * (_in.width) + _h_str * h_stride;

            // compute dLdX
            for (size_t j = 0; j < _field.height; j++)
            {
                for (size_t i = 0; i < _field.width; i++)
                {
                    dLdX[offset + j * _in.width + i] = (1.0 / (_field.width * _field.height)) * dLdY[v_stride * num_h_strides + h_stride];
                }
            }

        }}

}

template<typename T>
double MeanPool::avg_value(T *arr, size_t n)
{
    // initialize sum
    double sum = 0;

    // compute sum
    for (size_t i = 0; i < n; i++) {sum += arr[i];}

    // return average
    return sum/n;
}

#endif //ANN_MEAN_POOL_IMPL_HXX
