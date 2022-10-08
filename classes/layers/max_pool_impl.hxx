//
// Created by malav on 6/19/2022.
//

#ifndef ANN_MAX_POOL_IMPL_HXX
#define ANN_MAX_POOL_IMPL_HXX

#include "max_pool.hxx"

MaxPool::MaxPool(size_t in_width, size_t in_height, size_t fld_width, size_t fld_height, size_t h_stride, size_t v_stride)
: _in(in_width, in_height, 1),
  _field(fld_width, fld_height),
  _h_str(h_stride),
  _v_str(v_stride)
{
    // calculate total number of vertical and horizontal strides
    size_t num_v_strides = std::floor((_in.height - _field.height) / _v_str) + 1;
    size_t num_h_strides = std::floor((_in.width - _field.width) / _h_str) + 1;

    _winners =  new size_t[num_h_strides*num_v_strides]{0};

    // specify the output shape of the max pool layer
    _out = {num_h_strides, num_v_strides, 1};
}

//! move assignment operator---------------------------------------------------------------------
MaxPool &MaxPool::operator=(MaxPool &&other) noexcept {

    // check for self-assignment
    if (this != &other)
    {

        // Free the existing resource.
        delete[] _winners;

        // Copy the _data pointer and its _length from the
        // source object.
        _winners   = other._winners;
        _in = other._in;
        _out = other._out;
        _field = other._field;
        _h_str = other._h_str;
        _v_str = other._v_str;

        // Release the _data pointer from the source object so that
        // the destructor does not free the memory multiple times.
        other._winners = nullptr;
    }

    // return result
    return *this;
}
//! ---------------------------------------------------------------------

//! --- move constructor
MaxPool::MaxPool(MaxPool &&other) noexcept
: _in(other._in),
 _out(other._out),
 _field(other._field),
 _h_str(other._h_str),
 _v_str(other._v_str),
 _winners(other._winners)

{
    other._winners = nullptr;
}
//! -----------------------------



__host__ __device__
void
MaxPool::Forward(Vector<double> &input, Vector<double> &output)
{
    // check to make sure input length matches the input shape of pooling layer
    assert(input.get_len() == _in.width*_in.height);


    // initialize array for temporary value storage
    Pair* buffer = new Pair[_field.width*_field.height];

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
            output[v_stride*_out.width + h_stride] = winner.height;

            // store max value's index
            _winners[v_stride * _out.width + h_stride] = winner.width;
        }}

    delete[] buffer;
}


void MaxPool::Backward(Vector<double> &dLdY, Vector<double> &dLdX)
{

    // ensure that dLdY has same length as output of pooling layer
    assert(_out.width*_out.height == dLdY.get_len());

    // do backpropagation
    for (size_t i=0; i < _out.width*_out.height; i++)
    {
        dLdX[_winners[i]] = dLdY[i];
    }


}

__host__ __device__
Pair MaxPool::max_value(Pair *arr, size_t n)
{
    // assume winner is first element
    Pair answer = arr[0];

    // find the winner (max value's index, max value)
    for (size_t i = 1; i < n; i++)
    {
        if (answer.height < arr[i].height) {answer = arr[i];}
    }

    // return result
    return answer;
}

#endif //ANN_MAX_POOL_IMPL_HXX
