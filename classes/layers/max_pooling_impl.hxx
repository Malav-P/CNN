//
// Created by Malav Patel on 9/29/22.
//

#ifndef CNN_MAX_POOLING_IMPL_HXX
#define CNN_MAX_POOLING_IMPL_HXX

#include "max_pooling.hxx"

MaxPooling::MaxPooling(size_t in_maps, size_t in_width, size_t in_height, size_t fld_width, size_t fld_height,
                       size_t h_stride, size_t v_stride):
                       pool_vector(in_maps),
                       _in(in_width, in_height),
                       _field(fld_width, fld_height),
                       _h_str(h_stride),
                       _v_str(v_stride)
{
    for (size_t i = 0; i < in_maps; i++)
    {
        pool_vector[i] = MaxPool(in_width, in_height, fld_width, fld_height, h_stride, v_stride);
    }

    _out = pool_vector[0].out_shape();
}

void MaxPooling::Forward(std::vector<Vector<double>> &input, std::vector<Vector<double>> &output)
{

    for (size_t i = 0; i < pool_vector.size() ; i++)
    {
        // allocate memory for output vector
        output[i] = Vector<double>(_out.width*_out.height);

        pool_vector[i].Forward(input[i], output[i]);
    }
}

void MaxPooling::Backward(std::vector<Vector<double>> &dLdY, std::vector<Vector<double>> &dLdX)
{
    for (size_t i = 0; i < pool_vector.size() ; i++)
    {
        pool_vector[i].Backward(dLdY[i], dLdX[i]);
    }
}

#endif //CNN_MAX_POOLING_IMPL_HXX
