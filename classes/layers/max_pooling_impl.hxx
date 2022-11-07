//
// Created by Malav Patel on 9/29/22.
//

#ifndef CNN_MAX_POOLING_IMPL_HXX
#define CNN_MAX_POOLING_IMPL_HXX

#include "max_pooling.hxx"

MaxPooling::MaxPooling(size_t in_maps, size_t in_width, size_t in_height, size_t fld_width, size_t fld_height,
                       size_t h_stride, size_t v_stride):
                       pool_vector(in_maps),
                       _in(in_width, in_height, in_maps),
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

    Vector<double> OUTPUT;
    // allocate memory for output vector
    Vector<double> out(_out.width*_out.height);
    for (size_t i = 0; i < pool_vector.size() ; i++)
    {

        Vector<double> in(_in.height * _in.width, input.get_data() + i *_in.height * _in.width);
        pool_vector[i].Forward(in, out);

        OUTPUT = OUTPUT.merge(out);
    }

    output = OUTPUT;
}

void MaxPooling::Backward(Vector<double> &dLdY, Vector<double> &dLdX)
{
    Vector<double> OUTPUT;
    Vector<double> out(_in.height*_in.width);
    for (size_t i = 0; i < pool_vector.size() ; i++)
    {
        Vector<double> in(_out.height * _out.width, dLdY.get_data() + i *_out.height * _out.width);
        pool_vector[i].Backward(in, out);

        OUTPUT = OUTPUT.merge(out);
    }
    dLdX = OUTPUT;
}

#endif //CNN_MAX_POOLING_IMPL_HXX
