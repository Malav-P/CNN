//
// Created by Malav Patel on 11/7/22.
//

#ifndef CNN_LAYER_HXX
#define CNN_LAYER_HXX

#include "../lin_alg/data_types.hxx"

class Layer {
public:

    // default constructor should not exist
    Layer() = delete;

    Layer(size_t in_width, size_t in_height, size_t in_depth, size_t out_width, size_t out_height, size_t out_depth)
    : _in(in_width, in_height, in_depth),
      _out(out_width, out_height, out_depth)
    {}

    //! BOOST::APPLY_VISITOR FUNCTIONS ----------------------------------------------------------------------------

    // send feature through the convolutional layer
    virtual void Forward(Vector<double> &input, Vector<double> &output) = 0;

    // send feature backward through convolutional layer, keeping track of gradients
    virtual void Backward(Vector<double> &dLdYs, Vector<double> &dLdXs) = 0;

    // get output shape of convolution
    Dims3 const& out_shape() const {return _out;}

    // get input shape of convolution (without padding!)
    Dims3  in_shape() const {return _in;}

protected:

    // input shape of layer
    Dims3 _in {0,0,0};

    // output shape of layer
    Dims3 _out {0,0,0};

};

#endif //CNN_LAYER_HXX
