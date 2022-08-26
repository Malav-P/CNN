//
// Created by malav on 5/2/2022.
//

#ifndef ANN_LINEAR_HXX
#define ANN_LINEAR_HXX

#include "../lin_alg/data_types.hxx"

//! a linear class which applies a transformation of the form
//! z = f( Wx + b). W is a weight matrix and b is a bias. f is an activation function

template<typename activ_func>
class Linear {
    public:

        //! CONSTRUCTORS, DESTRUCTORS, MOVE CONSTRUCTORS, ASSIGNMENT OPERATORS, ETC ------------------------------------

        // default constructor
        Linear() = default;

        // construct Linear Layer with specified input and output sizes
        Linear(size_t in_size, size_t out_size);

        // destructor
        ~Linear() = default;

        //! -----------------------------------------------------------------------------------------------------------

        //! BOOST::APPLY_VISITOR FUNCTIONS ---------------------------------------------------------------------------

        // send vector forward through this layer
        void Forward(const Vector<double>& input, Vector<double>& output);

        // send vector backwards through layer, computing gradients and input error dLdX
        void Backward(const Vector<double>& dLdZ, Vector<double>& dLdX);

        // return out shape of layer
        Dims const& out_shape() const {return _out;}

        // return in shape of layer
        Dims const& in_shape() const {return _in;}

        // update the weights and biases according to their gradients
        template<typename Optimizer>
        void Update_Params(Optimizer* optimizer, size_t normalizer);

        //! ----------------------------------------------------------------------------------------------------------

        //! OTHER ----------------------------------------------------------------------------------------------------

        // get the weight matrix (FOR TESTING)
        Mat<double> const& get_weights() const {return _weights;}

        // get local output (FOR TESTING)
        Vector<double> const& get_local_output() const {return _local_output;}

        //! ---------------------------------------------------------------------------------------------------------
    private:
        // NOTE: empty braces call default constructor for that class ( at least i hope it does)
        // activation function
        activ_func f {};

        // input shape
        Dims _in {0, 0};

        // output shape
        Dims _out {0,0};

        // locally stored input
        Vector<double> _local_input {};

        // locally stored output Y = Wx + B
        Vector<double> _local_output {};

        // weight matrix W
        Mat<double> _weights {};

        // bias vector
        Vector<double> _biases {};

        // locally stored gradients dL/dW
        Mat<double> _dLdW {};

        // locally stored gradients dL/dB
        Vector<double> _dLdB {};

};


#include "linear_impl.hxx"
#endif //ANN_LINEAR_HXX
