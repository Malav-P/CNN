//
// Created by malav on 5/2/2022.
//

#ifndef ANN_LINEAR_HXX
#define ANN_LINEAR_HXX

namespace CNN {


//! a linear class which applies a transformation of the form
//! Y = Wx + b. W is a weight matrix and b is a bias.
    class Linear : public Layer {
    public:

        //! CONSTRUCTORS, DESTRUCTORS, MOVE CONSTRUCTORS, ASSIGNMENT OPERATORS, ETC ------------------------------------

        // default constructor should not exist
        Linear() = delete;

        // construct Linear Layer with specified input and output sizes
        Linear(size_t in_size, size_t out_size, double *weights = nullptr);


        //! -----------------------------------------------------------------------------------------------------------

        //! BOOST::APPLY_VISITOR FUNCTIONS ---------------------------------------------------------------------------

        // send vector forward through this layer
        void Forward(Vector<double> &input, Vector<double> &output) override;

        // send vector backwards through layer, computing gradients and input error dLdX
        void Backward(Vector<double> &dLdY, Vector<double> &dLdX) override;

        // update the weights and biases according to their gradients
        template<typename Optimizer>
        void Update_Params(Optimizer *optimizer, size_t normalizer);

        //! ----------------------------------------------------------------------------------------------------------

        //! OTHER ----------------------------------------------------------------------------------------------------

        // get the weight matrix
        Mat<double> const &get_weights() const { return _weights; }

        // get the biases
        Vector<double> const &get_biases() const { return _biases; }


        //! ---------------------------------------------------------------------------------------------------------
    private:
        // NOTE: empty braces call default constructor for that class ( at least i hope it does)

        // locally stored input
        Vector<double> _local_input{};

        // weight matrix W
        Mat<double> _weights{};

        // bias vector
        Vector<double> _biases{};

        // locally stored gradients dL/dW
        Mat<double> _dLdW{};

        // locally stored gradients dL/dB
        Vector<double> _dLdB{};

    };

}
#include "linear_impl.hxx"
#endif //ANN_LINEAR_HXX
