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
        Linear(size_t in_size, size_t out_size, double *weights = nullptr, double* biases = nullptr);


        //! -----------------------------------------------------------------------------------------------------------

        //! BOOST::APPLY_VISITOR FUNCTIONS ---------------------------------------------------------------------------

        // send vector forward through this layer
        void Forward(Array<double> &input, Array<double> &output) override;

        // send vector backwards through layer, computing gradients and input error dLdX
        void Backward(Array<double> &dLdY, Array<double> &dLdX) override;

        // update the weights and biases according to their gradients
        template<typename Optimizer>
        void Update_Params(Optimizer *optimizer, size_t normalizer);

        //! ----------------------------------------------------------------------------------------------------------

        //! OTHER ----------------------------------------------------------------------------------------------------

        // get the weight matrix
        Array<double> const &get_weights() const { return _weights; }

        // get the biases
        Array<double> const &get_biases() const { return _biases; }


        //! ---------------------------------------------------------------------------------------------------------
    private:
        // NOTE: empty braces call default constructor for that class ( at least i hope it does)

        // locally stored input
        Array<double> _local_input{};

        // weight matrix W
        Array<double> _weights{};

        // bias vector
        Array<double> _biases{};

        // locally stored gradients dL/dW
        Array<double> _dLdW{};

        // locally stored gradients dL/dB
        Array<double> _dLdB{};

    };

}
#include "linear_impl.hxx"
#endif //ANN_LINEAR_HXX
