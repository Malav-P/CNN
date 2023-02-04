//
// Created by malav on 5/2/2022.
//

#ifndef ANN_LINEAR_HXX
#define ANN_LINEAR_HXX

namespace CNN {



/**
 * An implementation of a standard densely connected layer.
 *
 * This layer performs a vector operation y = Ax + b, with \n A being a matrix of locally stored weights, \n b
 * a column vector of locally stored biases,  \n x and y being input and output columns vectors, respectively
 */
    class Linear : public Layer {
    public:

        /**
         * Default constructor is explicitly deleted, it should never be called
         */
        Linear() = delete;

        /**
         * Constructor for the Linear class
         *
         * @param in_size input vector size
         * @param out_size output vector size
         * @param weights (optional) a pointer to weights, usually loaded from a file from previously trained models
         * @param biases (optional) a pointer to biases, usually loaded from a file from previously trained models
         *
         * @note it is ill-advised to provide your own pointers for weights and biases when initializing the network
         * for training. Improperly distributed weights and biases may lead to exploding/vanishing gradient problems
         */
        Linear(size_t in_size, size_t out_size, double *weights = nullptr, double* biases = nullptr);


        /**
         * Propagate input data through the layer and store results in the output
         *
         * @param input the input data
         * @param output the output, where the results of the operation are stored
         */
        void Forward(Array<double> &input, Array<double> &output) override;

        /**
         * Propagate gradients backwards through layer, storing the gradients for the weights and biases
         *
         * @param dLdY the loss gradient with respect to the output
         * @param dLdX where the loss gradient with respect to the inputs will be stored
         */
        void Backward(Array<double> &dLdY, Array<double> &dLdX) override;

        /**
         * Update the parameters in the layer
         *
         * @tparam Optimizer the optimizer used for updating the weights (adam, rmsprop, sgd, etc.). Optimizers
         * will be defined as classes.
         * @param optimizer pointer to an Optimizer class containing the necessary parameters
         * @param normalizer a constant used to normalize the update, usually equal to the batch size
         */
        template<typename Optimizer>
        void Update_Params(Optimizer *optimizer, size_t normalizer);

        /**
         * Utility function to access the weights (read-only)
         * @return a const reference to the weights in the layer
         */
        Array<double> const &get_weights() const { return _weights; }

        /**
         * Utility function to access the biases (read-only)
         * @return a const reference to the biases in the layer
         */
        Array<double> const &get_biases() const { return _biases; }

    private:

        /// locally stored input
        Array<double> _local_input{};

        /// weight matrix
        Array<double> _weights{};

        /// bias vector
        Array<double> _biases{};

        /// locally stored gradients dL/dW
        Array<double> _dLdW{};

        /// locally stored gradients dL/dB
        Array<double> _dLdB{};

    };

}
#include "linear_impl.hxx"
#endif //ANN_LINEAR_HXX
