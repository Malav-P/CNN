//
// Created by malav on 9/25/2022.
//

#ifndef CNN_SIGMOID_HXX
#define CNN_SIGMOID_HXX

namespace CNN {

    /**
     * An implementation of the Sigmoid activation function y = 1 / 1 - exp(-x).
     */
    class Sigmoid : public Layer {
    public:

        /**
         * Default constructor should not exist, only the defined constructor can be called
         */
        Sigmoid() = delete;

        /**
         * Constructor for Sigmoid class
         * @param input_width input width of data
         * @param input_height input height of data
         * @param input_depth input depth (also referred to as number of channels) of data
         */
        Sigmoid(size_t input_width, size_t input_height, size_t input_depth);

        /**
         * Propagate data forward through this layer
         * @param input input image
         * @param output output where results of Sigmoid on input will be stored
         */
        void Forward(Array<double> &input, Array<double> &output) override;

        /**
         * Propagate gradients backwards through layer and compute loss gradient wrt input
         *
         * @param dLdY loss gradient wrt output
         * @param dLdX where the loss gradient wrt input will be stored
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
        void Update_Params(Optimizer *optimizer, size_t normalizer) {/* nothing to do, no parameters to be learned in this layer*/}
        //! ----------------------------------------------------------------------------------------------------------

    private:
        /// local input to layer
        Array<double> _local_input{};

        /// apply function to input
        double func(double input);

        /// apply derivative to input
        double deriv(double input);
    };


}

#include "./sigmoid_impl.hxx"
#endif //CNN_SIGMOID_HXX
