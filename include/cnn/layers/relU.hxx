//
// Created by malav on 5/3/2022.
//

#ifndef ANN_RELU_HXX
#define ANN_RELU_HXX

namespace CNN {

    /**
     * An implementation of the RelU activation function y = max(0,x).
     *
     * @note we have added functionality for a leaky RelU parameter which defaults to zero.
     * @see [Leaky RelU](https://paperswithcode.com/method/leaky-relu)
     */
    class RelU : public Layer {
    public:

        /**
         * Default constructor should not exist, only the defined constructor can be called
         */
        RelU() = delete;

        /**
         * Constructor for the RelU class
         *
         * @param input_width input width of image
         * @param input_height input height of image
         * @param input_depth input depth (also referred to as number of channels)
         * @param Alpha leaky parameters, defaults to zero
         *
         * @note some models exhibit vanishing gradient when 'Alpha' is set to zero. Consider setting
         * 'Alpha' = 0.1 in this case.
         */
        RelU(size_t input_width, size_t input_height, size_t input_depth, double Alpha=0.0);

        /**
         * Propagate data forward through this layer
         * @param input input image
         * @param output output where results of RelU on input will be stored
         */
        void Forward(Array<double> &input, Array<double> &output) override;

        /**
         * Propagate gradients backward through this layer.
         *
         * There are no weights or biases in the RelU layer, so we back propagate the gradient wrt output to obtain the
         * gradient wrt the input
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

        /**
         * Utility function to access the leaky relu parameter
         *
         * @return const reference to the leaky relu parameter
         */
        double const &get_leaky_param() const { return alpha; }

    private:

        /// leaky reLU parameter
        double alpha{0};

        /// local input to layer
        Array<double> _local_input;

        /// apply function to input
        double func(double input);

        /// apply derivative to input
        double deriv(double input);
    };


}
#include "relU_impl.hxx"
#endif //ANN_RELU_HXX
