//
// Created by malav on 5/3/2022.
//

#ifndef ANN_RELU_HXX
#define ANN_RELU_HXX

namespace CNN {

    class RelU : public Layer {
    public:

        // default constructor shouldn't exist;
        RelU() = delete;

        // default constructor
        RelU(size_t input_width, size_t input_height, size_t input_depth, double Alpha=0.0);
        //! BOOST::APPLY_VISITOR FUNCTIONS ---------------------------------------------------------------------------

        // send vector forward through this layer
        void Forward(Array<double> &input, Array<double> &output) override;

        // send vector backwards through layer, computing gradients and input error dLdX
        void Backward(Array<double> &dLdY, Array<double> &dLdX) override;

        // update the weights and biases according to their gradients
        template<typename Optimizer>
        void Update_Params(Optimizer *optimizer,
                           size_t normalizer) {/* nothing to do, no parameters to be learned in this layer*/}

        //! ----------------------------------------------------------------------------------------------------------

        double const &get_leaky_param() const { return alpha; }

    private:

        // leaky reLU parameter
        double alpha{0};

        // local input to layer
        Array<double> _local_input;

        // apply function to input
        double func(double input);

        // apply derivative to input
        double deriv(double input);
    };


}
#include "relU_impl.hxx"
#endif //ANN_RELU_HXX
