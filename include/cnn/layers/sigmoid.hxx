//
// Created by malav on 9/25/2022.
//

#ifndef CNN_SIGMOID_HXX
#define CNN_SIGMOID_HXX

namespace CNN {


    class Sigmoid : public Layer {
    public:

        // default constructor shouldnt exist
        Sigmoid() = delete;

        // default constructor
        Sigmoid(size_t input_width, size_t input_height, size_t input_depth);

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

    private:
        // local input to layer
        Array<double> _local_input{};

        // apply function to input
        double func(double input);

        // apply derivative to input
        double deriv(double input);
    };


}

#include "./sigmoid_impl.hxx"
#endif //CNN_SIGMOID_HXX
