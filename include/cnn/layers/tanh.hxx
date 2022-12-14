//
// Created by malav on 9/25/2022.
//

#ifndef CNN_TANH_HXX
#define CNN_TANH_HXX


namespace CNN {

    class Tanh : public Layer {
    public:

        // default constructor shouldnt exist
        Tanh() = delete;

        // default constructor
        Tanh(size_t input_width, size_t input_height, size_t input_depth);

        //! BOOST::APPLY_VISITOR FUNCTIONS ---------------------------------------------------------------------------

        // send vector forward through this layer
        void Forward(Vector<double> &input, Vector<double> &output) override;

        // send vector backwards through layer, computing gradients and input error dLdX
        void Backward(Vector<double> &dLdY, Vector<double> &dLdX) override;

        // update the weights and biases according to their gradients
        template<typename Optimizer>
        void Update_Params(Optimizer *optimizer,
                           size_t normalizer) {/* nothing to do, no parameters to be learned in this layer*/}
        //! ----------------------------------------------------------------------------------------------------------

    private:

        // local input to layer
        Vector<double> _local_input;

        // apply function to input
        double func(double input);

        // apply derivative to input
        double deriv(double input);
    };

}

#include "tanh_impl.hxx"
#endif //CNN_TANH_HXX
