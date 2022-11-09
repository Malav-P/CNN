//
// Created by malav on 5/3/2022.
//

#ifndef ANN_RELU_HXX
#define ANN_RELU_HXX

class RelU : public Layer {
    public:

        // default constructor shouldn't exist;
        RelU() = delete;

        // default constructor
        RelU(size_t input_width, size_t input_height, size_t input_depth);

        // constructor, alpha =/= 0 implies a leaky relU unit, alpha usually greater than 0
        RelU(double Alpha, size_t input_width, size_t input_height, size_t input_depth);

        //! BOOST::APPLY_VISITOR FUNCTIONS ---------------------------------------------------------------------------

        // send vector forward through this layer
        void Forward(Vector<double>& input, Vector<double>& output) override;

        // send vector backwards through layer, computing gradients and input error dLdX
        void Backward(Vector<double>& dLdY, Vector<double>& dLdX) override;

        // update the weights and biases according to their gradients
        template<typename Optimizer>
        void Update_Params(Optimizer* optimizer, size_t normalizer){/* nothing to do, no parameters to be learned in this layer*/}

        //! ----------------------------------------------------------------------------------------------------------

    private:

        // leaky reLU parameter
        double alpha {0};

        // local input to layer
        Vector<double> _local_input;

        // apply function to input
        double func(double input);

        // apply derivative to input
        double deriv(double input);
};



#include "relU_impl.hxx"
#endif //ANN_RELU_HXX
