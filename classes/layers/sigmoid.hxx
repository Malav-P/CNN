//
// Created by malav on 9/25/2022.
//

#ifndef CNN_SIGMOID_HXX
#define CNN_SIGMOID_HXX


class Sigmoid {
    public:

        // default constructor shouldnt exist
        Sigmoid() = delete;

        // default constructor
        Sigmoid(size_t input_width, size_t input_height);

        //! BOOST::APPLY_VISITOR FUNCTIONS ---------------------------------------------------------------------------

        // send vector forward through this layer
        void Forward(const Vector<double>& input, Vector<double>& output);

        // send vector backwards through layer, computing gradients and input error dLdX
        void Backward(Vector<double>& dLdY, Vector<double>& dLdX);

        // update the weights and biases according to their gradients
        template<typename Optimizer>
        void Update_Params(Optimizer* optimizer, size_t normalizer){/* nothing to do, no parameters to be learned in this layer*/}

        // return out shape of layer
        Dims3 const& out_shape() const {return _out;}

        // return in shape of layer
        Dims3 const& in_shape() const {return _in;}
        //! ----------------------------------------------------------------------------------------------------------

    private:


        // local input to layer
        Vector<double> _local_input {};

        // input shape of layer
        Dims3 _in {0,0,1};

        // output shape of layer

        Dims3 _out{0,0,1};

        // apply function to input
        double func(double input);

        // apply derivative to input
        double deriv(double input);
};




#include "./sigmoid_impl.hxx"
#endif //CNN_SIGMOID_HXX
