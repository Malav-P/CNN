//
// Created by malav on 5/3/2022.
//

#ifndef ANN_TANH_HXX
#define ANN_TANH_HXX

#include "../lin_alg/vector.hxx"

class Tanh {
    public:

        // default constructor
        Tanh() = default;

        //! apply function to input
        double func(double input) {return (2 / (1 + exp(-2*input))) - 1;}

        void func(const Vector<double>& input, Vector<double>& output)
        {
            assert(input.get_len() == output.get_len());
            for (size_t i = 0; i<input.get_len() ; i++) {output[i] = func(input[i]);}
        }

        //! apply derivative to input
        double deriv(double input) {return 1 - func(input)* func(input);}

        void deriv(const Vector<double>& input, Vector<double>& output)
        {
            assert(input.get_len() == output.get_len());
            for (size_t i = 0; i<input.get_len(); i++) {output[i] = deriv(input[i]);}
        }

};

#endif //ANN_TANH_HXX
