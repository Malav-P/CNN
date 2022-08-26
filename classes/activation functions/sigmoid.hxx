//
// Created by malav on 5/3/2022.
//

#ifndef ANN_SIGMOID_HXX
#define ANN_SIGMOID_HXX

#include "../lin_alg/vector.hxx"

class Sigmoid {
    public:

        //! default constructor
        Sigmoid() = default;

        //! apply function to input
        double func(double input) {return 1 / (1 + exp(-input));}

        void func(const Vector<double>& input, Vector<double>& output)
        {
            assert(input.get_len() == output.get_len());
            for (size_t i = 0; i<input.get_len() ; i++) {output[i] = func(input[i]);}
        }

        //! apply derivative to input
        double deriv(double input) {return func(input) * (1 - func(input));}

        void deriv(const Vector<double>& input, Vector<double>& output)
        {
            assert(input.get_len() == output.get_len());
            for (size_t i = 0; i<input.get_len(); i++) {output[i] = deriv(input[i]);}
        }

};

#endif //ANN_SIGMOID_HXX
