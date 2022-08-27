//
// Created by malav on 5/3/2022.
//

#ifndef ANN_RELU_HXX
#define ANN_RELU_HXX

#include "../lin_alg/vector.hxx"

class RelU {
    public:

        // default constructor
        RelU() = default;

        //! constructor, alpha =/= 0 implies a leaky relU unit, alpha usually greater than 0
        explicit RelU(double Alpha):
        alpha(Alpha)
        {
            if (Alpha < 0) {alpha = 0;} //! alpha cannot be negative
        }

        //! apply function to input
        double func(double input)
        {
            if (input < 0) {return alpha*input;}
            else {return input;}
        }

        void func(const Vector<double>& input, Vector<double>& output)
        {
            assert(input.get_len() == output.get_len());
            for (size_t i = 0; i<input.get_len() ; i++) {output[i] = func(input[i]);}
        }

        //! apply derivative to input
        double deriv(double input)
        {
            if (input < 0) {return alpha;}
            else {return 1;}
        }

        void deriv(const Vector<double>& input, Vector<double>& output)
        {
            assert(input.get_len() == output.get_len());
            for (size_t i = 0; i<input.get_len(); i++) {output[i] = deriv(input[i]);}
        }


    private:

        //! leaky reLU parameter
        double alpha {0};

};

#endif //ANN_RELU_HXX
