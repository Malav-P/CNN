//
// Created by malav on 7/15/2022.
//

#ifndef ANN_CROSS_ENTROPY_HXX
#define ANN_CROSS_ENTROPY_HXX

namespace CNN {

    class CrossEntropy {

    public:

        double loss(Array<double> &output, Array<double> &target) {
            // assert that number of elements in other vector and this vector are equal
            assert(output.getsize() == target.getsize());

            // initialize return variable
            double loss = 0;

            // compute loss
            for (size_t i = 0; i < output.getsize(); i++) { loss -= target[{0,i}] * log(output[{0,i}]); }

            // return result
            return loss;
        }

        // TODO
        Array<double> grad(Array<double> &output, Array<double> &target) { return (target).edivide(output, -1); }

    private:

    };

}
#endif //ANN_CROSS_ENTROPY_HXX
