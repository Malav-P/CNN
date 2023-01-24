//
// Created by malav on 7/15/2022.
//

#ifndef ANN_MEAN_SQUARE_ERROR_HXX
#define ANN_MEAN_SQUARE_ERROR_HXX

namespace CNN {


    class MSE {

    public:

        double loss(Array<double> &output, Array<double> &target) {
            // assert that number of elements in other vector and this vector are equal
            assert(output.getsize() == target.getsize());

            // initialize return variable
            double loss = 0;

            // compute loss
            for (size_t i = 0; i < output.getsize(); i++) {
                loss += 0.5 * (output[{0,i}] - target[{0,i}]) * (output[{0,i}] - target[{0,i}]);
            }

            // return result
            return loss;
        }

        Array<double> grad(Array<double> &output, Array<double> &target) { return output - target; }

    private:

    };


}
#endif //ANN_MEAN_SQUARE_ERROR_HXX
