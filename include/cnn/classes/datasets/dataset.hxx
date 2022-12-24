//
// Created by malav on 7/6/2022.
//

// This class defines a container for storing datapoints and their labels. The datapoint and its label are
// stored as a pair of Vector<double>.

#ifndef ANN_DATASET_HXX
#define ANN_DATASET_HXX

#include "../lin_alg/data_types.hxx"

// <datapoint, label>
using Vector_Pair = std::pair<CNN::Vector<double>, CNN::Vector<double>>;

namespace CNN {

    class DataSet {
    public:

        // constructor
        explicit DataSet(Dims shape)
                : shape(shape) {}

        // the data
        std::vector<Vector_Pair> datapoints;

        // size of training set, columns = number of samples, rows = number of elements in each sample
        Dims shape;

    };

}
#endif //ANN_DATASET_HXX
