//
// Created by malav on 7/6/2022.
//

#ifndef ANN_DATASET_HXX
#define ANN_DATASET_HXX

#include "../prereqs.hxx"
#include <vector>
#include "../lin_alg/data_types.hxx"

using Vector_Pair = std::pair<Vector<double>, Vector<double>>;

class DataSet
{
    public:

    // constructor
    explicit DataSet(Dims shape)
    : shape(shape)
    {}

    // the data
    std::vector<Vector_Pair> datapoints;

    // size of training set, columns = number of samples, rows = number of elements in each sample
    Dims shape;

};
#endif //ANN_DATASET_HXX
