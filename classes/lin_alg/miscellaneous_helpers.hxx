//
// Created by malav on 9/28/2022.
//

#ifndef CNN_MISCELLANEOUS_HELPERS_HXX
#define CNN_MISCELLANEOUS_HELPERS_HXX

#include "data_types.hxx"

// cubify an array of matrices
template<typename T>
Cuboid<T> cubify(std::vector<Mat<T>> &matrix_list)
{
    //TODO : ensure that the dimensions of all matrices are the same before we cubify!

    // initialize return variable
    Cuboid<T> obj(matrix_list[0].get_rows(), matrix_list[0].get_cols(), matrix_list.size());

    // do cubify operation
    for (size_t i = 0; i < obj.get_rows() ; i++){ for (size_t j = 0 ; j < obj.get_cols() ; j++) { for (size_t k = 0; k< obj.get_depth(); k++){
                obj(i,j,k) = matrix_list[k](i,j);
            }}}

    // return object
    return obj;
}

template<typename T>
Cuboid<T> cubify(Mat<T>& matrix, size_t n)
{

    // initialize return variable
    Cuboid<T> obj(matrix.get_rows(), matrix.get_cols(), n);

    // do cubify operation
    for (size_t i = 0; i < obj._rows ; i++){ for (size_t j = 0 ; j < obj._cols ; j++) { for (size_t k = 0; k< obj._depth; k++){
                obj(i,j,k) = matrix(i,j);
            }}}

    // return object
    return obj;
}

template<typename T>
std::vector<Mat<T>> cube_to_matarray(Cuboid<T>& cube)
{
    // intialize return variable
    std::vector<Mat<T>> obj_list(cube.get_depth());

    // do operation
    for (size_t k = 0; k< cube.get_depth(); k++)
    {
        Mat<double> mat(cube.get_rows(), cube.get_cols());
        for (size_t i = 0; i < cube.get_rows(); i++)
        {
            for (size_t j = 0; j < cube.get_cols(); j++)
            {
                mat(i,j) = cube(i,j,k);
            }
        }
        obj_list[k] = mat;
    }

    // return object
    return obj_list;
}

#endif //CNN_MISCELLANEOUS_HELPERS_HXX
