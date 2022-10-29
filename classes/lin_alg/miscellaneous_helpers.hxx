//
// Created by malav on 9/28/2022.
//

#ifndef CNN_MISCELLANEOUS_HELPERS_HXX
#define CNN_MISCELLANEOUS_HELPERS_HXX

#include "data_types.hxx"

// cubify an array of matrices
template<typename T>
Cuboid<T> cubify(Mat<T>* matrix_list, size_t len)
{
    //TODO : ensure that the dimensions of all matrices are the same before we cubify!

    // initialize return variable
    Cuboid<T> obj(matrix_list[0].get_rows(), matrix_list[0].get_cols(), len);

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
__host__ void port_to_GPU(Cuboid<T> *& d_cuboids, Cuboid<T>*& cuboids, T**& d_arrays, size_t N)
{

    // length of device array
    int d_data_len = cuboids[0].get_depth() * cuboids[0].get_rows() * cuboids[0].get_cols();

    // Allocate device struct array
    cudaMalloc( (void**)&d_cuboids, N*sizeof(Cuboid<T>));

    // copy over data from pool_vector to d_poolvec
    for (size_t i = 0; i< N; i++)
    {
        // host struct
        Cuboid<T>* elem = &(cuboids[i]);

        // device struct
        Cuboid<T>* d_elem = &(d_cuboids[i]);

        // copy struct from host to device
        cudaMemcpy(d_elem, elem, sizeof(Cuboid<T>), cudaMemcpyHostToDevice);

        // Allocate device pointer
        cudaMalloc((void**)&(d_arrays[i]), d_data_len*sizeof(T));

        // copy pointer content from host to device
        cudaMemcpy((d_arrays[i]), elem->_data, d_data_len*sizeof(T), cudaMemcpyHostToDevice);


        cudaMemcpy(&(d_elem->_data), &(d_arrays[i]), sizeof(T*), cudaMemcpyHostToDevice);
    }
}

template<typename T>
__host__ void copy_to_GPU(Cuboid<T> *& d_cuboids, Cuboid<T>*& cuboids, T**& d_arrays, size_t N)
{

    // length of device array
    int d_data_len = cuboids[0].get_depth() * cuboids[0].get_rows() * cuboids[0].get_cols();

    // copy over data from pool_vector to d_poolvec
    for (size_t i = 0; i< N; i++)
    {
        // host struct
        Cuboid<T>* elem = &(cuboids[i]);

        // device struct
        Cuboid<T>* d_elem = &(d_cuboids[i]);

        // copy struct from host to device
        cudaMemcpy(d_elem, elem, sizeof(Cuboid<T>), cudaMemcpyHostToDevice);

        // copy pointer content from host to device
        cudaMemcpy((d_arrays[i]), elem->_data, d_data_len*sizeof(T), cudaMemcpyHostToDevice);

        cudaMemcpy(&(d_elem->_data), &(d_arrays[i]), sizeof(T*), cudaMemcpyHostToDevice);
    }
}

template<typename T>
__host__ void port_to_GPU(Mat<T> *& d_mats, Mat<T>*& mats, T**& d_arrays, size_t N)
{

    // length of device array
    int d_data_len =  mats[0].get_rows() * mats[0].get_cols();

    // Allocate device struct array
    cudaMalloc( (void**)&d_mats, N*sizeof(Mat<T>));

    // copy over data from pool_vector to d_poolvec
    for (size_t i = 0; i< N; i++)
    {
        // host struct
        Mat<T>* elem = &(mats[i]);

        // device struct
        Mat<T>* d_elem = &(d_mats[i]);

        // copy struct from host to device
        cudaMemcpy(d_elem, elem, sizeof(Mat<T>), cudaMemcpyHostToDevice);

        // Allocate device pointer
        cudaMalloc((void**)&(d_arrays[i]), d_data_len*sizeof(T));

        // copy pointer content from host to device
        cudaMemcpy((d_arrays[i]), elem->_data, d_data_len*sizeof(T), cudaMemcpyHostToDevice);


        cudaMemcpy(&(d_elem->_data), &(d_arrays[i]), sizeof(T*), cudaMemcpyHostToDevice);
    }
}

template<typename T>
__host__ void copy_to_GPU(Mat<T> *& d_mats, Mat<T>*& mats, T**& d_arrays, size_t N)
{

    // length of device array
    int d_data_len =  mats[0].get_rows() * mats[0].get_cols();

    // copy over data from pool_vector to d_poolvec
    for (size_t i = 0; i< N; i++)
    {
        // host struct
        Mat<T>* elem = &(mats[i]);

        // device struct
        Mat<T>* d_elem = &(d_mats[i]);

        // copy struct from host to device
        cudaMemcpy(d_elem, elem, sizeof(Mat<T>), cudaMemcpyHostToDevice);

        // copy pointer content from host to device, there is an error here
        cudaMemcpy((d_arrays[i]), elem->_data, d_data_len*sizeof(T), cudaMemcpyHostToDevice);

        cudaMemcpy(&(d_elem->_data), &(d_arrays[i]), sizeof(T*), cudaMemcpyHostToDevice);
    }
}

#endif //CNN_MISCELLANEOUS_HELPERS_HXX
