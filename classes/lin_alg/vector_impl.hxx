//
// Created by malav on 5/3/2022.
//

#ifndef ANN_VECTOR_CPP
#define ANN_VECTOR_CPP

#include "vector.hxx"
#include "mat.hxx"

//!  constructor ---------------------------------------------------------------------------
template<typename T>
__host__ __device__
Vector<T>::Vector(size_t n, T *arr)
: _length(n)
, _data(new T[n]{0})
{
    if (arr != nullptr)
    {
        // TODO : currently no way to ensure arr has n elements in it
        std::memcpy(_data, arr, sizeof(T) * n);
//        for (size_t i = 0; i< n ; i++)
//        {
//            _data[i] = arr[i];
//        }
    }
}
//! ---------------------------------------------------------------------------

//! copy constructor ----------------------------------------------------------------------
template<typename T>
__host__ __device__
Vector<T>::Vector(const Vector &other)
: _length(other._length),
  _data(new T[_length])
{ std::memcpy(_data, other._data, sizeof(T) * _length); }
//! ----------------------------------------------------------------------

//! move constructor ----------------------------------------------------
template<typename T>
__host__ __device__
Vector<T>::Vector(Vector<T> &&other) noexcept
        : _length(other._length)
        , _data(other._data)
{
    other._data = nullptr;
}
//! ---------------------------------------------------------------------

//! copy assignment operator---------------------------------------------------------------------
template<typename T>
Vector<T> &Vector<T>::operator=(const Vector<T> &rhs) {
    // check for self assignment
    if(this != &rhs)
    {
        // make copy of input
        Vector<T> tmp(rhs);

        // no need to reallocate or swap '_length' if same size
        if (_length != tmp._length)
        {
            std::swap(_length, tmp._length);
            delete[] _data;
            _data = new T[_length];
        }

        // copy lin_alg to new pointer
        // equivalent to   for (size_t k = 0; k < _length; k++) { _data[k] = tmp._data[k]; }
        std::memcpy(_data, tmp._data, sizeof(T) * _length);
    }

    //  return result
    return (*this);
}
//! ---------------------------------------------------------------------

//! move assignment operator---------------------------------------------------------------------
template<typename T>
Vector<T> &Vector<T>::operator=(Vector<T> &&other) noexcept {

    // check for self-assignment
    if (this != &other)
    {

        // Free the existing resource.
        delete[] _data;

        // Copy the _data pointer and its _length from the
        // source object.
        _data   = other._data;
        _length = other._length;

        // Release the _data pointer from the source object so that
        // the destructor does not free the memory multiple times.
        other._data = nullptr;
    }

    // return result
    return *this;
}
//! ---------------------------------------------------------------------

//! += operator ---------------------------------------------------------
template<typename T>
Vector<T> &Vector<T>::operator+=(const Vector<T> &other)
{
    // if this vector has just been instantiated it will have no data and no length, thus we must use
    // the copy assignment operator for this case.
    if (_length == 0)
    {
        (*this) = other;
        return (*this);
    }

    // assert that RHS and LHS have same dimensions
    assert(_length == other._length);

    // do += operation
    for (size_t i = 0; i < _length ; i++) { _data[i] += other[i];}

    // return the current object
    return (*this);
}
//! ----------------------------------------------------------------------
//! ---------------------------------------------------------------------

//! indexing operator ---------------------------------------------------------------------
template<typename T>
__host__ __device__ T &Vector<T>::operator[](size_t idx) { return _data[idx]; }
//! ---------------------------------------------------------------------

//! const indexing operator ---------------------------------------------------------------------
template<typename T>
__host__ __device__ const T &Vector<T>::operator[](size_t idx) const { return _data[idx]; }
//! ---------------------------------------------------------------------

//! multiply operator (matrix)---------------------------------------------------------------------
template<typename T>
Vector<T> Vector<T>::operator*(Mat<T> &other)
{
    // assert vector length equal number of rows of matrix
    assert(_length == other._rows);

    // initialize return variable
    Vector<T> obj(other._cols);

//    // helper variable
//    Vector<T> col(_length);

    // do multiplication operation
    for (size_t col = 0; col < other._cols; col++)
    {
        obj[col] = 0;
        for (size_t row = 0; row < other._rows; row++)
        {
            obj[col] += other(row, col) * _data[row];
        }
    }

    // return result
    return obj;
}
//! ---------------------------------------------------------------------

//! multiply operator (scalar) ------------------------------------------
template<typename T>
Vector<T> Vector<T>::operator*(double c)
{

    // initialize return variable
    Vector<T> obj(_length);

    // do multiplication operation
    for (size_t i = 0; i < _length; i++)
    { obj[i] = _data[i] * c; }


    // return result
    return obj;
}

//! ----------------------------------------------------------------

//! multiply operator (vector)---------------------------------------------------------------------
template<typename T>
Mat<T> Vector<T>::operator*(const Vector<T> &other)
{

    // initialize return variable
    Mat<T> obj(_length, other._length);

    // do multiplication operation
    for (size_t i = 0; i < _length; i++) {for(size_t j = 0; j < other._length; j++)
        { obj(i,j) = _data[i] * other._data[j]; }
    }

    // return result
    return obj;
}
//! ---------------------------------------------------------------------

//! + operator ---------------------------------------------------------------------
template<typename T>
Vector<T> Vector<T>::operator+(const Vector<T> &other)
{
    // check that vectors are of equal length
    assert(_length == other._length);

    // initialize return variable
    Vector<T> obj(_length);

    // do addition operation
    for (size_t i = 0; i < _length; i++)
    { obj[i] = _data[i] + other._data[i]; }

    // return result
    return obj;
}
//! ---------------------------------------------------------------------
//! - operator ---------------------------------------------------------------------
template<typename T>
Vector<T> Vector<T>::operator-(const Vector<T> &other)
{
    // check that vectors are of equal length
    assert(_length == other._length);

    // initialize return variable
    Vector<T> obj(_length);

    // do subtraction operation
    for (size_t i = 0; i < _length; i++)
    { obj[i] = _data[i] - other._data[i]; }

    // return result
    return obj;
}
//! ---------------------------------------------------------------------
//! dot product between two vectors ---------------------------------------------------------------------
template<typename T>
T Vector<T>::dot(const Vector<T> &other)
{
    // vector lengths must be equal
    assert(_length == other._length);

    // initialize return variable
    T answer = 0;

    // do dot operation
    for (size_t i = 0; i < _length; i++) { answer += (_data[i] * (other._data)[i]);}

    // return result
    return answer;
}
//! ---------------------------------------------------------------------

//! fill vector with value ---------------------------------------------------------------------
template<typename T>
void Vector<T>::fill(T fill) { for (size_t i = 0; i < _length; i++) { _data[i] = fill;} }
//! ---------------------------------------------------------------------

//! reshape vector into matrix---------------------------------------------------------------------
template<typename T>
Mat<T> Vector<T>::reshape(size_t n_rows, size_t n_cols)
{
    // assert that number of elements in matrix and vector are equal
    assert(n_rows*n_cols == _length);

    // initialize return variable
    Mat<T> obj(n_rows, n_cols);

    // do reshape operation
    std::memcpy(obj._data, _data, sizeof(T) * _length);

    // return result
    return obj;
}
//! ---------------------------------------------------------------------

//! element-wise product of two vectors---------------------------------------------------------------------
template<typename T>
Vector<T> Vector<T>::eprod(const Vector<T> &other) const
{
    // assert that number of elements in other vector and this vector are equal
    assert(other._length == _length);

    // initialize return variable
    Vector<T> obj(_length);

    // do element-wise multiplication operation
    for (size_t i = 0; i < _length; i++) { obj._data[i] = _data[i] * other._data[i]; }

    // return result
    return obj;
}
//! ---------------------------------------------------------------------

//! element-wise quotient of two vectors---------------------------------------------------------------------

template<typename T>
Vector<T> Vector<T>::edivide(const Vector<T> &other)
{
    // assert that number of elements in other vector and this vector are equal
    assert(other._length == _length);

    // initialize return variable
    Vector<T> obj(_length);

    // do element-wise multiplication operation
    for (size_t i = 0; i < _length; i++) { obj._data[i] = _data[i] / other._data[i]; }

    // return result
    return obj;
}
//! ---------------------------------------------------------------------

//!  print vector --------------------------------------------------------------------
template<typename T>
void Vector<T>::print() const
{
    for (size_t i = 0; i < _length; i++)
    {
        std::cout << _data[i] << ", ";
    }

    std::cout << "\n";
}

template<typename T>
void Vector<T>::operator*=(double c)
{
    for (size_t i = 0; i < _length; i++)
    {
        _data[i] *= c;
    }
}

//! -----------------------------------------------------------------------------------

template<typename T>
Vector<T> Vector<T>::merge(const Vector<T> &other)
{

    // initialize return variable
    Vector<T> obj(_length + other._length);

    // copy over data
    std::memcpy(obj._data, _data, sizeof(T) * _length);
    std::memcpy(obj._data + _length, other._data, sizeof(T) * other._length);

    // return result
    return obj;
}

template<typename T>
T *Vector<T>::port_to_GPU()
{

    //! THIS FUNCTION RETURNS MALLOCED MEMORY, EVERY TIME IT IS CALLED, CUDAFREE MUST BE CALLED SOMEHWERE ELSE
    T* d_ptr;
    cudaMalloc(&d_ptr, _length*sizeof(T));
    cudaMemcpy( d_ptr, _data, _length*sizeof(T), cudaMemcpyHostToDevice);

    return d_ptr;
}


#endif //ANN_VECTOR_CPP
