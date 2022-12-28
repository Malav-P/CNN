//
// Created by malav on 9/28/2022.
//

#ifndef CNN_CUBOID_IMPL_HXX
#define CNN_CUBOID_IMPL_HXX

#include "cuboid.hxx"


namespace CNN {

//! constructor -------------------------------------------------------------------------
    template<typename T>
    Cuboid<T>::Cuboid(size_t side_len, T *arr)
            : _rows(side_len), _cols(side_len), _depth(side_len), _data(new T[side_len * side_len * side_len]{0}) {
        if (arr != nullptr) {
            // TODO : currently no way to ensure arr has side_len*side_len elements in it
            std::memcpy(_data, arr, sizeof(T) * _rows * _cols * _depth);
        }
    }
//! --------------------------------------------------------------------------------------

//! constructor --------------------------------------------------------------------------
    template<typename T>
    Cuboid<T>::Cuboid(size_t num_r, size_t num_c, size_t num_d, T *arr)
            : _rows(num_r), _cols(num_c), _depth(num_d), _data(new T[num_r * num_c * num_d]{0}) {
        if (arr != nullptr) {
            // TODO : currently no way to ensure arr has num_r*num_c elements in it
            std::memcpy(_data, arr, sizeof(T) * _rows * _cols * _depth);
        }
    }
//!--------------------------------------------------------------------------------------

//! copy constructor --------------------------------------------------------------------
    template<typename T>
    Cuboid<T>::Cuboid(const Cuboid<T> &other)
            : _rows(other._rows), _cols(other._cols), _depth(other._depth), _rot(other._rot),
              _data(new T[_rows * _cols * _depth]{0}) {
        std::memcpy(_data, other._data, sizeof(T) * _rows * _cols * _depth);
    }

//! ---------------------------------------------------------------------------------------
//! move constructor -------------------------------------------------------------
    template<typename T>
    Cuboid<T>::Cuboid(Cuboid<T> &&other) noexcept
            : _data(other._data), _rows(other._rows), _cols(other._cols), _depth(other._depth), _rot(other._rot) {
        // remove pointer to other data
        other._data = nullptr;
    }
//! -------------------------------------------------------------


//! copy assignment operator--------------------------------------------------------------------------------------
    template<typename T>
    Cuboid<T> &Cuboid<T>::operator=(const Cuboid<T> &rhs) {

        // check for self-assignment
        if (this != &rhs) {
            // make copy of input
            Cuboid<T> tmp(rhs);

            // swap values with tmp variable
            std::swap(_cols, tmp._cols);
            std::swap(_rows, tmp._rows);
            std::swap(_rot, tmp._rot);
            std::swap(_depth, tmp._depth);

            // no need to reallocate if same size
            if (_rows * _cols * _depth != tmp._cols * tmp._rows * tmp._depth) {
                delete[] _data;
                _data = new T[_rows * _cols * _depth];
            }

            // copy lin_alg to new pointer
            //  equivalent to   for (size_t k = 0; k < _rows * _cols; k++) { _data[k] = tmp._data[k]; }
            std::memcpy(_data, tmp._data, sizeof(T) * _rows * _cols * _depth);
        }

        // return result
        return (*this);
    }
//! -------------------------------------------------------------------------------------------------


//! ---------------------------------------------------------------------------
    template<typename T>
    Cuboid<T> &Cuboid<T>::operator=(Cuboid<T> &&other) noexcept {
        if (this != &other) {
            // Free the existing resource.
            delete[] _data;

            // Copy the _data pointer and its _length from the
            // source object.
            _data = other._data;
            _rows = other._rows;
            _cols = other._cols;
            _depth = other._depth;
            _rot = other._rot;

            // Release the _data pointer from the source object so that
            // the destructor does not free the memory multiple times.
            other._data = nullptr;
        }
        return *this;
    }
//! ---------------------------------------------------------------------------

//! += operator----------------------------------------------------------------
    template<typename T>
    Cuboid<T> &Cuboid<T>::operator+=(const Cuboid<T> &other) {
        // assert that RHS and LHS have same dimensions
        assert(_rows == other._rows && _cols == other._cols && _depth == other._depth);

        // do += operation
        for (size_t i = 0; i < _rows; i++) {
            for (size_t j = 0; j < _cols; j++) {
                for (size_t k = 0; k < _depth; k++) {
                    (*this)(i, j, k) += other(i, j, k);
                }
            }
        }

        // return the current object
        return (*this);
    }

//! ------------------------------------------------------------------------------
//! += operator----------------------------------------------------------------
    template<typename T>
    Cuboid<T> Cuboid<T>::operator+(const Cuboid<T> &rhs) {

        // assert that matrix dimensions are the same
        assert(_rows == rhs._rows && _cols == rhs._cols && _depth == rhs._depth);

        // initialize return variable
        Cuboid<T> obj(_rows, _cols, _depth);

        // do + operation
        for (size_t i = 0; i < _rows; i++) {
            for (size_t j = 0; j < _cols; j++) {
                for (size_t k = 0; k < _depth; k++) {
                    obj(i, j, k) = rhs(i, j, k) + (*this)(i, j, k);
                }
            }
        }

        // return the current object
        return obj;
    }

//! ------------------------------------------------------------------------------
//! cuboid index operator myObj(i,j,k) returns the i-j-kth element of cuboid ------------------
    template<typename T>
    T &Cuboid<T>::operator()(size_t i, size_t j, size_t k) { return _data[k * (_cols * _rows) + (i * _cols + j)]; }
//! ---------------------------------------------------------------------------------------

//! same as above but for const objects ---------------------------------------------------
    template<typename T>
    const T &Cuboid<T>::operator()(size_t i, size_t j, size_t k) const {
        return _data[k * (_cols * _rows) + (i * _cols + j)];
    }
//! ---------------------------------------------------------------------------------------

//! multiply cuboid with scalar --------------------------------------------------------
    template<typename T>
    Cuboid<T> Cuboid<T>::operator*(const double c) {

        // initialize return variable
        Cuboid<T> obj(_rows, _cols, _depth);

        // do * operation
        for (size_t i = 0; i < _rows; i++) {
            for (size_t j = 0; j < _cols; j++) {
                for (size_t k = 0; k < _depth; k++) {
                    obj(i, j, k) = c * (*this)(i, j, k);
                }
            }
        }

        // return the current object
        return obj;
    }
//! ----------------------------------------------------------------------------------------

//! compute dot product between two cuboids -----------------------------------------------
    template<typename T>
    T Cuboid<T>::dot(const Cuboid<T> &other) {
        // ensure matrices are same size
        assert((other._cols == _cols) && (other._rows == _rows) && (other._depth) == _depth);

        // initialize return variable
        T answer = 0;

        // do dot product
        for (size_t k = 0; k < _rows * _cols * _depth; k++) { answer += _data[k] * (other._data)[k]; }

        // return result
        return answer;
    }
//! -------------------------------------------------------------------------------------------

//! compute dot product between overlapping parts of cuboids ---------------------------------
    template<typename T>
    T Cuboid<T>::partial_dot(const Cuboid<T> &other, Dims3 p) {
        // starting indices must be within bounds of matrix
        assert(p.height < _rows && p.width < _cols && p.depth < _depth);

        // overlapping matrix must be within bounds of this matrix
        assert(p.height + other._rows <= _rows && p.width + other._cols <= _cols && p.depth + other._depth <= _depth);

        // initialize return variable;
        T answer = 0;

        // do partial dot operation
        for (size_t i = 0; i < other._rows; i++) {
            for (size_t j = 0; j < other._cols; j++) {
                for (size_t k = 0; k < other._depth; k++) {
                    answer += other(i, j, k) * (*this)(p.width + i, p.height + j, p.depth + k);
                }
            }
        }

        // return result
        return answer;
    }
//! -----------------------------------------------------------------------------------------

//! fill matrix with value -------------------------------------------------------------------
    template<typename T>
    void Cuboid<T>::fill(T t) { for (size_t i = 0; i < _cols * _rows * _depth; i++) { _data[i] = t; }}
//! ------------------------------------------------------------------------------------------

//! set rotation state of cuboid ------------------------------------------------------------
    template<typename T>
    void Cuboid<T>::set_rot(size_t n) {
        // rotation state can be either 0, 1, 2, or 3
        n %= 4;

        // need to rotate clockwise
        if (_rot < n) {
            size_t diff = n - _rot;
            for (size_t i = 0; i < diff; i++) { rotate_once(); } // clockwise rotation is good here
        }
            // need to rotate counterclockwise
        else if (_rot > n) {
            size_t diff = 4 - (_rot - n);
            for (size_t i = 0;
                 i < diff; i++) { rotate_once(); } // eventually change to rotate CCW, it is more efficient
        }
            // Nothing to do, _rot == n
        else {}
    }

//! ---------------------------------------------------------------------------------------------
//! flatten matrix into vector ------------------------------------------------------------------
    template<typename T>
    Vector<T> Cuboid<T>::flatten() {
        // initialize return variable
        Vector<T> obj(_rows *_cols
        *_depth);

        // do flatten operation
        for (size_t i = 0; i < _rows * _cols * _depth; i++) {
            std::memcpy(obj._data, _data, sizeof(T) * _rows * _cols * _depth);
        }

        // return result
        return obj;
    }
//! --------------------------------------------------------------------------------------------

    template<typename T>
    void Cuboid<T>::keep(Dims3 *indices) {
        Dims3 *current = indices;

        Dims3 candidate;

        for (size_t i = 0; i < _rows; i++) {
            for (size_t j = 0; j < _cols; j++) {
                for (size_t k = 0; k < _cols; k++) {
                    candidate = {i, j, k};

                    if (candidate == *current) {
                        current += 1;
                    } else { (*this)(i, j, k) = 0; }
                }
            }
        }
    }
//! ------------------------------------------------------------------------------------

//! rotate matrix once (helper for set_rotate) -------------------------------------------------
    template<typename T>
    void Cuboid<T>::rotate_once() {
        // copy data to local variable
        T tmp[_rows * _cols * _depth];
        for (size_t k = 0; k < _rows * _cols * _depth; k++) { tmp[k] = _data[k]; }

        // swap rows and cols
        std::swap(_rows, _cols);

        for (size_t k = 0; k < _depth; k++) {
            // do the rotation operation for each "matrix in the cuboid"
            for (size_t new_i = 0; new_i < _rows; new_i++) {
                for (size_t new_j = 0; new_j < _cols; new_j++) {
                    _data[(k * _cols * _rows) + (new_i * _cols + new_j)] = tmp[(k * _cols * _rows) +
                                                                               ((_cols - 1 - new_j) * _rows) + new_i];
                }
            }
        }

        // change member variable state
        _rot += 1;

        // rotation state can be one of 0, 1, 2, or 3
        _rot %= 4;
    }

//! print the cuboid------------------------------------------------------------------------------------
    template<typename T>
    void Cuboid<T>::print() const {
        std::cout.precision(4);
        std::cout.setf(std::ios::fixed, std::ios::floatfield);

        for (size_t k = 0; k < _depth; k++) {
            std::cout << "Depth: " << k << "\n";
            std::cout << "\n";
            for (size_t i = 0; i < _rows; i++) {
                for (size_t j = 0; j < _cols; j++) {
                    std::cout << std::setw(5) << (*this)(i, j, k) << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }

}

#endif //CNN_CUBOID_IMPL_HXX
