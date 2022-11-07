//
// Created by malav on 5/2/2022.
//

#ifndef ANN_VECTOR_HXX
#define ANN_VECTOR_HXX

#include "data_types.hxx"

template<typename T = double>
class Vector {
    public:

        //! CONSTRUCTORS, DESTRUCTORS, MOVE CONSTRUCTORS, ETC --------------------------------------------------------

        // constructor
        Vector() = default;

        // constructor
        explicit Vector(size_t n, T* arr = nullptr);

        // copy constructor
        Vector(const Vector& other);

        // move constructor
        Vector(Vector<T>&& other) noexcept;

        // destructor
        ~Vector() {delete[] _data;}
        //! ----------------------------------------------------------------------------------------------------------

        //! ASSIGNMENT, BINARY, UNARY OPERATORS -----------------------------------------------------------------------
        // copy assignment operator
        Vector<T>& operator=(const Vector<T>& rhs);

        // move assignment operator
        Vector<T>& operator=(Vector<T>&& other) noexcept;

        // += operator
        Vector<T>& operator+=(const Vector<T>& other);

        // indexing operator
        T& operator[](size_t idx);

        // const indexing operator
        const T& operator[](size_t idx) const;

        // multiply operator (matrix)
        Vector<T> operator * (Mat<T>& other);

        // multiply operator (scalar)
        Vector<T> operator * (double c);

        // *= operator (scalar)
        void operator *= (double c);

        // multiply operator (vector)
        Mat<T> operator * (const Vector<T>& other);

        // add operator
        Vector<T> operator + (const Vector<T>& other);

        // minus operator
        Vector<T> operator - (const Vector<T>& other);

        // merge two vectors
        Vector<T> merge (const Vector<T>& other);

        //! ----------------------------------------------------------------------------------------------------------


        //! OTHER ----------------------------------------------------------------------------------------------------
        // compute dot product between two Vectors
        T dot(const Vector<T>& other);

        // write into the vector at a specified start point
        void write(const Vector<T>& other, T* start);

        // fill vector with a value
        void fill(T fill);

        // turn a vector into a Mat object with _rows and _cols
        Mat<T> reshape(size_t n_rows, size_t n_cols);

        // compute element-wise product between two Vectors
        Vector<T> eprod(const Vector<T>& other) const;

        // compute the element-wise quotient of two Vectors
        Vector<T> edivide(const Vector<T>& other);

        // get the _length of vector (read only)
        size_t const& get_len() const {return _length;}

        // get the data (read only)
        T* const& get_data() const {return _data;}

        // print the vector elements
        void print() const;

        //! ----------------------------------------------------------------------------------------------------------

    private:

        // _length of vector (number of elements)
        size_t _length {0};

        // pointer to _data;
        T* _data {nullptr};

        friend class Mat<T>;

};



#include "vector_impl.hxx"
#endif //ANN_VECTOR_HXX
