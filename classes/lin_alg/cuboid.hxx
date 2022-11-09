//
// Created by malav on 9/28/2022.
//

#ifndef CNN_CUBOID_HXX
#define CNN_CUBOID_HXX


#include "data_types.hxx"


template<typename T = double>
class Cuboid {
    public:

        //! CONSTRUCTORS, DESTRUCTORS, MOVE CONSTRUCTORS, ETC --------------------------------------------------------

        // default constructor
        Cuboid() = default;

        // constructor
        explicit Cuboid(size_t side_len, T* arr = nullptr);

        // constructor
        Cuboid(size_t num_r, size_t num_c, size_t num_d, T* arr = nullptr);

        // copy constructor
        Cuboid(const Cuboid<T>& other);

        // move constructor
        Cuboid(Cuboid<T>&& other) noexcept;

        // destructor
        ~Cuboid() { delete[] _data; }

        //! --------------------------------------------------------------------------------------------------------

        //! ASSIGNMENT, BINARY, UNARY OPERATORS -------------------------------------------------------------------
        // copy assignment operator
        Cuboid<T>& operator=(const Cuboid<T>& rhs);

        // move assignment operator
        Cuboid<T>& operator=(Cuboid<T>&& other) noexcept ;

        // += operator
        Cuboid<T>& operator+=(const Cuboid<T>& other);

        // + operator
        Cuboid<T> operator+(const Cuboid<T>& rhs);

        // index operator
        T& operator()(size_t i, size_t j, size_t k);

        // const index operator
        const T& operator()(size_t i, size_t j, size_t k) const;

        // multiply operator (scalar)
        Cuboid<T> operator * (double c);


        //! -------------------------------------------------------------------------------------------------------

        //! OTHER -------------------------------------------------------------------------------------------------

        // compute inner product between two Matrices
        T dot(const Cuboid<T>& other);

        // compute partial inner product with given Dims of starting indices
        T partial_dot(const Cuboid<T>& other, Dims3 p);

        // keep values in certain indices, and set all others to zero. indices must be sorted in increasing row, increasing column order
        void keep(Dims3* indices);

        // fill cuboid with a value
        void fill(T t);

        // set the rotation state of the cuboid and restructure _data
        void set_rot(size_t n = 0);

        // create a Vector object out of this object
        Vector<T> flatten();

        // get number of rows of matrix (read only)
        size_t const& get_rows() const {return _rows;}

        // get number of cols of matrix (read only)
        size_t const& get_cols() const {return _cols;}

        // get depth of cuboid
        size_t const& get_depth() const {return _depth;}

        // get rotation state
        size_t const& get_rot()  const {return _rot;}

        // print the cuboid
        void print() const;

        //! --------------------------------------------------------------------------------------------------------

    private:

        // helper for set_rot, rotate matrix CW by 90 degrees
        void rotate_once();

        // number of columns
        size_t _cols {0};

        // number of rows
        size_t _rows {0};

        // depth
        size_t _depth {0};

        // rotation state, 0 = no rotate; 1 = 90 degrees clockwise; 2 = 180 degrees clockwise; 3 = 270 degrees clockwise
        size_t _rot {0};

    public:
        // pointer to _data
        T* _data {nullptr};

        friend class Vector<T>;
        friend class Mat<T>;
};

#include "cuboid_impl.hxx"
#endif //CNN_CUBOID_HXX
