
// Created by Malav Patel on 12/28/22.
//

#ifndef CNN_EXAMPLE_ARRAY_IMPL_HXX
#define CNN_EXAMPLE_ARRAY_IMPL_HXX

#include "array.hxx"

namespace CNN {
    template<typename T>
    Array<T>::Array(const vector<int> &shape, T *data)
            : shape_(shape),
              strides_(shape.size())
    {
        size_ = 1;
        //for (auto dim: shape) { size_ *= dim; }
        for (int i = shape.size() - 1 ; i > -1; i--) {strides_[i] = size_ ; size_ *= shape[i];}

        // capacity of array in bytes
        capacity_ = size_ * sizeof(T);

        // set pointer to data if provided
        if (data) {
            data_ = data;
            owndata_ = false;
        }

            // allocate resources if no data provided
        else {
            data_ = new T[size_];
            memset(data_, 0, capacity_);
            owndata_ = true;
        }
    }

    template<typename T>
    int Array<T>::offset(const vector<int> &indices) {
        // need to check if indices are within array dimensions


        int offset = 0;
        for (int i = 0 ; i < shape_.size(); i++) {offset += strides_[i] * indices[i];}
        return offset;
    }

// copy constructor
    template<typename T>
    Array<T>::Array(const Array &other)
            : shape_(other.shape_), strides_(other.strides_), size_(other.size_), capacity_(other.capacity_) {
        if (size_ != 0) {
            // resource allocation
            data_ = new T[size_];
            owndata_ = true;

            // copy data from src to dst
            std::memcpy(data_, other.data_, capacity_);
        }

            // if no data exists...
        else {
            data_ = nullptr;
            owndata_ = false;
        }
    }

// copy assignment operator
    template<typename T>
    Array<T> &Array<T>::operator=(const Array<T> &other) {

        if (this != &other) {
            // free existing resource
            if (owndata_) { delete[] data_; }

            // copy over member values
            shape_ = other.shape_;
            strides_ = other.strides_;
            size_ = other.size_;
            capacity_ = other.capacity_;

            // we own this data
            owndata_ = true;

            // resource allocation and transition
            data_ = new T[size_];
            std::memcpy(data_, other.data_, capacity_);
        }

        return *this;

    }

// move constructor
    template<typename T>
    Array<T>::Array(Array &&other) noexcept
            : shape_(other.shape_), strides_(other.strides_), size_(other.size_), capacity_(other.capacity_), owndata_(other.owndata_),
              data_(other.data_) {
        other.shape_ = {};
        other.strides_ = {};
        other.size_ = 0;
        other.capacity_ = 0;
        other.owndata_ = false;
        other.data_ = nullptr;
    }

// move assignment operator
    template<typename T>
    Array<T> &Array<T>::operator=(Array &&other) noexcept {
        // check for self assignment
        if (this != &other) {
            // free existing resource
            if (owndata_) { delete[] data_; }

            // copy member variables and data
            shape_ = other.shape_;
            strides_ = other.strides_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            owndata_ = other.owndata_;
            data_ = other.data_;

            // release memory from other variables
            other.shape_ = {};
            other.strides_ = {};
            other.size_ = 0;
            other.capacity_ = 0;
            other.owndata_ = false;
            other.data_ = nullptr;
        }

        return *this;

    }

    template<typename T>
    void Array<T>::Reshape(const vector<int> &newshape) {
        int count = 1;

        strides_.resize(newshape.size());
        for (int i = newshape.size() - 1 ; i > -1; i--) {strides_[i] = count ; count *= newshape[i];}

        if (count != size_) {
            if (owndata_) { delete[] data_; }
            size_ = count;
            capacity_ = size_ * sizeof(T);
            shape_ = newshape;

            data_ = new T[size_];
            memset(data_, 0, capacity_);
            owndata_ = true;
        } else {
            shape_ = newshape;
        }

    }

    template<typename T>
    T Array<T>::partial_dot(Array<T> &other, const vector<int> &startpos) {
        // assert that array dimensions are within bounds -- TODO

        T answer = 0;

        // define lambda function
        auto pdot = [&](vector<int> index) {
            vector<int> newindex(shape_.size());
            for (int i = 0; i < newindex.size(); i++) { newindex[i] = index[i] + startpos[i]; }

            answer += (*this)[index] * other[newindex];
        };


        vector<int> index(shape_.size());
        vector<int> start(shape_.size(), 0);

        recurse(index, 0, pdot, start, shape_);

        return answer;
    }

    template<typename T>
    Array<T> Array<T>::pad(const vector<int> &padims) {
        // only working with 3-d array
        assert(padims.size() == 6);

        vector<int> newshape(shape_);
        // new shape
        for (int i = 0; i < shape_.size(); i++) {
            newshape[i] += padims[2 * i];
            newshape[i] += padims[2 * i + 1];
        }

        Array<T> newarray(newshape);

        double* newdata = newarray.getdata();
        vector<int> newstride = newarray.getstride();

        for (int k = padims[0]; k < newshape[0] - padims[1]; k++ )
        {
            if (k < 0) {continue;}
            for (int i = padims[2]; i < newshape[1] - padims[3]; i++)
            {
                if (i < 0) { continue;}
                for (int j = padims[4]; j < newshape[2] - padims[5]; j++)
                {
                    if (j <0) {continue;}
                    newdata[k*newstride[0] + i * newstride[1] + j* newstride[2]] = data_[(k - padims[0]) * strides_[0] + ( i - padims[2])*strides_[1] + ( j - padims[4]) * strides_[2]];
                }
            }
        }


        return newarray;

    }

    template<typename T>
    Array<T> Array<T>::rotate() {
        assert(shape_.size() == 4);

        vector<int> newshape(shape_);

        Array<T> newarray(newshape);

        for (int l = 0 ; l < shape_[0]; l++)
        {
            for (int k = 0 ; k < shape_[1]; k++)
            {
                for (int i = 0 ; i < shape_[2]; i++)
                {
                    for (int j = 0 ; j < shape_[3]; j++)
                    {
                        newarray[{l,k,i,j}] = (*this)[{l,k,shape_[2] - 1 - i, shape_[3] - 1 - j}];
                    }
                }
            }
        }
        return newarray;
    }

    template<typename T>
    void Array<T>::operator*=(T alpha) {

        if (owndata_) {
            for (int i = 0; i < size_; i++) { data_[i] *= alpha; }
        } else {
            // TODO
        }
    }

    template<typename T>
    void Array<T>::fill(T value) {
        if (owndata_) {
            for (int i = 0; i < size_; i++) { data_[i] = value; }

        } else {
            // TODO
        }
    }

    template<typename T>
    void Array<T>::resetdata(T *newdata) {
        if (owndata_) { delete[] data_; }

        data_ = newdata;

        owndata_ = false;
    }

    template<typename T>
    void Array<T>::write(const Array<T> &other, size_t start) {
        T *start_ptr = data_ + start;
        std::memcpy(start_ptr, other.data_, sizeof(T) * other.getsize());

    }

    template<typename T>
    Array<T> Array<T>::edivide(const Array &other, T scalar) {
        // assert that arrays have same dimensions TODO

        Array<T> C = Array<T>(shape_);

        for (size_t i = 0; i < size_; i++) { C.data_[i] = scalar * data_[i] / other.data_[i]; }

        return C;
    }

    template<typename T>
    Array<T> Array<T>::operator-(const Array<T> &other) {
        // assert that arrays have same dimensions TODO

        Array<T> C = Array<T>(shape_);

        for (size_t i = 0; i < size_; i++) { C.data_[i] = data_[i] - other.data_[i]; }

        return C;
    }

    template<typename T>
    Array<T> Array<T>::as_strided(const vector<int> &shape, const vector<int> &strides) {

        Array<T> C(shape, data_);

        C.strides_ = strides;

        return C;
    }
}

#endif //CNN_EXAMPLE_ARRAY_IMPL_HXX
