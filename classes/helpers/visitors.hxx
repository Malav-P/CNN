//
// Created by malav on 6/23/2022.
//

#ifndef ANN_VISITORS_HXX
#define ANN_VISITORS_HXX

#include <boost/variant.hpp>
#include "pair.hxx"
#include "../lin_alg/data_types.hxx"

using Dims = Dimensions<>;

class Outshape_visitor : public boost::static_visitor<Dims>
{
public:

    template<typename T>
    Dims operator()(T* operand) const { return (*operand).out_shape(); }
};

class Inshape_visitor : public boost::static_visitor<Dims>
{
public:

    template<typename T>
    Dims operator()(T* operand) const { return (*operand).in_shape(); }
};

class Forward_visitor : public boost::static_visitor<>
{
public:

    template<typename T>
    void operator()(T* operand) const {(*operand).Forward(*input, *output);}

    Vector<double>* input;
    Vector<double>* output;
};

class Backward_visitor : public boost::static_visitor<>
{
public:

    template<typename T>
    void operator()(T* operand) const {(*operand).Backward(*dLdY, *dLdX);}

    Vector<double>* dLdY;
    Vector<double>* dLdX;
};

template<typename Optimizer>
class Update_parameters_visitor : public boost::static_visitor<>
{
public:

    template<typename T>
    void operator()(T* operand) const { (*operand).Update_Params(optimizer, normalizer);}

    Optimizer* optimizer {nullptr};
    size_t normalizer {1};
};
#endif //ANN_VISITORS_HXX
