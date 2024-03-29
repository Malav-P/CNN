//
// Created by malav on 4/26/2022.
//

#ifndef ANN_PREREQS_HXX
#define ANN_PREREQS_HXX

#include <cstdlib>
#include <cassert>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>


#include "helpers/pair.hxx"

#if   defined(APPLE)
    #include <Accelerate/Accelerate.h>
#elif defined(OTHER)
    #include <cblas.h>
#endif

#endif //ANN_PREREQS_HXX
