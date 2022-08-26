
#include "../classes/lin_alg/mat.hxx"
#include <iostream>

int main()
{
    int arr[8] = {4, 2, 1, 6, 8, 4, 0, 1};
    int vec[4] = {1,2,3,4};

    Mat<int> v{2, 4, arr};
    v.padding(0, 1, 2, 3);

    for (size_t i = 0; i<v.get_rows(); i++)
    {
        for (size_t j = 0; j<v.get_cols(); j++)
        {
            std::cout << v(i,j) << " ";
        }
        std::cout << "\n";
    }


//    Vector<int> vector{4, vec};
//    Vector<int> result = v*vector;
//
//    for (size_t i = 0; i<2; i++)
//    {
//        std::cout << result[i] << " ";
//    }
}