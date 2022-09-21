
#include "../classes/lin_alg/data_types.hxx"
#include <iostream>

int main()
{
    int arr[8] = {4, 2, 1, 6, 8, 4, 0, 1};
    int vec[4] = {1,2,3,4};

    Mat<int> v(2, 4, arr);
    v.padding(1, 1, 1, 1);

    v.print();

    v.crop(1,1,1,1);
    v.print();


}