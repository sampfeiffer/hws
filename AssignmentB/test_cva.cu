#include <vector>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/functional.h>


struct inv1_functor
{
  const int pos;

  inv1_functor(int _pos) : pos(_pos) {}

  __host__ __device__
  double operator()(const double &x, const int &i) const {
    if (i == pos)
      return x*100;
    else
      return -x;
  }
};

int main()
{
    // allocate 2 device_vectors with 10 elements
    thrust::device_vector<int> X(10);
    thrust::fill(X.begin(), X.end(), 2);

    int pos = 3;
    thrust::transform(X.begin(), X.end(), X.begin(), inv1_functor(pos));

    // print contents of X
    for(int i = 0; i < X.size(); i++)
        std::cout << "X[" << i << "] = " << X[i] << "\n";

    return 0;
}




