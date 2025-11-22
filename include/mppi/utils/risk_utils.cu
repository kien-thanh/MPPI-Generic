#include <mppi/utils/risk_utils.cuh>

// #include <thrust/device_vector.h>
// #include <thrust/sort.h>

#include <algorithm>

namespace mppi
{
template <class T = float>
__host__ __device__ void insertionSort(T* __restrict__ array, const int N)
{
  T temp;
  int j;
  for (int i = 1; i < N; i++)
  {
    temp = array[i];
    j = i - 1;
    while (j >= 0 && array[j] > temp)
    {
      array[j + 1] = array[j];
      --j;
    }
    array[j + 1] = temp;
  }
}

}  // namespace mppi
