#ifndef _BUFFER_DATA_MATCH_HPP_
#define _BUFFER_DATA_MATCH_HPP_


#include <hip/hip_runtime.h>
#include <sstream>
#include <

// Here flag can be a constant, variable or function call
#define MY_HIP_CHECK(flag)                                                                                                      \
    do  {                                                                                                                       \
        hipError_t _tmpVal;                                                                                                     \
	if ( (_tmpVal = flag) != hipSuccess) {                                                                                  \
            std::ostringstream ostr;                                                                                            \
	    ostr << "HIP Function Failed (" <<  __FILE__ << "," <<  __LINE__ << ") " <<  hipGetErrorString(_tmpVal);            \
            throw std::runtime_error(ostr.str());                                                                            \
	}                                                                                                                       \
    }                                                                                                                           \
    while (0)

#endif 

namespace kernelOutputVerify {

struct bufferCheckResult
{
   int numNans; 
   int numInfs; 
}; 

template <typename T>
class BufferMatcher
{
    BufferMatcher(float tolerance, int dataSize)
    {
       _epsilon = std::numeric_limits<float>::epsilon() * tolerance; 
       _dataSize = dataSize;  
    }; 

    int checkBuffer(T *devBuffer, struct bufferCheckResult *result); 

    float computeRMS(T *toCheckBuffer, T *refBuffer); 

private:

    int _dataSize; 
    float _epsilon; 
}; 


template <typename T>
__device__ checkBufferData(T *devBuffer, struct bufferCheckResult *result, int elemSize)
{
}; 

// all-block kernel
template <typename T> 
__device__  getMaxAbsValue_first_call(T *devBuffer, T *workspace, int elemSize)
{
}; 

// single-block kernel
template <typename T>
__device__  getMaxAbsValue_second_call(T *devBuffer, T *result, int elemSize)
{
};

// all-block kernel
template <typename T>
__device__ getSquareDiffSum_first_call(T *toCheckBuffer, T *refBuffer, T *worksp int elemSize)
{
}; 

// single-block kernel
template <typename T>
__device__ getSquareDiffSum_second_call(T *toCheckBuffer, T *refBuffer, int elemSize)
{
}; 



}; // end of namespace 

#endif  // end of _BUFFER_DATA_MATCH_HPP_
