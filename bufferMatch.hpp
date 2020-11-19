/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef _BUFFER_DATA_MATCH_HPP_
#define _BUFFER_DATA_MATCH_HPP_

#include <hip/hip_runtime.h>
#include <sstream>
#include <stdexcept>
#include <limits>
#include <type_traits>
#include <cstdint>   // for int8_t
#include <half.hpp>    // for host side fp16 

#include <hip/hip_bfloat16.h>     // kernel-used bp16

#include "bfloat16.hpp"           // host-used bp16

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

namespace kernelVerify {

static const int BlockSize = 256; 	

enum DataTypeId_t {
   DT_FP32 = 1,
   DT_FP16 = 2,
   DT_BP16 = 3,
   DT_INT8 = 4,
}; 

struct bufferCheckResult
{
   int numNans; 
   int numInfs; 
   float maxAbsVal; 
}; 

template <typename T>
static constexpr DataTypeId_t getDataTypeId(); 

template <>
constexpr DataTypeId_t getDataTypeId<float>()
{
   return(DT_FP32); 
}; 

template <>
constexpr DataTypeId_t getDataTypeId<half_float::half>()
{
   return(DT_FP16); 
}; 

template <>
constexpr DataTypeId_t getDataTypeId<bfloat16>()
{
   return(DT_BP16); 
}; 

template <>
constexpr DataTypeId_t getDataTypeId<int8_t>()
{
   return(DT_INT8); 
}; 

class BufferMatcher
{
public:
    BufferMatcher(hipStream_t stream, float tolerance, size_t dataSize);

    ~BufferMatcher() noexcept(false); 

    template <typename dataType> 
    int simpleCheckBuffer(dataType *devBuffer, struct bufferCheckResult *bufferResult); 

    template <typename workType, typename refType>
    int checkBuffers(workType *workBuffer, struct bufferCheckResult *workBufferResult, refType *refBuffer, struct bufferCheckResult *refBufferResult); 

    template <typename workType, typename refType>
    void computeRMS(workType *devBuffer1, refType *devBuffer2, float *rms); 

    template <typename workType, typename refType>
    int evaluateAndSimpleReport(workType *workBuffer, refType *refBuffer);

private:
    size_t _dataSize;
    float _epsilon;
    hipStream_t _stream;
    int _deviceId; 

    int blocks; // the number of blocks to dispatch for all-block kernels 
};

BufferMatcher::BufferMatcher(hipStream_t stream, float tolerance, size_t dataSize)
{
    _epsilon = std::numeric_limits<float>::epsilon() * tolerance; 
    _dataSize = dataSize;  
    _stream = stream;

    MY_HIP_CHECK( hipGetDevice(&this->_deviceId) );

    hipDeviceProp_t prop; 

    MY_HIP_CHECK( hipGetDeviceProperties(&prop, this->_deviceId) ); 

    this->blocks = prop.multiProcessorCount * 4;  // to dispatch 4 blocks per CU for all-block kernels
};

BufferMatcher::~BufferMatcher() noexcept(false)
{
}; 

// In-advance declarations
template <DataTypeId_t Tid>
__global__ void checkBufferData_wrapper(void *devBuffer, struct bufferCheckResult *result, size_t bufferSize);

template <DataTypeId_t Tid>
__global__ void getMaxAbsValue_first_call_wrapper(void *devBuffer, size_t bufferSize, float *workspace);

__global__  void getMaxAbsValue_second_call(float *workspace, size_t bufferSize, float *result);

template <DataTypeId_t Tid1, DataTypeId_t Tid2>
__global__ void getSquareDiffSum_first_call_wrapper(void *devBuffer1, void *devBuffer2, size_t bufferSize, double *workspace);

__global__ void getSquareDiffSum_second_call(double *workspace, size_t bufferSize, double *result);


template <typename dataType>
int BufferMatcher::simpleCheckBuffer(dataType *devBuffer, struct bufferCheckResult *bufferResult)
{
    constexpr DataTypeId_t Tid = getDataTypeId<dataType>(); 
    bufferCheckResult *kernelResult;
    int retVal = 0; 

    MY_HIP_CHECK( hipHostMalloc(reinterpret_cast<void**>(&kernelResult), sizeof(struct bufferCheckResult), hipHostMallocDefault) );

    *kernelResult = {0, 0};
    hipLaunchKernelGGL(checkBufferData_wrapper<Tid>, dim3(this->blocks), dim3(BlockSize), 0, this->_stream, reinterpret_cast<void*>(devBuffer), kernelResult, this->_dataSize);
    MY_HIP_CHECK( hipStreamSynchronize(this->_stream) );
    *bufferResult = *kernelResult;

    if (bufferResult->numNans > 0 || bufferResult->numInfs > 0 ) 
        retVal = -1; 	

    MY_HIP_CHECK( hipHostFree(reinterpret_cast<void*>(kernelResult)) );

    return(retVal); 
}; 


template <typename workType, typename refType>
int BufferMatcher::checkBuffers(workType *workBuffer, struct bufferCheckResult *workBufferResult, refType *refBuffer, struct bufferCheckResult *refBufferResult)
{
    constexpr DataTypeId_t refTid = getDataTypeId<refType>(); 
    constexpr DataTypeId_t workTid = getDataTypeId<workType>(); 

    bufferCheckResult *kernelResult; 

    MY_HIP_CHECK( hipHostMalloc(reinterpret_cast<void**>(&kernelResult), sizeof(struct bufferCheckResult), hipHostMallocDefault) ); 

    *kernelResult = {0, 0}; 
    hipLaunchKernelGGL(checkBufferData_wrapper<refTid>, dim3(this->blocks), dim3(BlockSize), 0, this->_stream, reinterpret_cast<void*>(refBuffer), kernelResult, this->_dataSize); 
    MY_HIP_CHECK( hipStreamSynchronize(this->_stream) ); 
    *refBufferResult = *kernelResult; 

    if ( refBufferResult->numNans > 0 || refBufferResult->numInfs > 0 ) {
         MY_HIP_CHECK( hipHostFree(reinterpret_cast<void*>(kernelResult)) ); 
         return(-1); 
    }; 

    *kernelResult = {0, 0};
    hipLaunchKernelGGL(checkBufferData_wrapper<workTid>, dim3(this->blocks), dim3(BlockSize), 0, this->_stream, reinterpret_cast<void*>(workBuffer), kernelResult, this->_dataSize);
    MY_HIP_CHECK( hipStreamSynchronize(this->_stream) );
    *workBufferResult = *kernelResult;

    if ( workBufferResult->numNans > 0 || workBufferResult->numInfs > 0 ) {
         MY_HIP_CHECK( hipHostFree(reinterpret_cast<void*>(kernelResult)) ); 
         return(-2); 
    }; 

    float *kernelMaxAbs; // the maximum abs calculated by the kernel
    float *workspace;   // workspace used by the getMaxAbsValue kernel

    MY_HIP_CHECK( hipHostMalloc(reinterpret_cast<void**>(&kernelMaxAbs), sizeof(float), hipHostMallocDefault) ); 
    MY_HIP_CHECK( hipMalloc(reinterpret_cast<void**>(&workspace), this->blocks * sizeof(float)) ); 

    hipLaunchKernelGGL(getMaxAbsValue_first_call_wrapper<refTid>, dim3(this->blocks), dim3(BlockSize), 0, this->_stream, 
			                          reinterpret_cast<void*>(refBuffer), this->_dataSize, workspace); 
    hipLaunchKernelGGL(getMaxAbsValue_second_call, dim3(1), dim3(BlockSize), 0, this->_stream, 
			                          reinterpret_cast<float*>(workspace), static_cast<size_t>(this->blocks), kernelMaxAbs); 
    MY_HIP_CHECK( hipStreamSynchronize(this->_stream) );

    refBufferResult->maxAbsVal = *kernelMaxAbs; 

    hipLaunchKernelGGL(getMaxAbsValue_first_call_wrapper<workTid>, dim3(this->blocks), dim3(BlockSize), 0, this->_stream, 
			                          reinterpret_cast<void*>(workBuffer), this->_dataSize, workspace); 
    hipLaunchKernelGGL(getMaxAbsValue_second_call, dim3(1), dim3(BlockSize), 0, this->_stream, 
			                          reinterpret_cast<float*>(workspace), static_cast<size_t>(this->blocks), kernelMaxAbs);
    MY_HIP_CHECK( hipStreamSynchronize(this->_stream) );

    workBufferResult->maxAbsVal = *kernelMaxAbs; 

    MY_HIP_CHECK( hipHostFree(reinterpret_cast<void*>(kernelResult)) );
    MY_HIP_CHECK( hipHostFree(reinterpret_cast<void*>(kernelMaxAbs)) );
    MY_HIP_CHECK( hipFree(reinterpret_cast<void*>(workspace)) ); 

    float epsilon = std::numeric_limits<float>::epsilon(); 

    if ( refBufferResult->maxAbsVal < epsilon && workBufferResult->maxAbsVal < epsilon ) 
	 return(1);   // both buffers are all-zero 
 
    if ( refBufferResult->maxAbsVal <= epsilon && workBufferResult->maxAbsVal > epsilon ) 
	 return(-3); 

    return(0); 
}; 

template <typename workType, typename refType>
void BufferMatcher::computeRMS(workType *workBuffer, refType *refBuffer, float *rms)
{
    constexpr DataTypeId_t refTid = getDataTypeId<refType>(); 
    constexpr DataTypeId_t workTid = getDataTypeId<workType>(); 
    double *workspace; 
    double *kernelSum;     // the sum of the square difference calculated by the kernel 

    MY_HIP_CHECK( hipMalloc(reinterpret_cast<void**>(&workspace), this->blocks * sizeof(double)) ); 
    MY_HIP_CHECK( hipHostMalloc(reinterpret_cast<void**>(&kernelSum), sizeof(double), hipHostMallocDefault) ); 

    auto kernelName = getSquareDiffSum_first_call_wrapper<workTid,refTid>;

    hipLaunchKernelGGL(kernelName, dim3(this->blocks), dim3(BlockSize), 0, this->_stream, 
		               reinterpret_cast<void*>(workBuffer), reinterpret_cast<void*>(refBuffer), this->_dataSize, workspace); 
    hipLaunchKernelGGL(getSquareDiffSum_second_call, dim3(1), dim3(BlockSize), 0, this->_stream,
			       reinterpret_cast<double*>(workspace), static_cast<size_t>(this->blocks), kernelSum); 
    MY_HIP_CHECK( hipStreamSynchronize(this->_stream) );

    *rms = sqrt(static_cast<float>(*kernelSum) / (float)this->_dataSize);   
};

    
template <typename workType, typename refType>
int BufferMatcher::evaluateAndSimpleReport(workType *workBuffer, refType *refBuffer)
{ 
    struct bufferCheckResult refBufferResult = {0,0,0.0f};  	
    struct bufferCheckResult workBufferResult = {0,0,0.0f}; 

    
    int ret = checkBuffers<workType, refType>(workBuffer, &workBufferResult, refBuffer, &refBufferResult); 

    if ( ret == -1 ) {
         if ( refBufferResult.numNans > 0 )
              std::cerr << refBufferResult.numNans << " NaN values found in the referrence data buffer." << std::endl; 
         if ( refBufferResult.numInfs > 0 )
              std::cerr << refBufferResult.numInfs << " Infinite values found in referrence data buffer." << std::endl;

         std::cerr << "The evaluation could not be executed!" << std::endl; 
         return(-2);         
    }

    if ( ret == -2 ) {
         if ( workBufferResult.numNans > 0 )
              std::cerr << refBufferResult.numNans << " NaN values found in the work data buffer." << std::endl;
         if ( workBufferResult.numInfs > 0 )
              std::cerr << refBufferResult.numInfs << " Infinite values found in work data buffer." << std::endl;

         std::cerr << "The evaluation could not be executed!" << std::endl;
         return(-2);
    }

    if ( ret == 1 ) {
         std::cout << "Both the work buffer and referrence buffer are all-zero, they are regarded as equal! " << std::endl; 
	 return(1); 
    }; 

    if ( ret == -3 ) {
         std::cerr << "The referrence buffer is found to be all-zero, but the work buffer is not all-zero, they are regarded as not equal! " << std::endl; 
         return(-1); 
    }; 

    float rms; 

    computeRMS<workType,refType>(workBuffer, refBuffer, &rms); 

    if ( rms / refBufferResult.maxAbsVal  > this->_epsilon ) {
          std::cerr << "The evaluated data seems not consistent with that of the referrence buffer!" << std::endl; 
	  std::cerr << "The calculated RMS is " << std::dec << rms << ", the relative error is " << rms / refBufferResult.maxAbsVal << ", the tolerance threshold is " << this->_epsilon << std::endl;  
          return(-2); 	      
    }; 

    std::cout << "The evaluated data seems be consistent with that of the referrence buffer!" << std::endl; 

    return(0); 
};

template <DataTypeId_t Tid>
struct get_type_from_id
{
    using type = float;
};

template <>
struct get_type_from_id<DT_FP32>
{
    using type = float;
};

template <>
struct get_type_from_id<DT_FP16>
{
    using type = _Float16; 
};

template <>
struct get_type_from_id<DT_BP16>
{
    using type = hip_bfloat16;
}; 

template <>
struct get_type_from_id<DT_INT8>
{
    using type = int8_t;
};

template <typename T>
__device__ void checkBufferData(T *devBuffer, struct bufferCheckResult *result, size_t bufferSize)
{
    int gridSize = hipGridDim_x * hipBlockDim_x; 
    size_t index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; 

    __shared__ bufferCheckResult blkResult; 

    if ( hipThreadIdx_x == 0 ) {
	 blkResult.numNans = 0; 
	 blkResult.numInfs = 0;
    };  

    __syncthreads(); 

    while ( index < bufferSize ) {
         if ( isnan(static_cast<float>(devBuffer[index])) ) 
	      (void)atomicAdd(&blkResult.numNans,1);

         if ( isinf(static_cast<float>(devBuffer[index])) ) 
	      (void)atomicAdd(&blkResult.numInfs,1);

	 index += gridSize;  
    }; 

    __syncthreads(); 

    if ( hipThreadIdx_x == 0 ) {
         if ( blkResult.numNans > 0 ) 
	      (void)atomicAdd(&result->numNans,blkResult.numNans); 
	 if ( blkResult.numInfs > 0 )
              (void)atomicAdd(&result->numInfs,blkResult.numInfs); 
    };  
}; 

template <DataTypeId_t Tid>
__global__ void checkBufferData_wrapper(void *devBuffer, struct bufferCheckResult *result, size_t bufferSize)
{
    using DataType = typename get_type_from_id<Tid>::type;

    checkBufferData<DataType>(reinterpret_cast<DataType*>(devBuffer), result, bufferSize);
};

// all-block kernel
template <typename T> 
__device__ void getMaxAbsValue_first_call(T *devBuffer, size_t bufferSize, float *workspace)
{
    int gridSize = hipGridDim_x * hipBlockDim_x; 
    size_t index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; 

    float myMaxAbs = 0.0f; 

    __shared__ float maxValues[BlockSize]; 

    while ( index < bufferSize ) {
        if ( myMaxAbs < fabsf(static_cast<float>(devBuffer[index])) )
	     myMaxAbs = fabsf(static_cast<float>(devBuffer[index])); 

        index += gridSize; 	
    }; 

    maxValues[hipThreadIdx_x] = myMaxAbs; 

    __syncthreads(); 

    for (int reduceThreads=BlockSize/2; reduceThreads > 0; reduceThreads = reduceThreads/2) {
	 if ( (hipThreadIdx_x < reduceThreads) && (maxValues[hipThreadIdx_x] < maxValues[hipThreadIdx_x+reduceThreads]) )
	      maxValues[hipThreadIdx_x] = maxValues[hipThreadIdx_x+reduceThreads]; 
	 __syncthreads(); 
    }; 

    if ( hipThreadIdx_x == 0 )
	 workspace[hipBlockIdx_x] = maxValues[0]; 
}; 

template <DataTypeId_t Tid>
__global__ void getMaxAbsValue_first_call_wrapper(void *devBuffer, size_t bufferSize, float *workspace)
{
    using DataType = typename get_type_from_id<Tid>::type;

    getMaxAbsValue_first_call<DataType>(reinterpret_cast<DataType*>(devBuffer), bufferSize, workspace);
};

// single-block kernel
__global__  void getMaxAbsValue_second_call(float *workspace, size_t bufferSize, float *result)
{
    size_t index = hipThreadIdx_x; 

    float myMaxAbs = 0.0f; 

    __shared__ float maxValues[BlockSize]; 

    while ( index < bufferSize ) {
        if ( myMaxAbs < workspace[index] )
	     myMaxAbs = workspace[index]; 

        index += BlockSize; 	
    }; 

    maxValues[hipThreadIdx_x] = myMaxAbs; 

    __syncthreads(); 

    for (int reduceThreads=BlockSize/2; reduceThreads > 0; reduceThreads = reduceThreads/2) {
	 if ( (hipThreadIdx_x < reduceThreads) && (maxValues[hipThreadIdx_x] < maxValues[hipThreadIdx_x+reduceThreads]) )
	      maxValues[hipThreadIdx_x] = maxValues[hipThreadIdx_x+reduceThreads]; 
	 __syncthreads(); 
    }; 

    if ( hipThreadIdx_x == 0 )
         *result = maxValues[0]; 
};

// all-block kernel
template <typename T1, typename T2>
__device__ void getSquareDiffSum_first_call(T1 *devBuffer1, T2 *devBuffer2, size_t bufferSize, double *workspace)
{
    int gridSize = hipGridDim_x * hipBlockDim_x; 
    size_t index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; 

    double mySum = 0.0; 

    __shared__ double sumValues[BlockSize]; 

    while ( index < bufferSize ) {
        float diff = static_cast<T1>(devBuffer1[index]) -static_cast<T2>(devBuffer2[index]); 
        mySum += static_cast<double>(diff) * static_cast<double>(diff); 

        index += gridSize; 	
    }; 
 
    sumValues[hipThreadIdx_x] = mySum; 

    __syncthreads(); 

    for (int reduceThreads=BlockSize/2; reduceThreads > 0; reduceThreads = reduceThreads/2) {
	 if ( hipThreadIdx_x < reduceThreads )
	      sumValues[hipThreadIdx_x] += sumValues[hipThreadIdx_x+reduceThreads]; 
	 __syncthreads(); 
    }; 

    if ( hipThreadIdx_x == 0 ) 
         workspace[hipBlockIdx_x] = sumValues[0]; 
}; 

template <DataTypeId_t Tid1, DataTypeId_t Tid2>
__global__ void getSquareDiffSum_first_call_wrapper(void *devBuffer1, void *devBuffer2, size_t bufferSize, double *workspace)
{
    using DataType1 = typename get_type_from_id<Tid1>::type;
    using DataType2 = typename get_type_from_id<Tid2>::type;

    getSquareDiffSum_first_call<DataType1, DataType2>(reinterpret_cast<DataType1*>(devBuffer1), reinterpret_cast<DataType2*>(devBuffer2), bufferSize, workspace);
};

// single-block kernel for getSquareDiffSum
__global__ void getSquareDiffSum_second_call(double *workspace, size_t bufferSize, double *result)
{
    size_t index = hipThreadIdx_x; 

    double mySum = 0.0; 

    __shared__ double sumValues[BlockSize]; 

    while ( index < bufferSize ) {
        mySum += workspace[index]; 

        index += BlockSize; 	
    }; 

    sumValues[hipThreadIdx_x] = mySum; 

    __syncthreads(); 

    for (int reduceThreads=BlockSize/2; reduceThreads > 0; reduceThreads = reduceThreads/2) {
	 if ( hipThreadIdx_x < reduceThreads )
	      sumValues[hipThreadIdx_x] += sumValues[hipThreadIdx_x+reduceThreads]; 
	 __syncthreads(); 
    }; 

    if ( hipThreadIdx_x == 0 )
         *result = sumValues[0]; 
}; 

}; // end of namespace 

#endif  // end of _BUFFER_DATA_MATCH_HPP_
