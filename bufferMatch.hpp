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
#include <cstdint>         // for int8_t
#include <half.hpp>        // for host side fp16 
#include <hip/hip_fp16.h>  // for kernel side fp16

namespace kernelVerify {

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

namespace kernelVerifyImpl {

static const int BlockSize = 256; 	

enum DataTypeId_t {
   DT_FP32 = 1,
   DT_FP16 = 2,
   DT_INT8 = 4,
}; 

struct bufferCheckStat
{
   int numNans; 
   int numInfs; 
   float maxAbs; 
   bool allZero; 
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
constexpr DataTypeId_t getDataTypeId<int8_t>()
{
   return(DT_INT8); 
}; 

class BufferCompare
{
public:
    BufferCompare();

    BufferCompare(const BufferCompare&) = delete;
    BufferCompare& operator=(BufferCompare&) = delete;    

    ~BufferCompare() noexcept(false); 

    template <typename workType, typename refType>
    int checkComparedBuffers(hipStream_t stream, workType *workBuffer, refType *refBuffer, size_t dataSize); 

    template <typename workType, typename refType>
    double computeRMSD(hipStream_t stream, workType *workBuffer, refType *refBuffer, size_t dataSize, bool buffersChecked); 

    void showComparedBuffersStat(std::ostream &os); 

    template <typename dataType>
    int checkSingleBuffer(hipStream_t stream, dataType *workBuffer, size_t dataSize);

    void showSingleBufferStat(std::ostream &os); 

private:
    int deviceId; 

    int blocks; // the number of blocks to dispatch for all-block kernels 
    bufferCheckStat *kernelStat;   // pointer of host-pinned data used by kernel

    float *kernelMaxAbs;           // the maximum abs calculated by kernel, pointer of host-pinned data used by kernel
    float *workspace1;             // workspace used by the getMaxAbs kernel, pointer of device memory

    double *kernelSum;             // the sum of the square difference calculated by kernel, pointer of host-pinned data used by kernel
    double *workspace2;            // workspace used by the getSquareDiffSum kernel, pointer of device memory 

public: 
    bufferCheckStat buffStat;      // used by checkSingleBuffer() 
    bufferCheckStat refBuffStat;   // reference buffer usually holds data that are assumed to be accurate 
    bufferCheckStat workBuffStat;  // work buffer holds data which we intend to verify
};

BufferCompare::BufferCompare()
{
    MY_HIP_CHECK( hipGetDevice(&deviceId) );

    hipDeviceProp_t prop; 

    MY_HIP_CHECK( hipGetDeviceProperties(&prop, this->deviceId) ); 

    blocks = prop.multiProcessorCount * 4;  // to dispatch 4 blocks per CU for all-block kernels

    MY_HIP_CHECK( hipHostMalloc(reinterpret_cast<void**>(&kernelStat), sizeof(struct bufferCheckStat), hipHostMallocDefault) );

    MY_HIP_CHECK( hipHostMalloc(reinterpret_cast<void**>(&kernelMaxAbs), sizeof(float), hipHostMallocDefault) );
    MY_HIP_CHECK( hipMalloc(reinterpret_cast<void**>(&workspace1), blocks * sizeof(float)) );

    MY_HIP_CHECK( hipHostMalloc(reinterpret_cast<void**>(&kernelSum), sizeof(double), hipHostMallocDefault) ); 
    MY_HIP_CHECK( hipMalloc(reinterpret_cast<void**>(&workspace2), blocks * sizeof(double)) ); 
};

BufferCompare::~BufferCompare() noexcept(false)
{
    MY_HIP_CHECK( hipHostFree(reinterpret_cast<void*>(kernelStat)) );

    MY_HIP_CHECK( hipHostFree(reinterpret_cast<void*>(kernelMaxAbs)) );
    MY_HIP_CHECK( hipFree(reinterpret_cast<void*>(workspace1)) ); 

    MY_HIP_CHECK( hipHostFree(reinterpret_cast<void*>(kernelSum)) );
    MY_HIP_CHECK( hipFree(reinterpret_cast<void*>(workspace2)) ); 
}; 

// In-advance declarations
template <DataTypeId_t Tid>
__global__ void checkBuffer(void *devBuffer, struct bufferCheckStat *stat, size_t bufferSize);

template <DataTypeId_t Tid>
__global__ void getMaxAbsCall1(void *devBuffer, size_t bufferSize, float *workspace);

__global__  void getMaxAbsCall2(float *workspace, size_t bufferSize, float *stat);

template <DataTypeId_t Tid1, DataTypeId_t Tid2>
__global__ void getSquareDiffSumCall1(void *devBuffer1, void *devBuffer2, size_t bufferSize, double *workspace);

__global__ void getSquareDiffSumCall2(double *workspace, size_t bufferSize, double *stat);

template <typename workType, typename refType>
int BufferCompare::checkComparedBuffers(hipStream_t stream, workType *workBuffer, refType *refBuffer, size_t dataSize)
{
    constexpr DataTypeId_t refTid = getDataTypeId<refType>(); 
    constexpr DataTypeId_t workTid = getDataTypeId<workType>(); 

    if ( workBuffer == nullptr || refBuffer == nullptr)
	 throw std::runtime_error("checkBuffers() is passed with empty buffers"); 

    this->refBuffStat = {0, 0, 0.0f, false}; 
    this->workBuffStat = {0, 0, 0.0f, false}; 

    *this->kernelStat = {0, 0, 0.0f, false}; 
    hipLaunchKernelGGL(checkBuffer<refTid>, dim3(this->blocks), dim3(BlockSize), 0, stream, reinterpret_cast<void*>(refBuffer), kernelStat, dataSize); 
    MY_HIP_CHECK( hipStreamSynchronize(stream) ); 
    refBuffStat = *kernelStat; 

    if ( refBuffStat.numNans > 0 || refBuffStat.numInfs > 0 ) 
         return(-1); 

    *kernelStat = {0, 0};
    hipLaunchKernelGGL(checkBuffer<workTid>, dim3(this->blocks), dim3(BlockSize), 0, stream, reinterpret_cast<void*>(workBuffer), kernelStat, dataSize);
    MY_HIP_CHECK( hipStreamSynchronize(stream) );
    workBuffStat = *kernelStat;

    if ( workBuffStat.numNans > 0 || workBuffStat.numInfs > 0 ) {
         return(-2); 
    }; 

    hipLaunchKernelGGL(getMaxAbsCall1<refTid>, dim3(this->blocks), dim3(BlockSize), 0, stream, reinterpret_cast<void*>(refBuffer), dataSize, this->workspace1); 
    hipLaunchKernelGGL(getMaxAbsCall2, dim3(1), dim3(BlockSize), 0, stream, this->workspace1, static_cast<size_t>(this->blocks), kernelMaxAbs); 
    MY_HIP_CHECK( hipStreamSynchronize(stream) );

    this->refBuffStat.maxAbs = *kernelMaxAbs; 

    hipLaunchKernelGGL(getMaxAbsCall1<workTid>, dim3(this->blocks), dim3(BlockSize), 0, stream, reinterpret_cast<void*>(workBuffer), dataSize, this->workspace1); 
    hipLaunchKernelGGL(getMaxAbsCall2, dim3(1), dim3(BlockSize), 0, stream, this->workspace1, static_cast<size_t>(this->blocks), kernelMaxAbs);
    MY_HIP_CHECK( hipStreamSynchronize(stream) );

    this->workBuffStat.maxAbs = *kernelMaxAbs; 

    float epsilon = std::numeric_limits<float>::epsilon(); 

    if ( this->refBuffStat.maxAbs < epsilon )
	 this->refBuffStat.allZero = true; 

    if ( this->workBuffStat.maxAbs < epsilon )
	 this->workBuffStat.allZero = true; 

    if ( this->refBuffStat.allZero && this->workBuffStat.allZero) 
	 return(1);   // both buffers are all-zero 
 
    if ( this->refBuffStat.maxAbs <= epsilon && this->workBuffStat.maxAbs > epsilon ) 
	 return(-3); 

    return(0); 
}; 

template <typename dataType>
int BufferCompare::checkSingleBuffer(hipStream_t stream, dataType *devBuffer, size_t dataSize)
{
    constexpr DataTypeId_t Tid = getDataTypeId<dataType>();

    if ( devBuffer == nullptr )
         throw std::runtime_error("checkSingleBuffer() is passed with empty buffer");

    this->buffStat = {0, 0, 0.0f, false};

    *this->kernelStat = {0, 0, 0.0f, false};
    hipLaunchKernelGGL(checkBuffer<Tid>, dim3(this->blocks), dim3(BlockSize), 0, stream, reinterpret_cast<void*>(devBuffer), kernelStat, dataSize);
    MY_HIP_CHECK( hipStreamSynchronize(stream) );
    buffStat = *kernelStat;

    if ( buffStat.numNans > 0 || buffStat.numInfs > 0 )
         return(-1);

    hipLaunchKernelGGL(getMaxAbsCall1<Tid>, dim3(this->blocks), dim3(BlockSize), 0, stream, reinterpret_cast<void*>(devBuffer), dataSize, this->workspace1);
    hipLaunchKernelGGL(getMaxAbsCall2, dim3(1), dim3(BlockSize), 0, stream, this->workspace1, static_cast<size_t>(this->blocks), kernelMaxAbs);
    MY_HIP_CHECK( hipStreamSynchronize(stream) );

    this->buffStat.maxAbs = *kernelMaxAbs;

    float epsilon = std::numeric_limits<float>::epsilon();

    if ( this->buffStat.maxAbs < epsilon )
         this->buffStat.allZero = true;

    if ( this->buffStat.allZero )
         return(1);   // both buffers are all-zero

    return(0);
};

template <typename workType, typename refType>
double BufferCompare::computeRMSD(hipStream_t stream, workType *workBuffer, refType *refBuffer, size_t dataSize, bool buffersChecked)
{
    constexpr DataTypeId_t refTid = getDataTypeId<refType>(); 
    constexpr DataTypeId_t workTid = getDataTypeId<workType>(); 

    float maxAbs1, maxAbs2, maxAbs; 

    if (!buffersChecked) {
        hipLaunchKernelGGL(getMaxAbsCall1<refTid>, dim3(this->blocks), dim3(BlockSize), 0, stream, reinterpret_cast<void*>(refBuffer), dataSize, this->workspace1);
        hipLaunchKernelGGL(getMaxAbsCall2, dim3(1), dim3(BlockSize), 0, stream, this->workspace1, static_cast<size_t>(this->blocks), this->kernelMaxAbs);
        MY_HIP_CHECK( hipStreamSynchronize(stream) );
	
        maxAbs1 = *kernelMaxAbs; 

        hipLaunchKernelGGL(getMaxAbsCall1<workTid>, dim3(this->blocks), dim3(BlockSize), 0, stream, reinterpret_cast<void*>(workBuffer), dataSize, this->workspace1);
        hipLaunchKernelGGL(getMaxAbsCall2, dim3(1), dim3(BlockSize), 0, stream, this->workspace1, static_cast<size_t>(this->blocks), this->kernelMaxAbs);
        MY_HIP_CHECK( hipStreamSynchronize(stream) );

        maxAbs = std::max(maxAbs1, maxAbs2); 
    } 
    else 
	maxAbs = std::max(this->refBuffStat.maxAbs, this->workBuffStat.maxAbs); 

    if ( maxAbs <= std::numeric_limits<float>::epsilon() ) 
	 throw std::runtime_error("RMS could not be computed when both buffers are all-zero!"); 

    auto kernelName = getSquareDiffSumCall1<workTid,refTid>; 
    hipLaunchKernelGGL(kernelName, dim3(this->blocks), dim3(BlockSize), 0, stream, 
		                                         reinterpret_cast<void*>(workBuffer), reinterpret_cast<void*>(refBuffer), dataSize, this->workspace2); 
    hipLaunchKernelGGL(getSquareDiffSumCall2, dim3(1), dim3(BlockSize), 0, stream, this->workspace2, static_cast<size_t>(this->blocks), this->kernelSum); 
    MY_HIP_CHECK( hipStreamSynchronize(stream) );

    return sqrt(*kernelSum) / (static_cast<double>(dataSize) * static_cast<double>(maxAbs));   
};
    
void BufferCompare::showComparedBuffersStat(std::ostream &os)
{ 
    if ( this->refBuffStat.numNans > 0 )
         os << this->refBuffStat.numNans << " NaN values found in the reference buffer." << std::endl; 

    if ( this->refBuffStat.numInfs > 0 )
         os << this->refBuffStat.numInfs << " Infinite values found in reference buffer." << std::endl;

    if ( this->workBuffStat.numNans > 0 )
         os << this->refBuffStat.numNans << " NaN values found in the work buffer." << std::endl;

    if ( this->workBuffStat.numInfs > 0 )
         os << this->refBuffStat.numInfs << " Infinite values found in work buffer." << std::endl;

    if ( this->refBuffStat.allZero && this->workBuffStat.allZero )
         os << "Both the work buffer and reference buffer are all-zero, they are regarded as equal! " << std::endl; 

    if ( this->refBuffStat.allZero && !this->workBuffStat.allZero )
         os << "The reference buffer is all-zero, but the work buffer is not, they are regarded as not equal, no RMS computation needed! " << std::endl; 
};

void BufferCompare::showSingleBufferStat(std::ostream &os)
{
    if ( this->buffStat.numNans > 0 )
         os << this->buffStat.numNans << " NaN values found in the checked buffer." << std::endl;

    if ( this->buffStat.numInfs > 0 )
         os << this->buffStat.numInfs << " Infinite values found in the checked buffer." << std::endl;

    if ( this->buffStat.allZero )
         os << "The checked buffer has all values being zero!" << std::endl;
}; 

typedef _Float16 half_t;

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
    using type = half_t; 
};

template <>
struct get_type_from_id<DT_INT8>
{
    using type = int8_t;
};

static inline __device__ half_t abs(half_t x)
{
    return __habs(x); 
}; 

static inline __device__ bool isnan(half_t x)
{
    return __hisnan(x); 
}; 

static inline __device__ bool isfinite(half_t x)
{
    return __hisinf(x); 
};

template <typename T>
__device__ void checkBufferImpl(T *devBuffer, struct bufferCheckStat *pStat, size_t bufferSize)
{
    int gridSize = hipGridDim_x * hipBlockDim_x; 
    size_t index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; 

    __shared__ bufferCheckStat blkResult; 

    if ( hipThreadIdx_x == 0 ) {
	 blkResult.numNans = 0; 
	 blkResult.numInfs = 0;
    };  

    __syncthreads(); 

    while ( index < bufferSize ) {
         if ( isnan(devBuffer[index]) ) 
	      (void)atomicAdd(&blkResult.numNans,1);

         if ( isfinite(devBuffer[index]) ) 
	      (void)atomicAdd(&blkResult.numInfs,1);

	 index += gridSize;  
    }; 

    __syncthreads(); 

    if ( hipThreadIdx_x == 0 ) {
         if ( blkResult.numNans > 0 ) 
	      (void)atomicAdd(&pStat->numNans,blkResult.numNans); 
	 if ( blkResult.numInfs > 0 )
              (void)atomicAdd(&pStat->numInfs,blkResult.numInfs); 
    };  
}; 

template <DataTypeId_t Tid>
__global__ void checkBuffer(void *devBuffer, struct bufferCheckStat *pStat, size_t bufferSize)
{
    using DataType = typename get_type_from_id<Tid>::type;

    checkBufferImpl(reinterpret_cast<DataType*>(devBuffer), pStat, bufferSize);
};

// all-block kernel
template <typename T> 
__device__ void getMaxAbsCall1Impl(T *devBuffer, size_t bufferSize, float *workspace)
{
    int gridSize = hipGridDim_x * hipBlockDim_x; 
    size_t index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; 

    float myMaxAbs = 0.0f; 

    __shared__ float maxValues[BlockSize]; 

    while ( index < bufferSize ) {
        if ( myMaxAbs < static_cast<float>(abs(devBuffer[index])) ) 
	     myMaxAbs = static_cast<float>(abs(devBuffer[index])); 

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
__global__ void getMaxAbsCall1(void *devBuffer, size_t bufferSize, float *workspace)
{
    using DataType = typename get_type_from_id<Tid>::type;

    getMaxAbsCall1Impl(reinterpret_cast<DataType*>(devBuffer), bufferSize, workspace);
};

// single-block kernel
__global__  void getMaxAbsCall2(float *workspace, size_t bufferSize, float *result)
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
__device__ void getSquareDiffSumCall1Impl(T1 *devBuffer1, T2 *devBuffer2, size_t bufferSize, double *workspace)
{
    int gridSize = hipGridDim_x * hipBlockDim_x; 
    size_t index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; 

    double mySum = 0.0; 

    __shared__ double sumValues[BlockSize]; 

    while ( index < bufferSize ) {
        double diff = static_cast<float>(devBuffer1[index]) - static_cast<float>(devBuffer2[index]); 
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
__global__ void getSquareDiffSumCall1(void *devBuffer1, void *devBuffer2, size_t bufferSize, double *workspace)
{
    using DataType1 = typename get_type_from_id<Tid1>::type;
    using DataType2 = typename get_type_from_id<Tid2>::type;

    getSquareDiffSumCall1Impl(reinterpret_cast<DataType1*>(devBuffer1), reinterpret_cast<DataType2*>(devBuffer2), bufferSize, workspace);
};

// single-block kernel for getSquareDiffSum
__global__ void getSquareDiffSumCall2(double *workspace, size_t bufferSize, double *result)
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

}; // end of namespace kernelVerifyImpl

using kernelVerifyImpl::BufferCompare;

static thread_local BufferCompare bufferCompare{}; 

template <typename workType, typename refType>
static int check_compared_buffers(hipStream_t stream, workType *workBuffer, refType *refBuffer, size_t dataSize)
{
    static_assert(!std::is_void<workType>::value && !std::is_void<refType>::value, "Explictly typed buffer is required!"); 

    return(  bufferCompare.checkComparedBuffers(stream, workBuffer, refBuffer, dataSize) ); 
}; 

template <typename workType, typename refType>
static double rmsd_device_buffers(hipStream_t stream, workType *workBuffer, refType *refBuffer, size_t dataSize, bool buffersChecked=false)
{
    static_assert(!std::is_void<workType>::value && !std::is_void<refType>::value, "Explicitly typed buffer is required!"); 

    return ( bufferCompare.computeRMSD(stream, workBuffer, refBuffer, dataSize, buffersChecked) );  
}; 

static void show_compared_buffers_stat(std::ostream &os)
{
    return ( bufferCompare.showComparedBuffersStat(os) ); 
};

template <typename dataType>
static int check_single_buffer(hipStream_t stream, dataType *devBuffer, size_t dataSize)
{
    static_assert(!std::is_void<dataType>::value, "Explicitly typed buffer is required!"); 

    return ( bufferCompare.checkSingleBuffer(stream, devBuffer, dataSize) ); 
};

static void show_single_buffer_stat(std::ostream &os)
{
    return ( bufferCompare.showComparedBuffersStat(os) ); 
};

}; // end of namespace kernelVerify

#endif  // end of _BUFFER_DATA_MATCH_HPP_
