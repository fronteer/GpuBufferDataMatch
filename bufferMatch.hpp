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

namespace kernelVerify {

struct bufferCheckResult
{
   int numNans; 
   int numInfs; 
   float maxAbsVal; 
}; 

class BufferMatcher
{
    static const int BlockSize = 256; 	

    BufferMatcher(hipStream_t stream, float tolerance, size_t dataSize);

    ~BufferMatcher() noexcept(false); 

    template <typename workType, typename refType>
    int checkBuffer(workType *workBuffer, struct bufferCheckResult *workBufferResult, refType *refBuffer, struct bufferCheckResult *refBufferResult); 

    template <typename workType, typename refType>
    void computeRMS(workType *devBuffer1, refType *devBuffer2, float *rms); 

    template <typename workType, typename refType>
    int evaluteAndSimpleReport(workType *workBuffer, refType *refBuffer);

private:
    size_t _dataSize;
    float _epsilon;
    hipStream_t _stream;
    int _deviceId; 

    int blocks; // the number of blocks to dispatch for all-block kernels 
};

BufferMatcher::BufferMatcher(hipStream_t stream, float tolerance, int dataSize)
{
    _epsilon = std::numeric_limits<float>::epsilon() * tolerance; 
    _dataSize = dataSize;  
    _stream = stream;

    MY_HIP_CHECK( hipGetDevice(&this->_deviceId) );

    hipDeviceProperty prop; 

    MY_HIP_CHECK( hipGetDeviceProperty(&prop, this->_deviceId) ); 

    this->blocks = prop.multiProcessorCount * 4;  // to dispatch 4 blocks per CU for all-block kernels
};

BufferMatcher::~BufferMatcher() noexcept(false)
{
}; 

template <typename workType, typename refType>
int checkBuffer(workType *workBuffer, struct bufferCheckResult *workBufferResult, refType *refBuffer, struct bufferCheckResult *refBufferResult);
{
    bufferCheckResult *kernelResult; 

    MY_HIP_CHECK( hipHostMalloc(reinterpret_cast<void**>(&kernelResult), sizeof(struct bufferCheckResult), hipHostMallocDefault) ); 

    *kernelResult = {0, 0}; 
    MY_HIP_CHECK( hipLaunchKernelGGL(checkBufferData<refType>, dim3(this->blocks), dim3(BlockSize), 0, this->_stream, refBuffer, kernelResult) ); 
    MY_HIP_CHECK( hipStreamSynchronize(this->_stream) ); 
    *refBufferResult = *kernelResult; 

    if ( refBufferResult->numNans > 0 || refBufferResult->numInf > 0 ) {
         MY_HIP_CHECK( hipHostFree(reinterpret_cast<void*>(kernelResult)) ); 
         return(-1); 
    }; 

    *kernelResult = {0, 0};
    MY_HIP_CHECK( hipLaunchKernelGGL(checkBufferData<workType>, dim3(this->blocks), dim3(BlockSize), 0, this->_stream, workBuffer, kernelResult) );
    MY_HIP_CHECK( hipStreamSynchronize(this->_stream) );
    *workBufferResult = *kernelResult;

    if ( workBufferResult->numNans > 0 || workBufferResult->numInf > 0 ) {
         MY_HIP_CHECK( hipHostFree(reinterpret_cast<void*>(kernelResult)) ); 
         return(-2); 
    }; 

    float *kernelMaxAbs; // the maximum abs calculated by the kernel
    refType *workspace;   // workspace used by the getMaxAbsValue kernel
    int allocUnitSize = sizeof(refType) < sizeof(workType)? sizeof(workType) : sizeof(refType); 

    MY_HIP_CHECK( hipHostMalloc(reinterpret_cast<void**>(&kernelMaxAbs), sizeof(float), hipHostMallocDefault) ); 
    MY_HIP_CHECK( hipMalloc(reinterpret_cast<void**>(&workspace), allocUnitSize) ); 

    MY_HIP_CHECK( hipLaunchKernelGGL(getMaxAbsValue_first_call<refType>, dim3(this->blocks), dim3(BlockSize), 0, this->_stream, refBuffer, this->_dataSize, workspace) ); 
    MY_HIP_CHECK( hipLaunchKernelGGL(getMaxAbsValue_second_call<refType>, dim3(1), dim3(BlockSize), 0, this->_stream, workspace, static_cast<size_t>(this->blocks), kernelMaxAbs) ); 
    MY_HIP_CHECK( hipStreamSynchronize(this->_stream) );

    refBufferResult->maxAbsVal = *kernelMaxAbs; 

    MY_HIP_CHECK( hipLaunchKernelGGL(getMaxAbsValue_first_call<workType>, dim3(this->blocks), dim3(BlockSize), 0, this->_stream, workBuffer, this->_dataSize, workspace) ); 
    MY_HIP_CHECK( hipLaunchKernelGGL(getMaxAbsValue_second_call<workType>, dim3(1), dim3(BlockSize), 0, this->_stream, workspace, static_cast<size_t>(this->blocks), kernelMaxAbs) ); 
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

template <typename workType, typname refType>
void BufferMatcher::computeRMS(workType *workBuffer, refType *refBuffer, float *rms)
{
    double *workspace; 
    double *kernelSum;     // the sum of the square difference calculated by the kernel 

    MY_HIP_CHECK( hipMalloc(reinterpret_cast<void**>(&workspace2), this->blocks * sizeof(double)) ); 

    MY_HIP_CHECK( hipLaunchKernelGGL(getSquareDiffSum_first_call<workType, refType>, dim3(this->blocks), dim3(BlockSize), 0, this->_stream, workBuffer, refBuffer, this->_dataSize, workspace) ); 
    MY_HIP_CHECK( hipLaunchKernelGGL(getSquareDiffSum_second_call<workType, refType>, dim3(1), dim3(BlockSize), 0, this->_stream, workspace, static_cast<size_t>(this->blocks), kernelSum) ); 
    MY_HIP_CHECK( hipStreamSynchronize(this->_stream) );

    *rms = static_cast<float>(*kernelSum);  
};

    
template <typename workType, typename refType>
int BufferMatcher::evaluteAndSimpleReport(workType *workBuffer, refType *refBuffer);
{ 
    struct bufferCheckResult refBufferResult = {0,0,0.0f};  	
    struct bufferCheckResult workBufferResult = {0,0,0.0f}; 

    
    int ret = checkBuffer<workType, refType>(workBuffer, &workBufferResult, refBuffer, &refBufferResult); 

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

    computeRMS(workBuffer, refBuffer, &rms); 

    if ( (sqrtf(rms/(float)this->_dataSize)) / refBufferResult.maxAbsVal  > this->_epsilon ) {
          std::cerr < "The evaluated data seems not consistent with that of the referrence buffer!" << std::endl; 
          return(-2); 	      
    }; 

    std::cout << "The evaluated data seems be consistent with that of the referrence buffer!" << std::endl; 

    return(0); 
};

template <typename T>
__global__ checkBufferData(T *devBuffer, struct bufferCheckResult *result, size_t bufferSize)
{
    int gridSize = hipGridDim_x * hipBlockDim_x; 
    size_t index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; 

    __shared__ blkResult; 

    if ( hipThreadIdx_x == 0 ) {
	 blkResult.numNans = 0; 
	 blkResult.numInfs = 0;
    };  

    __syncthreads(); 

    while ( index < bufferSize ) {
         if ( isnan(static_cast<float>(devBuffer[index])) ) 
	      (void)atomicAdd(&blkResult.numNans,1)

         if ( isinf(static_cast<float>(devBuffer[index])) ) 
	      (void)atomicAdd(&blkResult.numInfs,1)

	 index += gridSize;  
    }; 

    __syncthreads(); 

    if ( hipThreadIdx_x == 0 ) {
         if ( blkResult.numNans > 0 ) 
	      (void)atomicAdd(result->numNans,blkResult.numNans); 
	 if ( blkResult.numInfs > 0 )
              (void)atomicAdd(result->numInfs,blkResult.numInfs); 
    };  
}; 

// all-block kernel
template <typename T> 
__global__  getMaxAbsValue_first_call(T *devBuffer, size_t bufferSize, T *workspace)
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

    __syncthreas(); 

    for (int reduceThreads=BlockSize/2; reduceThreads > 0; reduceThreads = reduceThreads/2) {
	 if ( (hipThreadIdx_x < reduceThreads) && (maxValues[hipThreadIdx_x] < maxValues[hipThreadIdx_x+reduceThreads]) )
	      maxValues[hipThreadIdx_x] = maxValues[hipThreadIdx_x+reduceThreads]; 
	 __syncthreads(); 
    }; 

    if ( hipThreadIdx_x == 0 )
	 workspace[hipBlockIdx_x] = maxValues[0]; 
}; 

// single-block kernel
template <typename T>
__global__  getMaxAbsValue_second_call(T *workspace, size_t bufferSize, float *result)
{
    size_t index = hipThreadIdx_x; 

    float myMaxAbs = 0.0f; 

    __shared__ float maxValues[BlockSize]; 

    while ( index < bufferSize ) {
        if ( myMaxAbs < fabsf(static_cast<float>(workspace[index])) )
	     myMaxAbs = fabsf(static_cast<float>(workspace[index])); 

        index += BlockSize; 	
    }; 

    maxValues[hipThreadIdx_x] = myMaxAbs; 

    __syncthreas(); 

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
__global__ getSquareDiffSum_first_call(T1 *devBuffer1, T2 *devBuffer2, size_t bufferSize, double *workspace)
{
    int gridSize = hipGridDim_x * hipBlockDim_x; 
    size_t index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; 

    double mySum = 0.0; 

    __shared double sumValues[BlockSize]; 

    while ( index < bufferSize ) {
        float diff = static_cast<T>(devBuffer1[index]) -static_cast<T>(devBuffer2[index]); 
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

// single-block kernel for getSquareDiffSum
__global__ getSquareDiffSum_second_call(double *workspace, size_t bufferSize, double *result)
{
    size_t index = hipThreadIdx_x; 

    double mySum = 0.0; 

    __shared double sumValues[BlockSize]; 

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
