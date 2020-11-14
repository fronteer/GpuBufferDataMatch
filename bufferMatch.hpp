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

class BufferMatcher
{
    static const int BlockSize = 256; 	

    BufferMatcher(hipStream_t stream, float tolerance, int dataSize);

    ~BufferMatcher() noexcept(false); 

    template <typename T>
    int checkBuffer(T *devBuffer, struct bufferCheckResult *result); 

    template <typename T1, typname T2>
    void computeRMS(T1 *devBuffer1, T2 *devBuffer2, float *rms); 

    template <typename workType, typename refType>
    int evaluteAndSimpleReport(workType *workBuffer, refType *refBuffer);

private:
    int _dataSize;
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

template <typename T>
int BufferMatcher::checkBuffer(T *devBuffer, struct bufferCheckResult *result)
{
    bufferCheckResult *kernelResult; 

    MY_HIP_CHECK( hipHostMalloc(reinterpret_cast<void**>(&kernelResult, sizeof(struct bufferCheckResult), hipHostMallocDefault) ); 

    *kernelResult = {0, 0}; 

    MY_HIP_CHECK( hipLaunchKernelGGL(checkBufferData<T>, dim3(this->blocks), dim3(BlockSize), 0, this->_stream, devBuffer, kernelResult) ); 

    MY_HIP_CHECK( hipStreamSynchronize(this->_stream) ); 

    *result = *kernelResult; 

    if ( result->numNans > 0 && result->numInf > 0 ) 
         return(-1); 
 
    return(-1); 
}; 

template <typename T1, typname T2>
void BufferMatcher::computeRMS(T1 *devBuffer1, T2 *devBuffer2, float *rms)
{
};

    
template <typename workType, typename refType>
int BufferMatcher::evaluteAndSimpleReport(workType *workBuffer, refType *refBuffer);
    { 
         struct bufferCheckResult result = {0,0}; 

         if ( checkBuffer(refBuffer, &result) != 0 ) {
  	      std::cerr << result.numNans << " Some NaN/Infinite values found in the referrence data buffer." << std::endl; 
              std::cerr << "The evaluation could not be executed!" << std::endl; 
              return(-2);         
	 }

         if ( checkBuffer(workBuffer, &result) != 0 ) {
              if ( result.numNans > 0 )
                   std::cerr << result.numNans << " NaN values found in the evaluated data buffer." << std::endl;
              if ( result.numInfs > 0 )
                   std::cerr << result.numInfs << " Infinite values found in the evaluated data buffer." << std::endl;
              return(-3);
         }

	 float rms; 

         computeRMS(workBuffer, refBuffer, &rms); 

	 if ( rms > this->_epsilon ) {
              std::cerr < "The evaluated data seems not consistent with that of the referrence buffer!" << std::endl; 
              return(-1); 	      
	 }; 

	 std::cout << "The evaluated data seems be consistent with that of the referrence buffer!" << std::endl; 

         return(0); 
    }; 
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
__global__  getMaxAbsValue_first_call(T *devBuffer, T *workspace, size_t bufferSize)
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
__global__  getMaxAbsValue_second_call(T *workspace, T *result, size_t bufferSize)
{
    size_t index = hipThreadIdx_x; 

    float myMaxAbs = 0.0f; 

    __shared__ float maxValues[BlockSize]; 

    while ( index < bufferSize ) {
        if ( myMaxAbs < fabsf(static_cast<float>(devBuffer[index])) )
	     myMaxAbs = fabsf(static_cast<float>(devBuffer[index])); 

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
         *result = static_cast<T>(maxValues[0]); 
};

// all-block kernel
template <typename T1, typename T2>
__global__ getSquareDiffSum_first_call(T1 *devBuffer1, T2 *devBuffer2, double *workspace, size_t bufferSize)
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
__global__ getSquareDiffSum_second_call(double *workspace, double *result, int bufferSize)
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
