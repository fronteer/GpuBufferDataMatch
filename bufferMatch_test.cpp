#include "bufferMatch.hpp"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <cstdio>
#include <random>
#include <chrono>

const size_t dataSize = 1024*2048;  

static void *workHostBuffer;
static void *refHostBuffer; 
static void *workDevBuffer; 
static void *refDevBuffer; 

int main()
{
   using namespace kernelVerify; 

   hipStream_t  stream;

   MY_HIP_CHECK( hipStreamCreate(&stream) ); 

   BufferMatcher bufferMatch(stream, 80.0f, dataSize);  

   try {
       // test case 1 -- check the NaN and Infinite data in the float buffer
       {
           std::cout << std::endl; 
           workHostBuffer = reinterpret_cast<void*>( new float[dataSize] );  
          
           MY_HIP_CHECK( hipMalloc(&workDevBuffer, dataSize * sizeof(float)) ); 

	   unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count(); 
           std::default_random_engine generator(seed);
           std::uniform_int_distribution<size_t> distribution(0,dataSize-1);

           for (int i=0; i < dataSize; i++) {
                float val = (float)distribution(generator)/(float)dataSize - 0.5f; 

                reinterpret_cast<float*>(workHostBuffer)[i] = val; 
           }; 

           // randomly set the values of three elements to NaN
           for (int i=0; i < 3; i++) {
	        size_t index = distribution(generator); 

                reinterpret_cast<float*>(workHostBuffer)[index] = std::numeric_limits<float>::quiet_NaN(); 
	   }; 

           // randomly set the values of three elements to Infinite
           for (int i=0; i < 3; i++) {
                size_t index = distribution(generator);

                reinterpret_cast<float*>(workHostBuffer)[index] = std::numeric_limits<float>::infinity();
           };	   

           MY_HIP_CHECK( hipMemcpy(workDevBuffer, workHostBuffer, dataSize * sizeof(float), hipMemcpyHostToDevice) ); 

           struct bufferCheckResult result; 
           if ( bufferMatch.simpleCheckBuffer<float>(reinterpret_cast<float*>(workDevBuffer), &result) < 0 ) {
                std::cout << "Successfully found " << result.numNans << " NaN values and " << result.numInfs << " infinite values ! " << std::endl; 
	   }
	   else 
		std::cout << "Failed to found NaN values and infinite values in the checked buffer! " << std::endl; 

           MY_HIP_CHECK( hipFree(workDevBuffer) );
           delete [] reinterpret_cast<float*>(workHostBuffer); 
       };  

       // test case 2 -- check the situation where the referrence buffer consists of all-zero values, and the other buffer have one non-zero value with 50% probability
       {
           std::cout << std::endl; 
           workHostBuffer = reinterpret_cast<void*>( new float[dataSize] );
           refHostBuffer = reinterpret_cast<void*>( new float[dataSize] );

           MY_HIP_CHECK( hipMalloc(&workDevBuffer, dataSize * sizeof(float)) );
           MY_HIP_CHECK( hipMalloc(&refDevBuffer, dataSize * sizeof(float)) );

	   unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count(); 
           std::default_random_engine generator(seed);
           std::uniform_int_distribution<size_t> distribution1(0,dataSize-1);
           std::uniform_int_distribution<int> distribution2(0,1);

           for (int i=0; i < dataSize; i++) {
                reinterpret_cast<float*>(workHostBuffer)[i] = 0.0f;
                reinterpret_cast<float*>(refHostBuffer)[i] = 0.0f;
           };

           size_t index = distribution1(generator); 
           reinterpret_cast<float*>(workHostBuffer)[index] = (float) distribution2(generator);  // 0, 1

           MY_HIP_CHECK( hipMemcpy(workDevBuffer, workHostBuffer, dataSize * sizeof(float), hipMemcpyHostToDevice) );
           MY_HIP_CHECK( hipMemcpy(refDevBuffer, refHostBuffer, dataSize * sizeof(float), hipMemcpyHostToDevice) );

           struct bufferCheckResult workBufferResult, refBufferResult; 
           int retVal = bufferMatch.checkBuffers<float,float>(reinterpret_cast<float*>(workDevBuffer), &workBufferResult, reinterpret_cast<float*>(refDevBuffer), &refBufferResult);
             
           bufferMatch.evaluateAndSimpleReport<float,float>(reinterpret_cast<float*>(workDevBuffer),reinterpret_cast<float*>(refDevBuffer)); 

           MY_HIP_CHECK( hipFree(workDevBuffer) );
           MY_HIP_CHECK( hipFree(refDevBuffer) );
           delete [] reinterpret_cast<float*>(workHostBuffer); 
           delete [] reinterpret_cast<float*>(refHostBuffer); 
       }; 

       // test case 3 -- compare between buffers of same type which are initialized to same data
       {
           std::cout << std::endl; 
           workHostBuffer = reinterpret_cast<void*>( new float[dataSize] );  
           refHostBuffer = reinterpret_cast<void*>( new float[dataSize] );  
   
           MY_HIP_CHECK( hipMalloc(&workDevBuffer, dataSize * sizeof(float)) ); 
           MY_HIP_CHECK( hipMalloc(&refDevBuffer, dataSize * sizeof(float)) ); 

	   unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count(); 
           std::default_random_engine generator(seed);
           std::uniform_int_distribution<size_t> distribution(0,dataSize-1);	   
   
           // set up the two buffers with complete same data
           for (int i=0; i < dataSize; i++) {
                float val = (float)distribution(generator) / (float)dataSize - 0.5f; 

                reinterpret_cast<float*>(workHostBuffer)[i] = val; 
                reinterpret_cast<float*>(refHostBuffer)[i] = val; 
           }; 
   
           MY_HIP_CHECK( hipMemcpy(workDevBuffer, workHostBuffer, dataSize * sizeof(float), hipMemcpyHostToDevice) ); 
           MY_HIP_CHECK( hipMemcpy(refDevBuffer, refHostBuffer, dataSize * sizeof(float), hipMemcpyHostToDevice) ); 

           bufferMatch.evaluateAndSimpleReport<float,float>(reinterpret_cast<float*>(workDevBuffer),reinterpret_cast<float*>(refDevBuffer)); 

           MY_HIP_CHECK( hipFree(workDevBuffer) ); 
           MY_HIP_CHECK( hipFree(refDevBuffer) ); 
           delete [] reinterpret_cast<float*>(workHostBuffer); 
           delete [] reinterpret_cast<float*>(refHostBuffer); 
       }; 

       // test case 4 -- compare between buffers of same type which are initialized to same data, but the working buffer has 10 values corruptted to zero
       {
           std::cout << std::endl; 
           workHostBuffer = reinterpret_cast<void*>( new float[dataSize] );
           refHostBuffer = reinterpret_cast<void*>( new float[dataSize] );

           MY_HIP_CHECK( hipMalloc(&workDevBuffer, dataSize * sizeof(float)) );
           MY_HIP_CHECK( hipMalloc(&refDevBuffer, dataSize * sizeof(float)) );

	   unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count(); 
           std::default_random_engine generator(seed);
           std::uniform_int_distribution<size_t> distribution(0,dataSize-1);

           // set up the two buffers with complete same data
           for (int i=0; i < dataSize; i++) {
                float val = (float)distribution(generator) / (float)dataSize - 0.5f;

                reinterpret_cast<float*>(workHostBuffer)[i] = val;
                reinterpret_cast<float*>(refHostBuffer)[i] = val;
           };

           for (int i=0; i < 10; i++) {
                size_t index = distribution(generator) % dataSize; 

                reinterpret_cast<float*>(workHostBuffer)[index] = 0.0f; 
	   }; 

           MY_HIP_CHECK( hipMemcpy(workDevBuffer, workHostBuffer, dataSize * sizeof(float), hipMemcpyHostToDevice) );
           MY_HIP_CHECK( hipMemcpy(refDevBuffer, refHostBuffer, dataSize * sizeof(float), hipMemcpyHostToDevice) );

           bufferMatch.evaluateAndSimpleReport<float,float>(reinterpret_cast<float*>(workDevBuffer),reinterpret_cast<float*>(refDevBuffer));

           MY_HIP_CHECK( hipFree(workDevBuffer) );
           MY_HIP_CHECK( hipFree(refDevBuffer) );
           delete [] reinterpret_cast<float*>(workHostBuffer);
           delete [] reinterpret_cast<float*>(refHostBuffer);
       };

       // test case 5 -- check the NaN and Infinite data in the fp16 buffer
       {
           std::cout << std::endl;
           workHostBuffer = reinterpret_cast<void*>( new half_float::half[dataSize] );

           MY_HIP_CHECK( hipMalloc(&workDevBuffer, dataSize * sizeof(half_float::half)) );

           unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
           std::default_random_engine generator(seed);
           std::uniform_int_distribution<size_t> distribution(0,dataSize-1);

           for (int i=0; i < dataSize; i++) {
		half_float::half val = static_cast<half_float::half>( (float)distribution(generator)/(float)dataSize - 0.5f ); 

                reinterpret_cast<half_float::half*>(workHostBuffer)[i] = val;
           };

           // randomly set the values of three elements to NaN
           for (int i=0; i < 3; i++) {
                size_t index = distribution(generator);

                reinterpret_cast<half_float::half*>(workHostBuffer)[index] = static_cast<half_float::half>( std::numeric_limits<float>::quiet_NaN() );
           };

           // randomly set the values of three elements to Infinite
           for (int i=0; i < 3; i++) {
                size_t index = distribution(generator);

                reinterpret_cast<half_float::half*>(workHostBuffer)[index] = static_cast<half_float::half>( std::numeric_limits<float>::infinity() );
           };

           MY_HIP_CHECK( hipMemcpy(workDevBuffer, workHostBuffer, dataSize * sizeof(half_float::half), hipMemcpyHostToDevice) );

           struct bufferCheckResult result;
           if ( bufferMatch.simpleCheckBuffer<half_float::half>(reinterpret_cast<half_float::half*>(workDevBuffer), &result) < 0 ) {
                std::cout << "Successfully found " << result.numNans << " NaN values and " << result.numInfs << " infinite values ! " << std::endl;
           }
           else
                std::cout << "Failed to found NaN values and infinite values in the checked buffer! " << std::endl;

           MY_HIP_CHECK( hipFree(workDevBuffer) );
           delete [] reinterpret_cast<half_float::half*>(workHostBuffer);
       };

   } 
   catch ( std::exception &ex ) {
      std::cerr << ex.what() << std::endl; 
      return(-1); 
   }; 

   return(0); 
}; 

