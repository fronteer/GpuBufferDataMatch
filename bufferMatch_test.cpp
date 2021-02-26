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
#include "bufferMatch.hpp"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <cstdio>
#include <random>
#include <chrono>

const size_t dataSize = 8192*8192;  

static void *workHostBuffer;
static void *refHostBuffer; 
static void *workDevBuffer; 
static void *refDevBuffer; 

int main()
{
   using namespace kernelVerify; 
   using half_float::half; 

   hipStream_t  stream;

   MY_HIP_CHECK( hipStreamCreate(&stream) ); 

   BufferCompare bufferMatch(stream, 80.0f, dataSize);  
   BufferCompare bufferMatch2(stream, 200.0f, dataSize);  // the tolerance value should be further improved if conversion loss to be considered in specific situation 

   int testNo = 0; 

   try {
       // test case 1 -- check the NaN and Infinite data in the float buffer
       {
           testNo++; 
           std::cout << std::endl << "------- Testing case " << testNo << " ------- " << std::endl; 
           workHostBuffer = reinterpret_cast<void*>( new float[dataSize] );  
          
           MY_HIP_CHECK( hipMalloc(&workDevBuffer, dataSize * sizeof(float)) ); 

	   unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count(); 
           std::default_random_engine generator(seed);
           std::uniform_int_distribution<size_t> distribution(0,dataSize-1);

           for (int i=0; i < dataSize; i++) {
                auto val = static_cast<float>(distribution(generator)) / dataSize - 0.5f; 

                reinterpret_cast<float*>(workHostBuffer)[i] = val; 
           }; 

           // randomly set the values of three elements to NaN
           for (int i=0; i < 3; i++) {
	        size_t index = distribution(generator); 

                reinterpret_cast<float*>(workHostBuffer)[index] = std::numeric_limits<float>::quiet_NaN(); 
	   }; 

           // randomly set the values of three elements to Infinite
           for (int i=0; i < 5; i++) {
                size_t index = distribution(generator);

                reinterpret_cast<float*>(workHostBuffer)[index] = std::numeric_limits<float>::infinity();
           };	   

           MY_HIP_CHECK( hipMemcpy(workDevBuffer, workHostBuffer, dataSize * sizeof(float), hipMemcpyHostToDevice) ); 

           struct bufferCheckStat stat; 
           if ( bufferMatch.simpleCheckBuffer(reinterpret_cast<float*>(workDevBuffer), stat) < 0 ) {
                std::cout << "Successfully found " << stat.numNans << " NaN values and " << stat.numInfs << " infinite values ! " << std::endl; 
	   }
	   else 
		std::cout << "Failed to found NaN values and infinite values in the checked buffer! " << std::endl; 

           MY_HIP_CHECK( hipFree(workDevBuffer) );
           delete [] reinterpret_cast<float*>(workHostBuffer); 
       };  

       // test case 2 -- check the situation where the referrence buffer consists of all-zero values, and the other buffer have one non-zero value with 50% probability
       {
           testNo++;
           std::cout << std::endl << "------- Testing case " << testNo << " ------- " << std::endl; 
           workHostBuffer = reinterpret_cast<void*>( new float[dataSize] );  
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
           reinterpret_cast<float*>(workHostBuffer)[index] = static_cast<float>(distribution2(generator));  // 0, 1

           MY_HIP_CHECK( hipMemcpy(workDevBuffer, workHostBuffer, dataSize * sizeof(float), hipMemcpyHostToDevice) );
           MY_HIP_CHECK( hipMemcpy(refDevBuffer, refHostBuffer, dataSize * sizeof(float), hipMemcpyHostToDevice) );

           struct bufferCheckStat workBufferStat, refBufferStat; 
           int retVal = bufferMatch.checkBuffers(reinterpret_cast<float*>(workDevBuffer), workBufferStat, reinterpret_cast<float*>(refDevBuffer), refBufferStat);
             
           bufferMatch.evaluateAndSimpleReport(reinterpret_cast<float*>(workDevBuffer),reinterpret_cast<float*>(refDevBuffer)); 

           MY_HIP_CHECK( hipFree(workDevBuffer) );
           MY_HIP_CHECK( hipFree(refDevBuffer) );
           delete [] reinterpret_cast<float*>(workHostBuffer); 
           delete [] reinterpret_cast<float*>(refHostBuffer); 
       }; 

       // test case 3 -- compare between buffers of same float type which are initialized to same data
       {
           testNo++;
           std::cout << std::endl << "------- Testing case " << testNo << " ------- " << std::endl; 
           workHostBuffer = reinterpret_cast<void*>( new float[dataSize] );  
           workHostBuffer = reinterpret_cast<void*>( new float[dataSize] );  
           refHostBuffer = reinterpret_cast<void*>( new float[dataSize] );  
   
           MY_HIP_CHECK( hipMalloc(&workDevBuffer, dataSize * sizeof(float)) ); 
           MY_HIP_CHECK( hipMalloc(&refDevBuffer, dataSize * sizeof(float)) ); 

	   unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count(); 
           std::default_random_engine generator(seed);
           std::uniform_int_distribution<size_t> distribution(0,dataSize-1);	   
   
           // set up the two buffers with completely same data values
           for (int i=0; i < dataSize; i++) {
                auto val = static_cast<float>(distribution(generator)) / dataSize - 0.5f; 

                reinterpret_cast<float*>(workHostBuffer)[i] = val; 
                reinterpret_cast<float*>(refHostBuffer)[i] = val; 
           }; 
   
           MY_HIP_CHECK( hipMemcpy(workDevBuffer, workHostBuffer, dataSize * sizeof(float), hipMemcpyHostToDevice) ); 
           MY_HIP_CHECK( hipMemcpy(refDevBuffer, refHostBuffer, dataSize * sizeof(float), hipMemcpyHostToDevice) ); 

           bufferMatch.evaluateAndSimpleReport(reinterpret_cast<float*>(workDevBuffer),reinterpret_cast<float*>(refDevBuffer)); 

           MY_HIP_CHECK( hipFree(workDevBuffer) ); 
           MY_HIP_CHECK( hipFree(refDevBuffer) ); 
           delete [] reinterpret_cast<float*>(workHostBuffer); 
           delete [] reinterpret_cast<float*>(refHostBuffer); 
       }; 

       // test case 4 -- compare between buffers of same type which are initialized to same data, but the working buffer has 10 values corruptted to zero
       {
           testNo++;
           std::cout << std::endl << "------- Testing case " << testNo << " ------- " << std::endl; 
           workHostBuffer = reinterpret_cast<void*>( new float[dataSize] );
           refHostBuffer = reinterpret_cast<void*>( new float[dataSize] );

           MY_HIP_CHECK( hipMalloc(&workDevBuffer, dataSize * sizeof(float)) );
           MY_HIP_CHECK( hipMalloc(&refDevBuffer, dataSize * sizeof(float)) );

	   unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count(); 
           std::default_random_engine generator(seed);
           std::uniform_int_distribution<size_t> distribution(0,dataSize-1);

           // set up the two buffers with completely same data values
           for (int i=0; i < dataSize; i++) {
                auto val = static_cast<float>(distribution(generator)) / dataSize - 0.5f;

                reinterpret_cast<float*>(workHostBuffer)[i] = val;
                reinterpret_cast<float*>(refHostBuffer)[i] = val;
           };

           // corrupt the some values in the working buffer
           for (int i=0; i < 10; i++) {
                size_t index = distribution(generator) % dataSize; 

                reinterpret_cast<float*>(workHostBuffer)[index] = 0.0f; 
	   }; 

           MY_HIP_CHECK( hipMemcpy(workDevBuffer, workHostBuffer, dataSize * sizeof(float), hipMemcpyHostToDevice) );
           MY_HIP_CHECK( hipMemcpy(refDevBuffer, refHostBuffer, dataSize * sizeof(float), hipMemcpyHostToDevice) );

           bufferMatch.evaluateAndSimpleReport(reinterpret_cast<float*>(workDevBuffer),reinterpret_cast<float*>(refDevBuffer));

           MY_HIP_CHECK( hipFree(workDevBuffer) );
           MY_HIP_CHECK( hipFree(refDevBuffer) );
           delete [] reinterpret_cast<float*>(workHostBuffer);
           delete [] reinterpret_cast<float*>(refHostBuffer);
       };

       // test case 5 -- check the NaN and Infinite data in the fp16 buffer
       {
           testNo++;
           std::cout << std::endl << "------- Testing case " << testNo << " ------- " << std::endl; 
           workHostBuffer = reinterpret_cast<void*>( new half[dataSize] );

           MY_HIP_CHECK( hipMalloc(&workDevBuffer, dataSize * sizeof(half)) );

           unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
           std::default_random_engine generator(seed);
           std::uniform_int_distribution<size_t> distribution(0,dataSize-1);

           for (int i=0; i < dataSize; i++) {
		auto val = static_cast<half>(static_cast<float>(distribution(generator)) / dataSize - 0.5f); 

                reinterpret_cast<half*>(workHostBuffer)[i] = val;
           };

           // randomly set the values of three elements to NaN
           for (int i=0; i < 3; i++) {
                size_t index = distribution(generator);

                reinterpret_cast<half*>(workHostBuffer)[index] = std::numeric_limits<half>::quiet_NaN();
           };

           // randomly set the values of three elements to Infinite
           for (int i=0; i < 5; i++) {
                size_t index = distribution(generator);

                reinterpret_cast<half*>(workHostBuffer)[index] = std::numeric_limits<half>::infinity();
           };

           MY_HIP_CHECK( hipMemcpy(workDevBuffer, workHostBuffer, dataSize * sizeof(half), hipMemcpyHostToDevice) );

           struct bufferCheckStat stat;
           if ( bufferMatch.simpleCheckBuffer(reinterpret_cast<half*>(workDevBuffer), stat) < 0 ) {
                std::cout << "Successfully found " << stat.numNans << " NaN values and " << stat.numInfs << " infinite values ! " << std::endl;
           }
           else
                std::cout << "Failed to found NaN values and infinite values in the checked buffer! " << std::endl;

           MY_HIP_CHECK( hipFree(workDevBuffer) );
           delete [] reinterpret_cast<half*>(workHostBuffer);
       };

       // test case 6 -- compare between a working buffer of fp16 type and a referrence buffer of float type, data of which are initialized to same values; 
       {
           testNo++;
           std::cout << std::endl << "------- Testing case " << testNo << " ------- " << std::endl; 
           workHostBuffer = reinterpret_cast<void*>( new half[dataSize] );
           refHostBuffer = reinterpret_cast<void*>( new float[dataSize] );

           MY_HIP_CHECK( hipMalloc(&workDevBuffer, dataSize * sizeof(half)) );
           MY_HIP_CHECK( hipMalloc(&refDevBuffer, dataSize * sizeof(float)) );

           unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
           std::default_random_engine generator(seed);
           std::uniform_int_distribution<size_t> distribution(0,dataSize-1);

           // set up the two buffers with completely same data values
           for (int i=0; i < dataSize; i++) {
		half val = static_cast<half>(static_cast<float>(distribution(generator)) / dataSize - 0.5f); 

                reinterpret_cast<half*>(workHostBuffer)[i] = val;
                reinterpret_cast<float*>(refHostBuffer)[i] = static_cast<float>(val);
           };

           MY_HIP_CHECK( hipMemcpy(workDevBuffer, workHostBuffer, dataSize * sizeof(half), hipMemcpyHostToDevice) );
           MY_HIP_CHECK( hipMemcpy(refDevBuffer, refHostBuffer, dataSize * sizeof(float), hipMemcpyHostToDevice) );

           bufferMatch2.evaluateAndSimpleReport(reinterpret_cast<half*>(workDevBuffer),reinterpret_cast<float*>(refDevBuffer));

           MY_HIP_CHECK( hipFree(workDevBuffer) );
           MY_HIP_CHECK( hipFree(refDevBuffer) );
           delete [] reinterpret_cast<half*>(workHostBuffer);
           delete [] reinterpret_cast<float*>(refHostBuffer);
       };
       
       // test case 7 -- compare between a working buffer of fp16 type and a referrence buffer of float type, data of which are initialized to same values, 
       //                but the working buffer has 10 values corruptted to zero              
       {
           testNo++;
           std::cout << std::endl << "------- Testing case " << testNo << " ------- " << std::endl;
           workHostBuffer = reinterpret_cast<void*>( new half[dataSize] );
           refHostBuffer = reinterpret_cast<void*>( new float[dataSize] );

           MY_HIP_CHECK( hipMalloc(&workDevBuffer, dataSize * sizeof(half)) );
           MY_HIP_CHECK( hipMalloc(&refDevBuffer, dataSize * sizeof(float)) );

           unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
           std::default_random_engine generator(seed);
           std::uniform_int_distribution<size_t> distribution(0,dataSize-1);

           // set up the two buffers with completely same data values
           for (int i=0; i < dataSize; i++) {
		half val = static_cast<half>(static_cast<float>(distribution(generator)) / dataSize - 0.5f); 

                reinterpret_cast<half*>(workHostBuffer)[i] = val; 
                reinterpret_cast<float*>(refHostBuffer)[i] = static_cast<float>(val);
           };

           // corrupt the some values in the working buffer
           for (int i=0; i < 10; i++) {
                size_t index = distribution(generator) % dataSize;

                reinterpret_cast<half*>(workHostBuffer)[index] = static_cast<half>(0.0f);
           };

           MY_HIP_CHECK( hipMemcpy(workDevBuffer, workHostBuffer, dataSize * sizeof(half), hipMemcpyHostToDevice) );
           MY_HIP_CHECK( hipMemcpy(refDevBuffer, refHostBuffer, dataSize * sizeof(float), hipMemcpyHostToDevice) );

           bufferMatch2.evaluateAndSimpleReport(reinterpret_cast<half*>(workDevBuffer),reinterpret_cast<float*>(refDevBuffer));

           MY_HIP_CHECK( hipFree(workDevBuffer) );
           MY_HIP_CHECK( hipFree(refDevBuffer) );
           delete [] reinterpret_cast<half*>(workHostBuffer);
           delete [] reinterpret_cast<float*>(refHostBuffer);
       };

       // test case 10 -- compare between a working buffer of int8_t type and a referrence buffer of float type, data of which are initialized to same values; 
       {
           testNo++;
           std::cout << std::endl << "------- Testing case " << testNo << " ------- " << std::endl;
           workHostBuffer = reinterpret_cast<void*>( new int8_t[dataSize] );
           refHostBuffer = reinterpret_cast<void*>( new float[dataSize] );

           MY_HIP_CHECK( hipMalloc(&workDevBuffer, dataSize * sizeof(int8_t)) );
           MY_HIP_CHECK( hipMalloc(&refDevBuffer, dataSize * sizeof(float)) );

           unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
           std::default_random_engine generator(seed);
           std::uniform_int_distribution<size_t> distribution(0,dataSize-1);

           // set up the two buffers with completely same data values
           int8_t maxVal = std::numeric_limits<int8_t>::max(); 
           for (int i=0; i < dataSize; i++) {
                int8_t val = static_cast<int8_t>( (static_cast<float>(distribution(generator)) / dataSize - 0.5f) * maxVal );

                reinterpret_cast<int8_t*>(workHostBuffer)[i] = val;
                reinterpret_cast<float*>(refHostBuffer)[i] = static_cast<float>(val);
           };

           MY_HIP_CHECK( hipMemcpy(workDevBuffer, workHostBuffer, dataSize * sizeof(int8_t), hipMemcpyHostToDevice) );
           MY_HIP_CHECK( hipMemcpy(refDevBuffer, refHostBuffer, dataSize * sizeof(float), hipMemcpyHostToDevice) );

           bufferMatch2.evaluateAndSimpleReport(reinterpret_cast<int8_t*>(workDevBuffer),reinterpret_cast<float*>(refDevBuffer));

           MY_HIP_CHECK( hipFree(workDevBuffer) );
           MY_HIP_CHECK( hipFree(refDevBuffer) );
           delete [] reinterpret_cast<int8_t*>(workHostBuffer);
           delete [] reinterpret_cast<float*>(refHostBuffer);
       };

       // test case 11 -- compare between a working buffer of int8_t type and a referrence buffer of float type, data of which are initialized to same values, 
       //                but the working buffer has 10 values corruptted to zero              
       {
           testNo++;
           std::cout << std::endl << "------- Testing case " << testNo << " ------- " << std::endl;
           workHostBuffer = reinterpret_cast<void*>( new int8_t[dataSize] );
           refHostBuffer = reinterpret_cast<void*>( new float[dataSize] );

           MY_HIP_CHECK( hipMalloc(&workDevBuffer, dataSize * sizeof(int8_t)) );
           MY_HIP_CHECK( hipMalloc(&refDevBuffer, dataSize * sizeof(float)) );

           unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
           std::default_random_engine generator(seed);
           std::uniform_int_distribution<size_t> distribution(0,dataSize-1);

           // set up the two buffers with completely same data values
           int8_t maxVal = std::numeric_limits<int8_t>::max(); 
           for (int i=0; i < dataSize; i++) {
                int8_t val = static_cast<int8_t>( (static_cast<float>(distribution(generator)) / dataSize - 0.5f) * maxVal );

                reinterpret_cast<int8_t*>(workHostBuffer)[i] = val;
                reinterpret_cast<float*>(refHostBuffer)[i] = static_cast<float>(val);
           };

           // corrupt some values in the working buffer
           for (int i=0; i < 10; i++) {
                size_t index = distribution(generator) % dataSize;

                reinterpret_cast<int8_t*>(workHostBuffer)[index] = static_cast<int8_t>(0.0f);
           };

           MY_HIP_CHECK( hipMemcpy(workDevBuffer, workHostBuffer, dataSize * sizeof(int8_t), hipMemcpyHostToDevice) );
           MY_HIP_CHECK( hipMemcpy(refDevBuffer, refHostBuffer, dataSize * sizeof(float), hipMemcpyHostToDevice) );

           bufferMatch2.evaluateAndSimpleReport(reinterpret_cast<int8_t*>(workDevBuffer),reinterpret_cast<float*>(refDevBuffer));

           MY_HIP_CHECK( hipFree(workDevBuffer) );
           MY_HIP_CHECK( hipFree(refDevBuffer) );
           delete [] reinterpret_cast<int8_t*>(workHostBuffer);
           delete [] reinterpret_cast<float*>(refHostBuffer);
       };

   } 
   catch ( std::exception &ex ) {
      std::cerr << ex.what() << std::endl; 
      return(-1); 
   }; 

   return(0); 
}; 

