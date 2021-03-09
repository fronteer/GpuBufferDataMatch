
1) Define internal APIs to be used by MIOpen driver/ and test/.

2) The testing codes use the APIs to judge whether the output (which are usually float values) from the tested kernel is valid by comparing 
   this output to the output of the other kernel (which is usually a reference kernel easier to implement and validate).

3) The kernel buffers are compared by computing Root Mean Square Difference of the them. If the computed RMSD value is less than user specified
   tolerance value, the two buffers are regarded as equal. 

4) One usage situation is to compare the output of the tested convolution kernel to the output of the naive-conv kernel.

