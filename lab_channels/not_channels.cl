//ACL Kernel

__kernel void process_data(
		  __global const float *restrict input, 
		  __global float *restrict output, 
		  unsigned N)

{
	for (unsigned i = 0; i < N; i++)	
	{
		output[i] = input[i] * input[i];
	}
}
