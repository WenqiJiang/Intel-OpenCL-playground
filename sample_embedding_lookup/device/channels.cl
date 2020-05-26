//ACL Kernel
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel float c0 __attribute__((depth(128)));
channel float c1 __attribute__((depth(128)));

__attribute__((max_global_work_dim(0)))
__kernel void host_reader(__global const float * restrict input, unsigned N) 
{
	for (unsigned i=0; i<N; i++)
	{
		write_channel_intel(c0, input[i]);
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void process_data(unsigned N)
{
	for (int i=0; i<N; i++) {
		float idata =  read_channel_intel(c0);
		idata = idata*idata;
		write_channel_intel(c1, idata);
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void host_writer(__global float * restrict output, unsigned N)
{
	int err;
	float value=0;
	for (unsigned i=0; i<N; i++)
	{
		 value = read_channel_intel(c1);
		 output[i] = value;
	}
}
