//ACL Kernel


__kernel void host_reader(__global const float *input) 
{
}

__kernel void process_data(unsigned N)
{
	for (i=0; i<N; i++) {
		idata =  read_channel_intel(c0);
		idata = idata*idata;
		write_channel_intel(c1,idata);
	}
}

__kernel void host_writer(__global float *output)
{
}
