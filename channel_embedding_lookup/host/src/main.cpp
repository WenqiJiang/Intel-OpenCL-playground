#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

#include "constants.hpp"

using namespace aocl_utils;

#define KERNEL_NUM 1
// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
cl_program program = NULL;
cl_kernel kernel_embedding_lookup;
cl_kernel kernel_reduction_sum;
cl_command_queue queue_embedding_lookup;
cl_command_queue queue_reduction_sum;
// cl_command_queue queue;
cl_mem input_a_buf, input_b_buf, output_buf;

// Problem data.
unsigned N = BANK_SIZE; // problem size

// CPU side 
scoped_aligned_ptr<D_TYPE> input_a, input_b; // num_devices elements
scoped_aligned_ptr<D_TYPE> output; // num_devices elements
scoped_array<D_TYPE> ref_output; // num_devices elements

// Function prototypes
D_TYPE rand_int();

// Entry point.
int main(int argc, char **argv) {
  Options options(argc, argv);

  /////////////////////////     Init OpenCL     /////////////////////////
  cl_int status;

  printf("Initializing OpenCL\n");

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL device.
  device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  printf("Platform: %s\n", getPlatformName(platform).c_str());
  printf("Using %d device(s)\n", num_devices);
  for(unsigned i = 0; i < num_devices; ++i) {
    printf("  %s\n", getDeviceName(device[i]).c_str());
  }
  // WENQI: only 1 device is supported
  if (num_devices != 1) {
    return -1;
  }

  // Create the context.
  context = clCreateContext(NULL, num_devices, device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
  std::string binary_file = getBoardBinaryFile("embedding_lookup", device[0]);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Command queue.
  queue_embedding_lookup = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue_embedding_lookup");
  queue_reduction_sum = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue_reduction_sum");
  // queue = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);
  // checkError(status, "Failed to create command queue");

  // Kernel.
  // here, kernel name is the top-level function name
  const char *kernel_embedding_lookup_name = "embedding_lookup";
  kernel_embedding_lookup = clCreateKernel(program, kernel_embedding_lookup_name, &status);
  checkError(status, "Failed to create kernel_embedding_lookup");
  const char *kernel_reduction_sum_name = "reduction_sum";
  kernel_reduction_sum = clCreateKernel(program, kernel_reduction_sum_name, &status);
  checkError(status, "Failed to create kernel_reduction_sum");

  // Input buffers.
  input_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(D_TYPE), NULL, &status);
  checkError(status, "Failed to create buffer for input A");

  // Output buffer.
  output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, BATCH_SIZE * sizeof(D_TYPE), NULL, &status);
  checkError(status, "Failed to create buffer for output");
  std::cout << "finished creating buffers." << std::endl;
  /////////////////////////     Init Problem     /////////////////////////
  if(num_devices == 0) {
    checkError(-1, "No devices");
  }

  const int access_idx[] = {3, 99, 38, 72, 29, 57, 1, 72, 36, 76, 35, 50, 37, 57, 
      13, 66, 26, 70, 41, 93, 48, 82, 44, 78, 25, 52, 3, 92, 36, 56, 46, 88};

  input_a.reset(N);
  output.reset(N);
  ref_output.reset(BATCH_SIZE);

  for(unsigned j = 0; j < N; ++j) {
    input_a[j] = rand_int();
  }

  // software results
  for (int item = 0; item < BATCH_SIZE; item++) {
    int idx = access_idx[item];
    // 3 tables
    int result = 0;
    for (int count = 0; count < DATA_SIZE_0; count++) {
      result += input_a[ADDR_START_TABLE_0 + idx * DATA_SIZE_0 + count];
    }
    for (int count = 0; count < DATA_SIZE_1; count++) {
      result += input_a[ADDR_START_TABLE_1 + idx * DATA_SIZE_1 + count];
    }
    for (int count = 0; count < DATA_SIZE_2; count++) {
      result += input_a[ADDR_START_TABLE_2 + idx * DATA_SIZE_2 + count];
    }
    ref_output[item] = result;
  }
  std::cout << "finished computing sw results." << std::endl;

  /////////////////////////     Run Kernel     /////////////////////////

  const double start_time = getCurrentTimestamp();

  // Launch the problem for each device.
  cl_event kernel_embedding_lookup_event;
  cl_event kernel_reduction_sum_event;  
  // write: host -> device embedding
  // finish: device -> host results
  cl_event write_event;
  cl_event finish_event; 
  std::cout << "finished declare events." << std::endl;

  status = clEnqueueWriteBuffer(queue_embedding_lookup, input_a_buf, CL_FALSE,
      0, N * sizeof(D_TYPE), input_a, 0, NULL, &write_event);
  // status = clEnqueueWriteBuffer(queue, input_a_buf, CL_FALSE,
  //     0, N * sizeof(D_TYPE), input_a, 0, NULL, &write_event);
  checkError(status, "Failed to transfer input A");

  // Set kernel arguments
  unsigned argi_embedding_lookup = 0;
  status = clSetKernelArg(kernel_embedding_lookup, argi_embedding_lookup++, sizeof(cl_mem), &input_a_buf);
  checkError(status, "Failed to set argument %d", argi_embedding_lookup - 1);

  unsigned argi_reduction_sum = 0;
  status = clSetKernelArg(kernel_reduction_sum, argi_reduction_sum++, sizeof(cl_mem), &output_buf);
  checkError(status, "Failed to set argument %d", argi_reduction_sum - 1);

  const size_t global_work_size = 1;
  const size_t local_work_size = 1;
  // printf("Launching for device (%zd elements)\n", global_work_size);
  // status = clEnqueueNDRangeKernel(queue_embedding_lookup, kernel_embedding_lookup, 1, NULL,
  //     &global_work_size, NULL, 1, write_event, &kernel_embedding_lookup_event);

  status = clEnqueueNDRangeKernel(queue_embedding_lookup, kernel_embedding_lookup, 1, NULL,
      &global_work_size , &local_work_size, 1, &write_event, &kernel_embedding_lookup_event);
  checkError(status, "Failed to launch kernel_embedding_lookup");
  status = clEnqueueNDRangeKernel(queue_reduction_sum, kernel_reduction_sum, 1, NULL, 
      &global_work_size , &local_work_size, 1, &write_event, &kernel_reduction_sum_event);
  checkError(status, "Failed to launch kernel_reduction_sum");

  // Read the result. This the final operation.
  status = clEnqueueReadBuffer(queue_reduction_sum, output_buf, CL_FALSE,
      0, BATCH_SIZE * sizeof(D_TYPE), output, 1, &kernel_reduction_sum_event, &finish_event);
  
  // Wait for all devices to finish.
  clWaitForEvents(1, &finish_event);

  const double end_time = getCurrentTimestamp();

  // Wall-clock time taken.
  printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);

  // Get kernel times using the OpenCL event profiling API.
  cl_ulong time_ns = getStartEndTime(kernel_embedding_lookup_event);
  printf("Kernel time (device %d): %0.3f ms\n", double(time_ns) * 1e-6);

  // Release all events.  
  clReleaseEvent(write_event);
  clReleaseEvent(kernel_embedding_lookup_event);
  clReleaseEvent(kernel_reduction_sum_event);
  clReleaseEvent(finish_event);

  // Verify results.
  bool pass = true;
    for(unsigned j = 0; j < BATCH_SIZE && pass; ++j) {
      if(output[j] != ref_output[j]) {
        printf("Failed verification @ device %d, index %d\nOutput: %f\nReference: %f\n",
            j, output[j], ref_output[j]);
        pass = false;
      }
  }

  printf("\nVerification: %s\n", pass ? "PASS" : "FAIL");

  /////////////////////////     Clenup     /////////////////////////

  if(kernel_embedding_lookup) {
    clReleaseKernel(kernel_embedding_lookup);
  }
  if(queue_embedding_lookup) {
    clReleaseCommandQueue(queue_embedding_lookup);
  }
  if(kernel_reduction_sum) {
    clReleaseKernel(kernel_reduction_sum);
  }
  if(queue_reduction_sum) {
    clReleaseCommandQueue(queue_reduction_sum);
  }
  // if(queue) {
  //   clReleaseCommandQueue(queue);
  // }
  if(input_a_buf) {
    clReleaseMemObject(input_a_buf);
  }
  if(input_b_buf) {
    clReleaseMemObject(input_b_buf);
  }
  if(output_buf) {
    clReleaseMemObject(output_buf);
  }

  if(program) {
    clReleaseProgram(program);
  }
  if(context) {
    clReleaseContext(context);
  }

  return 0;
}

/////// HELPER FUNCTIONS ///////

// Randomly generate a floating-point number between -10 and 10.
D_TYPE rand_int() {
  return rand() % 15 - 7;
}

// Free the resources allocated during initialization
void cleanup() {
  for(unsigned i = 0; i < num_devices; ++i) {
    if(kernel_embedding_lookup) {
      clReleaseKernel(kernel_embedding_lookup);
    }
    if(queue_embedding_lookup) {
      clReleaseCommandQueue(queue_embedding_lookup);
    }
    if(kernel_reduction_sum) {
      clReleaseKernel(kernel_reduction_sum);
    }
    if(queue_reduction_sum) {
      clReleaseCommandQueue(queue_reduction_sum);
    }
    // if(queue) {
    //   clReleaseCommandQueue(queue);
    // }
    if(input_a_buf) {
      clReleaseMemObject(input_a_buf);
    }
    if(input_b_buf) {
      clReleaseMemObject(input_b_buf);
    }
    if(output_buf) {
      clReleaseMemObject(output_buf);
    }
  }

  if(program) {
    clReleaseProgram(program);
  }
  if(context) {
    clReleaseContext(context);
  }
}

