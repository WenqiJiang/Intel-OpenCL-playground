#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
cl_kernel kernel;
cl_command_queue queue;
cl_mem input_a_buf, input_b_buf, output_buf;

// Problem data.
const int input_size = BANK_SIZE; // problem size
const int output_size = BATCH_SIZE;

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
  queue = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  // Kernel.
  // here, kernel name is the top-level function name
  const char *kernel_name = "embedding_lookup";
  kernel = clCreateKernel(program, kernel_name, &status);
  checkError(status, "Failed to create kernel");

  // Input buffers.
  input_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size * sizeof(D_TYPE), NULL, &status);
  checkError(status, "Failed to create buffer for input A");

  input_b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size * sizeof(D_TYPE), NULL, &status);
  checkError(status, "Failed to create buffer for input B");

  // Output buffer.
  output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, input_size * sizeof(D_TYPE), NULL, &status);
  checkError(status, "Failed to create buffer for output");

  /////////////////////////     Init Problem     /////////////////////////
  if(num_devices == 0) {
    checkError(-1, "No devices");
  }

  const int access_idx[] = {3, 99, 38, 72, 29, 57, 1, 72, 36, 76, 35, 50, 37, 57, 
      13, 66, 26, 70, 41, 93, 48, 82, 44, 78, 25, 52, 3, 92, 36, 56, 46, 88};

  // Generate input vectors A and B and the reference output consisting
  // of a total of input_size elements.
  // We create separate arrays for each device so that each device has an
  // aligned buffer.
  input_a.reset(input_size);
  input_b.reset(input_size);
  output.reset(input_size);
  ref_output.reset(output_size);

  for(unsigned j = 0; j < input_size; ++j) {
    input_a[j] = rand_int();
    input_b[j] = rand_int();
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
  

  /////////////////////////     Run Kernel     /////////////////////////

  const double start_time = getCurrentTimestamp();

  // Launch the problem for each device.
  cl_event kernel_event;
  cl_event finish_event;

  // Transfer inputs to each device. Each of the host buffers supplied to
  // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
  // for the host-to-device transfer.
  // 2 buffers a and b
  cl_event write_event;
  status = clEnqueueWriteBuffer(queue, input_a_buf, CL_FALSE,
      0, input_size * sizeof(D_TYPE), input_a, 0, NULL, &write_event);
  checkError(status, "Failed to transfer input A");

  // Set kernel arguments.
  unsigned argi = 0;

  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &input_a_buf);
  checkError(status, "Failed to set argument %d", argi - 1);

  // status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &input_b_buf[i]);
  // checkError(status, "Failed to set argument %d", argi - 1);

  status = clSetKernelArg(kernel, argi++, sizeof(cl_mem), &output_buf);
  checkError(status, "Failed to set argument %d", argi - 1);

  // Enqueue kernel.
  // Use a global work size corresponding to the number of elements to add
  // for this device.
  //
  // We don't specify a local work size and let the runtime choose
  // (it'll choose to use one work-group with the same size as the global
  // work-size).
  //
  // Events are used to ensure that the kernel is not launched until
  // the writes to the input buffers have completed.
  // const size_t global_work_size = input_size;
  const size_t global_work_size = 1;
  printf("Launching for device %d (%zd elements)\n", global_work_size);

  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL,
      &global_work_size, NULL, 1, &write_event, &kernel_event);
  checkError(status, "Failed to launch kernel");

  // Read the result. This the final operation.
  status = clEnqueueReadBuffer(queue, output_buf, CL_FALSE,
      0, output_size * sizeof(D_TYPE), output, 1, &kernel_event, &finish_event);

  // Release local events.
  clReleaseEvent(write_event);
  
  // Wait for all devices to finish.
  clWaitForEvents(num_devices, &finish_event);

  const double end_time = getCurrentTimestamp();

  // Wall-clock time taken.
  printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);

  // Get kernel times using the OpenCL event profiling API.
  cl_ulong time_ns = getStartEndTime(kernel_event);
  printf("Kernel time (device %d): %0.3f ms\n", double(time_ns) * 1e-6);

  // Release all events.
  clReleaseEvent(kernel_event);
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

  if(kernel) {
    clReleaseKernel(kernel);
  }
  if(queue) {
    clReleaseCommandQueue(queue);
  }
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
    if(kernel) {
      clReleaseKernel(kernel);
    }
    if(queue) {
      clReleaseCommandQueue(queue);
    }
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

