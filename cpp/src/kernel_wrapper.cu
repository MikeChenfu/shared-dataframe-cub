/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cudf/utilities/type_dispatcher.hpp"
#include "kernel_wrapper.hpp"
#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define DEF_THREAD_CNT 96
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__global__ void initOffsets( int total, int seg_len, uint32_t *out)
{
   int idx=blockIdx.x*blockDim.x+threadIdx.x;

   if(idx<total)
   {
      out[idx]=seg_len;
   }
   return;
}

int getMaxMeanSales( float *d_in_array, int in_len, int seg_len, float **d_out_array)
{
    //cudaStream_t stream1, stream2, stream3, stream4 ;

    //cudaStreamCreate ( &stream2) ;

    size_t bytes_scan=0;
    size_t temp_storage_bytes=0;
    uint32_t *d_A;
    uint32_t *d_offsets;
    void *d_temp = NULL;


    gpuErrchk (cudaMalloc (
        (void **) &d_A,
        (in_len+1) * sizeof (uint32_t)
    ));

    gpuErrchk (cudaMalloc (
        (void **) &d_offsets,
        (in_len+1) * sizeof (uint32_t)
    ));

    initOffsets<<<(in_len+1)/DEF_THREAD_CNT+1,DEF_THREAD_CNT>>>(in_len+1, seg_len, d_A);

    gpuErrchk (cudaDeviceSynchronize ());


    gpuErrchk (cub::DeviceScan::ExclusiveSum (
            NULL, bytes_scan, d_A, d_offsets, in_len + 1
    ));
    gpuErrchk (cudaDeviceSynchronize ());

    gpuErrchk (cudaMalloc (&d_temp, bytes_scan));

    gpuErrchk (cub::DeviceScan::ExclusiveSum (
            d_temp, bytes_scan, d_A, d_offsets, in_len + 1
    ));
   
    gpuErrchk (cudaFree (d_temp));
    gpuErrchk (cudaFree (d_A));
    gpuErrchk (cudaFree (d_offsets));


    return cudaSuccess;

}

 
 CudfWrapper::CudfWrapper(cudf::mutable_table_view table_view, cudf::mutable_table_view result) {
   mtv = table_view;
   res = result;
 }
 
 void CudfWrapper::tenth_mm_to_inches() {
 /*
  // Example of showing num_columns and num_rows only for potential debugging
  printf("kernel_wrapper.cu input: # of columns: %lu\n", mtv.num_columns());
  printf("kernel_wrapper.cu input: # of rows: %lu\n", mtv.num_rows());
  printf("kernel_wrapper.cu output: # of columns: %lu\n", res.num_columns());
  printf("kernel_wrapper.cu output: # of rows: %lu\n", res.num_rows());
 
  // get the target columns from the table and index starting from 1
  // e.g. ['station_id','data', 'type','val'], the index of val is 4.
  std::unique_ptr<cudf::mutable_column_device_view, std::function<void(cudf::mutable_column_device_view*)>> 
       val = cudf::mutable_column_device_view::create(mtv.column(4));
        
  std::unique_ptr<cudf::mutable_column_device_view, std::function<void(cudf::mutable_column_device_view*)>> 
       data = cudf::mutable_column_device_view::create(mtv.column(2));

  // get the result columns from the table
  std::unique_ptr<cudf::mutable_column_device_view, std::function<void(cudf::mutable_column_device_view*)>> 
       res_val = cudf::mutable_column_device_view::create(res.column(1));
        
  std::unique_ptr<cudf::mutable_column_device_view, std::function<void(cudf::mutable_column_device_view*)>> 
       res_data = cudf::mutable_column_device_view::create(res.column(2));
*/
  float *d_wkly_mean = NULL;
  float *d_max_mean=NULL;
  getMaxMeanSales( d_wkly_mean, 3000*52, 52, &d_max_mean );
  cudaError_t err = cudaStreamSynchronize(0);
  printf("cudaStreamSynchronize Response = %d\n", (int)err);

 }
 
 CudfWrapper::~CudfWrapper() {
   // It is important to note that CudfWrapper does not own the underlying Dataframe 
   // object and that will be freed by the Python/Cython layer later.
 }
