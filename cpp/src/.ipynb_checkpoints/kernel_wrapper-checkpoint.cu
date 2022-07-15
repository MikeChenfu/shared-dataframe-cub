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

static constexpr double mm_to_inches = 0.0393701;

__global__ void kernel_tenth_mm_to_inches(cudf::mutable_column_device_view val, cudf::mutable_column_device_view data, cudf::mutable_column_device_view res_data, cudf::mutable_column_device_view res_val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < val.size()) {
      res_val.element<double>(i) = val.element<double>(i) * mm_to_inches;
      res_data.element<int>(i) = data.element<int>(i) + 1;
    }
}
 
 CudfWrapper::CudfWrapper(cudf::mutable_table_view table_view, cudf::mutable_table_view result) {
   mtv = table_view;
   res = result;
   
 }
 
 void CudfWrapper::tenth_mm_to_inches() {
 
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


  // Invoke the Kernel to convert tenth_mm -> inches
  kernel_tenth_mm_to_inches<<<(mtv.num_rows()+255)/256, 256>>>(*val, *data, *res_data, *res_val);
  cudaError_t err = cudaStreamSynchronize(0);
  printf("cudaStreamSynchronize Response = %d\n", (int)err);
 }
 
 CudfWrapper::~CudfWrapper() {
   // It is important to note that CudfWrapper does not own the underlying Dataframe 
   // object and that will be freed by the Python/Cython layer later.
 }
