# Shareable Dataframes
## Overview
Real world development teams are quite varied. Team members often have different knowledge in toolsets and business procedures. This example's intention is the solve the challenge of team members with different software skill sets by allowing each team member to use their primary development language while contributing to solving a common business requirement with Nvidia RAPIDS.

This example demonstrates how to share cudf dataframes between Python and custom CUDA kernels. This is useful for performing custom CUDA-accelerated business logic on cuDF dataframes and handling certain tasks in Python and others in CUDA.

Dataframes that are created in Python cuDF are already present in GPU memory and accessible to CUDA code. This makes it straightforward to write a CUDA kernel to work with a dataframe columns. In fact this is how libcudf processes dataframes in CUDA kernels; the only difference in this example is that we invoke CUDA kernels that exist outside the cuDF code base. The term User Defined Function (UDF) could be loosely used to describe what this example is demonstrating.

This example provides a Cython `kernel_wrapper` implementation to make sharing the dataframes between Python and our custom CUDA kernel easier. This wrapper allows Python users to seamlessly invoke those CUDA kernels with a single function call and also provides a clear place to implement the C++ "glue code".

The example CUDA kernel accepts a data column (PRCP) containing rainfall values stored as 1/10th of a mm and converts those values to inches. The dataframe is read from a local CSV file using Python. Python then invokes the CUDA mm->inches conversion kernel via the Cython `kernel_wrapper`, passing it the dataframe object. The converted data can then be accessed from Python, e.g. using `df.head()`.

This is similar to an existing [weather notebook](https://github.com/rapidsai/notebooks-contrib/blob/branch-0.19/intermediate_notebooks/examples/weather.ipynb), which provides a reference for understanding the implementation. 

## Building (Inside Docker container)

0. Change the installed path for following commands, current path is `/rapids/rapids-examples/shareable-dataframes/`
1. Compile C++ `kernel_wrapper` code and CUDA kernels 

    - ```cmake -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DCMAKE_CUDA_ARCHITECTURES= -S /rapids/rapids-examples/shareable-dataframes/cpp -B /rapids/rapids-examples/shareable-dataframes/cpp/build```
    - ```cmake --build /rapids/rapids-examples/shareable-dataframes/cpp/build -j${PARALLEL_LEVEL} -v```
    - ```cmake --install /rapids/rapids-examples/shareable-dataframes/cpp/build -v```

2. Build the cython kernel_wrapper code, this will also link against the previously compiled C++ code. 
    - ```cd /rapids/rapids-examples/shareable-dataframes/python && python setup.py build install``` 

3. Run the Python example script. It expects an input of a single Weather year file. - EX: ```python /rapids/rapids-examples/shareable-dataframes/python/python_kernel_wrapper.py /rapids/rapids-examples/shareable-dataframes/test.csv```

CUDA Kernel with existing business logic:
``` cpp
#include <stdio.h>
#include <cudf/column/column_device_view.cuh> // cuDF component
#include <cudf/table/table_device_view.cuh> // cuDF component

static constexpr float mm_to_inches = 0.0393701;

// cudf::mutable_column_device_view used in place of device memory buffer
__global__ void kernel_tenth_mm_to_inches(cudf::mutable_column_device_view val, cudf::mutable_column_device_view data,
cudf::mutable_column_device_view res_data, cudf::mutable_column_device_view res_val
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < val.size()) {
      res_val.element<double>(i) = val.element<double>(i) * mm_to_inches;
      res_data.element<int>(i) = data.element<int>(i) + 1;
    }
}
```

Cython wrapper "glue":
```python
from cudf._lib.cpp.table.table_view cimport mutable_table_view
from cudf._lib.column cimport Column

from libcpp.string cimport string
from cudf._lib.utils cimport data_from_unique_ptr, table_view_from_table, table_view_from_columns
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.cpp.column.column_view cimport column_view, mutable_column_view
from libcpp.vector cimport vector

cdef mutable_table_view make_mutable_table_view(columns) except*:
    cdef vector[mutable_column_view] mutable_column_views
    cdef Column col
    for col in columns:
        mutable_column_views.push_back(col.mutable_view())
    return mutable_table_view(mutable_column_views)

cdef mutable_table_view mutable_view_from_table(tbl, ignore_index=False) except*:
    return make_mutable_table_view(
        tbl._index._data.columns + tbl._data.columns
        if not ignore_index and tbl._index is not None
        else tbl._data.columns
        )

cdef extern from "src/kernel_wrapper.hpp":
    cdef cppclass C_CudfWrapper "CudfWrapper":
        C_CudfWrapper(mutable_table_view tbl, mutable_table_view res)
        void tenth_mm_to_inches()

cdef class CudfWrapper:
    cdef C_CudfWrapper* gdf

    def __cinit__(self, t, result, ignore_index=False):
        cdef mutable_table_view input_table = mutable_view_from_table(t, ignore_index)
        cdef mutable_table_view output = mutable_view_from_table(result, ignore_index)
        self.gdf = new C_CudfWrapper(input_table, output)

    def tenth_mm_to_inches(self):
        self.gdf.tenth_mm_to_inches()
```

Python logic using CUDA kernel:
``` python
import cudf
import cudfkernel  # Cython bindings to execute existing CUDA Kernels

# CSV reader options; names of columns from weather data csv file
column_names = [
    "station_id",
    "date",
    "type",
    "val"
]
usecols = column_names[0:4]

# Create input dataframe
weather_df = cudf.read_csv(
    weather_file_path, names=column_names, usecols=usecols
)

#Create output dataframe
res = cudf.DataFrame({'res_val': [0.0]*len(weather_df.data), 'res_data': [0]*len(weather_df.data)})
res.res_data = res.res_data.astype('int32')

# Rainfall is stored as 1/10ths of MM.
rainfall_df = weather_df[weather_df["type"] == "PRCP"]

# Wrap the rainfall_df for CUDA to consume
rainfall_kernel = cudfkernel.CudfWrapper(rainfall_df, res)  

# Run the custom Kernel on the whole Dataframe 
rainfall_kernel.tenth_mm_to_inches()

# Shows head() after rainfall totals have been altered
print(rainfall_df.head())
print(res.head())

```
