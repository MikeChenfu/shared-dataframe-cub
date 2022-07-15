#pragma once

#include <string>
#include <assert.h>
#include <cstdio>

#include <stdio.h>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/table/table_device_view.cuh>

class CudfWrapper {
  cudf::mutable_table_view mtv;
  cudf::mutable_table_view res;

  public:
    // Creates a Wrapper around an existing cuDF Dataframe object
    CudfWrapper(cudf::mutable_table_view table_view, cudf::mutable_table_view result);

    ~CudfWrapper();

    void tenth_mm_to_inches();
};
