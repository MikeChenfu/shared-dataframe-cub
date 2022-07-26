import cudf
from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column_view cimport mutable_column_view
from cudf._lib.cpp.table.table_view cimport mutable_table_view
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "kernel_wrapper.hpp":
    cdef cppclass C_CudfWrapper "CudfWrapper":
        C_CudfWrapper(mutable_table_view tbl1, mutable_table_view tbl2)
        void tenth_mm_to_inches()

cdef class CudfWrapper:
    cdef C_CudfWrapper* gdf

    def __cinit__(self, columns):
        cdef vector[mutable_column_view] column_views1
        cdef vector[mutable_column_view] column_views2

        cdef Column col
        for col in columns[:1]:
            column_views1.push_back(col.mutable_view())
         
        for col in columns[1:]:
            column_views2.push_back(col.mutable_view())
         
        cdef mutable_table_view tv1 = mutable_table_view(column_views1)
        cdef mutable_table_view tv2 = mutable_table_view(column_views2)

        self.gdf = new C_CudfWrapper(tv1, tv2)

    def cython_tenth_mm_to_inches(self):
        self.gdf.tenth_mm_to_inches()