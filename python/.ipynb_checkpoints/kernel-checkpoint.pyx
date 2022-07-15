import cudf
from cudf._lib.cpp.table.table_view cimport mutable_table_view
from cudf._lib.table cimport Table
from libcpp.string cimport string

cdef extern from "kernel_wrapper.hpp":
    cdef cppclass C_CudfWrapper "CudfWrapper":
        C_CudfWrapper(mutable_table_view tbl, mutable_table_view res)
        void tenth_mm_to_inches()

cdef class CudfWrapper:
    cdef C_CudfWrapper* gdf
    cdef C_CudfWrapper* res

    def __cinit__(self, Table t, Table result):
        self.gdf = new C_CudfWrapper(t.mutable_view(), result.mutable_view())

    def cython_tenth_mm_to_inches(self):
        self.gdf.tenth_mm_to_inches()
