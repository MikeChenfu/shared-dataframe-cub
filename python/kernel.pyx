from cudf._lib.cpp.table.table_view cimport mutable_table_view
from cudf._lib.column cimport Column

from libcpp.string cimport string
from cudf._lib.utils cimport data_from_unique_ptr, table_view_from_table, table_view_from_columns
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column_view cimport column_view, mutable_column_view
from libcpp.vector cimport vector

cdef mutable_table_view make_mutable_table_view(columns) except*:
    """
    Helper function to create a cudf::mutable_table_view from
    a list of Columns
    """
    cdef vector[mutable_column_view] mutable_column_views

    cdef Column col
    for col in columns:
        mutable_column_views.push_back(col.mutable_view())

    return mutable_table_view(mutable_column_views)

cdef mutable_table_view mutable_view_from_table(tbl, ignore_index=False) except*:
    """Create a cudf::table_view from a Table.
    Parameters
    ----------
    ignore_index : bool, default False
        If True, don't include the index in the columns.
    """
    return make_mutable_table_view(
        tbl._index._data.columns + tbl._data.columns
        if not ignore_index and tbl._index is not None
        else tbl._data.columns
        )
        
cdef extern from "kernel_wrapper.hpp":
    cdef cppclass C_CudfWrapper "CudfWrapper":
        C_CudfWrapper(mutable_table_view tbl, mutable_table_view res)
        void tenth_mm_to_inches()

cdef class CudfWrapper:
    cdef C_CudfWrapper* gdf
    def __cinit__(self, t, result, ignore_index=False):
        cdef mutable_table_view input_table = mutable_view_from_table(t, ignore_index)
        cdef mutable_table_view output = mutable_view_from_table(result, ignore_index)
        self.gdf = new C_CudfWrapper(input_table, output)

    def cython_tenth_mm_to_inches(self):
        self.gdf.tenth_mm_to_inches()
