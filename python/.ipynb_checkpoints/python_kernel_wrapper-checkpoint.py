import sys
import cudf
import cudfkernel  # Cython bindings to execute existing CUDA Kernels


def read_df(weather_file_path):
    print(
        "Reading Weather file '"
        + weather_file_path
        + "' to Dataframe with Python"
    )

    # CSV reader options
    column_names = [
        "station_id",
        "data",
        "type",
        "val"
    ]
    usecols = column_names[0:4]
    dtype_map = {'station_id': 'object', 'data': 'int32', 'type': 'object', 'val': 'float'}
    # All 2010 weather recordings
    weather_df = cudf.read_csv(
        weather_file_path, names=column_names, usecols=usecols, dtype=dtype_map, index=False
    )
    
    res = cudf.DataFrame({'res_val': [0.0]*len(weather_df.data), 'res_data': [0]*len(weather_df.data)})
    res.res_data = res.res_data.astype('int32')
    rainfall_df = weather_df[weather_df["type"] == "PRCP"]

    # Run the custom Kernel on Dataframe 
    rainfall_kernel = cudfkernel.CudfWrapper(
        rainfall_df,  res
    )  # Wrap the dataframe you want to perform Kernel calls on
    
    #kenel function : res_val = val * mm_to_inches, res_data = data + 1
    rainfall_kernel.cython_tenth_mm_to_inches()  

    # Shows head() after rainfall totals have been altered
    print(rainfall_df.head())
    print(res.head())

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Input Weather datafile path missing")

    read_df(sys.argv[1])
