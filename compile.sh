export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

rm -rf cpp/build && cmake -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DCMAKE_CUDA_ARCHITECTURES=70 -S /root/c_python/rapids-examples/shareable-dataframes/cpp -B /root/c_python/rapids-examples/shareable-dataframes/cpp/build

cmake --build /root/c_python/rapids-examples/shareable-dataframes/cpp/build -j${PARALLEL_LEVEL}

cmake --install /root/c_python/rapids-examples/shareable-dataframes/cpp/build

cd python && python setup.py build install
cd .. && python python/python_kernel_wrapper.py test.csv