cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$nproc
./build/deviceQuery