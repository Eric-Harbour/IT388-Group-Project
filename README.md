# IT388-Group-Project
IT388 Group Project: Parallelization of the Gaussian Blur

## Steps for building
1. Enter project root folder in terminal
2. Execute `cmake -B build` (Only needed when add/removed files or changed CMake)
3. Execute `cmake --build build` (Run after any change to source code)
4. Move to `./build/bin` and run specific executable

## Steps for running on Expanse
1. Enter project root
2. Load CPU modules
3. Execute `cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF -B build`
4. Execute `cmake --build build`
5. Logout and log back in to refresh modules
6. Load GPU modules
7. Move to `src` and execute `../scripts/compile_nvidia.sh`
8. Move `cuda_exec` to `../build/bin`
9. Move to `./build/bin` and run specific executable
10. You can now use the job scripts with these executables