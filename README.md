# IT388-Group-Project
IT388 Group Project: Parallelising Gaussian Blur

Names: Garrett Blankenship, Jeffrey Armour, Eric Harbour, and Nathan Hilbert 

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

## Description
This project implements Gaussian blur using three parallel programming models: [OpenMP](./src/omp_main.cpp), [MPI](./src/mpi_main.cpp), and [CUDA](./src/cuda_main.cu). All three implementations use the same approach to compute the blurred image: 
1. Split the rows of an image to run in parallel.
2. Compute the new RGB value for each pixel using the values of the pixels around it on this row.
3. Rotate the image so columns of pixels are now rows of the image.
4. Perform steps 1 and 2 again.
5. Transpose back to the original orientation to get the final result.

---

All three implementations use an `Image` struct to hold image data:

```cpp
struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::byte* data = nullptr;
    bool transposed = false;
};
```

`width` and `height`: Dimensions of the image by pixels.

`channels`: Color chanels (Red, Green, Blue, Alpha). Alpha is the opacity of the pixel.

`data`: array of bytes. Each byte represents a color channel within a pixel bytes. Making 4 bytes per pixel. e.g. [R, G, B, A, R, G, B, A, ...].

`transposed`: Tracks whether the image has been rotated or not.


## Main Functions
The weight for each neighboring pixel is calculated using the Gaussian function:
```cpp
float weight = std::exp(-(dx * dx) / (2.0f * sigma * sigma));
```
`dx`: Horizontal distance from the current pixel to the neighbor being sampled.

`sigma`: Standard deviation — controls the strength/spread of the blur.

- `horizontal_blur`: This function is what performs the Gausian blur. 

- `create_image`: Loads a image using the stb_image library. This allows us to operate on the images pixels indivually stored in the custom Image struct. The CUDA version loads the image directly into the GPU's memory

- `transpose`: Loops through each color channel of each pixel of the image and swaps every (i,j) with (j,i). The openMP implementation parallelizes this with static scheduling.
- `save_image`:
