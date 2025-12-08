#include <iostream>
#include <cuda_runtime.h>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// =====================================================================
// KERNEL: RGB to Grayscale
// =====================================================================
__global__ 
void rgbToGrayKernel(const unsigned char* rgb, unsigned char* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    
    // RGB images have 3 bytes per pixel
    int rgb_idx = idx * 3;

    float r = rgb[rgb_idx + 0];
    float g = rgb[rgb_idx + 1];
    float b = rgb[rgb_idx + 2];

    // Compute luminance
    unsigned char val = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);

    gray[idx] = val;
}

// =====================================================================
// MAIN
// =====================================================================
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./grayscale <input.jpg> <output.jpg>\n";
        return 1;
    }

    const char* inputFile = argv[1];
    const char* outputFile = argv[2];

    // Load Image in COLOR
    int width, height, channels;
    unsigned char* h_rgb = stbi_load(inputFile, &width, &height, &channels, 3);

    if (!h_rgb) {
        std::cerr << "Error loading image.\n";
        return 1;
    }

    int numPixels = width * height;
    std::cout << "Loaded Image: " << width << "x" << height << " (RGB)\n";

    // Allocate GPU Memory
    unsigned char *d_rgb, *d_gray;
    
    // Allocate RGB buffer (Size * 3)
    cudaMalloc(&d_rgb, numPixels * 3);
    // Allocate Grayscale buffer (Size * 1)
    cudaMalloc(&d_gray, numPixels);

    // Copy RGB to GPU
    cudaMemcpy(d_rgb, h_rgb, numPixels * 3, cudaMemcpyHostToDevice);

    // Launch Kernel
    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);

    rgbToGrayKernel<<<blocks, threads>>>(d_rgb, d_gray, width, height);
    cudaDeviceSynchronize();

    // Download Grayscale Result
    std::vector<unsigned char> h_gray(numPixels);
    cudaMemcpy(h_gray.data(), d_gray, numPixels, cudaMemcpyDeviceToHost);

    stbi_write_jpg(outputFile, width, height, 1, h_gray.data(), 100);

    std::cout << "Saved grayscale image to " << outputFile << "\n";

    // Cleanup
    cudaFree(d_rgb);
    cudaFree(d_gray);
    stbi_image_free(h_rgb);

    return 0;
}