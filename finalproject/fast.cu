#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// simple CUDA error check
#define CHECK_CUDA(call) do {                              \
    cudaError_t err = (call);                              \
    if (err != cudaSuccess) {                              \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1);                                      \
    }                                                      \
} while(0)

__device__ const int circleX[16] =
    {0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1};
__device__ const int circleY[16] =
    {-3,-3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3};

// ----------------------------------------------------
// FAST KERNEL: score per pixel 
// ----------------------------------------------------
#define BLOCK_W 16
#define BLOCK_H 16
#define HALO 3
// Shared memory dimensions: Block size + 3 pixels on each side
#define SMEM_W (BLOCK_W + 2 * HALO)
#define SMEM_H (BLOCK_H + 2 * HALO)

// Device helper: Checks if a bitmask contains a run of 'len' consecutive 1s 
__device__ __forceinline__ bool has_circular_run(unsigned int mask, int len) {
    unsigned int extended = mask | (mask << 16);
    unsigned int res = extended;
    for (int i = 1; i < len; i++) {
        res = res & (extended << i);
    }
    return (res & 0xFFFF) != 0;
}

__global__ void fastKernel(const unsigned char* __restrict__ in, 
                                     unsigned char* __restrict__ score,
                                     int width, int height, 
                                     int intensity_thresh, int run_thresh)
{
    // Setup Shared Memory
    __shared__ unsigned char smem[SMEM_H][SMEM_W];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;
    int tile_base_x = blockIdx.x * blockDim.x - HALO;
    int tile_base_y = blockIdx.y * blockDim.y - HALO;

    // Cooperative Load 
    for (int i = tid; i < SMEM_H * SMEM_W; i += blockSize) {
        int smem_y = i / SMEM_W;
        int smem_x = i % SMEM_W;
        int global_x = tile_base_x + smem_x;
        int global_y = tile_base_y + smem_y;

        unsigned char val = 0;
        if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
            val = in[global_y * width + global_x];
        }
        smem[smem_y][smem_x] = val;
    }

    __syncthreads();

    // Setup Coordinates
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx < 3 || gx >= width - 3 || gy < 3 || gy >= height - 3 || 
        threadIdx.x >= BLOCK_W || threadIdx.y >= BLOCK_H) return;

    int sx = threadIdx.x + HALO;
    int sy = threadIdx.y + HALO;
    int center = smem[sy][sx];

    
    unsigned int mask_bright = 0;
    unsigned int mask_dark = 0;

    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        int neighbor = smem[sy + circleY[k]][sx + circleX[k]];

        // If neighbor is brighter, set the k-th bit
        if (neighbor > center + intensity_thresh) mask_bright |= (1u << k);
        
        // If neighbor is darker, set the k-th bit
        if (neighbor < center - intensity_thresh) mask_dark |= (1u << k);
    }


    int final_score = 0;

    if (__popc(mask_bright) >= run_thresh) {
        if (has_circular_run(mask_bright, run_thresh)) {
            // Calculate score 
            final_score = __popc(mask_bright); 
        }
    }
    
    // Only check dark if bright didn't trigger 
    if (final_score == 0 && __popc(mask_dark) >= run_thresh) {
        if (has_circular_run(mask_dark, run_thresh)) {
            final_score = __popc(mask_dark);
        }
    }

    if (gx < width && gy < height) {
        score[gy * width + gx] = (unsigned char)final_score;
    }
}

// ----------------------------------------------------
// NMS KERNEL: keep only local maxima in a window
// ----------------------------------------------------
#define NMS_SMEM_W (BLOCK_W + 2 * NMS_RADIUS)
#define NMS_SMEM_H (BLOCK_H + 2 * NMS_RADIUS)
#define NMS_WINDOW 5
#define NMS_RADIUS 2

__global__ void nmsKernel(const unsigned char* __restrict__ score, 
                                   unsigned char* __restrict__ out,
                                   int width, int height)
{
    // Shared Memory Setup
    __shared__ unsigned char smem[NMS_SMEM_H][NMS_SMEM_W];
    // Global pixel coordinates
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    // Linear thread ID for cooperative loading
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;
    // Top-left of the tile in Global Memory
    int tile_base_x = blockIdx.x * blockDim.x - NMS_RADIUS;
    int tile_base_y = blockIdx.y * blockDim.y - NMS_RADIUS;

    // Threads cooperate to load the SMEM_W x SMEM_H area
    int total_smem_pixels = NMS_SMEM_W * NMS_SMEM_H;
    for (int i = tid; i < total_smem_pixels; i += blockSize) {
        int local_y = i / NMS_SMEM_W;
        int local_x = i % NMS_SMEM_W;

        int global_y = tile_base_y + local_y;
        int global_x = tile_base_x + local_x;

        unsigned char val = 0;
        if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
            val = score[global_y * width + global_x];
        }
        smem[local_y][local_x] = val; 
    }
    __syncthreads(); // Wait for the tile to be fully loaded

    // Early exit for threads outside the valid computation area
    if (gx < NMS_RADIUS || gx >= width - NMS_RADIUS || gy < NMS_RADIUS || gy >= height - NMS_RADIUS ||
        threadIdx.x >= BLOCK_W || threadIdx.y >= BLOCK_H) 
    {
        if (gx < width && gy < height) out[gy * width + gx] = 0;
        return;
    }
    // Center coordinates in Shared Memory
    int smem_cx = threadIdx.x + NMS_RADIUS;
    int smem_cy = threadIdx.y + NMS_RADIUS;
    unsigned char center_val = smem[smem_cy][smem_cx];

    // Early exit for background pixels
    if (center_val == 0) {
        out[gy * width + gx] = 0;
        return;
    }
    bool isMax = true;
    // Iterate over the 5x5 window using Shared Memory
    for (int dy = -NMS_RADIUS; dy <= NMS_RADIUS && isMax; ++dy) {
        for (int dx = -NMS_RADIUS; dx <= NMS_RADIUS; ++dx) {
            if (dx == 0 && dy == 0) continue;
            // Access neighbor from static shared memory array
            unsigned char n = smem[smem_cy + dy][smem_cx + dx];
            if (n > center_val) {
                isMax = false;
                break; // Stop checking if we find a larger neighbor
            }
        }
    }
    out[gy * width + gx] = isMax ? 255 : 0;
}

// ------------------------------------------------------------
// SOBEL KERNEL: compute Ix, Iy (float arrays)
// ------------------------------------------------------------
#define SOBEL_RADIUS 1

// Shared memory dimensions: Block size + 2 * Radius (2*1 = 2)
#define SOBEL_SMEM_W (BLOCK_W + 2 * SOBEL_RADIUS)
#define SOBEL_SMEM_H (BLOCK_H + 2 * SOBEL_RADIUS)

__global__ void sobelKernel(const unsigned char* __restrict__ in, 
                                  float* __restrict__ Ix, 
                                  float* __restrict__ Iy, 
                                  int width, int height)
{
    // Shared Memory Setup
    __shared__ unsigned char smem[SOBEL_SMEM_H][SOBEL_SMEM_W];

    // Global coordinates
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int global_idx = gy * width + gx;
    
    // Linear thread ID for cooperative loading
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;

    // Top-left of the tile in Global Memory (including the 1-pixel halo/apron)
    int tile_base_x = blockIdx.x * blockDim.x - SOBEL_RADIUS;
    int tile_base_y = blockIdx.y * blockDim.y - SOBEL_RADIUS;
    

    int total_smem_pixels = SOBEL_SMEM_W * SOBEL_SMEM_H;
    
    for (int i = tid; i < total_smem_pixels; i += blockSize) {
        int local_y = i / SOBEL_SMEM_W;
        int local_x = i % SOBEL_SMEM_W;

        int global_y = tile_base_y + local_y;
        int global_x = tile_base_x + local_x;

        unsigned char val = 0;

        if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
            val = in[global_y * width + global_x];
        }
        smem[local_y][local_x] = val; 
    }

    __syncthreads(); // Wait for the tile to be fully loaded

    // Compute Sobel Gradient (using Shared Memory)
    
    // Early exit for threads computing outside the valid image boundaries
    if (gx <= 0 || gx >= width - 1 || gy <= 0 || gy >= height - 1 ||
        threadIdx.x >= BLOCK_W || threadIdx.y >= BLOCK_H) 
    {
        // Zero out boundary results
        if (gx < width && gy < height) {
            Ix[global_idx] = 0.0f;
            Iy[global_idx] = 0.0f;
        }
        return;
    }
    
    // Center coordinates in Shared Memory (offset by SOBEL_RADIUS = 1)
    int smem_cx = threadIdx.x + SOBEL_RADIUS;
    int smem_cy = threadIdx.y + SOBEL_RADIUS;
    
    // Sobel Gx Calculation
    float sx =
        -smem[smem_cy-1][smem_cx-1] + smem[smem_cy-1][smem_cx+1] -2.0f*smem[smem_cy][smem_cx-1] 
        + 2.0f*smem[smem_cy][smem_cx+1] -smem[smem_cy+1][smem_cx-1] + smem[smem_cy+1][smem_cx+1];

    // Sobel Gy Calculation
    float sy =
         smem[smem_cy-1][smem_cx-1] + 2.0f*smem[smem_cy-1][smem_cx] + smem[smem_cy-1][smem_cx+1]
        -smem[smem_cy+1][smem_cx-1] - 2.0f*smem[smem_cy+1][smem_cx] - smem[smem_cy+1][smem_cx+1];

    // Write results to global memory
    Ix[global_idx] = sx;
    Iy[global_idx] = sy;
}

// ------------------------------------------------------------
// HARRIS KERNEL: compute windowed sums Sxx, Syy, Sxy and R
// ------------------------------------------------------------
__global__
void harrisWindowKernel(const unsigned char* fastMask,
                        const float* Ix, const float* Iy,
                        float* harris, int width, int height,
                        int window, float k)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 0 || x >= width || y < 0 || y >= height) return;

    int idx = y*width + x;
    if (fastMask[idx] == 0) { harris[idx] = 0.0f; return; }

    int r = window/2;
    float Sxx = 0.0f, Syy = 0.0f, Sxy = 0.0f;

    // sum products over window
    for (int dy = -r; dy <= r; ++dy) {
        int ny = y + dy;
        if (ny < 1 || ny >= height-1) continue; 
        for (int dx = -r; dx <= r; ++dx) {
            int nx = x + dx;
            if (nx < 1 || nx >= width-1) continue;
            int nidx = ny*width + nx;
            float ix = Ix[nidx];
            float iy = Iy[nidx];
            Sxx += ix * ix;
            Syy += iy * iy;
            Sxy += ix * iy;
        }
    }

    float det = Sxx * Syy - Sxy * Sxy;
    float trace = Sxx + Syy;
    float R = det - k * trace * trace;
    harris[idx] = R;
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
int main(int argc, char** argv)
{
    if (argc < 5) {
        std::cerr << "Usage: ./fast input.png output.png runThresh topN\n";
        return 1;
    }

    const char* inputPath = argv[1];
    const char* outputPath = argv[2];
    int runThresh = std::atoi(argv[3]);
    int topN = std::atoi(argv[4]);

    int width, height, ch;
    unsigned char* h_img = stbi_load(inputPath, &width, &height, &ch, 1);
    if (!h_img) { std::cerr << "Failed to load " << inputPath << "\n"; return 1; }

    int imgSize = width * height;

    // Device buffers
    unsigned char *d_in = nullptr, *d_score = nullptr, *d_nms = nullptr;
    float *d_Ix = nullptr, *d_Iy = nullptr, *d_harris = nullptr;

    CHECK_CUDA(cudaMalloc(&d_in, imgSize));
    CHECK_CUDA(cudaMalloc(&d_score, imgSize));
    CHECK_CUDA(cudaMalloc(&d_nms, imgSize));
    CHECK_CUDA(cudaMalloc(&d_Ix, imgSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Iy, imgSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_harris, imgSize * sizeof(float)));

    // init score & nms & harris to zero to avoid garbage at borders
    CHECK_CUDA(cudaMemset(d_score, 0, imgSize));
    CHECK_CUDA(cudaMemset(d_nms, 0, imgSize));
    CHECK_CUDA(cudaMemset(d_harris, 0, imgSize * sizeof(float)));

    // copy input
    CHECK_CUDA(cudaMemcpy(d_in, h_img, imgSize, cudaMemcpyHostToDevice));

    dim3 threads(BLOCK_W,BLOCK_H);
    dim3 blocks((width + BLOCK_W-1)/BLOCK_W, (height + BLOCK_H-1)/BLOCK_H);

    int intensityThresh = 30;
    // FAST
    fastKernel<<<blocks, threads>>>(d_in, d_score, width, height, intensityThresh, runThresh);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // NMS 
    nmsKernel<<<blocks, threads>>>(d_score, d_nms, width, height);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Sobel gradients
    sobelKernel<<<blocks, threads>>>(d_in, d_Ix, d_Iy, width, height);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Harris using windowed sums 
    int harrisWindow = 3; 
    float harrisK = 0.04f;
    harrisWindowKernel<<<blocks, threads>>>(d_nms, d_Ix, d_Iy, d_harris, width, height, harrisWindow, harrisK);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // copy Harris back to host
    std::vector<float> h_harris(imgSize);
    CHECK_CUDA(cudaMemcpy(h_harris.data(), d_harris, imgSize * sizeof(float), cudaMemcpyDeviceToHost));

    // gather >0 Harris responses into a vector (score, idx)
    struct Corner { float score; int idx; };
    std::vector<Corner> corners;
    corners.reserve(1024);
    for (int i = 0; i < imgSize; ++i) {
        if (h_harris[i] > 0.0f) corners.push_back({h_harris[i], i});
    }

    // sort descending and keep topN
    std::sort(corners.begin(), corners.end(), [](const Corner& a, const Corner& b){
        return a.score > b.score;
    });
    if ((int)corners.size() > topN) corners.resize(topN);

    // build output mask
    std::vector<unsigned char> out(imgSize, 0);
    for (const auto &c : corners) out[c.idx] = 255;

    stbi_write_png(outputPath, width, height, 1, out.data(), width);

    // cleanup
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_score));
    CHECK_CUDA(cudaFree(d_nms));
    CHECK_CUDA(cudaFree(d_Ix));
    CHECK_CUDA(cudaFree(d_Iy));
    CHECK_CUDA(cudaFree(d_harris));
    stbi_image_free(h_img);

    std::cout << "Saved top " << corners.size() << " Harris-ranked FAST corners to " << outputPath << "\n";
    return 0;
}