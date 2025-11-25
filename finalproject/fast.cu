// fast_harris_topN.cu
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
// FAST KERNEL: score per pixel (best contiguous run length)
// ----------------------------------------------------
__global__
void fastKernel(const unsigned char* in, unsigned char* score,
                int width, int height, int intensity_thresh,
                int run_thresh)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 3 || x >= width-3 || y < 3 || y >= height-3) return;

    int center = in[y*width + x];
    int b = 0, d = 0, maxB = 0, maxD = 0;

    // iterate 16 + (run_thresh - 1) to allow wrap runs up to run_thresh length
    for (int k = 0; k < 16 + run_thresh - 1; ++k) {
        int m = k % 16;
        int nx = x + circleX[m];
        int ny = y + circleY[m];
        int val = in[ny*width + nx];

        if (val > center + intensity_thresh) { b++; if (b > maxB) maxB = b; } else b = 0;
        if (val < center - intensity_thresh) { d++; if (d > maxD) maxD = d; } else d = 0;
    }

    int best = maxB > maxD ? maxB : maxD;
    score[y*width + x] = (best >= run_thresh) ? (unsigned char)best : 0;
}

// ----------------------------------------------------
// NMS KERNEL: keep only local maxima in a window
// Input: score (uchar), Output: mask (uchar: 255 or 0)
// ----------------------------------------------------
__global__
void nmsKernel(const unsigned char* score, unsigned char* out,
               int width, int height, int window)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int r = window / 2;
    if (x < r || x >= width - r || y < r || y >= height - r) return;

    unsigned char center = score[y*width + x];
    if (center == 0) { out[y*width + x] = 0; return; }

    bool isMax = true;
    for (int dy = -r; dy <= r && isMax; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            if (dx == 0 && dy == 0) continue;
            unsigned char n = score[(y+dy)*width + (x+dx)];
            if (n > center) { isMax = false; break; }
        }
    }
    out[y*width + x] = isMax ? 255 : 0;
}

// ------------------------------------------------------------
// SOBEL KERNEL: compute Ix, Iy (float arrays)
// ------------------------------------------------------------
__global__
void sobelKernel(const unsigned char* in, float* Ix, float* Iy, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= width-1 || y <= 0 || y >= height-1) return;
    int idx = y*width + x;

    float gx =
        -in[(y-1)*width + (x-1)] + in[(y-1)*width + (x+1)]
        -2*in[(y)*width   + (x-1)] + 2*in[(y)*width   + (x+1)]
        -in[(y+1)*width + (x-1)] + in[(y+1)*width + (x+1)];

    float gy =
         in[(y-1)*width + (x-1)] + 2*in[(y-1)*width + x] + in[(y-1)*width + (x+1)]
        -in[(y+1)*width + (x-1)] - 2*in[(y+1)*width + x] - in[(y+1)*width + (x+1)];

    Ix[idx] = gx;
    Iy[idx] = gy;
}

// ------------------------------------------------------------
// HARRIS KERNEL: compute windowed sums Sxx, Syy, Sxy and R
// Only evaluate at locations where fastMask != 0
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

    // sum products over window (small window: 3x3 or 5x5)
    for (int dy = -r; dy <= r; ++dy) {
        int ny = y + dy;
        if (ny < 1 || ny >= height-1) continue; // skip borders where sobel invalid
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
        std::cerr << "Usage: ./fast_harris input.png output.png runThresh topN\n";
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

    dim3 threads(16,16);
    dim3 blocks((width + 15)/16, (height + 15)/16);

    int intensityThresh = 30;
    // 1) FAST
    fastKernel<<<blocks, threads>>>(d_in, d_score, width, height, intensityThresh, runThresh);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 2) NMS (choose window 5 or 3)
    int nmsWindow = 5;
    nmsKernel<<<blocks, threads>>>(d_score, d_nms, width, height, nmsWindow);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 3) Sobel gradients
    sobelKernel<<<blocks, threads>>>(d_in, d_Ix, d_Iy, width, height);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 4) Harris using windowed sums (use window 3 or 5)
    int harrisWindow = 3; // small window often used; try 3 or 5
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
