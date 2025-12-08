#include "kernels.h" 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <limits>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>

__device__ const int circleX[16] =
    {0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1};
__device__ const int circleY[16] =
    {-3,-3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3};


// Device helper: Checks if a bitmask contains a run of 'len' consecutive 1s (circular)
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


__global__ void nmsKernel(const unsigned char* __restrict__ score, 
                                   unsigned char* __restrict__ out,
                                   int width, int height)
{
    // Static Shared Memory Setup
    __shared__ unsigned char smem[NMS_SMEM_H][NMS_SMEM_W];
    // Global pixel coordinates
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    // Linear thread ID for cooperative loading
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;
    int tile_base_x = blockIdx.x * blockDim.x - NMS_RADIUS;
    int tile_base_y = blockIdx.y * blockDim.y - NMS_RADIUS;

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
    // Iterate over the window using Shared Memory
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

__global__ void sobelKernel(const unsigned char* __restrict__ in, 
                                  float* __restrict__ Ix, 
                                  float* __restrict__ Iy, 
                                  int width, int height)
{

    // Stores the input image data (unsigned char) for fast access
    __shared__ unsigned char smem[SOBEL_SMEM_H][SOBEL_SMEM_W];

    // Global coordinates
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int global_idx = gy * width + gx;
    
    // Linear thread ID for cooperative loading
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;


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
        smem[local_y][local_x] = val; // Store in shared memory
    }

    __syncthreads(); // Wait for the tile to be fully loaded

    // Compute Sobel Gradient 
    
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
    
    // Center coordinates in Shared Memory 
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

// Bilinear Interpolation for better Rotation Invariance
__device__ float bilinearAt(const unsigned char* smem_patch, float x, float y, int patch_dim) {
    int x1 = (int)x;
    int y1 = (int)y;
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    // Strict bounds check relative to the shared memory patch
    if (x1 < 0 || x2 >= patch_dim || y1 < 0 || y2 >= patch_dim) return 0.0f;

    float p11 = smem_patch[y1 * patch_dim + x1];
    float p12 = smem_patch[y1 * patch_dim + x2];
    float p21 = smem_patch[y2 * patch_dim + x1];
    float p22 = smem_patch[y2 * patch_dim + x2];

    float wx = x - x1;
    float wy = y - y1;

    return (1.0f - wy) * ((1.0f - wx) * p11 + wx * p12) +
           (wy) * ((1.0f - wx) * p21 + wx * p22);
}

__global__ 
void gaussianBlur(const unsigned char* __restrict__ input, 
                        unsigned char* __restrict__ output, 
                        int width, int height) 
{

    __shared__ unsigned char s_tile[BLUR_SMEM_DIM][BLUR_SMEM_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;


    int tile_base_x = blockIdx.x * blockDim.x - BLUR_RADIUS;
    int tile_base_y = blockIdx.y * blockDim.y - BLUR_RADIUS;

    int thread_id_linear = ty * blockDim.x + tx;
    int total_smem_pixels = BLUR_SMEM_DIM * BLUR_SMEM_DIM;

    for (int i = thread_id_linear; i < total_smem_pixels; i += (blockDim.x * blockDim.y)) {
        int smem_y = i / BLUR_SMEM_DIM;
        int smem_x = i % BLUR_SMEM_DIM;

        int global_x = tile_base_x + smem_x;
        int global_y = tile_base_y + smem_y;

        global_x = max(0, min(global_x, width - 1));
        global_y = max(0, min(global_y, height - 1));

        s_tile[smem_y][smem_x] = input[global_y * width + global_x];
    }

    __syncthreads(); // Wait for tile to load

    if (x < width && y < height) {
        int kernel[5] = {1, 4, 6, 4, 1};
        int sum = 0;
        int weightSum = 256; 

        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                int val = s_tile[ty + BLUR_RADIUS + ky][tx + BLUR_RADIUS + kx];
                int w = kernel[ky + 2] * kernel[kx + 2];
                sum += val * w;
            }
        }
        output[y * width + x] = (unsigned char)(sum / weightSum);
    }
}

__global__ void orbKernel(
    const unsigned char* __restrict__ img,
    const unsigned char* __restrict__ kp,
    const int* __restrict__ pattern,
    unsigned char* __restrict__ descriptors,
    int width, int height)
{
    extern __shared__ int smem[]; 
    int* s_pattern = smem; 
    unsigned char* s_patch = (unsigned char*)&s_pattern[PATTERN_SIZE];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;
    
    for (int i = tid; i < PATTERN_SIZE; i += blockDim.x) {
        s_pattern[i] = pattern[i];
    }
    __syncthreads();

    if (global_idx >= width * height) return;

    bool is_kp = (kp[global_idx] != 0);

    int y = global_idx / width;
    int x = global_idx % width;
    const int BORDER = 24;
    if (x < BORDER || x >= width - BORDER || y < BORDER || y >= height - BORDER) {
        is_kp = false;
    }

    unsigned int active_mask = __ballot_sync(0xFFFFFFFF, is_kp);
    int lane_id = tid % 32;

    while (active_mask != 0) {
        // Select the leader (lowest active lane)
        int leader_lane = __ffs(active_mask) - 1;
        
        // Broadcast leader's pixel index to everyone
        int leader_idx = __shfl_sync(0xFFFFFFFF, global_idx, leader_lane);
        int center_x = leader_idx % width;
        int center_y = leader_idx / width;

        int patch_base_x = center_x - PATCH_R;
        int patch_base_y = center_y - PATCH_R;

        // Entire warp loads the 33x33 patch together
        for (int i = lane_id; i < PATCH_DIM * PATCH_DIM; i += 32) {
            int py = i / PATCH_DIM;
            int px = i % PATCH_DIM;

            int src_idx = (patch_base_y + py) * width + (patch_base_x + px);
            s_patch[py * PATCH_DIM + px] = img[src_idx];
        }
        __syncwarp(); // Wait for patch to load

        // Compute orientation
        float local_m10 = 0, local_m01 = 0;
        
        for (int dy = -PATCH_R + lane_id; dy <= PATCH_R; dy += 32) {
            for (int dx = -PATCH_R; dx <= PATCH_R; dx++) {
                if (dx*dx + dy*dy > PATCH_R*PATCH_R) continue;

                unsigned char val = s_patch[(dy + PATCH_R) * PATCH_DIM + (dx + PATCH_R)];
                local_m10 += dx * val;
                local_m01 += dy * val;
            }
        }

        // Reduce within warp
        for (int offset = 16; offset > 0; offset /= 2) {
            local_m10 += __shfl_down_sync(0xFFFFFFFF, local_m10, offset);
            local_m01 += __shfl_down_sync(0xFFFFFFFF, local_m01, offset);
        }
        
        float m10 = __shfl_sync(0xFFFFFFFF, local_m10, 0);
        float m01 = __shfl_sync(0xFFFFFFFF, local_m01, 0);
        
        float angle = atan2f(m01, m10);
        float ca, sa;
        __sincosf(angle, &sa, &ca);

        // Compute descriptors
        if (lane_id < 32) {
            unsigned char outByte = 0;
            for (int bit = 0; bit < 8; bit++) {
                int k = 8 * lane_id + bit;
                
                int px1 = s_pattern[k * 4 + 0];
                int py1 = s_pattern[k * 4 + 1];
                int px2 = s_pattern[k * 4 + 2];
                int py2 = s_pattern[k * 4 + 3];

                float rx1 = PATCH_R + (ca * px1 - sa * py1);
                float ry1 = PATCH_R + (sa * px1 + ca * py1);
                float rx2 = PATCH_R + (ca * px2 - sa * py2);
                float ry2 = PATCH_R + (sa * px2 + ca * py2);

                float val1 = bilinearAt(s_patch, rx1, ry1, PATCH_DIM);
                float val2 = bilinearAt(s_patch, rx2, ry2, PATCH_DIM);

                if (val1 < val2) outByte |= (1 << bit);
            }
            descriptors[leader_idx * 32 + lane_id] = outByte;
        }

        active_mask &= ~(1 << leader_lane);
    }
}

__global__ void matchKernel(const unsigned int* d_desc1,  
                 const unsigned int* d_desc2,  
                 MatchResult* d_matches,       
                 int numDesc1, 
                 int numDesc2)
{
    // Each thread takes ONE descriptor from Image 1 and finds its best match in Image 2
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numDesc1) return;

    unsigned int myDesc[8];

    int myOffset = idx * 8;
    
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        myDesc[k] = d_desc1[myOffset + k];
    }

    // Initialize best distances to max possible
    int bestDist = 256;
    int secondBestDist = 256; 
    int bestIdx = -1;

    // Loop through ALL descriptors in Image 2 
    for (int j = 0; j < numDesc2; j++) {
        
        int dist = 0;
        int otherOffset = j * 8;

        // Compute Hamming Distance (XOR + Population Count)
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            unsigned int other = d_desc2[otherOffset + k];
            dist += __popc(myDesc[k] ^ other);
        }

        // Track Best and Second Best 
        if (dist < bestDist) {
            secondBestDist = bestDist;
            bestDist = dist;
            bestIdx = j;
        } else if (dist < secondBestDist) {
            secondBestDist = dist;
        }
    }

    // Save result to global memory
    d_matches[idx].trainIdx = bestIdx;
    d_matches[idx].distance = bestDist;
    d_matches[idx].secondDist = secondBestDist;
}

bool loadDescriptorFile(const char* fname, std::vector<unsigned int>& buffer) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        std::cerr << "Error opening " << fname << "\n";
        return false;
    }
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);

    if (size % 4 != 0) {
        std::cerr << "Error: File size not aligned to 4 bytes. Is this a valid descriptor file?\n";
        fclose(f);
        return false;
    }

    buffer.resize(size / 4); 
    size_t read = fread(buffer.data(), 1, size, f);
    fclose(f);
    
    return read == size;
}

void normalizePoints(const std::vector<Point2f>& in, std::vector<Point2f>& out, Mat3x3& T) {
    float meanx = 0, meany = 0;
    for (const auto& p : in) {
        meanx += p.x;
        meany += p.y;
    }
    meanx /= in.size();
    meany /= in.size();

    float meanDist = 0;
    for (const auto& p : in) {
        meanDist += sqrtf((p.x - meanx)*(p.x - meanx) + (p.y - meany)*(p.y - meany));
    }
    meanDist /= in.size();

    float scale = sqrtf(2) / meanDist;

    T = Mat3x3::identity();
    T.data[0] = scale; T.data[1] = 0;     T.data[2] = -scale * meanx;
    T.data[3] = 0;     T.data[4] = scale; T.data[5] = -scale * meany;
    T.data[6] = 0;     T.data[7] = 0;     T.data[8] = 1;

    out.resize(in.size());
    for (size_t i = 0; i < in.size(); i++) {
        out[i].x = T.data[0] * in[i].x + T.data[1] * in[i].y + T.data[2];
        out[i].y = T.data[3] * in[i].x + T.data[4] * in[i].y + T.data[5];
    }
}

// Lightweight Random Number Generator (Xorshift)
__device__ unsigned int xorshift(unsigned int& state) {
    unsigned int x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

// Linear Solver for RANSAC
__device__ bool solveHomography8x8(float A[8][9], float h[9]) {
    const int N = 8;
    for (int i = 0; i < N; i++) {
        int pivot = i;
        float maxVal = fabsf(A[i][i]);
        for (int k = i + 1; k < N; k++) {
            if (fabsf(A[k][i]) > maxVal) { maxVal = fabsf(A[k][i]); pivot = k; }
        }
        if (pivot != i) {
            for (int j = i; j <= N; j++) {
                float temp = A[i][j]; A[i][j] = A[pivot][j]; A[pivot][j] = temp;
            }
        }
        if (fabsf(A[i][i]) < 1e-7f) return false;
        for (int k = i + 1; k < N; k++) {
            float factor = A[k][i] / A[i][i];
            for (int j = i; j <= N; j++) A[k][j] -= factor * A[i][j];
        }
    }
    for (int i = N - 1; i >= 0; i--) {
        float sum = 0.0f;
        for (int j = i + 1; j < N; j++) sum += A[i][j] * h[j];
        h[i] = (A[i][N] - sum) / A[i][i];
    }
    h[8] = 1.0f;
    return true;
}


__global__ void ransacBlockKernel(const Point2f* __restrict__ srcPts, 
                                  const Point2f* __restrict__ dstPts, 
                                  int numMatches,
                                  float* __restrict__ bestH, 
                                  int* __restrict__ maxInliers, 
                                  int* __restrict__ mutex, 
                                  float threshold) 
{
    // We store the Homography and the block's total inliers here
    // so all threads can access them instantly.
    __shared__ float s_H[9];
    __shared__ int s_blockInliers;

    int tid = threadIdx.x;
    
    // Initialize shared counter
    if (tid == 0) s_blockInliers = 0;


    // Only one thread per block needs to do the heavy matrix math.
    // The others wait.
    if (tid == 0) {

        unsigned int rngState = blockIdx.x * 19349663 + (unsigned int)clock64();
        int indices[4];

        for (int i = 0; i < 4; i++) {
            while (true) {
                int randIdx = xorshift(rngState) % numMatches;
                bool duplicate = false;
                for (int k = 0; k < i; k++) if (indices[k] == randIdx) duplicate = true;
                if (!duplicate) { indices[i] = randIdx; break; }
            }
        }

        float A[8][9]; 
        for (int i = 0; i < 4; i++) {
            Point2f s = srcPts[indices[i]];
            Point2f d = dstPts[indices[i]];
            
            A[2*i][0] = s.x; A[2*i][1] = s.y; A[2*i][2] = 1;
            A[2*i][3] = 0;   A[2*i][4] = 0;   A[2*i][5] = 0;
            A[2*i][6] = -s.x * d.x; A[2*i][7] = -s.y * d.x; A[2*i][8] = d.x;

            A[2*i+1][0] = 0;   A[2*i+1][1] = 0;   A[2*i+1][2] = 0;
            A[2*i+1][3] = s.x; A[2*i+1][4] = s.y; A[2*i+1][5] = 1;
            A[2*i+1][6] = -s.x * d.y; A[2*i+1][7] = -s.y * d.y; A[2*i+1][8] = d.y;
        }

        // Solve and write directly to Shared Memory
        // If it fails, we set a flag (s_H[8] = 0) to tell other threads to abort
        if (!solveHomography8x8(A, s_H)) {
             s_H[8] = 0.0f; 
        }
    }

    // All threads pause here until Thread 0 finishes writing s_H
    __syncthreads();

    // Early exit if the model generation failed
    if (fabsf(s_H[8]) < 1e-6f) return;

    // Copy H (Homography) from shared memory to local registers
    float h0=s_H[0], h1=s_H[1], h2=s_H[2];
    float h3=s_H[3], h4=s_H[4], h5=s_H[5];
    float h6=s_H[6], h7=s_H[7], h8=s_H[8];

    float threshSq = threshold * threshold;
    int localInliers = 0;

    // Threads split the work
    for (int i = tid; i < numMatches; i += blockDim.x) {
        Point2f s = srcPts[i];
        Point2f d = dstPts[i];

        // --- (Your original inlier check logic) ---
        float w = h6 * s.x + h7 * s.y + h8;
        if (fabsf(w) > 1e-5f) {
            float invW = 1.0f / w;
            float px = (h0 * s.x + h1 * s.y + h2) * invW;
            float py = (h3 * s.x + h4 * s.y + h5) * invW;
            float distSq = (px - d.x)*(px - d.x) + (py - d.y)*(py - d.y);
            
            if (distSq < threshSq) {
                localInliers++;
            }
        }
    }


    // Aggregate local thread counts into the block counter
    if (localInliers > 0) {
        atomicAdd(&s_blockInliers, localInliers);
    }
    
    // Wait for all threads to finish counting
    __syncthreads();

    if (tid == 0) {
        // Quick check: Is our block result better than the current global max?
        if (s_blockInliers > *maxInliers) {
            
            bool isSet = false;
            do {
                if (isSet = (atomicCAS(mutex, 0, 1) == 0)) {                
                    // Check again now that we have the lock
                    if (s_blockInliers > *maxInliers) {
                        *maxInliers = s_blockInliers;
                        // Copy shared memory model to global memory
                        for(int k=0; k<9; k++) bestH[k] = s_H[k];
                    }
                    
                    atomicExch(mutex, 0);
                }
            } while (!isSet);
        }
    }
}

__device__ float bilinearAtStitcher(const unsigned char* img, int w, int h, float x, float y, int c) {
    int x1 = (int)x; int y1 = (int)y;
    int x2 = x1 + 1; int y2 = y1 + 1;
    if (x1 < 0 || x2 >= w || y1 < 0 || y2 >= h) return 0.0f;
    
    int stride = w * 3;
    float p11 = img[y1 * stride + x1 * 3 + c];
    float p12 = img[y1 * stride + x2 * 3 + c];
    float p21 = img[y2 * stride + x1 * 3 + c];
    float p22 = img[y2 * stride + x2 * 3 + c];
    
    float wx = x - x1; float wy = y - y1;
    return (1.0f - wy) * ((1.0f - wx) * p11 + wx * p12) + (wy) * ((1.0f - wx) * p21 + wx * p22);
}

__global__ void warpTextureKernel(cudaTextureObject_t texObj, 
                                  unsigned char* __restrict__ dst, 
                                  int dstW, int dstH,
                                  const float* __restrict__ h_inv, 
                                  int offsetX, int offsetY) 
{
    int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    int dstY = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds check
    if (dstX >= dstW || dstY >= dstH) return;


    float u = dstX + offsetX + 0.5f;
    float v = dstY + offsetY + 0.5f;

    // Calculate the 'w' (depth) component first
    float w = h_inv[6]*u + h_inv[7]*v + h_inv[8];

    if (fabs(w) < 1e-5) return;
    
    float invW = 1.0f / w;
    float srcX = (h_inv[0]*u + h_inv[1]*v + h_inv[2]) * invW;
    float srcY = (h_inv[3]*u + h_inv[4]*v + h_inv[5]) * invW;


    // tex2D allows for hardware accelerated interpolation
    // a. Fetches the 4 neighboring pixels from L1 Texture Cache
    // b. Handles boundary checks 
    // c. Performs bilinear interpolation logic
    // d. Returns a normalized float4 
    float4 color = tex2D<float4>(texObj, srcX, srcY);

    int dstIdx = (dstY * dstW + dstX) * 3;
    
    // Alpha blending
    if (color.w > 0.0f) {
        dst[dstIdx + 0] = (unsigned char)(color.x * 255.0f);
        dst[dstIdx + 1] = (unsigned char)(color.y * 255.0f);
        dst[dstIdx + 2] = (unsigned char)(color.z * 255.0f);
    }
}

__global__ void pasteKernel(const unsigned char* src, unsigned char* dst, 
                            int srcW, int srcH, int dstW, int dstH, int offsetX, int offsetY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= srcW || y >= srcH) return;
    
    int destX = x - offsetX;
    int destY = y - offsetY;
    
    if (destX >= 0 && destX < dstW && destY >= 0 && destY < dstH) {
        int srcIdx = (y * srcW + x) * 3;
        int dstIdx = (destY * dstW + destX) * 3;
        
        dst[dstIdx + 0] = src[srcIdx + 0];
        dst[dstIdx + 1] = src[srcIdx + 1];
        dst[dstIdx + 2] = src[srcIdx + 2];
    }
}