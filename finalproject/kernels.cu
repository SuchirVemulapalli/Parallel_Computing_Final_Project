#include "kernels.h" // Include the header
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

// ----------------------------------------------------
// FAST KERNEL: score per pixel (best contiguous run length)
// ----------------------------------------------------

// Device helper: Checks if a bitmask contains a run of 'len' consecutive 1s (circular)
__device__ __forceinline__ bool has_circular_run(unsigned int mask, int len) {
    // Duplicate the mask to handle the "wrap around" (e.g., run crossing index 15->0)
    // 0000...1111111111111111 (16 bits) -> becomes 32 bits with wrap
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
    // 1. Setup Shared Memory
    __shared__ unsigned char smem[SMEM_H][SMEM_W];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;
    int tile_base_x = blockIdx.x * blockDim.x - HALO;
    int tile_base_y = blockIdx.y * blockDim.y - HALO;

    // 2. Cooperative Load (Unchanged - maximizes memory throughput)
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

    // 3. Setup Coordinates
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx < 3 || gx >= width - 3 || gy < 3 || gy >= height - 3 || 
        threadIdx.x >= BLOCK_W || threadIdx.y >= BLOCK_H) return;

    int sx = threadIdx.x + HALO;
    int sy = threadIdx.y + HALO;
    int center = smem[sy][sx];

    // 4. Branchless Ring Check (Optimized for Warp Scheduler)
    // We execute the exact same instructions for every pixel (no divergence).
    
    unsigned int mask_bright = 0;
    unsigned int mask_dark = 0;

    // Unroll the loop fully. The compiler converts this to independent load instructions.
    // The Warp Scheduler can issue these loads back-to-back, hiding latency perfectly.
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        int neighbor = smem[sy + circleY[k]][sx + circleX[k]];
        
        // Predicated execution (no branches)
        // If neighbor is brighter, set the k-th bit
        if (neighbor > center + intensity_thresh) mask_bright |= (1u << k);
        
        // If neighbor is darker, set the k-th bit
        if (neighbor < center - intensity_thresh) mask_dark |= (1u << k);
    }

    // 5. Evaluate Runs using Bitwise Logic
    // This replaces the nested if/else logic with pure ALU operations
    int final_score = 0;

    // Optimization: Popcount check
    // If total set bits < threshold, it is mathematically impossible to have a contiguous run.
    // __popc is a single hardware instruction.
    if (__popc(mask_bright) >= run_thresh) {
        if (has_circular_run(mask_bright, run_thresh)) {
            // Calculate score (sum of diffs or just max length? FAST usually uses Sum of Absolute Diff)
            // For this specific request, we just need to return a non-zero score.
            // Let's return the run length (conceptually) or a fixed value. 
            // The original kernel returned 'maxB'. We can approximate or recalculate if needed.
            // For speed, often just 255 or the bit count is used.
            // Let's stick to the original logic: return maxB (approximated by popcount or just valid)
            final_score = __popc(mask_bright); 
        }
    }
    
    // Only check dark if bright didn't trigger (FAST implies a corner is EITHER bright OR dark, rarely both)
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
// Input: score (uchar), Output: mask (uchar: 255 or 0)
// ----------------------------------------------------

__global__ void nmsKernel(const unsigned char* __restrict__ score, 
                                   unsigned char* __restrict__ out,
                                   int width, int height)
{
    // 1. Static Shared Memory Setup
    // Size is known at compile time, allocated automatically by CUDA
    __shared__ unsigned char smem[NMS_SMEM_H][NMS_SMEM_W];
    // Global pixel coordinates
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    // Linear thread ID for cooperative loading
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x * blockDim.y;
    // Top-left of the tile in Global Memory (including the halo/apron)
    int tile_base_x = blockIdx.x * blockDim.x - NMS_RADIUS;
    int tile_base_y = blockIdx.y * blockDim.y - NMS_RADIUS;
    // 2. Cooperative Global -> Shared Load (Warp Scheduling Optimization)
    // Threads cooperate to load the SMEM_W x SMEM_H area
    int total_smem_pixels = NMS_SMEM_W * NMS_SMEM_H;
    for (int i = tid; i < total_smem_pixels; i += blockSize) {
        int local_y = i / NMS_SMEM_W;
        int local_x = i % NMS_SMEM_W;

        int global_y = tile_base_y + local_y;
        int global_x = tile_base_x + local_x;

        unsigned char val = 0;
        // Boundary check
        if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
            val = score[global_y * width + global_x];
        }
        smem[local_y][local_x] = val; // Store in static shared memory
    }
    __syncthreads(); // Wait for the tile to be fully loaded

    // Early exit for threads outside the valid computation area (boundaries and padding)
    if (gx < NMS_RADIUS || gx >= width - NMS_RADIUS || gy < NMS_RADIUS || gy >= height - NMS_RADIUS ||
        threadIdx.x >= BLOCK_W || threadIdx.y >= BLOCK_H) 
    {
        if (gx < width && gy < height) out[gy * width + gx] = 0;
        return;
    }
    // Center coordinates in Shared Memory (offset by NMS_RADIUS)
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

// Shared memory dimensions: Block size + 2 * Radius (2*1 = 2)

__global__ void sobelKernel(const unsigned char* __restrict__ in, 
                                  float* __restrict__ Ix, 
                                  float* __restrict__ Iy, 
                                  int width, int height)
{
    // 1. Static Shared Memory Setup
    // Stores the input image data (unsigned char) for fast access
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
    
    // 2. Cooperative Global -> Shared Load (Warp Scheduling Optimization)
    // This process ensures coalesced global memory access, maximizing throughput.
    int total_smem_pixels = SOBEL_SMEM_W * SOBEL_SMEM_H;
    
    for (int i = tid; i < total_smem_pixels; i += blockSize) {
        int local_y = i / SOBEL_SMEM_W;
        int local_x = i % SOBEL_SMEM_W;

        int global_y = tile_base_y + local_y;
        int global_x = tile_base_x + local_x;

        unsigned char val = 0;
        // Boundary check (Handle image borders)
        if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
            val = in[global_y * width + global_x];
        }
        smem[local_y][local_x] = val; // Store in shared memory
    }

    __syncthreads(); // Wait for the tile to be fully loaded

    // 3. Compute Sobel Gradient (using Shared Memory)
    
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
    // Readings are now fast, low-latency accesses from shared memory
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

// =====================================================================
// DEVICE HELPER FUNCTIONS
// =====================================================================

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
    // Shared memory cache
    __shared__ unsigned char s_tile[BLUR_SMEM_DIM][BLUR_SMEM_DIM];

    // Global coordinates
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // 1. LOAD DATA INTO SHARED MEMORY (Handle Halo)
    // We need to load a 20x20 area using 16x16 threads.
    // Some threads will load more than one pixel.
    
    // Top-left corner of the shared memory tile in global space
    int tile_base_x = blockIdx.x * blockDim.x - BLUR_RADIUS;
    int tile_base_y = blockIdx.y * blockDim.y - BLUR_RADIUS;

    // Linearize loading to ensure all SMEM is filled
    int thread_id_linear = ty * blockDim.x + tx;
    int total_smem_pixels = BLUR_SMEM_DIM * BLUR_SMEM_DIM;

    for (int i = thread_id_linear; i < total_smem_pixels; i += (blockDim.x * blockDim.y)) {
        int smem_y = i / BLUR_SMEM_DIM;
        int smem_x = i % BLUR_SMEM_DIM;

        int global_x = tile_base_x + smem_x;
        int global_y = tile_base_y + smem_y;

        // Boundary Clamping (Safe Global Load)
        global_x = max(0, min(global_x, width - 1));
        global_y = max(0, min(global_y, height - 1));

        s_tile[smem_y][smem_x] = input[global_y * width + global_x];
    }

    __syncthreads(); // Wait for tile to load

    // 2. COMPUTE CONVOLUTION
    if (x < width && y < height) {
        int kernel[5] = {1, 4, 6, 4, 1};
        int sum = 0;
        int weightSum = 256; // Precomputed sum of (1+4+6+4+1)^2

        // Loop relative to thread's position in SMEM
        // The thread at tx, ty corresponds to s_tile[ty + RADIUS][tx + RADIUS]
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

// =====================================================================
// CUDA Kernel: ORB Descriptor
// =====================================================================
// Pattern constants

__global__ void orbKernel(
    const unsigned char* __restrict__ img,
    const unsigned char* __restrict__ kp,
    const int* __restrict__ pattern,
    unsigned char* __restrict__ descriptors,
    int width, int height)
{
    // --- 1. SHARED MEMORY ALLOCATION ---
    // smem contains: [Pattern Array (constant)] [Image Patch (overwritten per kp)]
    extern __shared__ int smem[]; 
    int* s_pattern = smem; 
    unsigned char* s_patch = (unsigned char*)&s_pattern[PATTERN_SIZE];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;
    
    // --- 2. CACHE PATTERN (Collaborative Load) ---
    // All threads help load the pattern once
    for (int i = tid; i < PATTERN_SIZE; i += blockDim.x) {
        s_pattern[i] = pattern[i];
    }
    __syncthreads();

    if (global_idx >= width * height) return;

    // --- 3. WARP SCHEDULING ---
    // Check if current thread has a keypoint
    bool is_kp = (kp[global_idx] != 0);

    // Boundary check (exclude borders)
    int y = global_idx / width;
    int x = global_idx % width;
    const int BORDER = 24;
    if (x < BORDER || x >= width - BORDER || y < BORDER || y >= height - BORDER) {
        is_kp = false;
    }

    // Leader Election: Who in this warp has work to do?
    unsigned int active_mask = __ballot_sync(0xFFFFFFFF, is_kp);
    int lane_id = tid % 32;

    while (active_mask != 0) {
        // Select the leader (lowest active lane)
        int leader_lane = __ffs(active_mask) - 1;
        
        // Broadcast leader's pixel index to everyone
        int leader_idx = __shfl_sync(0xFFFFFFFF, global_idx, leader_lane);
        int center_x = leader_idx % width;
        int center_y = leader_idx / width;

        // --- 4. LOAD PATCH TO SMEM (Collaborative) ---
        int patch_base_x = center_x - PATCH_R;
        int patch_base_y = center_y - PATCH_R;

        // Entire warp loads the 33x33 patch together
        for (int i = lane_id; i < PATCH_DIM * PATCH_DIM; i += 32) {
            int py = i / PATCH_DIM;
            int px = i % PATCH_DIM;
            // Safe Global Load
            int src_idx = (patch_base_y + py) * width + (patch_base_x + px);
            s_patch[py * PATCH_DIM + px] = img[src_idx];
        }
        __syncwarp(); // Wait for patch to load

        // --- 5. COMPUTE ORIENTATION (Parallel Reduction) ---
        float local_m10 = 0, local_m01 = 0;
        
        for (int dy = -PATCH_R + lane_id; dy <= PATCH_R; dy += 32) {
            for (int dx = -PATCH_R; dx <= PATCH_R; dx++) {
                if (dx*dx + dy*dy > PATCH_R*PATCH_R) continue;
                // Read from SMEM
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

        // --- 6. COMPUTE DESCRIPTOR (Parallel) ---
        // Each thread calculates 1 Byte of the descriptor
        if (lane_id < 32) {
            unsigned char outByte = 0;
            for (int bit = 0; bit < 8; bit++) {
                int k = 8 * lane_id + bit;
                
                int px1 = s_pattern[k * 4 + 0];
                int py1 = s_pattern[k * 4 + 1];
                int px2 = s_pattern[k * 4 + 2];
                int py2 = s_pattern[k * 4 + 3];

                // Coordinates relative to SMEM patch center (PATCH_R, PATCH_R)
                float rx1 = PATCH_R + (ca * px1 - sa * py1);
                float ry1 = PATCH_R + (sa * px1 + ca * py1);
                float rx2 = PATCH_R + (ca * px2 - sa * py2);
                float ry2 = PATCH_R + (sa * px2 + ca * py2);

                // Use the SMEM bilinear helper
                float val1 = bilinearAt(s_patch, rx1, ry1, PATCH_DIM);
                float val2 = bilinearAt(s_patch, rx2, ry2, PATCH_DIM);

                if (val1 < val2) outByte |= (1 << bit);
            }
            // Write to Global Memory
            descriptors[leader_idx * 32 + lane_id] = outByte;
        }

        // Done with this leader
        active_mask &= ~(1 << leader_lane);
    }
}

__global__ void matchKernel(const unsigned int* d_desc1,  // Image 1 Descriptors (Query)
                 const unsigned int* d_desc2,  // Image 2 Descriptors (Train)
                 MatchResult* d_matches,       // Output Array
                 int numDesc1, 
                 int numDesc2)
{
    // Each thread takes ONE descriptor from Image 1 and finds its best match in Image 2
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numDesc1) return;

    // 1. Load the Query Descriptor into Registers
    // Descriptors are 32 bytes. That is 8 integers (4 bytes * 8 = 32).
    // Storing this in local registers prevents reading global memory repeatedly.
    unsigned int myDesc[8];
    
    // We assume the input pointer is treated as an array of ints.
    // Stride is 8 ints per descriptor.
    int myOffset = idx * 8;
    
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        myDesc[k] = d_desc1[myOffset + k];
    }

    // Initialize best distances to max possible (256 bits)
    int bestDist = 256;
    int secondBestDist = 256; 
    int bestIdx = -1;

    // 2. Loop through ALL descriptors in Image 2 (Brute Force)
    for (int j = 0; j < numDesc2; j++) {
        
        int dist = 0;
        int otherOffset = j * 8;

        // Compute Hamming Distance (XOR + Population Count)
        // __popc is a hardware instruction that counts "1" bits instantly.
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            unsigned int other = d_desc2[otherOffset + k];
            dist += __popc(myDesc[k] ^ other);
        }

        // 3. Track Best and Second Best (Logic for Lowe's Ratio Test)
        if (dist < bestDist) {
            secondBestDist = bestDist;
            bestDist = dist;
            bestIdx = j;
        } else if (dist < secondBestDist) {
            secondBestDist = dist;
        }
    }

    // 4. Save result to global memory
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

    // Descriptor files are bytes (uint8), but we process them as ints (uint32).
    // Size must be divisible by 4.
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