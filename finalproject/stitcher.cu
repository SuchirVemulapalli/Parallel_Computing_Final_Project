#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <ctime>

// Use the header-only libraries included in previous steps
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct Point2f { float x; float y; };

// =====================================================================
// HOST HELPER: 3x3 Matrix Math
// =====================================================================
struct Mat3x3 {
    float data[9]; // Row-major

    static Mat3x3 identity() {
        return {1,0,0, 0,1,0, 0,0,1};
    }

    // Matrix Multiplication (A * B)
    Mat3x3 operator*(const Mat3x3& b) const {
        Mat3x3 res;
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                res.data[r*3 + c] = 
                    data[r*3 + 0] * b.data[0*3 + c] +
                    data[r*3 + 1] * b.data[1*3 + c] +
                    data[r*3 + 2] * b.data[2*3 + c];
            }
        }
        return res;
    }

    // Transform a Point (Perspective Division)
    Point2f transform(Point2f p) const {
        float x = p.x;
        float y = p.y;
        float Z = data[6]*x + data[7]*y + data[8];
        return { (data[0]*x + data[1]*y + data[2]) / Z, 
                 (data[3]*x + data[4]*y + data[5]) / Z };
    }

    // Invert 3x3 Matrix
    Mat3x3 inverse() const {
        float det = data[0] * (data[4] * data[8] - data[7] * data[5]) -
                    data[1] * (data[3] * data[8] - data[5] * data[6]) +
                    data[2] * (data[3] * data[7] - data[4] * data[6]);

        if (std::abs(det) < 1e-7) return identity(); 

        float invDet = 1.0f / det;
        Mat3x3 res;
        res.data[0] = (data[4] * data[8] - data[5] * data[7]) * invDet;
        res.data[1] = (data[2] * data[7] - data[1] * data[8]) * invDet;
        res.data[2] = (data[1] * data[5] - data[2] * data[4]) * invDet;
        res.data[3] = (data[5] * data[6] - data[3] * data[8]) * invDet;
        res.data[4] = (data[0] * data[8] - data[2] * data[6]) * invDet;
        res.data[5] = (data[2] * data[3] - data[0] * data[5]) * invDet;
        res.data[6] = (data[3] * data[7] - data[4] * data[6]) * invDet;
        res.data[7] = (data[1] * data[6] - data[0] * data[7]) * invDet;
        res.data[8] = (data[0] * data[4] - data[1] * data[3]) * invDet;
        return res;
    }
};

// =====================================================================
// HOST HELPER: Coordinate Normalization (Crucial for DLT Stability)
// =====================================================================
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

// =====================================================================
// DEVICE HELPER: Lightweight Random Number Generator (Xorshift)
// =====================================================================
__device__ unsigned int xorshift(unsigned int& state) {
    unsigned int x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

// =====================================================================
// DEVICE HELPER: Linear Solver (Gaussian Elimination for 8x8)
// =====================================================================
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

// =====================================================================
// KERNEL: RANSAC Homography
// =====================================================================
__global__ void ransacKernel(const Point2f* srcPts, const Point2f* dstPts, int numMatches,
                             float* bestH, int* maxInliers, 
                             int numIterations, float threshold) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numIterations) return;

    unsigned int rngState = idx * 19349663 + (unsigned int)clock64();
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
    float h_local[9];

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

    if (!solveHomography8x8(A, h_local)) return;

    int inliers = 0;
    float threshSq = threshold * threshold; // This threshold is now in NORMALIZED space

    for (int i = 0; i < numMatches; i++) {
        Point2f s = srcPts[i];
        Point2f d = dstPts[i];
        float w = h_local[6] * s.x + h_local[7] * s.y + h_local[8];
        if (fabsf(w) < 1e-5f) continue;
        float px = (h_local[0] * s.x + h_local[1] * s.y + h_local[2]) / w;
        float py = (h_local[3] * s.x + h_local[4] * s.y + h_local[5]) / w;
        float distSq = (px - d.x)*(px - d.x) + (py - d.y)*(py - d.y);
        if (distSq < threshSq) inliers++;
    }

    if (inliers > atomicAdd(maxInliers, 0)) {
        atomicMax(maxInliers, inliers);
        if (inliers == *maxInliers) {
            for(int k=0; k<9; k++) bestH[k] = h_local[k];
        }
    }
}

// =====================================================================
// DEVICE HELPER: Bilinear Interpolation (RGB Support)
// =====================================================================
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

// =====================================================================
// KERNEL: Warp RGB
// =====================================================================
__global__ void warpKernel(const unsigned char* src, unsigned char* dst, 
                           int srcW, int srcH, int dstW, int dstH,
                           const float* h_inv, int offsetX, int offsetY) 
{
    int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    int dstY = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dstX >= dstW || dstY >= dstH) return;

    float u = dstX + offsetX;
    float v = dstY + offsetY;
    float w = h_inv[6]*u + h_inv[7]*v + h_inv[8];

    if (fabs(w) < 1e-5) return;
    
    float srcX = (h_inv[0]*u + h_inv[1]*v + h_inv[2]) / w;
    float srcY = (h_inv[3]*u + h_inv[4]*v + h_inv[5]) / w;

    if (srcX >= 0 && srcX < srcW - 1 && srcY >= 0 && srcY < srcH - 1) {
        int dstIdx = (dstY * dstW + dstX) * 3;
        dst[dstIdx + 0] = (unsigned char)bilinearAtStitcher(src, srcW, srcH, srcX, srcY, 0); // R
        dst[dstIdx + 1] = (unsigned char)bilinearAtStitcher(src, srcW, srcH, srcX, srcY, 1); // G
        dst[dstIdx + 2] = (unsigned char)bilinearAtStitcher(src, srcW, srcH, srcX, srcY, 2); // B
    }
}

// =====================================================================
// KERNEL: Paste RGB
// =====================================================================
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

// =====================================================================
// MAIN
// =====================================================================
int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " <img1> <img2> <kp1> <kp2> <matches> <output.jpg>\n";
        return 1;
    }

    // 1. Load Images (RGB)
    int w1, h1, c1, w2, h2, c2;
    unsigned char* img1 = stbi_load(argv[1], &w1, &h1, &c1, 3); 
    unsigned char* img2 = stbi_load(argv[2], &w2, &h2, &c2, 3);
    if(!img1 || !img2) { std::cerr << "Images not found\n"; return 1; }

    // 2. Load Matches & Keypoints
    struct MatchEntry { int qIdx; int tIdx; float dist; float secDist; };
    std::vector<Point2f> hostPts1, hostPts2;
    std::vector<Point2f> allKp1, allKp2;

    auto loadKPs = [](const char* f, std::vector<Point2f>& out) {
        FILE* fp = fopen(f, "rb"); if(!fp) return;
        fseek(fp, 0, SEEK_END); long n = ftell(fp) / 8; rewind(fp);
        out.resize(n); fread(out.data(), 8, n, fp); fclose(fp);
    };
    loadKPs(argv[3], allKp1);
    loadKPs(argv[4], allKp2);

    FILE* f = fopen(argv[5], "rb");
    if(!f) { std::cerr << "Matches file not found\n"; return 1; }
    
    MatchEntry entry;
    while(fread(&entry, sizeof(entry), 1, f)) {
        if(entry.dist < 0.75f * entry.secDist) {
            if(entry.qIdx < allKp1.size() && entry.tIdx < allKp2.size()) {
                hostPts1.push_back(allKp1[entry.qIdx]); // Dst
                hostPts2.push_back(allKp2[entry.tIdx]); // Src
            }
        }
    }
    fclose(f);

    int numMatches = hostPts1.size();
    if (numMatches < 4) { std::cerr << "Not enough matches.\n"; return 1; }

    // =========================================================
    // NORMALIZE POINTS (CRITICAL FIX FOR GPU PRECISION)
    // =========================================================
    std::vector<Point2f> normPts1, normPts2;
    Mat3x3 T1, T2;
    normalizePoints(hostPts1, normPts1, T1);
    normalizePoints(hostPts2, normPts2, T2);

    // 3. GPU RANSAC
    Point2f *d_pts1, *d_pts2;
    float *d_bestH;
    int *d_maxInliers;
    
    cudaMalloc(&d_pts1, numMatches * sizeof(Point2f));
    cudaMalloc(&d_pts2, numMatches * sizeof(Point2f));
    cudaMalloc(&d_bestH, 9 * sizeof(float));
    cudaMalloc(&d_maxInliers, sizeof(int));

    // Upload NORMALIZED points
    cudaMemcpy(d_pts1, normPts1.data(), numMatches * sizeof(Point2f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pts2, normPts2.data(), numMatches * sizeof(Point2f), cudaMemcpyHostToDevice);
    cudaMemset(d_maxInliers, 0, sizeof(int));

    // Launch RANSAC using Normalized Threshold (e.g. 0.01 instead of 5.0 pixels)
    // Since we scaled by roughly 1/(width), 5.0 pixels becomes very small.
    // Standard approach: 5.0 * scale_factor. Or just use a small float like 0.05.
    ransacKernel<<<(2048+255)/256, 256>>>(d_pts2, d_pts1, numMatches, d_bestH, d_maxInliers, 2048, 0.01f);
    cudaDeviceSynchronize();

    float h_norm[9];
    int maxInliers = 0;
    cudaMemcpy(h_norm, d_bestH, 9 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxInliers, d_maxInliers, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "RANSAC Best Inliers: " << maxInliers << "\n";

    if (maxInliers < 4) {
        std::cerr << "RANSAC failed to find a valid model.\n";
        return 1;
    }

    // =========================================================
    // DENORMALIZE HOMOGRAPHY: H = T1^-1 * H_norm * T2
    // =========================================================
    Mat3x3 H_norm;
    for(int i=0; i<9; i++) H_norm.data[i] = h_norm[i];
    
    Mat3x3 H = T1.inverse() * H_norm * T2;

    // 4. Calculate Canvas
    Point2f corners[4] = {{0,0}, {0,(float)h2}, {(float)w2,(float)h2}, {(float)w2,0}};
    Point2f transCorners[4];
    float x_min = 0, x_max = w1, y_min = 0, y_max = h1;

    for(int i=0; i<4; i++) {
        transCorners[i] = H.transform(corners[i]);
        if(transCorners[i].x < x_min) x_min = transCorners[i].x;
        if(transCorners[i].x > x_max) x_max = transCorners[i].x;
        if(transCorners[i].y < y_min) y_min = transCorners[i].y;
        if(transCorners[i].y > y_max) y_max = transCorners[i].y;
    }

    int canvasW = ceil(x_max - x_min);
    int canvasH = ceil(y_max - y_min);
    int offX = round(x_min);
    int offY = round(y_min);

    Mat3x3 T_shift = Mat3x3::identity();
    T_shift.data[2] = -offX; 
    T_shift.data[5] = -offY; 
    
    Mat3x3 H_final = T_shift * H;
    Mat3x3 H_inv = H_final.inverse();

    // 5. Warp & Paste
    unsigned char *d_img1, *d_img2, *d_canvas;
    float *d_H_inv;
    
    cudaMalloc(&d_img1, w1 * h1 * 3);
    cudaMalloc(&d_img2, w2 * h2 * 3);
    cudaMalloc(&d_canvas, canvasW * canvasH * 3);
    cudaMalloc(&d_H_inv, 9 * sizeof(float));
    
    cudaMemset(d_canvas, 0, canvasW * canvasH * 3);
    cudaMemcpy(d_img1, img1, w1 * h1 * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, img2, w2 * h2 * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_H_inv, H_inv.data, 9 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 grid((canvasW + 15)/16, (canvasH + 15)/16);
    
    warpKernel<<<grid, threads>>>(d_img2, d_canvas, 
                                  w2, h2, canvasW, canvasH,
                                  d_H_inv, 0, 0); 
                                  
    dim3 gridPaste((w1 + 15)/16, (h1 + 15)/16);
    pasteKernel<<<gridPaste, threads>>>(d_img1, d_canvas,
                                        w1, h1, canvasW, canvasH,
                                        offX, offY);

    cudaDeviceSynchronize();
    
    std::vector<unsigned char> result(canvasW * canvasH * 3);
    cudaMemcpy(result.data(), d_canvas, canvasW * canvasH * 3, cudaMemcpyDeviceToHost);
    
    stbi_write_jpg(argv[6], canvasW, canvasH, 3, result.data(), 100);

    cudaFree(d_pts1); cudaFree(d_pts2); cudaFree(d_bestH); cudaFree(d_maxInliers);
    cudaFree(d_img1); cudaFree(d_img2); cudaFree(d_canvas); cudaFree(d_H_inv);
    stbi_image_free(img1); stbi_image_free(img2);

    std::cout << "Saved " << argv[6] << "\n";
    return 0;
}