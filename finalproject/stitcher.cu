#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <ctime>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct Point2f { float x; float y; };

// =====================================================================
// MATRIX HOST HELPERS
// =====================================================================
struct Mat3x3 {
    float data[9]; 

    static Mat3x3 identity() { return {1,0,0, 0,1,0, 0,0,1}; }

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

    Point2f transform(Point2f p) const {
        float Z = data[6]*p.x + data[7]*p.y + data[8];
        return { (data[0]*p.x + data[1]*p.y + data[2]) / Z, 
                 (data[3]*p.x + data[4]*p.y + data[5]) / Z };
    }

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

void normalizePoints(const std::vector<Point2f>& in, std::vector<Point2f>& out, Mat3x3& T) {
    float meanx = 0, meany = 0;
    for (const auto& p : in) { meanx += p.x; meany += p.y; }
    meanx /= in.size(); meany /= in.size();

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
// DEVICE HELPERS
// =====================================================================
__device__ unsigned int xorshift(unsigned int& state) {
    unsigned int x = state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    state = x;
    return x;
}

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
// KERNEL: RANSAC 
// =====================================================================
__global__ void ransacBlockKernel(const Point2f* __restrict__ srcPts, 
                                  const Point2f* __restrict__ dstPts, 
                                  int numMatches,
                                  float* __restrict__ bestH, 
                                  int* __restrict__ maxInliers, 
                                  int* __restrict__ mutex,
                                  float threshold) 
{
    __shared__ float s_H[9];
    __shared__ int s_blockInliers;

    int tid = threadIdx.x;
    if (tid == 0) s_blockInliers = 0;

    // Model Generation 
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

        if (!solveHomography8x8(A, s_H)) s_H[8] = 0.0f; 
    }
    
    __syncthreads();
    if (fabsf(s_H[8]) < 1e-6f) return;

    // Count Inliers 
    float h0=s_H[0], h1=s_H[1], h2=s_H[2];
    float h3=s_H[3], h4=s_H[4], h5=s_H[5];
    float h6=s_H[6], h7=s_H[7], h8=s_H[8];
    
    float threshSq = threshold * threshold;
    int localCount = 0;

    for (int i = tid; i < numMatches; i += blockDim.x) {
        Point2f s = srcPts[i];
        Point2f d = dstPts[i];
        float w = h6 * s.x + h7 * s.y + h8;
        
        if (fabsf(w) > 1e-5f) {
            float invW = 1.0f / w;
            float px = (h0 * s.x + h1 * s.y + h2) * invW;
            float py = (h3 * s.x + h4 * s.y + h5) * invW;
            float distSq = (px - d.x)*(px - d.x) + (py - d.y)*(py - d.y);
            if (distSq < threshSq) localCount++;
        }
    }

    if (localCount > 0) atomicAdd(&s_blockInliers, localCount);
    __syncthreads();

    if (tid == 0) {
        if (s_blockInliers > *maxInliers) {
            
            bool isSet = false;
            do {
                if (isSet = (atomicCAS(mutex, 0, 1) == 0)) {
                    if (s_blockInliers > *maxInliers) {
                        *maxInliers = s_blockInliers;
                        for(int k=0; k<9; k++) bestH[k] = s_H[k];
                    }
                    atomicExch(mutex, 0);
                }
            } while (!isSet);
        }
    }
}

// =====================================================================
// KERNEL: Warp with Texture 
// =====================================================================
__global__ void warpTextureKernel(cudaTextureObject_t texObj, 
                                  unsigned char* __restrict__ dst, 
                                  int dstW, int dstH,
                                  const float* __restrict__ h_inv, 
                                  int offsetX, int offsetY) 
{
    int dstX = blockIdx.x * blockDim.x + threadIdx.x;
    int dstY = blockIdx.y * blockDim.y + threadIdx.y;

    if (dstX >= dstW || dstY >= dstH) return;

    // Use +0.5f to sample pixel centers
    float u = dstX + offsetX + 0.5f;
    float v = dstY + offsetY + 0.5f;

    float w = h_inv[6]*u + h_inv[7]*v + h_inv[8];

    if (fabs(w) < 1e-5) return;
    
    float invW = 1.0f / w;
    float srcX = (h_inv[0]*u + h_inv[1]*v + h_inv[2]) * invW;
    float srcY = (h_inv[3]*u + h_inv[4]*v + h_inv[5]) * invW;

    float4 color = tex2D<float4>(texObj, srcX, srcY);

    int dstIdx = (dstY * dstW + dstX) * 3;
    if (color.w > 0.0f) {
        dst[dstIdx + 0] = (unsigned char)(color.x * 255.0f);
        dst[dstIdx + 1] = (unsigned char)(color.y * 255.0f);
        dst[dstIdx + 2] = (unsigned char)(color.z * 255.0f);
    }
}

__global__ void pasteKernel(const unsigned char* __restrict__ src, 
                            unsigned char* __restrict__ dst, 
                            int srcW, int srcH, int dstW, int dstH, 
                            int offsetX, int offsetY)
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

    int w1, h1, c1, w2, h2, c2;
    unsigned char* img1 = stbi_load(argv[1], &w1, &h1, &c1, 3); 
    unsigned char* img2 = stbi_load(argv[2], &w2, &h2, &c2, 3);
    if(!img1 || !img2) { std::cerr << "Images not found\n"; return 1; }

    // Load Matches
    struct MatchEntry { int qIdx; int tIdx; float dist; float secDist; };
    std::vector<Point2f> hostPts1, hostPts2, allKp1, allKp2;

    auto loadKPs = [](const char* f, std::vector<Point2f>& out) {
        FILE* fp = fopen(f, "rb"); if(!fp) return;
        fseek(fp, 0, SEEK_END); long n = ftell(fp) / 8; rewind(fp);
        out.resize(n); 
        fread(out.data(), 8, n, fp); 
        fclose(fp);
    };
    loadKPs(argv[3], allKp1);
    loadKPs(argv[4], allKp2);

    FILE* f = fopen(argv[5], "rb");
    if(!f) return 1;
    MatchEntry entry;
    while(fread(&entry, sizeof(entry), 1, f)) {
        if(entry.dist < 0.75f * entry.secDist) {
            if(entry.qIdx < allKp1.size() && entry.tIdx < allKp2.size()) {
                hostPts1.push_back(allKp1[entry.qIdx]); 
                hostPts2.push_back(allKp2[entry.tIdx]); 
            }
        }
    }
    fclose(f);

    int numMatches = hostPts1.size();
    if (numMatches < 4) { std::cerr << "Not enough matches.\n"; return 1; }

    std::vector<Point2f> normPts1, normPts2;
    Mat3x3 T1, T2;
    normalizePoints(hostPts1, normPts1, T1);
    normalizePoints(hostPts2, normPts2, T2);

    Point2f *d_pts1, *d_pts2;
    float *d_bestH;
    int *d_maxInliers;
    int *d_mutex; 
    
    cudaMalloc(&d_pts1, numMatches * sizeof(Point2f));
    cudaMalloc(&d_pts2, numMatches * sizeof(Point2f));
    cudaMalloc(&d_bestH, 9 * sizeof(float));
    cudaMalloc(&d_maxInliers, sizeof(int));
    cudaMalloc(&d_mutex, sizeof(int));

    cudaMemcpy(d_pts1, normPts1.data(), numMatches * sizeof(Point2f), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pts2, normPts2.data(), numMatches * sizeof(Point2f), cudaMemcpyHostToDevice);
    cudaMemset(d_maxInliers, 0, sizeof(int));
    cudaMemset(d_mutex, 0, sizeof(int));

    ransacBlockKernel<<<2048, 256>>>(d_pts2, d_pts1, numMatches, d_bestH, d_maxInliers, d_mutex, 0.01f);
    cudaDeviceSynchronize();

    float h_norm[9];
    int maxInliers = 0;
    cudaMemcpy(h_norm, d_bestH, 9 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxInliers, d_maxInliers, sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "RANSAC Inliers: " << maxInliers << "\n";
    if (maxInliers < 4) return 1;

    Mat3x3 H_norm;
    for(int i=0; i<9; i++) H_norm.data[i] = h_norm[i];
    Mat3x3 H = T1.inverse() * H_norm * T2;

    // Calculate Canvas
    Point2f corners[4] = {{0,0}, {0,(float)h2}, {(float)w2,(float)h2}, {(float)w2,0}};
    float x_min = 0, x_max = w1, y_min = 0, y_max = h1;

    for(int i=0; i<4; i++) {
        Point2f p = H.transform(corners[i]);
        if(p.x < x_min) x_min = p.x; if(p.x > x_max) x_max = p.x;
        if(p.y < y_min) y_min = p.y; if(p.y > y_max) y_max = p.y;
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

    unsigned char *d_img1, *d_canvas;
    float *d_H_inv;
    
    cudaMalloc(&d_img1, w1 * h1 * 3);
    cudaMalloc(&d_canvas, canvasW * canvasH * 3);
    cudaMalloc(&d_H_inv, 9 * sizeof(float));
    
    cudaMemset(d_canvas, 0, canvasW * canvasH * 3);
    cudaMemcpy(d_img1, img1, w1 * h1 * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_H_inv, H_inv.data, 9 * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<uchar4> host_img2_rgba(w2 * h2);
    for(int i=0; i < w2*h2; i++) {
        host_img2_rgba[i].x = img2[i*3 + 0];
        host_img2_rgba[i].y = img2[i*3 + 1];
        host_img2_rgba[i].z = img2[i*3 + 2];
        host_img2_rgba[i].w = 255; 
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, w2, h2);
    cudaMemcpyToArray(cuArray, 0, 0, host_img2_rgba.data(), w2*h2*sizeof(uchar4), cudaMemcpyHostToDevice);

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder; 
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 0; 
 
    float borderColor[4] = {0.0f, 0.0f, 0.0f, 0.0f}; 
    memcpy(texDesc.borderColor, borderColor, sizeof(borderColor));

    cudaTextureObject_t texObj;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    dim3 threads(16, 16);
    dim3 gridWarp((canvasW + 15)/16, (canvasH + 15)/16);
    dim3 gridPaste((w1 + 15)/16, (h1 + 15)/16);

    warpTextureKernel<<<gridWarp, threads>>>(texObj, d_canvas, canvasW, canvasH, d_H_inv, 0, 0);
    pasteKernel<<<gridPaste, threads>>>(d_img1, d_canvas, w1, h1, canvasW, canvasH, offX, offY);
    
    cudaDeviceSynchronize();
    
    std::vector<unsigned char> result(canvasW * canvasH * 3);
    cudaMemcpy(result.data(), d_canvas, canvasW * canvasH * 3, cudaMemcpyDeviceToHost);
    stbi_write_jpg(argv[6], canvasW, canvasH, 3, result.data(), 100);

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaFree(d_pts1); cudaFree(d_pts2); cudaFree(d_bestH); cudaFree(d_maxInliers); cudaFree(d_mutex);
    cudaFree(d_img1); cudaFree(d_canvas); cudaFree(d_H_inv);
    stbi_image_free(img1); stbi_image_free(img2);

    return 0;
}