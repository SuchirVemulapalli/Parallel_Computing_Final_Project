#include <iostream>
#include "kernels.h"
#include <vector>
#include <limits>
#include <string>
#include <cmath>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void getDescriptors(unsigned char* d_img, std::vector<unsigned char> feature_vector, const char* inputPath, int w1, int h1);

// simple CUDA error check
#define CHECK_CUDA(call) do {                              \
    cudaError_t err = (call);                              \
    if (err != cudaSuccess) {                              \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1);                                      \
    }                                                      \
} while(0)

void getFeatures(const char* inputPath, int topN) {
    int width, height, ch;
    unsigned char* h_img = stbi_load(inputPath, &width, &height, &ch, 1);
    int imgSize = width * height;

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

    CHECK_CUDA(cudaMemcpy(d_in, h_img, imgSize, cudaMemcpyHostToDevice));

    dim3 threads(BLOCK_W,BLOCK_H);
    dim3 blocks((width + BLOCK_W-1)/BLOCK_W, (height + BLOCK_H-1)/BLOCK_H);

    int intensityThresh = 30;
    int runThresh = 9;
    // 1) FAST
    fastKernel<<<blocks, threads>>>(d_in, d_score, width, height, intensityThresh, runThresh);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 2) NMS (choose window 5 or 3)
    nmsKernel<<<blocks, threads>>>(d_score, d_nms, width, height);
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

    // cleanup
    CHECK_CUDA(cudaFree(d_score));
    CHECK_CUDA(cudaFree(d_nms));
    CHECK_CUDA(cudaFree(d_Ix));
    CHECK_CUDA(cudaFree(d_Iy));
    CHECK_CUDA(cudaFree(d_harris));
    stbi_image_free(h_img);
    std::cout << "Got top " << corners.size() << " Harris-ranked FAST corners of " << inputPath << "\n";

    getDescriptors(d_in, out, inputPath, width, height);
    CHECK_CUDA(cudaFree(d_in));
}

void getDescriptors(unsigned char* d_img, std::vector<unsigned char> feature_vector, const char* inputPath, int w1, int h1) {
    int imgSize = w1*h1;
    unsigned char* h_kp = feature_vector.data();
    unsigned char* d_desc;
    unsigned char* d_blurred;
    unsigned char* d_kp; 
    int* d_pattern;

    std::string path(inputPath);
    size_t lastIndex = path.find_last_of(".");
    std::string baseName = (lastIndex == std::string::npos) ? path : path.substr(0, lastIndex);
    std::string descName = baseName + ".descriptors.bin";
    std::string kpName   = baseName + ".kp.bin";
    const char* outDesc = descName.c_str();
    const char* outKp   = kpName.c_str();

    cudaMalloc(&d_blurred, imgSize);
    cudaMalloc(&d_kp, imgSize);
    cudaMalloc(&d_desc, imgSize * 32);
    cudaMalloc(&d_pattern, sizeof(h_ORB_pattern)); 

    // Copy Data to GPU
    cudaMemcpy(d_kp,  h_kp,  imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern, h_ORB_pattern, sizeof(h_ORB_pattern), cudaMemcpyHostToDevice);

    // Initialize descriptors to 0
    cudaMemset(d_desc, 0, imgSize * 32);

    std::cout << "Applying Gaussian Blur..." << std::endl;

    // =================================================================
    // STEP 1: RUN GAUSSIAN BLUR (OPTIMIZED)
    // =================================================================
    dim3 blurBlock(16, 16);
    dim3 blurGrid((w1 + 15) / 16, (h1 + 15) / 16);
    
    gaussianBlur<<<blurGrid, blurBlock>>>(d_img, d_blurred, w1, h1);
    
    cudaDeviceSynchronize();

    std::cout << "Running ORB Kernel..." << std::endl;

    // =================================================================
    // STEP 2: RUN ORB KERNEL (OPTIMIZED WITH DYNAMIC SMEM)
    // =================================================================
    int threadsPerBlock = 256;
    int blocksPerGrid = (imgSize + threadsPerBlock - 1) / threadsPerBlock;
    size_t smemSize = (1024 * sizeof(int)) + (33 * 33 * sizeof(unsigned char));

    orbKernel<<<blocksPerGrid, threadsPerBlock, smemSize>>>(
        d_blurred, 
        d_kp, 
        d_pattern, 
        d_desc, 
        w1, h1
    );
    
    cudaDeviceSynchronize();

    // Copy descriptors back to Host
    std::vector<unsigned char> h_desc(imgSize * 32);
    cudaMemcpy(h_desc.data(), d_desc, imgSize * 32, cudaMemcpyDeviceToHost);

    // Save Logic (Descriptors + Keypoints)
    FILE* f_desc = fopen(outDesc, "wb");
    FILE* f_kp   = fopen(outKp,   "wb");

    int count = 0;
    // Must match the BORDER used in the kernel to avoid saving zeroed descriptors
    const int HOST_BORDER = 24; 

    for (int i=0; i<imgSize; i++) {
        if (h_kp[i] == 255) {
            
            // Calculate X and Y on CPU
            int px = i % w1;
            int py = i / w1;

            if (px < HOST_BORDER || px >= w1 - HOST_BORDER || 
                py < HOST_BORDER || py >= h1 - HOST_BORDER) {
                continue; 
            }

            // Write Descriptor
            fwrite(&h_desc[i*32], 1, 32, f_desc);

            // Write Coordinates
            float kx = (float)(i % w1); 
            float ky = (float)(i / w1);
            fwrite(&kx, sizeof(float), 1, f_kp);
            fwrite(&ky, sizeof(float), 1, f_kp);

            count++;
        }
    }

    fclose(f_desc);
    fclose(f_kp);

    std::cout << "Successfully saved " << count << " descriptors and keypoints.\n";

    cudaFree(d_blurred);
    cudaFree(d_desc);
    cudaFree(d_kp);
    cudaFree(d_pattern);
}

void match(const char* img1Path, const char* img2Path)
{
    auto stripExt = [](const std::string& s) {
    size_t i = s.find_last_of(".");
    return (i == std::string::npos) ? s : s.substr(0, i);
    };
    std::string base1 = stripExt(std::string(img1Path));
    std::string descName1 = base1 + ".descriptors.bin";
    std::string kpName1   = base1 + ".kp.bin";
    std::string base2 = stripExt(std::string(img2Path));
    std::string descName2 = base2 + ".descriptors.bin";
    std::string kpName2   = base2 + ".kp.bin";
    std::string outName = base1 + ".matches.bin";

    std::cout << "Loading descriptor files...\n";

    std::vector<unsigned int> h_desc1, h_desc2;
    loadDescriptorFile(descName1.c_str(), h_desc1);
    loadDescriptorFile(descName2.c_str(), h_desc2);

    // Calculate number of descriptors (Total Ints / 8 Ints per Desc)
    int numDesc1 = h_desc1.size() / 8;
    int numDesc2 = h_desc2.size() / 8;

    std::cout << "Image 1 Descriptors: " << numDesc1 << "\n";
    std::cout << "Image 2 Descriptors: " << numDesc2 << "\n";
    std::cout << "Total Comparisons: " << (long)numDesc1 * (long)numDesc2 << "\n";

    // -----------------------------------------------------
    // GPU Allocation
    // -----------------------------------------------------
    unsigned int *d_desc1, *d_desc2;
    MatchResult *d_matches;

    cudaMalloc(&d_desc1, h_desc1.size() * sizeof(unsigned int));
    cudaMalloc(&d_desc2, h_desc2.size() * sizeof(unsigned int));
    cudaMalloc(&d_matches, numDesc1 * sizeof(MatchResult));

    // Upload Descriptors
    cudaMemcpy(d_desc1, h_desc1.data(), h_desc1.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_desc2, h_desc2.data(), h_desc2.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // -----------------------------------------------------
    // Launch Kernel
    // -----------------------------------------------------
    dim3 threads(256);
    dim3 blocks((numDesc1 + 255) / 256);

    std::cout << "Launching Matching Kernel...\n";
    matchKernel<<<blocks, threads>>>(d_desc1, d_desc2, d_matches, numDesc1, numDesc2);
    
    cudaDeviceSynchronize();

    // -----------------------------------------------------
    // Download Results
    // -----------------------------------------------------
    std::vector<MatchResult> h_matches(numDesc1);
    cudaMemcpy(h_matches.data(), d_matches, numDesc1 * sizeof(MatchResult), cudaMemcpyDeviceToHost);

    // -----------------------------------------------------
    // Save to Disk (for Python)
    // -----------------------------------------------------
    // We write a simple binary format:
    // [QueryIndex (int), TrainIndex (int), Distance (float), SecondDistance (float)]
    // We use float for distances to match standard OpenCV DMatch structs commonly used in Python.
    
    FILE* f = fopen(outName.c_str(), "wb");

    int saved = 0;
    for (int i = 0; i < numDesc1; i++) {
        struct OutputFormat {
            int qIdx;
            int tIdx;
            float dist;
            float secDist;
        } entry;

        entry.qIdx = i;
        entry.tIdx = h_matches[i].trainIdx;
        entry.dist = (float)h_matches[i].distance;
        entry.secDist = (float)h_matches[i].secondDist;

        fwrite(&entry, sizeof(entry), 1, f);
        saved++;
    }
    fclose(f);

    // Cleanup
    cudaFree(d_desc1);
    cudaFree(d_desc2);
    cudaFree(d_matches);
}

void stitch(const char* img1Path, const char* img2Path, const char* outimg)
{
    auto stripExt = [](const std::string& s) {
    size_t i = s.find_last_of(".");
    return (i == std::string::npos) ? s : s.substr(0, i);
    };
    std::string base1 = stripExt(std::string(img1Path));
    std::string descName1 = base1 + ".descriptors.bin";
    std::string kpName1   = base1 + ".kp.bin";
    std::string base2 = stripExt(std::string(img2Path));
    std::string descName2 = base2 + ".descriptors.bin";
    std::string kpName2   = base2 + ".kp.bin";
    std::string matches = base1 + ".matches.bin";

    // 1. Load Images (RGB)
    int w1, h1, c1, w2, h2, c2;
    unsigned char* img1 = stbi_load(img1Path, &w1, &h1, &c1, 3); 
    unsigned char* img2 = stbi_load(img2Path, &w2, &h2, &c2, 3);

    // 2. Load Matches & Keypoints
    struct MatchEntry { int qIdx; int tIdx; float dist; float secDist; };
    std::vector<Point2f> hostPts1, hostPts2;
    std::vector<Point2f> allKp1, allKp2;

    auto loadKPs = [](const char* f, std::vector<Point2f>& out) {
        FILE* fp = fopen(f, "rb"); if(!fp) return;
        fseek(fp, 0, SEEK_END); long n = ftell(fp) / 8; rewind(fp);
        out.resize(n); fread(out.data(), 8, n, fp); fclose(fp);
    };
    loadKPs(kpName1.c_str(), allKp1);
    loadKPs(kpName2.c_str(), allKp2);

    FILE* f = fopen(matches.c_str(), "rb");
    
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
    
    stbi_write_jpg(outimg, canvasW, canvasH, 3, result.data(), 100);

    cudaFree(d_pts1); cudaFree(d_pts2); cudaFree(d_bestH); cudaFree(d_maxInliers);
    cudaFree(d_img1); cudaFree(d_img2); cudaFree(d_canvas); cudaFree(d_H_inv);
    stbi_image_free(img1); stbi_image_free(img2);

    std::cout << "Saved " << outimg << "\n";
}

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::cerr << "Usage: ./pipeline <img1.png> <img2.png> <out_img> <topN>\n";
        return 1;
    }
    const char* img1path = argv[1];
    const char* img2path = argv[2];
    const char* outimg = argv[3];
    int topN = std::atoi(argv[4]);

    getFeatures(img1path, topN);
    getFeatures(img2path, topN);
    match(img1path, img2path);
    stitch(img1path, img2path, outimg);
}