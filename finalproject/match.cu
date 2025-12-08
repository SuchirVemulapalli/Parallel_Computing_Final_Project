#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <limits>

// =====================================================================
// STRUCTURES
// =====================================================================

// Structure to hold the result for a single feature
struct MatchResult {
    int trainIdx;      // Index of the matching feature in Image 2
    int distance;      // Hamming distance of the best match
    int secondDist;    // Hamming distance of the 2nd best match (for Ratio Test)
    int padding;       // Keeps struct size aligned
};

// =====================================================================
// DEVICE KERNEL: Hamming Distance Matcher
// =====================================================================
__global__
void matchKernel(const unsigned int* d_desc1,  
                 const unsigned int* d_desc2,  
                 MatchResult* d_matches,      
                 int numDesc1, 
                 int numDesc2)
{
    // Each thread takes ONE descriptor from Image 1 and finds its best match in Image 2
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numDesc1) return;

    // Load the Query Descriptor into Registers
    unsigned int myDesc[8];
    
    int myOffset = idx * 8;
    
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        myDesc[k] = d_desc1[myOffset + k];
    }

    // Initialize best distances to max possible (256 bits)
    int bestDist = 256;
    int secondBestDist = 256; 
    int bestIdx = -1;

    // Loop through ALL descriptors in Image 2 (Brute Force)
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

// =====================================================================
// HOST HELPER FUNCTIONS
// =====================================================================
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

// =====================================================================
// MAIN
// =====================================================================
int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr << "Usage: ./match <desc1.bin> <desc2.bin> <matches.bin>\n";
        std::cerr << "Note: Keypoint files are not strictly needed for matching, only descriptors.\n";
        return 1;
    }

    std::cout << "Loading descriptor files...\n";

    std::vector<unsigned int> h_desc1, h_desc2;
    if (!loadDescriptorFile(argv[1], h_desc1) || !loadDescriptorFile(argv[2], h_desc2)) {
        return 1;
    }

    // Calculate number of descriptors (Total Ints / 8 Ints per Desc)
    int numDesc1 = h_desc1.size() / 8;
    int numDesc2 = h_desc2.size() / 8;

    if (numDesc1 == 0 || numDesc2 == 0) {
        std::cerr << "Error: One of the descriptor files is empty/invalid.\n";
        return 1;
    }

    std::cout << "Image 1 Descriptors: " << numDesc1 << "\n";
    std::cout << "Image 2 Descriptors: " << numDesc2 << "\n";
    std::cout << "Total Comparisons: " << (long)numDesc1 * (long)numDesc2 << "\n";

    unsigned int *d_desc1, *d_desc2;
    MatchResult *d_matches;

    cudaMalloc(&d_desc1, h_desc1.size() * sizeof(unsigned int));
    cudaMalloc(&d_desc2, h_desc2.size() * sizeof(unsigned int));
    cudaMalloc(&d_matches, numDesc1 * sizeof(MatchResult));

    // Upload Descriptors
    cudaMemcpy(d_desc1, h_desc1.data(), h_desc1.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_desc2, h_desc2.data(), h_desc2.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

    dim3 threads(256);
    dim3 blocks((numDesc1 + 255) / 256);

    std::cout << "Launching Matching Kernel...\n";
    matchKernel<<<blocks, threads>>>(d_desc1, d_desc2, d_matches, numDesc1, numDesc2);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    std::vector<MatchResult> h_matches(numDesc1);
    cudaMemcpy(h_matches.data(), d_matches, numDesc1 * sizeof(MatchResult), cudaMemcpyDeviceToHost);
    
    FILE* f = fopen(argv[3], "wb");
    if (!f) {
        std::cerr << "Error opening output file " << argv[3] << "\n";
        return 1;
    }

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

    std::cout << "Matches saved to " << argv[3] << "\n";

    // Cleanup
    cudaFree(d_desc1);
    cudaFree(d_desc2);
    cudaFree(d_matches);

    return 0;
}