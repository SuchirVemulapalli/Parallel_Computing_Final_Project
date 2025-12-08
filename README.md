# Parallel Computing Final Project
Panoramic Image Creation on CUDA GPUs Using ORB + RANSAC. The full pipeline to enable this functionality consists of four main kernels: FAST, BRIEF, matching, stitching.

# Running the project (complete pipeline)
cd finalproject
make clean
make
./pipeline <img1.png> <img2.png> <out_img.png> <topN>

# Every kernel in the pipeline can also be tested individually
cd finalproject
make clean
make
./fast <input.png> <output.png> <runThresh> <topN>
./brief <img.png> <kp_mask.png> <out_desc.bin> <out_keypoints.bin>
./match <desc1.bin> <desc2.bin> <matches.bin>
./stitcher <img1.png> <img2.png> <kp1.bin> <kp2.bin> <matches.bin> <output.png>


