# Parallel Computing Final Project  
## Panoramic Image Creation on CUDA GPUs (FAST + BRIEF + Matching + RANSAC Stitching)

This project implements a full panoramic image creation pipeline using CUDA GPU acceleration.  
The system uses the following four main kernels:

1. **FAST** — Keypoint detection  
2. **BRIEF** — Descriptor generation  
3. **Matching** — Feature matching across images  
4. **Stitching** — Homography estimation (RANSAC) + panorama creation  

Each kernel can be executed independently, or combined into a full end-to-end pipeline.

---

## Running the Complete Pipeline

```bash
cd finalproject
make clean
make
./pipeline <img1.png> <img2.png> <out_img.png> <topN>
```




