EXECUTABLE_FAST := fast
EXECUTABLE_BRIEF := brief
EXECUTABLE_MATCH := match
EXECUTABLE_STITCHER := stitcher
EXECUTABLE_GRAYSCALE := grayscale
OBJDIR := objs

CU_FAST := fast.cu
CU_BRIEF := brief.cu
CU_MATCH := match.cu
CU_STITCHER := stitcher.cu
CU_GRAYSCALE := grayscale.cu

OBJ_FAST := $(OBJDIR)/fast.o
OBJ_BRIEF := $(OBJDIR)/brief.o
OBJ_MATCH := $(OBJDIR)/match.o
OBJ_STITCHER := $(OBJDIR)/stitcher.o
OBJ_GRAYSCALE := $(OBJDIR)/grayscale.o

NVCC := nvcc
NVCCFLAGS := -O3 -m64 --gpu-architecture=compute_61 -ccbin /usr/bin/g++-11 -allow-unsupported-compiler

all: $(EXECUTABLE_FAST) $(EXECUTABLE_BRIEF) $(EXECUTABLE_MATCH) $(EXECUTABLE_STITCHER) $(EXECUTABLE_GRAYSCALE)

dirs:
	mkdir -p $(OBJDIR)

$(EXECUTABLE_FAST): dirs $(OBJ_FAST)
	$(NVCC) -o $@ $(OBJ_FAST)

$(EXECUTABLE_BRIEF): dirs $(OBJ_BRIEF)
	$(NVCC) -o $@ $(OBJ_BRIEF)

$(EXECUTABLE_MATCH): dirs $(OBJ_MATCH)
	$(NVCC) -o $@ $(OBJ_MATCH)

$(EXECUTABLE_STITCHER): dirs $(OBJ_STITCHER)
	$(NVCC) -o $@ $(OBJ_STITCHER)

$(EXECUTABLE_GRAYSCALE): dirs $(OBJ_GRAYSCALE)
	$(NVCC) -o $@ $(OBJ_GRAYSCALE)

$(OBJDIR)/%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(EXECUTABLE_FAST) $(EXECUTABLE_BRIEF) $(EXECUTABLE_MATCH) $(EXECUTABLE_STITCHER) $(EXECUTABLE_GRAYSCALE)

.PHONY: all clean dirs
