# =============================================================================
# EXECUTABLES
# =============================================================================
EXECUTABLE_FAST      := fast
EXECUTABLE_BRIEF     := brief
EXECUTABLE_MATCH     := match
EXECUTABLE_STITCHER  := stitcher
EXECUTABLE_GRAYSCALE := grayscale
EXECUTABLE_PIPELINE  := pipeline

# =============================================================================
# FILES
# =============================================================================
OBJDIR := objs

# Original independent source files
CU_FAST      := fast.cu
CU_BRIEF     := brief.cu
CU_MATCH     := match.cu
CU_STITCHER  := stitcher.cu
CU_GRAYSCALE := grayscale.cu

# New pipeline source files
CU_PIPELINE  := pipeline.cu
CU_KERNELS   := kernels.cu

# Object file paths
OBJ_FAST      := $(OBJDIR)/fast.o
OBJ_BRIEF     := $(OBJDIR)/brief.o
OBJ_MATCH     := $(OBJDIR)/match.o
OBJ_STITCHER  := $(OBJDIR)/stitcher.o
OBJ_GRAYSCALE := $(OBJDIR)/grayscale.o

# Pipeline needs two object files
OBJ_PIPELINE  := $(OBJDIR)/pipeline.o
OBJ_KERNELS   := $(OBJDIR)/kernels.o

# =============================================================================
# COMPILER SETTINGS
# =============================================================================
NVCC := nvcc
# Added -I. to ensure it finds kernels.h
NVCCFLAGS := -O3 -m64 --gpu-architecture=compute_61 -ccbin /usr/bin/g++-11 -allow-unsupported-compiler -I.

# =============================================================================
# TARGETS
# =============================================================================

all: $(EXECUTABLE_FAST) $(EXECUTABLE_BRIEF) $(EXECUTABLE_MATCH) $(EXECUTABLE_STITCHER) $(EXECUTABLE_GRAYSCALE) $(EXECUTABLE_PIPELINE)

dirs:
	mkdir -p $(OBJDIR)

# --- 1. NEW PIPELINE TARGET (Links Main + Kernels) ---
$(EXECUTABLE_PIPELINE): dirs $(OBJ_PIPELINE) $(OBJ_KERNELS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJ_PIPELINE) $(OBJ_KERNELS)

# --- 2. ORIGINAL TARGETS ---
$(EXECUTABLE_FAST): dirs $(OBJ_FAST)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJ_FAST)

$(EXECUTABLE_BRIEF): dirs $(OBJ_BRIEF)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJ_BRIEF)

$(EXECUTABLE_MATCH): dirs $(OBJ_MATCH)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJ_MATCH)

$(EXECUTABLE_STITCHER): dirs $(OBJ_STITCHER)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJ_STITCHER)

$(EXECUTABLE_GRAYSCALE): dirs $(OBJ_GRAYSCALE)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJ_GRAYSCALE)

# --- 3. COMPILATION RULES ---

# Generic rule for .cu -> .o
$(OBJDIR)/%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Header Dependency: Recompile pipeline if header changes
$(OBJ_PIPELINE) $(OBJ_KERNELS): kernels.h

clean:
	rm -rf $(OBJDIR) $(EXECUTABLE_FAST) $(EXECUTABLE_BRIEF) $(EXECUTABLE_MATCH) $(EXECUTABLE_STITCHER) $(EXECUTABLE_GRAYSCALE) $(EXECUTABLE_PIPELINE)

.PHONY: all clean dirs