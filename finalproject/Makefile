EXECUTABLE_FAST := fast
EXECUTABLE_BRIEF := brief
OBJDIR := objs

CU_FAST := fast.cu
CU_BRIEF := brief.cu

OBJ_FAST := $(OBJDIR)/fast.o
OBJ_BRIEF := $(OBJDIR)/brief.o

NVCC := nvcc
NVCCFLAGS := -O3 -m64 --gpu-architecture=compute_61 -ccbin /usr/bin/g++-11 -allow-unsupported-compiler

all: $(EXECUTABLE_FAST) $(EXECUTABLE_BRIEF)

dirs:
	mkdir -p $(OBJDIR)

$(EXECUTABLE_FAST): dirs $(OBJ_FAST)
	$(NVCC) -o $@ $(OBJ_FAST)

$(EXECUTABLE_BRIEF): dirs $(OBJ_BRIEF)
	$(NVCC) -o $@ $(OBJ_BRIEF)

$(OBJDIR)/%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(EXECUTABLE_FAST) $(EXECUTABLE_BRIEF)

.PHONY: all clean dirs
