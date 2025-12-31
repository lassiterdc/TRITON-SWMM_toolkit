ifndef BUILDDIR
	BUILDDIR=./build
endif
ifndef SRCDIR
	SRCDIR=./src
endif

export LD_LIBRARY_PATH=./Stormwater-Management-Model/build/bin:$LD_LIBRARY_PATH

hpc_gpu:
	make -f makefile.hpc ACTIVE_GPU=1 SRCDIR=$(SRCDIR) BUILDDIR=$(BUILDDIR)
	
hpc_omp:
	make -f makefile.hpc ACTIVE_OMP=1 SRCDIR=$(SRCDIR) BUILDDIR=$(BUILDDIR)

hpc_swmm_gpu:
	make -f makefile.hpc ACTIVE_GPU=1 SWMM_TRITON=1 SRCDIR=$(SRCDIR) BUILDDIR=$(BUILDDIR)
	
hpc_swmm_omp:
	make -f makefile.hpc ACTIVE_OMP=1 SWMM_TRITON=1 SRCDIR=$(SRCDIR) BUILDDIR=$(BUILDDIR)

summit_gpu:
	make -f makefile.summit ACTIVE_GPU=1 SRCDIR=$(SRCDIR) BUILDDIR=$(BUILDDIR)
	
summit_omp:
	make -f makefile.summit ACTIVE_OMP=1 SRCDIR=$(SRCDIR) BUILDDIR=$(BUILDDIR)

summit_swmm_gpu:
	make -f makefile.summit ACTIVE_GPU=1 SWMM_TRITON=1 SRCDIR=$(SRCDIR) BUILDDIR=$(BUILDDIR)
	
summit_swmm_omp:
	make -f makefile.summit ACTIVE_OMP=1 SWMM_TRITON=1 SRCDIR=$(SRCDIR) BUILDDIR=$(BUILDDIR)

afw_gpu:
	make -f makefile.afw ACTIVE_GPU=1 SRCDIR=$(SRCDIR) BUILDDIR=$(BUILDDIR)
	
afw_omp:
	make -f makefile.afw ACTIVE_OMP=1 SRCDIR=$(SRCDIR) BUILDDIR=$(BUILDDIR)

afw_swmm_gpu:
	make -f makefile.afw ACTIVE_GPU=1 SWMM_TRITON=1 SRCDIR=$(SRCDIR) BUILDDIR=$(BUILDDIR)
	
afw_swmm_omp:
	make -f makefile.afw ACTIVE_OMP=1 SWMM_TRITON=1 SRCDIR=$(SRCDIR) BUILDDIR=$(BUILDDIR)

frontier_gpu:
	make -f makefile.frontier ACTIVE_GPU=1 SRCDIR=$(SRCDIR) BUILDDIR=$(BUILDDIR)

frontier_omp:
	make -f makefile.frontier SRCDIR=$(SRCDIR) BUILDDIR=$(BUILDDIR)

frontier_swmm_gpu:
	make -f makefile.frontier ACTIVE_GPU=1 SWMM_TRITON=1 SRCDIR=$(SRCDIR) BUILDDIR=$(BUILDDIR)

frontier_swmm_omp: 
	make -f makefile.frontier SWMM_TRITON=1 SRCDIR=$(SRCDIR) BUILDDIR=$(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)/triton
