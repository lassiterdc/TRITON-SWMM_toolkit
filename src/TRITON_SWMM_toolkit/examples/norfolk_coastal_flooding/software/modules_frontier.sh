if [ ! command -v module &> /dev/null ];then
    source /usr/share/lmod/lmod/init/sh
fi
export HIP_PLATFORM=amd
export HIP_COMPILER=amdclang++
export MPICH_GPU_SUPPORT_ENABLED=1
export CRAY_CPU_TARGET=x86-64

module purge
module load PrgEnv-amd/8.5.0 amd/5.7.1 rocm/5.7.1 xpmem craype-accel-amd-gfx90a

module -t list
