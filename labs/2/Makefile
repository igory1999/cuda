run:	vectorAdd
	./vectorAdd
vectorAdd: vectorAdd.cu
	nvcc -o vectorAdd vectorAdd.cu

vectorAdd_samples: ${CUDA_SAMPLES}/0_Simple/vectorAdd/vectorAdd.cu
	nvcc -o vectorAdd_samples -I${CUDA_SAMPLES}/common/inc ${CUDA_SAMPLES}/0_Simple/vectorAdd/vectorAdd.cu
run_samples: vectorAdd_samples
	./vectorAdd_samples
clean:
	rm -f *~ vectorAdd vectorAdd_samples *.o
