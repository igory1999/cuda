SAMPLES=/software/cuda-10.0-el7-x86_64/samples
run0: build0
	./matrixMul0
build0: matrixMul0.cu
	nvcc -std=c++11 -o matrixMul0 matrixMul0.cu
run1: build1
	./matrixMul1
build1: matrixMul1.cu
	nvcc -std=c++11 -o matrixMul1 matrixMul1.cu
run2: build2
	./matrixMul
build2: ${SAMPLES}/0_Simple/matrixMul/matrixMul.cu
	nvcc -I${SAMPLES}/common/inc -o matrixMul ${SAMPLES}/0_Simple/matrixMul/matrixMul.cu
clean:
	rm -rf *~ *.out *.err matrixMul0 matrixMul1 matrixMul

