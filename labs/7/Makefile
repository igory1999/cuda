run1: build1
	./atomics1
build1: atomics1.cu
	nvcc -o atomics1 atomics1.cu
run2: build2
	./atomics2
build2: atomics2.cu
	nvcc -o atomics2 atomics2.cu
run3: build3
	./atomics3
build3: atomics3.cu
	nvcc -o atomics3 atomics3.cu
clean:
	rm -rf *~ atomics1 atomics2 atomics3
