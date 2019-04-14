run0: build0
	./occupancy
build0: occupancy.cu
	nvcc -lcurand -o occupancy occupancy.cu
run1: build1
	./occupancy1
build1: occupancy1.cu
	nvcc -lcurand -o occupancy1 occupancy1.cu
clean:
	rm -rf *~ occupancy occupancy1

