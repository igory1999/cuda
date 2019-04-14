run0: build0
	./coalesce
build0: coalesce.cu
	nvcc -lcurand -o coalesce coalesce.cu
clean:
	rm -rf *~ coalesce
