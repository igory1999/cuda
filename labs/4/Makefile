run_functions: build_functions
	echo "Standard functions"
	./functions_standard
	echo "================"
	echo "Instrinsic functions"
	./functions_intrinsic
build_functions: functions.cu
	nvcc -o functions_standard functions.cu
	nvcc --use_fast_math -o functions_intrinsic functions.cu
run_nostreams: build_nostreams
	./nostreams
build_nostreams: nostreams.cu
	nvcc -o nostreams nostreams.cu
run_streams: build_streams
	./streams
build_streams: streams.cu
	nvcc -o streams streams.cu
debug: 
	nvcc -g -G -o streams_debug streams.cu
clean:
	rm -rf *~ nostreams streams functions_standard functions_intrinsic streams_debug

