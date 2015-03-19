all:
	nvcc -O3 -Xptxas -dlcm=ca -m64 -arch=sm_35 -Iinclude/ -o harmonic_sum_test src/harmonic_sum_test.cu