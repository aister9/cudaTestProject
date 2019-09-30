#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "rmg.h"
#include <direct.h>
#include "ArrayQueue.h"

#define LAST_FILE_NUMBER 10
#define MATRIX_W 512
#define MATRIX_H 512
#define QUEUE_CAPACITY 6

#define PREPROCESS true

__global__ void addSample(int *inputData,int *gpuMap,int *outputData, int w, int h);

int main() {
	srand((unsigned int)time(NULL));

#if PREPROCESS == true
	if (mkdir("matrixs") == -1) {
		printf("matrixs: Folder is already exist..\n");
	}
	if (mkdir("output") == -1) {
		printf("output: Folder is already exist..\n");
	}

	cout << "Random Matrix Generation ... ";
	for (int i = 0; i < LAST_FILE_NUMBER; i++) {
		string filename = "matrixs\\matrix " + to_string(i) + ".txt";
		random_matrix_generator(filename, MATRIX_W, MATRIX_H);
		int persentage = (double)i / (double)10 * 100;
		if (persentage % 25 == 0) cout << persentage << "% ... ";
	}
	cout << "Complete" << endl;
#endif

	int *gpuMap; int *inputData[QUEUE_CAPACITY]; int *outputData[QUEUE_CAPACITY];
	for (int i = 0; i < QUEUE_CAPACITY; i++) {
		cudaMalloc(&inputData[i], sizeof(int) * MATRIX_W * MATRIX_H);
		cudaMalloc(&outputData[i], sizeof(int) * MATRIX_W * MATRIX_H);
	}
	cudaMalloc(&gpuMap, sizeof(int) * MATRIX_W * MATRIX_H);
	int *cpuMap = new int[MATRIX_W * MATRIX_H];

	for (int i = 0; i < MATRIX_W * MATRIX_H; i++) {
		cpuMap[i] = rand() % 10;
	}

	//gpuMap Generate
	cudaMemcpy(gpuMap, cpuMap, sizeof(int) * MATRIX_W * MATRIX_H, cudaMemcpyHostToDevice);
	delete[] cpuMap;

	cudaStream_t streams[QUEUE_CAPACITY];
	cudaEvent_t isEnd[QUEUE_CAPACITY];
	for (int i = 0; i < QUEUE_CAPACITY; i++) {
		cudaStreamCreate(&streams[i]);
		cudaEventCreate(&isEnd[i]);
	}
	ArrayQueue aq; //input Queue
	ArrayQueue rq; //output Queue
	aq.resize(QUEUE_CAPACITY);
	rq.resize(LAST_FILE_NUMBER);
	//Queue_capacity == 6, MATRIX_W * MATRIX_H * 4 = 31MB * 6 = 186MB
	//aq + rq = 186+310 = 496MB(2.42% use ram : total ram 20gb)
	aq.set_array_size_pinned(MATRIX_W, MATRIX_H);
	rq.set_array_size_pinned(MATRIX_W, MATRIX_H);

	int ind = 0;
	bool error = false;
#pragma omp parallel sections
	{
#pragma omp section // file read
		{
			while (true) {
				
				if (aq.isFull()) { //꽉찬상태면 무한루프
					continue;
				}
				//아니면
				string fname = "matrixs\\matrix " + to_string(ind) + ".txt";;
				aq.tailAdder(); //다음 큐의 위치로 tail을 이동
				if (!read_matrix_in_file(fname, aq.data[aq.tail], MATRIX_W, MATRIX_H)) {
					cout << "error options" << endl;
					error = true;
					break;
				}
				cout << "File read complete : " << fname << endl;
				ind++;

				if (ind == LAST_FILE_NUMBER) {
					break;
				}
			}
		}
#pragma omp section // run kernel
		{
			int streamInd = 0;
			while (true) {
				if (aq.isEmpty()) {
					if (ind == LAST_FILE_NUMBER) break;
					else continue;
				}
				if (error) break;
				
				cudaMemcpyAsync(inputData[streamInd], aq.dequeue(), sizeof(int)*MATRIX_W*MATRIX_H, cudaMemcpyHostToDevice, streams[streamInd]);
				addSample << <1, 512, 0, streams[streamInd] >> > (inputData[streamInd], gpuMap, outputData[streamInd], MATRIX_W, MATRIX_H);
				while (rq.isFull()) {
					continue;
				}
				rq.tailAdder();
				cudaMemcpyAsync(rq.getTailData(), outputData[streamInd], sizeof(int)*MATRIX_W*MATRIX_H, cudaMemcpyDeviceToHost, streams[streamInd]);
				cudaEventRecord(isEnd[streamInd], streams[streamInd]);

				streamInd = (streamInd + 1) % QUEUE_CAPACITY;
			}
		}
#pragma omp section //file output
		{
			int resultInd = 0;
			bool calcEnd = false;
			while (true) {
				if (rq.isEmpty()) {
					if (ind == LAST_FILE_NUMBER) break;
					else continue;
				}
				
				for (int j = 0; j < QUEUE_CAPACITY; j++) {
					if (cudaEventQuery(isEnd[j]) == cudaSuccess) {
						cout << "CUDA stream " << j << "end" << endl;
						calcEnd = true;
						break;
					}
				}
				if (calcEnd == false) continue;

				string filename = "output\\result " + to_string(resultInd) + ".txt";
				record_matrix_in_file(filename, rq.dequeue(), MATRIX_W, MATRIX_H);
				resultInd++;
				calcEnd = false;
			}
		}
	}


	for (int i = 0; i < QUEUE_CAPACITY; i++) {
		cudaStreamDestroy(streams[i]);
		cudaEventDestroy(isEnd[i]);
	}
	for (int i = 0; i < QUEUE_CAPACITY; i++) {
		cudaFree(inputData[i]);
		cudaFree(outputData[i]);
	}
	aq.cudaFreeAllMembers();
	cudaFree(gpuMap);
}

__global__ void addSample(int *inputData, int *gpuMap, int *outputData, int w, int h) {
	int tIdx = threadIdx.x;
	int totalThreads = blockDim.x;

	for (int i = tIdx; i < w*h; i += totalThreads) {
		outputData[i] = inputData[i] + gpuMap[i];
	}

	return;
}

