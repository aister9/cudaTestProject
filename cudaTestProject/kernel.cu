#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "rmg.h"
#include <direct.h>
#include "ArrayQueue.h"
#include "DS_timer.h"

#define LAST_FILE_NUMBER 10
#define MATRIX_W 512
#define MATRIX_H 512
#define QUEUE_CAPACITY 6

#define PREPROCESS true
#define NORMALCASE true
#define CASEMODE 0 // 0 is Matrix Sum, 1 is Matrix Multiplication
#define SAVEFORMAT 0 //0 is File, 1 is Vector

__global__ void addSample(int *inputData,int *gpuMap,int *outputData, int w, int h);
__global__ void matrixMultiSample(int *inputData, int *gpuMap, int *outputData, int w, int h);

int main() {
	srand((unsigned int)time(NULL));

#if PREPROCESS == true
	if (mkdir("matrixs") == -1) {
		printf("matrixs: Folder is already exist..\n");
	}
	if (mkdir("output") == -1) {
		printf("output: Folder is already exist..\n");
	}
	if (mkdir("case2-output") == -1) {
		printf("case2-output: Folder is already exist..\n");
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
	record_matrix_in_file("cpumap.txt", cpuMap, MATRIX_W, MATRIX_H);

	//gpuMap Generate
	cudaMemcpy(gpuMap, cpuMap, sizeof(int) * MATRIX_W * MATRIX_H, cudaMemcpyHostToDevice);
	delete[] cpuMap;

	cudaStream_t streams[QUEUE_CAPACITY];
	cudaEvent_t isEnd[LAST_FILE_NUMBER];
	for (int i = 0; i < QUEUE_CAPACITY; i++) {
		cudaStreamCreate(&streams[i]);
	}
	for (int i = 0; i < LAST_FILE_NUMBER; i++)
		cudaEventCreate(&isEnd[i]);
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

	DS_timer caseTimer(2);
	caseTimer.initTimers();
	caseTimer.setTimerName(0, "Use Concurrent Kernel(Heterogeneous parallel)");
	caseTimer.setTimerName(1, "Use Normal Case(1 Kernel "+to_string(LAST_FILE_NUMBER)+" LOOP)");
	vector<int *> case1_Vector;
	vector<int *> case2_Vector;

#if CASEMODE == 1
	dim3 bDim(16, 16); // 256 threads per block
	dim3 gDim(ceil((float)MATRIX_W / 16.0), ceil((float)MATRIX_H / 16.0));
#endif

	//////////////////////////////////////////////////////////////////////////
	/*Concurrent Kernel*/
	caseTimer.onTimer(0);
#pragma omp parallel sections
	{
#pragma omp section // file read
		{
			while (true) {
				
				if (aq.isFull()) { //꽉찬상태면 무한루프
					continue;
				}
				//아니면
				string fname = "matrixs\\matrix " + to_string(ind) + ".txt";
				aq.tailAdder(); //다음 큐의 위치로 tail을 이동
				if (!read_matrix_in_file(fname, aq.data[aq.tail], MATRIX_W, MATRIX_H)) {
					cout << "error options" << endl;
					error = true;
					break;
				}
				aq.chk=aq.tail;
				ind++;

				if (ind == LAST_FILE_NUMBER) {
					break;
				}
			}
		}
#pragma omp section // run kernel
		{
			int streamInd = 0;
			int fileInd = 0;
			while (true) {
				if (aq.isEmpty()) {
					if (ind == LAST_FILE_NUMBER) break;
					else continue;
				}
				if (error) break;
				if (aq.chk != aq.tail) continue;
				
				cudaMemcpyAsync(inputData[streamInd], aq.dequeue(), sizeof(int)*MATRIX_W*MATRIX_H, cudaMemcpyHostToDevice, streams[streamInd]);
#if CASEMODE == 0
				addSample << <1, 512, 0, streams[streamInd] >> > (inputData[streamInd], gpuMap, outputData[streamInd], MATRIX_W, MATRIX_H);
#elif CASEMODE == 1
				matrixMultiSample << <gDim, bDim, 0, streams[streamInd] >> > (inputData[streamInd], gpuMap, outputData[streamInd], MATRIX_W, MATRIX_H);
#endif
				
				while (rq.isFull()) {
					continue;
				}
				rq.tailAdder();
				cudaMemcpyAsync(rq.getTailData(), outputData[streamInd], sizeof(int)*MATRIX_W*MATRIX_H, cudaMemcpyDeviceToHost, streams[streamInd]);
				cudaEventRecord(isEnd[fileInd++], streams[streamInd]);

				streamInd = (streamInd + 1) % QUEUE_CAPACITY;
			}
		}
#pragma omp section //file output
		{
			int resultInd = 0;
			while (true) {
				if (rq.isEmpty()) {
					if (ind == LAST_FILE_NUMBER) break;
					else continue;
				}
				if (cudaEventQuery(isEnd[resultInd]) != cudaSuccess) continue;
				
#if SAVEFORMAT == 0
				string filename = "output\\result " + to_string(resultInd) + ".txt";
				if (!record_matrix_in_file(filename, rq.dequeue(), MATRIX_W, MATRIX_H)) {
					cout << "error options : can`t record matrix" << endl;
					error = true;
					break;
				}
#elif SAVEFORMAT == 1
				case1_Vector.push_back(rq.dequeue());
#endif	
				resultInd++;
			}
		}
	}
	caseTimer.offTimer(0);
	//////////////////////////////////////////////////////////////////////////
	
	cout << "--------------------------------------------------------" << endl;
	cout << "CASE 1 END" << endl;
	cout << "--------------------------------------------------------" << endl;

	//////////////////////////////////////////////////////////////////////////
	/*NORMAL CASE*/
	caseTimer.onTimer(1);
#if NORMALCASE == true
	for (int i = 0; i < LAST_FILE_NUMBER; i++) {
		string openfname = "matrixs\\matrix " + to_string(i) + ".txt";
		string outfname = "case2-output\\result " + to_string(i) + ".txt";
		int *inputMatrix = new int[MATRIX_W*MATRIX_H];
		int *outputMatrix = new int[MATRIX_W*MATRIX_H];
		DS_timer tm(5);
		tm.setTimerName(0, "File Read");
		tm.setTimerName(1, "HtoD");
		tm.setTimerName(2, "Kernel");
		tm.setTimerName(3, "DtoH");
		tm.setTimerName(4, "File Save");
		tm.onTimer(0);
		if (!read_matrix_in_file(openfname, inputMatrix, MATRIX_W, MATRIX_H)) {
			cout << "error options" << endl;
			break;
		}
		tm.offTimer(0);
		tm.onTimer(1);
		cudaMemcpy(inputData[0], inputMatrix, sizeof(int)*MATRIX_W*MATRIX_H, cudaMemcpyHostToDevice);
		tm.offTimer(1);
		tm.onTimer(2);
#if CASEMODE == 0
		addSample << <1, 512 >> > (inputData[0], gpuMap, outputData[0], MATRIX_W, MATRIX_H);
#elif CASEMODE == 1
		matrixMultiSample<<<gDim, bDim>>> (inputData[0], gpuMap, outputData[0], MATRIX_W, MATRIX_H);
#endif
		cudaDeviceSynchronize();
		tm.offTimer(2);
		tm.onTimer(3);
		cudaMemcpy(outputMatrix, outputData[0], sizeof(int)*MATRIX_W*MATRIX_H, cudaMemcpyDeviceToHost);
		tm.offTimer(3);
		tm.onTimer(4);
#if SAVEFORMAT == 0
		if (!record_matrix_in_file(outfname, outputMatrix, MATRIX_W, MATRIX_H)) {
			cout << "error options : can`t record matrix" << endl;
			error = true;
			break;
		}
#elif SAVEFORMAT == 1
		case2_Vector.push_back(outputMatrix);
#endif	
		tm.offTimer(4);
		tm.printTimer();
		
	}
#endif
	caseTimer.offTimer(1);
	//////////////////////////////////////////////////////////////////////////

	cout << "--------------------------------------------------------" << endl;
	cout << "CASE 2 END" << endl;
	cout << "--------------------------------------------------------" << endl;

	caseTimer.printTimer();

	ofstream timerlog;
	timerlog.open("Timer Log.txt", ios::app);

#if CASEMODE == 0
	timerlog << "CASE : ADD SAMPLE" << endl;
#elif CASEMODE == 1
	timerlog << "CASE : MATRIX MULTIPLICATION SAMPLE" << endl;
#endif

#if SAVEFORMAT == 0
	timerlog << "MODE : OUTPUT FILE" << endl;
#elif SAVEFORMAT == 1
	timerlog << "MODE : OUTPUT VECTOR" << endl;
#endif	
	timerlog << "MATRIX SIZE : " << MATRIX_W << "X" << MATRIX_H << " - LOOP COUNT : " << LAST_FILE_NUMBER << endl;
	timerlog.close();

	caseTimer.printToFile("Timer Log.txt");

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

__global__ void matrixMultiSample(int *inputData, int *gpuMap, int *outputData, int w, int h) {
	int row = threadIdx.y + blockDim.y*blockIdx.y;
	int col = threadIdx.x + blockDim.x*blockIdx.x;
	int ind = row * w + col;

	outputData[ind] = 0;

	if (row >= h || col >= w) {
		return;
	}

	for (int k = 0; k < h; k++) {
		outputData[ind] += inputData[row*h+k] + gpuMap[k*w+col];
	}

	return;
}