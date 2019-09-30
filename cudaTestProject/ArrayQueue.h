#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<vector>
using namespace std;

class ArrayQueue {
private:
	int width = 0;
	int height = 0;
	int capacity = 0;
public:
	vector<int *> data;
	int tail = 0;
	int head = 0;
	int chk = 0;

	ArrayQueue() {
	}
	void resize(const int &capacity);
	void set_array_size(const int &width, const int &height);
	void set_array_size_pinned(const int &width, const int &height);
	void cudaFreeAllMembers();
	void enqueue(int *_array);
	int* dequeue();
	bool isFull();
	bool isEmpty();
	int* getTailData();
	void tailAdder();
	int currentQueueSize();
};

void ArrayQueue::resize(const int &capacity) {
	this->capacity = capacity;
	data.resize(capacity);
}

void ArrayQueue::set_array_size(const int &width, const int &height) {
	this->width = width;
	this->height = height;
	for (int i = 0; i < data.size(); i++) {
		data[i] = new int[width*height];
	}
}
void ArrayQueue::set_array_size_pinned(const int &width, const int &height) {
	this->width = width;
	this->height = height;
	for (int i = 0; i < data.size(); i++) {
		cudaMallocHost(&data[i], sizeof(int)*width*height);
	}
}
void ArrayQueue::cudaFreeAllMembers() {
	for (int i = 0; i < data.size(); i++) {
		cudaFreeHost(data[i]);
	}
}

int ArrayQueue::currentQueueSize() {
	return abs(tail - head);
}
int* ArrayQueue::getTailData() {
	if (isEmpty()) return nullptr;
	return data[tail];
}
void ArrayQueue::tailAdder() {
	tail = (tail + 1) % capacity;
}

void ArrayQueue::enqueue(int *_array) {
	if (isFull()) {
		return;
	}
	tail = (tail + 1) % capacity;
	data[tail] = _array;
}

int* ArrayQueue::dequeue() {
	if (isEmpty()) {
		return nullptr;
	}
	head = (head + 1) % capacity;
	return data[head];
}

bool ArrayQueue::isFull() {
	return (tail+1)%capacity == head;
}
bool ArrayQueue::isEmpty() {
	return head == tail;
}