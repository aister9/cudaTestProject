#pragma once
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <omp.h>
#include <ctime>
#include <cstdlib>
using namespace std;

bool random_matrix_generator(string filename, int width, int height);
bool read_matrix_in_file(string filename, int *arrayData, int w, int h);


bool random_matrix_generator(string filename, int width, int height) {

	ofstream newfile;
	newfile.open(filename);

	if (!newfile.is_open()) {
		cout << "file open error!!" << endl;
		newfile.close();
		return false;
	}


	newfile << height << " " << width << endl;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int dice_roll = rand() % 10;

			newfile << dice_roll;
			if (x != width - 1) {
				newfile << " ";
			}
		}
		newfile << endl;
	}

	newfile.close();

	return true;
}


bool read_matrix_in_file(string filename, int *arrayData, int w, int h) {
	ifstream ofile;
	ofile.open(filename);
	if (!ofile.is_open()) {
		cout << "file open error!!" << endl;
		ofile.close();
		return false;
	}
	int width = 0, height = 0;
	int data;
	ofile >> height >> width;

	if (height != h) { cout << "matrix size wrong!" << endl; return false; }
	if (width != w) { cout << "matrix size wrong!" << endl; return false; }
	
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			ofile >> data;

			arrayData[y*width + x] = data;
		}
	}

	return true;
}