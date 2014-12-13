#include <vector>
#include <opencv2/core/core.hpp>
#include "ReleaseAssert.h"

void SaveVectorPoint3f(std::string filePath, std::vector<cv::Point3f> &vertices, std::string mode = "w")
{
	if (mode == "w") {
		FILE *f = fopen(filePath.c_str(), "w");
		ASSERT(f != NULL);
		fprintf(f, "%d\n", vertices.size());
		for (int i = 0; i < vertices.size(); i++) {
			cv::Point3f &p = vertices[i];
			fprintf(f, "%f  %f  %f\n", p.x, p.y, p.z);
		}
		fclose(f);
	}
	else if (mode == "wb") {
		FILE *f = fopen(filePath.c_str(), "wb");
		ASSERT(f != NULL);
		int size = vertices.size();
		fwrite(&size, sizeof(int), 1, f);
		if (size > 0) {
			fwrite(&vertices[0], sizeof(cv::Point3f), size, f);
		}
		fclose(f);
	}
	else {
		printf("Incorret mode writing mode.\n.");
		ASSERT(0);
	}
}

std::vector<cv::Point3f> LoadVectorPoint3f(std::string filePath, std::string mode = "r")
{

	std::vector<cv::Point3f> vertices;
	if (mode == "r") {
		FILE *f = fopen(filePath.c_str(), "r");
		ASSERT(f != NULL);
		int size = 0;
		fscanf(f, "%d", &size);
		vertices.resize(size);
		for (int i = 0; i < vertices.size(); i++) {
			cv::Point3f p;
			fscanf(f, "%f%f%f", &p.x, &p.y, &p.z);
			vertices[i] = p;
		}
		fclose(f);
	}
	else if (mode == "rb") {
		FILE *f = fopen(filePath.c_str(), "wb");
		ASSERT(f != NULL);
		int size = 0;
		fread(&size, sizeof(int), 1, f);
		vertices.resize(size);
		if (size > 0) {
			fread(&vertices[0], sizeof(cv::Point3f), size, f);
		}
		fclose(f);
	}
	else {
		printf("Incorret mode reading mode.\n.");
		ASSERT(0);
	}
	return vertices;
}

void SaveVectorPoint2f(std::string filePath, std::vector<cv::Point2f> &vertices, std::string mode = "w")
{
	if (mode == "w") {
		FILE *f = fopen(filePath.c_str(), "w");
		ASSERT(f != NULL);
		fprintf(f, "%d\n", vertices.size());
		for (int i = 0; i < vertices.size(); i++) {
			cv::Point2f &p = vertices[i];
			fprintf(f, "%f %f\n", p.x, p.y);
		}
		fclose(f);
	}
	else if (mode == "wb") {
		FILE *f = fopen(filePath.c_str(), "wb");
		ASSERT(f != NULL);
		int size = vertices.size();
		fwrite(&size, sizeof(int), 1, f);
		if (size > 0) {
			fwrite(&vertices[0], sizeof(cv::Point2f), size, f);
		}
		fclose(f);
	}
	else {
		printf("Incorret mode writing mode.\n.");
		ASSERT(0);
	}
}

std::vector<cv::Point2f> LoadVectorPoint2f(std::string filePath, std::string mode = "r")
{

	std::vector<cv::Point2f> vertices;
	if (mode == "r") {
		FILE *f = fopen(filePath.c_str(), "r");
		ASSERT(f != NULL);
		int size = 0;
		fscanf(f, "%d", &size);
		vertices.resize(size);
		for (int i = 0; i < vertices.size(); i++) {
			cv::Point2f p;
			fscanf(f, "%f%f", &p.x, &p.y);
			vertices[i] = p;
		}
		fclose(f);
	}
	else if (mode == "rb") {
		FILE *f = fopen(filePath.c_str(), "wb");
		ASSERT(f != NULL);
		int size = 0;
		fread(&size, sizeof(int), 1, f);
		vertices.resize(size);
		if (size > 0) {
			fread(&vertices[0], sizeof(cv::Point2f), size, f);
		}
		fclose(f);
	}
	else {
		printf("Incorret mode reading mode.\n.");
		ASSERT(0);
	}
	return vertices;
}

void SaveVectorVectorInt(std::string filePath, std::vector<std::vector<int>> &data, std::string mode = "w")
{
	if (mode == "w") {
		FILE *f = fopen(filePath.c_str(), "w");
		ASSERT(f != NULL);
		fprintf(f, "%d\n", data.size());
		for (int i = 0; i < data.size(); i++) {
			fprintf(f, "%d  ", data[i].size());
			for (int j = 0; j < data[i].size(); j++) {
				fprintf(f, "%d ", data[i][j]);
			}
			fprintf(f, "\n");
		}
		fclose(f);
	}
	/*else if (mode == "wb") {
		FILE *f = fopen(filePath.c_str(), "wb");
		ASSERT(f != NULL);
		int size = vertices.size();
		fwrite(&size, sizeof(int), 1, f);
		if (size > 0) {
			fwrite(&vertices[0], sizeof(cv::Point3f), size, f);
		}
		fclose(f);
	}*/
	else {
		printf("Incorret mode writing mode.\n.");
		ASSERT(0);
	}
}

std::vector<std::vector<int>> LoadVectorVectorInt(std::string filePath, std::string mode = "r")
{
	std::vector<std::vector<int>> data;
	if (mode == "r") {
		FILE *f = fopen(filePath.c_str(), "r");
		ASSERT(f != NULL);
		int sizeI = 0;
		fscanf(f, "%d", &sizeI);
		data.resize(sizeI);
		for (int i = 0; i < sizeI; i++) {
			int sizeJ = 0;
			fscanf(f, "%d", &sizeJ);
			data[i].resize(sizeJ);
			for (int j = 0; j < sizeJ; j++) {
				fscanf(f, "%d", &data[i][j]);
			}
		}
		fclose(f);
	}
	/*else if (mode == "wb") {
	FILE *f = fopen(filePath.c_str(), "wb");
	ASSERT(f != NULL);
	int size = vertices.size();
	fwrite(&size, sizeof(int), 1, f);
	if (size > 0) {
	fwrite(&vertices[0], sizeof(cv::Point3f), size, f);
	}
	fclose(f);
	}*/
	else {
		printf("Incorret mode writing mode.\n.");
		ASSERT(0);
	}
	return data;
}