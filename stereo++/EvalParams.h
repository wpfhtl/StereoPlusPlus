#ifndef __EVALPARAMS_H__
#define __EVALPARAMS_H__

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <stack>
#include <set>
#include <string>

#include "MCImg.h"
#include "SlantedPlane.h"

struct ARAPEvalParams {
	int numRows, numCols;
	cv::Mat *dispL;
	cv::Mat *labelMap;
	cv::Mat *confidenceImg;
	cv::Mat *GT;
	cv::Mat *canvas;
	std::vector<cv::Point2f> *baryCenters;
	std::vector<cv::Vec3f> *n;
	std::vector<float> *u;
	std::vector<std::vector<int>> *nbGraph;
	std::vector<std::vector<cv::Point2i>> *segPixelLists;
	MCImg<float> *dsiL;
	int numDisps;
	MCImg<SlantedPlane> *pixelwiseSlantedPlanesL;
	MCImg<float> *pixelwiseBestCostsL;
	ARAPEvalParams() : numRows(0), numCols(0), labelMap(0), baryCenters(0), dispL(0), canvas(0),
		n(0), u(0), nbGraph(0), segPixelLists(0), confidenceImg(0), GT(0), dsiL(0), numDisps(0),
		pixelwiseSlantedPlanesL(0), pixelwiseBestCostsL(0) {}
};

#endif