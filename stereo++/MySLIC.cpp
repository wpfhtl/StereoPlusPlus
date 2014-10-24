#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <stack>
#include <set>
#include <string>
#include <iostream>

#include "StereoAPI.h"
#include "ReleaseAssert.h"

void VisualizeSegmentation(cv::Mat &img, cv::Mat &labelMap)
{
	srand(12003);
	int numRows = img.rows, numCols = img.cols;
	double minVal, maxVal;
	cv::minMaxLoc(labelMap, &minVal, &maxVal);
	int numLabels = maxVal + 1;
	std::vector<cv::Vec3b> colors(numLabels);
	for (int i = 0; i < numLabels; i++) {
		colors[i][0] = rand() % 256;
		colors[i][1] = rand() % 256;
		colors[i][2] = rand() % 256;
	}

	cv::Mat segImg(numRows, numCols, CV_8UC3);
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			int id = labelMap.at<int>(y, x);
			segImg.at<cv::Vec3b>(y, x) = colors[id];
		}
	}

	cv::Mat canvas;
	cv::hconcat(img, segImg, canvas);
	cv::imshow("segmentation", canvas);
	cv::waitKey(0);
}

static inline float dist(cv::Vec3f &colorX, cv::Vec2f &posX, cv::Vec3f &colorY, cv::Vec2f &posY, float S, float m)
{
	cv::Vec3f colorDiff = colorX - colorY;
	cv::Vec2f posDiff = posX - posY;
	return colorDiff.dot(colorDiff) + posDiff.dot(posDiff) * (m*m) / (S*S);
}

void MySlicSegmentation(cv::Mat &img, int numPreferedRegions, float compactness, cv::Mat &labelMap)
{
	// Step 1 - Initialize labelMap by checkboard
	cv::Mat imgByte = img.clone();
	img.convertTo(img, CV_32FC3, 1.f / 1);
	int numRows = img.rows, numCols = img.cols;
	int segLen = std::sqrt((numRows * numCols * 1.f) / numPreferedRegions) + 0.5f;
	int M = numRows / segLen, N = numCols / segLen;
	M += (numRows % segLen > segLen / 3);
	N += (numCols % segLen > segLen / 3);

	int numClusters = M * N;
	std::vector<int> clusterSizes(numClusters, 0);
	std::vector<cv::Vec3f> muColors(numClusters, cv::Vec3f(0, 0, 0));
	std::vector<cv::Vec2f> muPositions(numClusters, cv::Vec2f(0, 0));

	labelMap.create(numRows, numCols, CV_32SC1);
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			int i = std::min(y / segLen, M - 1);
			int j = std::min(x / segLen, N - 1);
			int id = i * N + j;
			labelMap.at<int>(y, x) = id;
			muColors[id] += img.at<cv::Vec3f>(y, x);
			muPositions[id] += cv::Vec2f(x, y);
			clusterSizes[id]++;
		}
	}

	for (int id = 0; id < numClusters; id++) {
		muColors[id] /= clusterSizes[id];
		muPositions[id] /= clusterSizes[id];
	}

	cv::Mat costMap(numRows, numCols, CV_32FC1);
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			int id = labelMap.at<int>(y, x);
			costMap.at<float>(y, x) = dist(muColors[id], muPositions[id],
				cv::Vec3f(img.at<cv::Vec3f>(y, x)), cv::Vec2f(x, y), segLen, compactness);
		}
	}


	// Step 2 - K-mean clustering	
	const int maxKmeansIters = 50;
	for (int iter = 0; iter < maxKmeansIters; iter++) {
		printf("doing iter = %d\n", iter);

		// update pixel labels
		for (int id = 0; id < numClusters; id++) {
			int xc = muPositions[id][0] + 0.5;
			int yc = muPositions[id][1] + 0.5;
			for (int y = yc - 2 * segLen; y <= yc + 2 * segLen; y++) {
				for (int x = xc - 2 * segLen; x <= xc + 2 * segLen; x++) {
					if (InBound(y, x, numRows, numCols)) {
						float cost = dist(muColors[id], muPositions[id],
							cv::Vec3f(img.at<cv::Vec3f>(y, x)), cv::Vec2f(x, y), segLen, compactness);
						if (cost < costMap.at<float>(y, x)) {
							//printf("sfsdfsdf\n");
							labelMap.at<int>(y, x) = id;
							costMap.at<float>(y, x) = cost;
						}
					}
				}
			}
		}


		// update cluster centers
		std::fill(clusterSizes.begin(), clusterSizes.end(), 0);
		std::fill(muColors.begin(), muColors.end(), cv::Vec3f(0, 0, 0));
		std::fill(muPositions.begin(), muPositions.end(), cv::Vec2f(0, 0));

		for (int y = 0; y < numRows; y++) {
			for (int x = 0; x < numCols; x++) {
				int id = labelMap.at<int>(y, x);
				muColors[id] += img.at<cv::Vec3f>(y, x);
				muPositions[id] += cv::Vec2f(x, y);
				clusterSizes[id]++;
			}
		}

		for (int id = 0; id < numClusters; id++) {
			if (clusterSizes[id] == 0) {
				printf("cluster %d is empty.\n");
				continue;
			}
			muColors[id] /= clusterSizes[id];
			muPositions[id] /= clusterSizes[id];
		}
	}

	VisualizeSegmentation(imgByte, labelMap);
}

void TestMySlicSegmentation()
{
	extern std::string ROOTFOLDER;
	cv::Mat imL = cv::imread("D:/data/stereo/" + ROOTFOLDER + "/im2.png");
	cv::Mat imR = cv::imread("D:/data/stereo/" + ROOTFOLDER + "/im6.png");
	cv::Mat labelMap;
	MySlicSegmentation(imL, 500, 30, labelMap);
}