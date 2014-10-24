#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <stack>
#include <set>
#include <string>
#include <iostream>

#include "StereoAPI.h"
#include "SlantedPlane.h"
#include "PostProcess.h"
#include "ReleaseAssert.h"
#include "Timer.h"

#include "MeanShift/msImageProcessor.h"




static int GetLabelMapByFloodFill(cv::Mat &img, cv::Mat &labelMap)
{
	const cv::Point2i offsetN4[] = {
		cv::Point2i(-1, 0), cv::Point2i(+1, 0),
		cv::Point2i(0, -1), cv::Point2i(0, +1) };
	int numRows = img.rows, numCols = img.cols;
	labelMap = -1 * cv::Mat::ones(numRows, numCols, CV_32SC1);
	int runningLabel = 0;

	//cv::imshow("sdfsdfsd", img);
	//cv::waitKey(0);

	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			if (labelMap.at<int>(y, x) == -1) {
				labelMap.at<int>(y, x) = runningLabel;
				std::stack<cv::Point2i> stack;
				stack.push(cv::Point2i(x, y));
				while (!stack.empty()) {
					cv::Point2i p = stack.top();
					stack.pop();
					for (int r = 0; r < 4; r++) {
						cv::Point2i q = p + offsetN4[r];
						if (InBound(q, numRows, numCols)
							&& labelMap.at<int>(q.y, q.x) == -1) {

							cv::Vec3b &A = img.at<cv::Vec3b>(p.y, p.x);
							cv::Vec3b &B = img.at<cv::Vec3b>(q.y, q.x);
							if (A[0] == B[0] && A[1] == B[1] && A[2] == B[2]) {
								labelMap.at<int>(q.y, q.x) = runningLabel;
								stack.push(q);
							}
						}
					}
				}
				runningLabel++;
			}
		}
	}
	//printf("numLabels = %d\n", runningLabel);
	return runningLabel;
}

int MeanShiftSegmentation(cv::Mat &img, const float colorRadius, const float spatialRadius, const int minRegion, cv::Mat &labelMap)
{
	const int width = img.cols, height = img.rows;
	cv::Mat rgb;
	cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

	msImageProcessor ms;
	ms.DefineImage((unsigned char*)rgb.data, COLOR, height, width);
	if (ms.ErrorStatus)
	{
		printf("meanShift DefineImage returns an error!\n");
	}

	ms.Filter((int)spatialRadius, colorRadius, HIGH_SPEEDUP);
	ms.FuseRegions(spatialRadius, minRegion);
	if (ms.ErrorStatus)
	{
		printf("meanShift FuseRegions returns an error!\n");
	}

	cv::Mat segmImage(height, width, CV_8UC3), result;
	ms.GetResults(segmImage.data);
	if (ms.ErrorStatus)
	{
		printf("meanShift GetResults returns an error!\n");
	}
	cv::cvtColor(segmImage, result, CV_RGB2BGR);
	//cv::imshow("segments", result);
	//cv::waitKey(0);




	// The opencv flood fill seems buggy!, the newVal and decode value is different for some running labels, e.g., 26
	/*const cv::Scalar colorDiff = cv::Scalar::all(0);
	int runningLabel = 0;
	cv::Mat mask(result.rows + 2, result.cols + 2, CV_8UC1, cv::Scalar::all(0));
	for (int y = 0; y < result.rows; y++)
	{
		for (int x = 0; x < result.cols; x++)
		{
			if (mask.at<uchar>(y, x) == 0)
			{

				cv::Scalar newVal(runningLabel / (256 * 256), runningLabel / 256, runningLabel % 256);
				cv::floodFill(result, mask, cv::Point(x, y), newVal, 0, colorDiff, colorDiff);
				
				cv::Vec3b color = result.at<cv::Vec3b>(y, x);
				int label = 256 * 256 * (int)color[0] + 256 * (int)color[1] + (int)color[2];

				std::cout << "runningLabel = " << runningLabel << "\n";
				std::cout << "newVal = " << newVal << "\n";
				std::cout << "decode Value = " << color << "\n\n";

				runningLabel++;
				
			}
		}
	}

	cv::imshow("result", result);
	cv::waitKey(0);

	labelMap.create(img.rows, img.cols, CV_32SC1);
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			cv::Vec3b color = result.at<cv::Vec3b>(y, x);
			int label = 256 * 256 * (int)color[0] + 256 * (int)color[1] + (int)color[2];
			labelMap.at<int>(y, x) = label;
		}
	}*/

	int numLabels = GetLabelMapByFloodFill(result, labelMap);
#if 0
	void VisualizeSegmentation(cv::Mat &img, cv::Mat &labelMap);
	VisualizeSegmentation(img, labelMap);
#endif
	
	//int numLabels = runningLabel;
	std::vector<bool> visited(numLabels, false);
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			int label = labelMap.at<int>(y, x);
			if (label >= numLabels) {
				printf("(x,y,label) = (%d, %d, %d\n", x, y, label);
				printf("BUG11111111111111111111n");
			} 
			visited[label] = true;
		}
	}

	for (int i = 0; i < numLabels; i++) {
		if (!visited[i]) {
			printf("BUG!!!!!!!!!!!!\n");
		}
	}

	return numLabels;
}



static void RandomPermute(std::vector<cv::Point3f> &pointList, int N)
{
	for (int i = 0; i < N; i++) {
		int j = rand() % pointList.size();
		std::swap(pointList[i], pointList[j]);
	}
}

cv::Vec3f RansacPlaneFitting(std::vector<cv::Point3f> &pointList, float inlierThresh)
{
	if (pointList.size() < 5) {
		return cv::Vec3f(0, 0, 15);	// A frontal-parallel plane with constant disparity 15
	}

	int bestNumInliers = 0;
	cv::Vec3f bestAbc = cv::Vec3f(0, 0, 15);

	for (int retry = 0; retry < 200; retry++) {
		RandomPermute(pointList, 5);
		cv::Mat A(5, 3, CV_32FC1);
		cv::Mat B(5, 1, CV_32FC1);
		cv::Mat X;
		for (int i = 0; i < 5; i++) {
			A.at<float>(i, 0) = pointList[i].x;
			A.at<float>(i, 1) = pointList[i].y;
			A.at<float>(i, 2) = 1.f;
			B.at<float>(i, 0) = pointList[i].z;
		}

		cv::solve(A, B, X, cv::DECOMP_QR);

		cv::Vec3f abc(X.at<float>(0, 0), X.at<float>(1, 0), X.at<float>(2, 0));
		int numInliers = 0;
		for (int i = 0; i < pointList.size(); i++) {
			cv::Point3f p = pointList[i];
			float d = abc.dot(cv::Vec3f(p.x, p.y, 1.f));
			if (std::abs(d - p.z) <= inlierThresh) {
				numInliers++;
			}
		}

		if (numInliers > bestNumInliers) {
			bestNumInliers = numInliers;
			bestAbc = abc;
		}
	}

	return bestAbc;
}

cv::Vec3f RansacPlaneFitting(std::vector<cv::Vec3f> &vecList, float inlierThresh)
{
	std::vector<cv::Point3f> pointList(vecList.size());
	for (int i = 0; i < vecList.size(); i++) {
		pointList[i].x = vecList[i][0];
		pointList[i].y = vecList[i][1];
		pointList[i].z = vecList[i][2];
	}
	return RansacPlaneFitting(pointList, inlierThresh);
}


//meanShiftSegmentation(g_L, 3.0f, 3.0f, 16, g_segmentL);
void TestMeanShift()
{
	printf("TestMeanShift\n");
	std::string imgPath = "D:\\data\\MiddEval3\\trainingQ\\Shelves\\im0.png";
	//std::string imgPath = "D:\\data\\MiddEval3\\testQ\\Staircase\\im0.png";
	cv::Mat img = cv::imread(imgPath);
	cv::Mat segImg;
	bs::Timer::Tic("MeanShift");
	MeanShiftSegmentation(img, 3.f, 3.f, 300, segImg);
	bs::Timer::Toc();
}