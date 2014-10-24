#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"




#include "daisy/daisy.h"
// #include "kutility/kutility.h"
#include "ReleaseAssert.h"


#include <vector>
#include <stack>
#include <set>
#include <string>


static float L2Dist(float *a, float *b, int size)
{
	float dist = 0.f;
	for (int i = 0; i < size; i++) {
		dist += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return dist;
}

std::vector<int> MatchDescriptorsAlongScanline(
	std::vector<cv::Point2f> &subpixelKeypointsL, std::vector<cv::Point2f> &subpixelKeypointsR,
	float *descriptorsL, float *descriptorsR, int descriptorSize, int maxDisp, int sign)
{
	int numKeypointsL = subpixelKeypointsL.size();
	int numKeypointsR = subpixelKeypointsR.size();
	std::vector<int> matchIdxLtoR(numKeypointsL);


	const int maxMargin = 3;
	int idxStart = 0, idxEnd = 0;

	for (int i = 0; i < subpixelKeypointsL.size(); i++) {

		cv::Point2f p = subpixelKeypointsL[i];
		while (idxStart < numKeypointsR && subpixelKeypointsR[idxStart].y < p.y - maxMargin) {
			idxStart++;
		}
		while (idxEnd < numKeypointsR && subpixelKeypointsR[idxEnd].y <= p.y + maxMargin) {
			idxEnd++;
		}

		int bestIdx = idxStart;
		int secondBestIdx = idxStart;
		float bestDist = FLT_MAX;
		float secondBestDist = FLT_MAX;

		for (int j = idxStart; j < idxEnd; j++) {
			cv::Point2f q = subpixelKeypointsR[i];
			if (sign == -1 && (q.x < p.x - maxDisp || q.x > p.x)) {
				continue;
			}
			if (sign == +1 && (q.x > p.x + maxDisp || q.x < p.x)) {
				continue;
			}
			float dist = L2Dist(descriptorsL + i * descriptorSize, descriptorsR + j * descriptorSize, descriptorSize);
			if (dist < bestDist) {
				secondBestDist = bestDist;
				bestDist = dist;
				secondBestIdx = bestIdx;
				bestIdx = j;
			}
		}

		if (bestDist == FLT_MAX || secondBestDist == FLT_MAX || bestDist / secondBestDist >= 0.7) {
			matchIdxLtoR[i] = -1;
		}
		else {
			matchIdxLtoR[i] = bestIdx;
		}
	}

	return matchIdxLtoR;
}

template<typename T>
static void RandomPermute(std::vector<T> &pointList, int N)
{
	for (int i = 0; i < N; i++) {
		int j = rand() % pointList.size();
		std::swap(pointList[i], pointList[j]);
	}
}

static cv::Vec2f RansacFitVerticalAlignmentAdjustmentModel(std::vector<cv::Point2f> &subpixelKeypointsL,
	std::vector<cv::Point2f> &subpixelKeypointsR, std::vector<int> &matchIdxLtoR, int sign, int numRows, int numCols)
{

	std::vector<cv::Vec2f> pointList;
	pointList.reserve(subpixelKeypointsL.size());
	for (int i = 0; i < subpixelKeypointsL.size(); i++) {
		if (matchIdxLtoR[i] != -1) {
			// yRef + sign*dy = yTarget
			float yRef = subpixelKeypointsL[i].y;
			float yTarget = subpixelKeypointsR[matchIdxLtoR[i]].y;
			float dy = (yTarget - yRef) / sign;
			pointList.push_back(cv::Vec2f(yRef - numRows / 2, dy));

		}
	}

	printf("numPoints for RANSAC: %d\n", pointList.size());
	int numPoints = pointList.size();
	int minNumSamples = 4;
	cv::Mat A(minNumSamples, 2, CV_32FC1);
	cv::Mat B(minNumSamples, 1, CV_32FC1);
	const float errThreshold = 0.1;
	int bestNumInliers = 0;
	cv::Vec2f bestModel = cv::Vec2f(0, 0);

	for (int retry = 0; retry < 200; retry++) {
		RandomPermute<cv::Vec2f>(pointList, minNumSamples);
		for (int i = 0; i < minNumSamples; i++) {
			A.at<float>(i, 0) = pointList[i][0];
			A.at<float>(i, 1) = 1;
			B.at<float>(i, 0) = pointList[i][1];
		}
		cv::Mat x;
		cv::solve(A, B, x, cv::DECOMP_SVD);

		float a = x.at<float>(0, 0);
		float b = x.at<float>(1, 0);
		int numInliers = 0;
		for (int i = 0; i < numPoints; i++) {
			float err = std::abs(a * pointList[i][0] + b - pointList[i][1]);
			if (err <= errThreshold) {
				numInliers++;
			}
		}
		if (numInliers > bestNumInliers) {
			bestNumInliers = numInliers;
			bestModel = cv::Vec2f(a, b);
		}
	}

	printf("RANSAC re-estimating using all inliers...\n");
	for (int retry = 0; retry < 5; retry++) {
		printf("retry = %d\n", retry);
		printf("bestNumInliers = %d\n", bestNumInliers);
		float a = bestModel[0];
		float b = bestModel[1];
		A.create(bestNumInliers, 2, CV_32FC1);
		B.create(bestNumInliers, 1, CV_32FC1);

		int cnt = 0;
		for (int i = 0; i < numPoints; i++) {
			float err = std::abs(a * pointList[i][0] + b - pointList[i][1]);
			if (err <= errThreshold) {
				A.at<float>(cnt, 0) = pointList[i][0];
				A.at<float>(cnt, 1) = 1;
				B.at<float>(cnt, 0) = pointList[i][1];
				cnt++;
			}
		}

		cv::Mat x;
		cv::solve(A, B, x, cv::DECOMP_SVD);
		a = x.at<float>(0, 0);
		b = x.at<float>(1, 0);
		int bestNumInliers = 0;
		bestModel = cv::Vec2f(a, b);
		for (int i = 0; i < numPoints; i++) {
			float err = std::abs(a * pointList[i][0] + b - pointList[i][1]);
			if (err < errThreshold) {
				bestNumInliers++;
			}
		}
	}

	printf("bestNumInliers = %d\n", bestNumInliers);
	printf("bestModel: (%f, %f)\n", bestModel[0], bestModel[1]);
	return bestModel;
}

void Rectification(cv::Mat &imL, cv::Mat &imR)
{
	const int	maxCorners = 20000;
	const float qualityLevel = 0.000003f;
	const float minDistance = 2.f;
	const int	blockSize = 9;
	const float k = 0.04;
	cv::GoodFeaturesToTrackDetector detector(maxCorners, qualityLevel, minDistance, blockSize, true, k);

	cv::Mat imgGrayL, imgGrayR;
	cv::cvtColor(imL, imgGrayL, CV_BGR2GRAY);
	cv::cvtColor(imR, imgGrayR, CV_BGR2GRAY);

	std::vector<cv::KeyPoint> keypointsL, keypointsR;
	detector.detect(imL, keypointsL);
	detector.detect(imR, keypointsR);

	//cv::Mat keypointsImg;
	//cv::drawKeypoints(imL, keypointsL, keypointsImg, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//cv::imshow("keypionts", keypointsImg);
	//cv::waitKey(0);
	//return;

	std::vector<cv::Point2f> subpixelKeypointsL(keypointsL.size());
	for (int i = 0; i < keypointsL.size(); i++) {
		subpixelKeypointsL[i] = keypointsL[i].pt;
	}
	std::vector<cv::Point2f> subpixelKeypointsR(keypointsR.size());
	for (int i = 0; i < keypointsR.size(); i++) {
		subpixelKeypointsR[i] = keypointsR[i].pt;
	}


	cv::Size winSize = cv::Size(2, 2);
	cv::Size zeroZone = cv::Size(-1, -1);
	cv::TermCriteria criteria = cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);
	cv::cornerSubPix(imgGrayL, subpixelKeypointsL, winSize, zeroZone, criteria);
	cv::cornerSubPix(imgGrayR, subpixelKeypointsR, winSize, zeroZone, criteria);

	// Sort the keypoints by y first then x
	static struct SortCvPoint2f {
		bool operator ()(const cv::Point2f &a, const cv::Point2f &b) const {
			if (a.y == b.y) {
				return a.x < b.x;
			}
			return a.y < b.y;
		}
	};
	std::sort(subpixelKeypointsL.begin(), subpixelKeypointsL.end(), SortCvPoint2f());
	std::sort(subpixelKeypointsR.begin(), subpixelKeypointsR.end(), SortCvPoint2f());



	ASSERT(imgGrayL.isContinuous());
	ASSERT(imgGrayR.isContinuous());
	int numRows = imL.rows, numCols = imL.cols;
	daisy *daisyL = new daisy();
	daisy *daisyR = new daisy();
	daisyL->verbose(0);
	daisyR->verbose(0);
	daisyL->set_image<unsigned char>(imgGrayL.data, numRows, numCols);
	daisyR->set_image<unsigned char>(imgGrayR.data, numRows, numCols);
	daisyL->set_parameters(15, 3, 8, 8);
	daisyR->set_parameters(15, 3, 8, 8);
	daisyL->initialize_single_descriptor_mode();
	daisyR->initialize_single_descriptor_mode();



	printf("extracting descriptors...\n");
	int descriptorSize = daisyL->descriptor_size();
	ASSERT(descriptorSize == 200);
	float *descriptorsL = new float[keypointsL.size() * descriptorSize];
	float *descriptorsR = new float[keypointsR.size() * descriptorSize];
	for (int i = 0; i < keypointsL.size(); i++) {
		daisyL->get_descriptor(subpixelKeypointsL[i].y,
			subpixelKeypointsL[i].x, 0, descriptorsL + i * descriptorSize);
	}
	for (int i = 0; i < keypointsR.size(); i++) {
		daisyR->get_descriptor(subpixelKeypointsR[i].y,
			subpixelKeypointsR[i].x, 0, descriptorsR + i * descriptorSize);
	}


	int maxDisp = 256;
	ASSERT(maxDisp);
	std::vector<int> matchIdxLtoR = MatchDescriptorsAlongScanline(
		subpixelKeypointsL, subpixelKeypointsR, descriptorsL, descriptorsR, descriptorSize, maxDisp, -1);
	std::vector<int> matchIdxRtoL = MatchDescriptorsAlongScanline(
		subpixelKeypointsR, subpixelKeypointsL, descriptorsR, descriptorsL, descriptorSize, maxDisp, +1);

	delete[] descriptorsL, descriptorsR;

	for (int i = 0; i < keypointsL.size(); i++) {
		if (matchIdxLtoR[i] != -1) {
			if (matchIdxRtoL[matchIdxLtoR[i]] != i) {
				matchIdxLtoR[i] = -1;
			}
		}
	}
	for (int i = 0; i < keypointsR.size(); i++) {
		if (matchIdxRtoL[i] != -1) {
			if (matchIdxLtoR[matchIdxRtoL[i]] != i) {
				matchIdxRtoL[i] = -1;
			}
		}
	}

	printf("REACH HERE!!!!!!!\n");

	// FIXEM: add code to visualize the matching results.
	int numFalseMatches = 0;
	int numMatches = 0;
	extern std::string midd3Resolution;
	extern std::string midd3TestCaseId;
	//std::string filePathGT = "D:\\data\\MiddEval3\\" + midd3Resolution + "\\" + midd3TestCaseId + "\\disp0GT.pfm";
	//cv::Mat ReadFilePFM(std::string filePath);
	//cv::Mat dispGT = ReadFilePFM(filePathGT);
	cv::Mat canvas;
	cv::hconcat(imL, imR, canvas);
	for (int i = 0; i < subpixelKeypointsL.size(); i++) {
		if (matchIdxLtoR[i] != -1) {
			cv::Point2f p = subpixelKeypointsL[i];
			cv::Point2f q = subpixelKeypointsR[matchIdxLtoR[i]];
			q.x += numCols;
			cv::Scalar color(rand() % 256, rand() % 256, rand() % 256, 255);
			cv::circle(canvas, p, 3, color, 4, CV_AA);
			cv::circle(canvas, q, 3, color, 4, CV_AA);
			cv::line(canvas, p, q, color, 2, CV_AA);

			float estDisp = p.x - (q.x - numCols);
			//float gtDisps = dispGT.at<float>(p.y, p.x);
			//if (std::abs(estDisp - gtDisps) > 1.0) {
			//	numFalseMatches++;
			//	//matchIdxLtoR[i] = -1;
			//}
			numMatches++;
		}
	}
	printf("numMatches = %d\n", numMatches);
	printf("numFalseMatches = %d\n", numFalseMatches);
	cv::resize(canvas, canvas, cv::Size(canvas.cols / 2, canvas.rows / 2));
	cv::imshow("matches", canvas);
	cv::waitKey(0);


	// Ransac plane fit the rectification model
	cv::Vec2f modelL = RansacFitVerticalAlignmentAdjustmentModel(
		subpixelKeypointsL, subpixelKeypointsR, matchIdxLtoR, -1, numRows, numCols);
	cv::Vec2f modelR = RansacFitVerticalAlignmentAdjustmentModel(
		subpixelKeypointsR, subpixelKeypointsL, matchIdxRtoL, +1, numRows, numCols);


	for (int y = 0; y < 20; y++) {
		float rtfy = (y - numRows / 2) * modelL[0] + modelL[1];
		printf("%4d -> %6.2f\n", y, rtfy);
	}

	std::string filePathRectifyTxt = "D:\\data\\MiddEval3\\fastPreview\\" + midd3Resolution + "\\" + midd3TestCaseId + "_rectify_errThresh=0.1.txt";
	//std::string filePathRectifyTxt = "D:/data/MiddEval3/trainingH/Playtable/rectify.txt";
	FILE *fid = fopen(filePathRectifyTxt.c_str(), "w");
	fprintf(fid, "model = (%f, %f)\n\n", modelL[0], modelL[1]);
	for (int y = 0; y < numRows; y++) {
		float rtfy = (y - numRows / 2) * modelL[0] + modelL[1];
		fprintf(fid, "%4d -> %6.2f\n", y, rtfy);
	}
	fclose(fid);

	
	////cv::Mat canvas;
	//cv::hconcat(imL, imR, canvas);
	for (int i = 0; i < subpixelKeypointsL.size(); i++) {
		float a = modelL[0];
		float b = modelL[1];
		if (matchIdxLtoR[i] != -1) {
			cv::Point2f p = subpixelKeypointsL[i];
			cv::Point2f q = subpixelKeypointsR[matchIdxLtoR[i]];
			float dy = p.y - q.y;
			float err = std::abs(a * p.y + b - dy);
			printf("err = %f\n", err);
			if (err <= 0.1) {
				
				cv::Mat canvas;
				cv::hconcat(imL, imR, canvas);
				q.x += numCols;
				cv::Scalar color(rand() % 256, rand() % 256, rand() % 256, 255);
				cv::line(canvas, p, q, color, 4, CV_AA);
				
				cv::resize(canvas, canvas, cv::Size(canvas.cols / 2, canvas.rows / 2));
				cv::imshow("matches", canvas);
				cv::waitKey(0);
			}
		}
		
	}

	//cv::resize(canvas, canvas, cv::Size(canvas.cols / 2, canvas.rows / 2));
	//cv::imshow("matches", canvas);
	//cv::waitKey(0);
	
}