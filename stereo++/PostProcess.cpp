#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <set>

#include "StereoAPI.h"
#include "Timer.h"
#include "SlantedPlane.h"
#include "ReleaseAssert.h"



extern int					PATCHRADIUS;
extern int					PATCHWIDTH;
extern float				GRANULARITY;
extern float				SIMILARITY_GAMMA;
extern int					MAX_PATCHMATCH_ITERS;






cv::Mat CrossCheck(cv::Mat &dispL, cv::Mat &dispR, int sign, float thresh)
{
	int numRows = dispL.rows, numCols = dispL.cols;
	cv::Mat validPixelMapL(numRows, numCols, CV_8UC1);

	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			int xMatch = 0.5 + x + sign * dispL.at<float>(y, x);
			if (0 <= xMatch && xMatch < numCols
				&& std::abs(dispL.at<float>(y, x) - dispR.at<float>(y, xMatch)) <= thresh) {
				validPixelMapL.at<unsigned char>(y, x) = 255;
			}
			else {
				validPixelMapL.at<unsigned char>(y, x) = 0;
			}
		}
	}

	return validPixelMapL;
}

void DisparityHoleFilling(cv::Mat &disp, MCImg<SlantedPlane> &slantedPlanes, cv::Mat &validPixelMap)
{
	// This function fills the invalid pixel (y,x) by finding its nearst (left and right) 
	// valid neighbors on the same scanline, and select the one with lower disparity.
	int numRows = disp.rows, numCols = disp.cols;
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			if (!validPixelMap.at<unsigned char>(y, x)) {
				int xL = x - 1, xR = x + 1;
				float dL = FLT_MAX, dR = FLT_MAX;
				while (!validPixelMap.at<unsigned char>(y, xL) && 0 <= xL)		xL--;
				while (!validPixelMap.at<unsigned char>(y, xR) && xR < numCols)	xR++;
				if (xL >= 0)		dL = slantedPlanes[y][xL].ToDisparity(y, x);
				if (xR < numCols)	dR = slantedPlanes[y][xR].ToDisparity(y, x);
				//disp.at<float>(y, x) = std::min(dL, dR);
				if (dL < dR) {
					disp.at<float>(y, x) = dL;
					slantedPlanes[y][x] = slantedPlanes[y][xL];
				}
				else {
					disp.at<float>(y, x) = dR;
					slantedPlanes[y][x] = slantedPlanes[y][xR];
				}
			}
		}
	}
}

float SelectWeightedMedianFromPatch(cv::Mat &disp, int yc, int xc, float *w)
{
	int numRows = disp.rows, numCols = disp.cols;
	std::vector<std::pair<float, float>> depthWeightPairs;
	depthWeightPairs.reserve(PATCHWIDTH * PATCHWIDTH);

	for (int y = yc - PATCHRADIUS, id = 0; y <= yc + PATCHRADIUS; y++) {
		for (int x = xc - PATCHRADIUS; x <= xc + PATCHRADIUS; x++, id++) {
			if (InBound(y, x, numRows, numCols)) {
				depthWeightPairs.push_back(std::make_pair(disp.at<float>(y, x), w[id]));
			}
		}
	}

	std::sort(depthWeightPairs.begin(), depthWeightPairs.end());

	float wAcc = 0.f, wSum = 0.f;
	for (int i = 0; i < depthWeightPairs.size(); i++) {
		wSum += depthWeightPairs[i].second;
	}
	for (int i = 0; i < depthWeightPairs.size(); i++) {
		wAcc += depthWeightPairs[i].second;
		if (wAcc >= wSum / 2.f) {
			// Note that this line can always be reached
			if (i > 0) {
				return (depthWeightPairs[i - 1].first + depthWeightPairs[i].first) / 2.f;
			}
			else {
				return depthWeightPairs[i].first;
			}
			break;
		}
	}
	printf("BUGGG!!!\n");
	return disp.at<float>(yc, xc);
}

void WeightedMedianFilterInvalidPixels(cv::Mat &disp, cv::Mat &validPixelMap, cv::Mat &img)
{
	cv::Mat dispOut = disp.clone();
	int numRows = disp.rows, numCols = disp.cols;
	float *w = new float[PATCHWIDTH * PATCHWIDTH];

	// DO NOT USE PARALLEL FOR HERE !!
	// IT WILL CAUSE WRITING CONFLICT IN w !!!
	for (int yc = 0; yc < numRows; yc++) {
		for (int xc = 0; xc < numCols; xc++) {
			if (!validPixelMap.at<unsigned char>(yc, xc)) {

				memset(w, 0, PATCHWIDTH * PATCHWIDTH * sizeof(float));
				cv::Vec3b cc = img.at<cv::Vec3b>(yc, xc);
				for (int id = 0, y = yc - PATCHRADIUS; y <= yc + PATCHRADIUS; y++) {
					for (int x = xc - PATCHRADIUS; x <= xc + PATCHRADIUS; x++, id++) {
						if (InBound(y, x, numRows, numCols)) {
							w[id] = exp(-(float)L1Dist(cc, img.at<cv::Vec3b>(y, x)) / SIMILARITY_GAMMA);
						}
					}
				}

				disp.at<float>(yc, xc) = SelectWeightedMedianFromPatch(disp, yc, xc, w);
			}
		}
	}

	delete[] w;
}

std::vector<float> DetermineConfidence(cv::Mat &validPixelMap, std::vector<std::vector<cv::Point2i>> &segPixelLists)
{
	int numSegs = segPixelLists.size();
	std::vector<float> confidence(numSegs);
	for (int id = 0; id < numSegs; id++) {
		std::vector<cv::Point2i> &pixelList = segPixelLists[id];
		int numValidPixels = 0;
		for (int i = 0; i < pixelList.size(); i++) {
			cv::Point2i &p = pixelList[i];
			if (validPixelMap.at<bool>(p.y, p.x)) {
				numValidPixels++;
			}
		}
		confidence[id] = (float)numValidPixels / (float)pixelList.size();
	}
	return confidence;
}

cv::Mat DrawSegmentConfidenceMap(int numRows, int numCols, std::vector<float> &confidence,
	std::vector<std::vector<cv::Point2i>> &segPixelLists)
{
	ASSERT(confidence.size() == segPixelLists.size());
	cv::Mat confidenceMap(numRows, numCols, CV_32FC1);
	confidenceMap.setTo(0.f);
	int numSegs = confidence.size();
	for (int id = 0; id < numSegs; id++) {
		std::vector<cv::Point2i> &pixelList = segPixelLists[id];
		for (int k = 0; k < pixelList.size(); k++) {
			cv::Point2i &p = pixelList[k];
			ASSERT(InBound(p.y, p.x, numRows, numCols))
				confidenceMap.at<float>(p.y, p.x) = confidence[id];
		}
	}
	cv::cvtColor(confidenceMap, confidenceMap, CV_GRAY2BGR);
	confidenceMap.convertTo(confidenceMap, CV_8UC3, 255);
	return confidenceMap;
}

void SegmentOcclusionFilling(std::vector<SlantedPlane> &slantedPlanes, std::vector<cv::Point2f> &baryCenters,
	std::vector<std::vector<int>> &nbIndices, std::vector<float> &confidence)
{
	const float LOW_CONF_THRESH = 0.5f;
	const float HIGH_CONF_THRESH = 0.8f;
	int numSegs = slantedPlanes.size();

	for (int id = 0; id < numSegs; id++) {
		if (confidence[id] < LOW_CONF_THRESH) {

			float lowestDisp = FLT_MAX;
			int bestNbId = -1;
			float yc = baryCenters[id].y;
			float xc = baryCenters[id].x;

			for (int k = 0; k < nbIndices[id].size(); k++) {
				int nbId = nbIndices[id][k];
				if (confidence[nbId] > HIGH_CONF_THRESH) {
					float d = slantedPlanes[nbId].ToDisparity(yc, xc);
					if (d < lowestDisp) {
						lowestDisp = d;
						bestNbId = nbId;
					}
				}
			}
			if (bestNbId != -1) {
				slantedPlanes[id] = slantedPlanes[bestNbId];
			}
		}
	}
}

void SegmentOcclusionFilling(int numRows, int numCols, std::vector<SlantedPlane> &slantedPlanes,
	std::vector<cv::Point2f> &baryCenters, std::vector<std::vector<int>> &nbIndices,
	std::vector<float> &confidence, std::vector<std::vector<cv::Point2i>> &segPixelLists)
{
	const float LOW_CONF_THRESH = 0.4f;
	const float HIGH_CONF_THRESH = 0.85f;
	int numSegs = slantedPlanes.size();

	cv::Mat ownerMap(numRows, numCols, CV_32SC1);
	ownerMap.setTo(-1);
	for (int id = 0; id < numSegs; id++) {
		std::vector<cv::Point2i> &pixelList = segPixelLists[id];
		for (int k = 0; k < pixelList.size(); k++) {
			cv::Point2i &p = pixelList[k];
			ownerMap.at<int>(p.y, p.x) = id;
		}
	}

	for (int id = 0; id < numSegs; id++) {
		if (confidence[id] < LOW_CONF_THRESH) {

			int yc = baryCenters[id].y + 0.5;
			int xc = baryCenters[id].x + 0.5;
			std::set<int> nbIdSet;

			const int RADIUS = 17;
			for (int y = yc - RADIUS; y <= yc + RADIUS; y++) {
				for (int x = xc - RADIUS; x <= xc + RADIUS; x++) {
					if (InBound(y, x, numRows, numCols)) {
						nbIdSet.insert(ownerMap.at<int>(y, x));
					}
				}
			}

			float lowestDisp = FLT_MAX;
			int bestNbId = -1;

			for (std::set<int>::iterator it = nbIdSet.begin(); it != nbIdSet.end(); it++) {
				int nbId = *it;
				if (nbId != -1 && confidence[nbId] > HIGH_CONF_THRESH) {
					float d = slantedPlanes[nbId].ToDisparity(baryCenters[id].y, baryCenters[id].x);
					if (d < lowestDisp) {
						lowestDisp = d;
						bestNbId = nbId;
					}
				}
			}
			if (bestNbId != -1) {
				slantedPlanes[id] = slantedPlanes[bestNbId];
			}
		}
	}
}

