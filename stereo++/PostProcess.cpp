#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "StereoAPI.h"
#include "Timer.h"
#include "SlantedPlane.h"




extern int					PATCHRADIUS;
extern int					PATCHWIDTH;
extern float				GRANULARITY;
extern float				SIMILARITY_GAMMA;
extern int					MAX_PATCHMATCH_ITERS;




cv::Mat SlantedPlaneMapToDisparityMap(MCImg<SlantedPlane> &slantedPlanes)
{
	int numRows = slantedPlanes.h, numCols = slantedPlanes.w;
	cv::Mat dispMap(numRows, numCols, CV_32FC1);

	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			dispMap.at<float>(y, x) = slantedPlanes[y][x].ToDisparity(y, x);
		}
	}

	return dispMap;
}

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

static void DisparityHoleFilling(cv::Mat &disp, MCImg<SlantedPlane> &slantedPlanes, cv::Mat &validPixelMap)
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

static float SelectWeightedMedianFromPatch(cv::Mat &disp, int yc, int xc, float *w)
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

static void WeightedMedianFilterInvalidPixels(cv::Mat &disp, cv::Mat &validPixelMap, cv::Mat &img)
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

void PatchMatchOnPixelPostProcess(MCImg<SlantedPlane> &slantedPlanesL, MCImg<SlantedPlane> &slantedPlanesR,
	cv::Mat &imL, cv::Mat &imR, cv::Mat &dispL, cv::Mat &dispR)
{
	dispL = SlantedPlaneMapToDisparityMap(slantedPlanesL);
	dispR = SlantedPlaneMapToDisparityMap(slantedPlanesR);

	cv::Mat validPixelMapL = CrossCheck(dispL, dispR, -1);
	cv::Mat validPixelMapR = CrossCheck(dispR, dispL, +1);

	DisparityHoleFilling(dispL, slantedPlanesL, validPixelMapL);
	DisparityHoleFilling(dispR, slantedPlanesR, validPixelMapR);

	validPixelMapL = CrossCheck(dispL, dispR, -1);
	validPixelMapR = CrossCheck(dispR, dispL, +1);

	bs::Timer::Tic("WMF");
	WeightedMedianFilterInvalidPixels(dispL, validPixelMapL, imL);
	WeightedMedianFilterInvalidPixels(dispR, validPixelMapR, imR);
	bs::Timer::Toc();
}
