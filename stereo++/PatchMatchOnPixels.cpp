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

extern enum CostAggregationType { GRID, TOP50 };
extern enum MatchingCostType	{ ADGRADIENT, ADCENSUS };

extern CostAggregationType	gCostAggregationType;
extern MatchingCostType		gMatchingCostType;

extern MCImg<float>			gDsiL;
extern MCImg<float>			gDsiR;
extern MCImg<float>			gSimWeightsL;
extern MCImg<float>			gSimWeightsR;
extern MCImg<SimVector>		gSimVecsL;
extern MCImg<SimVector>		gSimVecsR;





static void ImproveGuess(int y, int x, SlantedPlane &oldGuess, SlantedPlane &newGuess, float &bestCost, int sign)
{
	float newCost = PatchMatchSlantedPlaneCost(y, x, newGuess, sign);
	if (newCost < bestCost) {
		bestCost = newCost;
		oldGuess = newGuess;
	}
}

static void PropagateAndRandomSearch(int round, int sign, float maxDisp, cv::Point2i srcPos,
	MCImg<SlantedPlane> &slantedPlanesL, MCImg<SlantedPlane> &slantedPlanesR, cv::Mat &bestCostsL, cv::Mat &bestCostsR)
{
	int step = (round % 2 == 0 ? +1 : -1);
	int y = srcPos.y, x = srcPos.x;
	int numRows = bestCostsL.rows, numCols = bestCostsL.cols;

	// Spatial Propagation
	if (0 <= y - step && y - step < numRows) {
		SlantedPlane newGuess = slantedPlanesL[y - step][x];
		ImproveGuess(y, x, slantedPlanesL[y][x], newGuess, bestCostsL.at<float>(y, x), sign);
	}

	if (0 <= x - step && x - step < numCols) {
		SlantedPlane newGuess = slantedPlanesL[y][x - step];
		ImproveGuess(y, x, slantedPlanesL[y][x], newGuess, bestCostsL.at<float>(y, x), sign);
	}

	// Random Search
	float zRadius = maxDisp / 2.f;
	float nRadius = 1.0f;
	while (zRadius >= 0.1) {
		SlantedPlane newGuess = SlantedPlane::ConstructFromRandomPertube(slantedPlanesL[y][x], y, x, nRadius, zRadius);
		ImproveGuess(y, x, slantedPlanesL[y][x], newGuess, bestCostsL.at<float>(y, x), sign);
		zRadius /= 2.0f;
		nRadius /= 2.0f;
	}

	// View Propagation
	int xMatch = 0.5 + x + sign * slantedPlanesL[y][x].ToDisparity(y, x);
	if (0 <= xMatch && xMatch < numCols) {
		SlantedPlane newGuess = SlantedPlane::ConstructFromOtherView(slantedPlanesL[y][x], sign);
		ImproveGuess(y, xMatch, slantedPlanesR[y][xMatch], newGuess, bestCostsR.at<float>(y, xMatch), -sign);
	}
}

static void PatchMatchRandomInit(MCImg<SlantedPlane> &slantedPlanes, cv::Mat &bestCosts, float maxDisp, int sign)
{
	int numRows = bestCosts.rows, numCols = bestCosts.cols;
	// Do not use parallelism here, it would results in "regular" random patterns
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			slantedPlanes[y][x] = SlantedPlane::ConstructFromRandomInit(y, x, maxDisp);
		}
	}

	#pragma omp parallel for
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			bestCosts.at<float>(y, x) = PatchMatchSlantedPlaneCost(y, x, slantedPlanes[y][x], sign);
		}
	}
}

MCImg<float> PrecomputeSimilarityWeights(cv::Mat &img, int patchRadius, int simGamma)
{
	int numRows = img.rows, numCols = img.cols;
	int patchSize = (2 * patchRadius + 1) * (2 * patchRadius + 1);

	MCImg<float> simWeights(numRows * numCols, patchSize);
	memset(simWeights.data, 0, numRows * numCols * patchRadius * sizeof(float));

	#pragma omp parallel for
	for (int yc = 0; yc < numRows; yc++) {
		for (int xc = 0; xc < numCols; xc++) {

			float *w = simWeights.line(yc * numCols + xc);
			cv::Vec3b center = img.at<cv::Vec3b>(yc, xc);

			for (int y = yc - patchRadius, id = 0; y <= yc + patchRadius; y++) {
				for (int x = xc - patchRadius; x <= xc + patchRadius; x++, id++) {
					if (InBound(y, x, numRows, numCols)) {
						float adDiff = L1Dist(center, img.at<cv::Vec3b>(y, x));
						w[id] = exp(-adDiff / simGamma);
					}
				}
			}
		}
	}

	return simWeights;
}

static cv::Mat SlantedPlaneMapToDisparityMap(MCImg<SlantedPlane> &slantedPlanes)
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

static cv::Mat CrossCheck(cv::Mat &dispL, cv::Mat &dispR, int sign, float thresh = 1.f)
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
				disp.at<float>(y, x) = std::min(dL, dR);
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
		}
	}

	return disp.at<float>(yc, xc);
}

static void WeightedMedianFilterInvalidPixels(cv::Mat &disp, cv::Mat &validPixelMap, cv::Mat &guideImg)
{
	cv::Mat dispOut = disp.clone();
	int numRows = disp.rows, numCols = disp.cols;
	float *w = new float[PATCHWIDTH * PATCHWIDTH];

	#pragma omp parallel for
	for (int yc = 0; yc < numRows; yc++) {
		for (int xc = 0; xc < numCols; xc++) {
			if (!validPixelMap.at<unsigned char>(yc, xc)) {
				memset(w, 0, PATCHWIDTH * PATCHWIDTH * sizeof(float));
				cv::Vec3b &center = guideImg.at<cv::Vec3b>(yc, xc);
				for (int y = yc - PATCHRADIUS, id = 0; y <= yc + PATCHRADIUS; y++) {
					for (int x = xc - PATCHRADIUS; x <= xc + PATCHRADIUS; x++, id++) {
						if (InBound(y, x, numRows, numCols)) {
							cv::Vec3b &c = guideImg.at<cv::Vec3b>(y, x);
							w[id] = exp(-L1Dist(center, c) / (float)SIMILARITY_GAMMA);
						}
					}
				}
				dispOut.at<float>(yc, xc) = SelectWeightedMedianFromPatch(disp, yc, xc, w);
			}
		}
	}
	
	delete[] w;
	dispOut.copyTo(disp);
}

void RunPatchMatchOnPixels(std::string rootFolder, cv::Mat &imL, cv::Mat &imR, cv::Mat &dispL, cv::Mat &dispR)
{
	int numRows = imL.rows, numCols = imL.cols, numPixels = imL.rows * imL.cols;
	int numDisps, maxDisp, visualizeScale;
	SetupStereoParameters(rootFolder, numDisps, maxDisp, visualizeScale);

	MCImg<SlantedPlane> slantedPlanesL(numRows, numCols);
	MCImg<SlantedPlane> slantedPlanesR(numRows, numCols);

	cv::Mat bestCostsL(imL.size(), CV_32FC1);
	cv::Mat bestCostsR(imL.size(), CV_32FC1);

	if (gMatchingCostType == ADCENSUS) {
		gDsiL = ComputeAdCensusCostVolume(imL, imR, numDisps, -1, GRANULARITY);
		gDsiR = ComputeAdCensusCostVolume(imR, imL, numDisps, +1, GRANULARITY);
	}
	else if (gMatchingCostType == ADGRADIENT) {
		gDsiL = ComputeAdGradientCostVolume(imL, imR, numDisps, -1, GRANULARITY);
		gDsiR = ComputeAdGradientCostVolume(imR, imL, numDisps, +1, GRANULARITY);
	}

	std::vector<SimVector> simVecsStdL;
	std::vector<SimVector> simVecsStdR;

	if (gCostAggregationType == GRID) {
		bs::Timer::Tic("Precompute Similarity Weights");
		gSimWeightsL = PrecomputeSimilarityWeights(imL, PATCHRADIUS, SIMILARITY_GAMMA);
		gSimWeightsR = PrecomputeSimilarityWeights(imR, PATCHRADIUS, SIMILARITY_GAMMA);
		bs::Timer::Toc();
	}
	else if (gCostAggregationType == TOP50) {
		bs::Timer::Tic("Begin SelfSimilarityPropagation");
		SelfSimilarityPropagation(imL, simVecsStdL);
		SelfSimilarityPropagation(imR, simVecsStdR);
		InitSimVecWeights(imL, simVecsStdL);
		InitSimVecWeights(imR, simVecsStdR);
		gSimVecsL = MCImg<SimVector>(numRows, numCols, 1, &simVecsStdL[0]);
		gSimVecsR = MCImg<SimVector>(numRows, numCols, 1, &simVecsStdR[0]);
		bs::Timer::Toc();
	}




	// Step 1 - Random initialization
	bs::Timer::Tic("Random initializing");
	PatchMatchRandomInit(slantedPlanesL, bestCostsL, maxDisp, -1);
	PatchMatchRandomInit(slantedPlanesR, bestCostsR, maxDisp, +1);
	bs::Timer::Toc();


	std::vector<cv::Point2i> pixelList(numRows * numCols);
	for (int y = 0, id = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++, id++) {
			pixelList[id] = cv::Point2i(x, y);
		}
	}


	// Step 2 - Spatial propagation and random search
	for (int round = 0; round < MAX_PATCHMATCH_ITERS; round++) {

		bs::Timer::Tic("Left View");
		#pragma omp parallel for
		for (int id = 0; id < numPixels; id++) {
			PropagateAndRandomSearch(round, -1, maxDisp, pixelList[id],
				slantedPlanesL, slantedPlanesR, bestCostsL, bestCostsR);
		}
		bs::Timer::Toc();

		bs::Timer::Tic("Right View");
		#pragma omp parallel for
		for (int id = 0; id < numPixels; id++) {
			PropagateAndRandomSearch(round, +1, maxDisp, pixelList[id],
				slantedPlanesR, slantedPlanesL, bestCostsR, bestCostsL);
		}
		bs::Timer::Toc();

		std::reverse(pixelList.begin(), pixelList.end());

		dispL = SlantedPlaneMapToDisparityMap(slantedPlanesL);
		dispR = SlantedPlaneMapToDisparityMap(slantedPlanesR);

		std::vector<std::pair<std::string, void*>> auxParams;
		auxParams.push_back(std::pair<std::string, void*>("slantedPlanesL", &slantedPlanesL));
		auxParams.push_back(std::pair<std::string, void*>("bestCostsL",     &bestCostsL));
		EvaluateDisparity(rootFolder, dispL, 0.5f, auxParams, "OnMousePatchMatchOnPixels");
	}


	// Step 3 - Cross check and post-process
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

	EvaluateDisparity(rootFolder, dispL, 0.5f);
}

void TestPatchMatchOnPixels()
{
	cv::Mat imL = cv::imread("D:/data/stereo/teddy/im2.png");
	cv::Mat imR = cv::imread("D:/data/stereo/teddy/im6.png");
	cv::Mat dispL, dispR;

	RunPatchMatchOnPixels("teddy", imL, imR, dispL, dispR);
}