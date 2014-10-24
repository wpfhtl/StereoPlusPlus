#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "StereoAPI.h"
#include "Timer.h"
#include "SlantedPlane.h"
#include "PostProcess.h"



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

void RunPatchMatchOnPixels(std::string rootFolder, cv::Mat &imL, cv::Mat &imR, cv::Mat &dispL, cv::Mat &dispR,
	std::string filePathImageL, std::string filePathImageR, std::string filePathImageOut)
{
	int numRows = imL.rows, numCols = imL.cols, numPixels = imL.rows * imL.cols;
	int numDisps, maxDisp, visualizeScale;
	SetupStereoParameters(rootFolder, numDisps, maxDisp, visualizeScale);
	//InitGlobalDsiAndSimWeights(imL, imR, numDisps);


	bs::Timer::Tic("Cost Volume");


	double tensorSize = (double)numRows * numCols * numDisps * sizeof(unsigned short) / (double)(1024 * 1024);
	printf("(numRows, numCols) = (%d, %d)\n", numRows, numCols);
	printf("tensorSize: %lf\n", tensorSize);
	printf("numDisps: %d\n", numDisps);

	if (tensorSize > 1000/*rootFolder == "Midd3"*/) {
		extern cv::Mat gSobelImgL, gSobelImgR, gCensusImgL, gCensusImgR;
		cv::Mat ComputeCappedSobelImage(cv::Mat &imgIn, int sobelCapValue);
		gSobelImgL = ComputeCappedSobelImage(imL, 15);
		gSobelImgR = ComputeCappedSobelImage(imR, 15);
		gCensusImgL = ComputeCensusImage(imL, 2, 2);
		gCensusImgR = ComputeCensusImage(imR, 2, 2);
	}
	else {
		extern MCImg<float> gDsiL, gDsiR;
		gDsiL = MCImg<float>(numRows, numCols, numDisps);
		gDsiR = MCImg<float>(numRows, numCols, numDisps);
		void CostVolumeFromYamaguchi(std::string &leftFilePath, std::string &rightFilePath,
			MCImg<float> &dsiL, MCImg<float> &dsiR, int numDisps);
		printf("%s\n%s\n", filePathImageL.c_str(), filePathImageR.c_str());
		CostVolumeFromYamaguchi(filePathImageL, filePathImageR, gDsiL, gDsiR, numDisps);

		//MCImg<float> ComputeBirchfieldTomasiCostVolume(cv::Mat &imL, cv::Mat &imR, int numDisps, int sign);
		//MCImg<float> Compute5x5CensusCostVolume(cv::Mat &imL, cv::Mat &imR, int numDisps, int sign);
		//void CombineBirchfieldTomasiAnd5x5Census(MCImg<float> &dsiBT, MCImg<float> &dsiCensus);
		//MCImg<float> dsiBTL = ComputeBirchfieldTomasiCostVolume(imL, imR, numDisps, -1);
		//MCImg<float> dsiBTR = ComputeBirchfieldTomasiCostVolume(imR, imL, numDisps, +1);
		//gDsiL = Compute5x5CensusCostVolume(imL, imR, numDisps, -1);
		//gDsiR = Compute5x5CensusCostVolume(imR, imL, numDisps, +1);
		//CombineBirchfieldTomasiAnd5x5Census(dsiBTL, gDsiL);
		//CombineBirchfieldTomasiAnd5x5Census(dsiBTR, gDsiR);
	}

	extern cv::Mat gImLabL, gImLabR;
	gImLabL = imL.clone();
	gImLabR = imR.clone();
	bs::Timer::Toc();


	MCImg<SlantedPlane> slantedPlanesL(numRows, numCols);
	MCImg<SlantedPlane> slantedPlanesR(numRows, numCols);

	cv::Mat bestCostsL(imL.size(), CV_32FC1);
	cv::Mat bestCostsR(imL.size(), CV_32FC1);



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

	struct PatchMatchOnPixelEvalParams {
		MCImg<SlantedPlane> *slantedPlanes;
		cv::Mat *bestCosts;
		PatchMatchOnPixelEvalParams() : slantedPlanes(NULL), bestCosts(NULL) {}
	};


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

		PatchMatchOnPixelEvalParams evalParams;
		evalParams.slantedPlanes	= &slantedPlanesL;
		evalParams.bestCosts		= &bestCostsL;
		//EvaluateDisparity(rootFolder, dispL, 0.5f, &evalParams, "OnMousePatchMatchOnPixels");
	}


	//////////////////////////////////////////////////////////////////////////////
	//                                   Output
	//////////////////////////////////////////////////////////////////////////////
	float upFactor = (rootFolder == "KITTI" ? 256 : 64);
	cv::Mat SetInvalidDisparityToZeros(cv::Mat &dispL, cv::Mat &validPixelMapL);
	cv::Mat CrossCheck(cv::Mat &dispL, cv::Mat &dispR, int sign, float thresh = 1.f);
	cv::Mat validPixelL = CrossCheck(dispL, dispR, -1, 1.f);
	cv::Mat validPixelR = CrossCheck(dispR, dispL, +1, 1.f);
	cv::Mat dispCrossCheckedL = SetInvalidDisparityToZeros(dispL, validPixelL);
	cv::Mat dispCrossCheckedR = SetInvalidDisparityToZeros(dispR, validPixelR);

	// Step 3 - Cross check and post-process
	cv::Mat dispWmfL = dispL.clone();
	cv::Mat dispWmfR = dispR.clone();
	PatchMatchOnPixelPostProcess(slantedPlanesL, slantedPlanesR, imL, imR, dispWmfL, dispWmfR);

	dispL.convertTo(dispL, CV_16UC1, upFactor);
	dispR.convertTo(dispR, CV_16UC1, upFactor);
	dispCrossCheckedL.convertTo(dispCrossCheckedL, CV_16UC1, upFactor);
	dispCrossCheckedR.convertTo(dispCrossCheckedR, CV_16UC1, upFactor);
	dispWmfL.convertTo(dispWmfL, CV_16UC1, upFactor);
	dispWmfR.convertTo(dispWmfR, CV_16UC1, upFactor);

	cv::imwrite(filePathImageOut + "_dispL.png", dispL);
	cv::imwrite(filePathImageOut + "_dispR.png", dispR);
	cv::imwrite(filePathImageOut + "_dispCrossCheckedL.png", dispCrossCheckedL);
	cv::imwrite(filePathImageOut + "_dispCrossCheckedR.png", dispCrossCheckedR);
	cv::imwrite(filePathImageOut + "_wmfL.png", dispWmfL);
	cv::imwrite(filePathImageOut + "_wmfR.png", dispWmfR);

	//PatchMatchOnPixelEvalParams evalParams;
	//evalParams.slantedPlanes = &slantedPlanesL;
	//evalParams.bestCosts = &bestCostsL;
	//EvaluateDisparity(rootFolder, dispL, 0.5f, &evalParams, "OnMousePatchMatchOnPixels");

	//slantedPlanesL.SaveToBinaryFile("d:/" + rootFolder + "SlantedPlanesL.bin");
	//slantedPlanesR.SaveToBinaryFile("d:/" + rootFolder + "SlantedPlanesR.bin");

	//cv::imwrite("d:/data/stereo/" + rootFolder + "/PatchMatchOnPixel_dispL.png", dispImgL);
	//cv::imwrite("d:/data/stereo/" + rootFolder + "/PatchMatchOnPixel_dispR.png", visualizeScale * dispR);
}

void TestPatchMatchOnPixels()
{
	extern std::string ROOTFOLDER;
	std::string rootFolder = ROOTFOLDER;
	cv::Mat imL = cv::imread("D:/data/stereo/" + rootFolder + "/im2.png");
	cv::Mat imR = cv::imread("D:/data/stereo/" + rootFolder + "/im6.png");
	
	cv::Mat dispL, dispR;
	RunPatchMatchOnPixels(ROOTFOLDER, imL, imR, dispL, dispR, "s", "s", "s");
}