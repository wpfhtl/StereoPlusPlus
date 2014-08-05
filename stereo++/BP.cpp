#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "StereoAPI.h"
#include "Timer.h"


#define		SIMILARITYGAMMA		10
#define		PAIRWISECUTOFF		5
#define		LAMBDA				0.002f

extern int					PATCHRADIUS;
extern int					PATCHWIDTH;
extern float				GRANULARITY;

extern enum CostAggregationType { REGULAR_GRID, TOP50 };
extern enum MatchingCostType	{ ADGRADIENT, ADCENSUS };

extern CostAggregationType	gCostAggregationType;
extern MatchingCostType		gMatchingCostType;

extern MCImg<float>			gDsiL;
extern MCImg<float>			gDsiR;
extern MCImg<float>			gSimWeightsL;
extern MCImg<float>			gSimWeightsR;
extern MCImg<SimVector>		gSimVecsL;
extern MCImg<SimVector>		gSimVecsR;


static const cv::Point2i dirDelta[4] = { cv::Point2i(0, -1), cv::Point2i(-1, 0), cv::Point2i(0, +1), cv::Point2i(+1, 0) };

static inline float PairwiseCost(int xs, int xt)
{
	return LAMBDA * std::min(PAIRWISECUTOFF, std::abs(xs - xt));
}

static inline float *GetMessage(cv::Point2i &s, cv::Point2i &t, MCImg<float> &allMessages, int numDisps)
{
	// This function assumes the position of s and t are both legal!
	cv::Point2i delta = t - s;
	for (int k = 0; k < 4; k++) {
		if (delta == dirDelta[k]) {
			return allMessages.get(t.y, t.x) + k * numDisps;
		}
	}
	printf("Bug: You should never reach here!\n");
	return allMessages.get(t.y, t.x);
}

static void UpdateMessage(cv::Point2i &s, cv::Point2i &t, MCImg<float> &oldMessages, MCImg<float> &newMessages, MCImg<float> &unaryCost)
{
	int  numRows = unaryCost.h, numCols = unaryCost.w, numDisps = unaryCost.n;
	if (!InBound(s, numRows, numCols) || !InBound(t, numRows, numCols)) {
		return;
	}

	float *s2tMsgNew = GetMessage(s, t, newMessages, numDisps);

	float minMsgVal = FLT_MAX;
	for (int xt = 0; xt < numDisps; xt++) {	

		float bestCost = FLT_MAX;
		for (int xs = 0; xs < numDisps; xs++) {
			float tmpCost = unaryCost.get(s.y, s.x)[xs] + PairwiseCost(xs, xt);
			for (int k = 0; k < 4; k++) {
				cv::Point2i q = s + dirDelta[k];
				if (InBound(q, numRows, numCols) && q != t) {
					tmpCost += GetMessage(q, s, oldMessages, numDisps)[xs];
				}
			}
			bestCost = std::min(bestCost, tmpCost);
		}

		s2tMsgNew[xt] = bestCost;
		//s2tMsg[xt] = 0.7 * s2tMsg[xt] + 0.3 * bestCost;
		minMsgVal = std::min(minMsgVal, bestCost);
	}

	// Normalize messages
	for (int xt = 0; xt < numDisps; xt++) {
		s2tMsgNew[xt] -= minMsgVal;
	}
}

static float UpdateBelief(cv::Point2i &t, MCImg<float> &allBeliefs, MCImg<float> &allMessages, MCImg<float> &unaryCost)
{
	int numRows = unaryCost.h, numCols = unaryCost.w, numDisps = unaryCost.n;
	float maxDiff = -1;
	float *s2tMsgs[4] = { 0 };


	for (int k = 0; k < 4; k++) {
		cv::Point2i s = t + dirDelta[k];
		if (InBound(s, numRows, numCols)) {
			s2tMsgs[k] = GetMessage(s, t, allMessages, numDisps);
		}
	}

	float *belief = allBeliefs.get(t.y, t.x);
	for (int xt = 0; xt < numDisps; xt++) {
		float newBelief = unaryCost.get(t.y, t.x)[xt];
		for (int k = 0; k < 4; k++) {
			if (s2tMsgs[k]) {
				newBelief += s2tMsgs[k][xt];
			}
		}
		maxDiff = std::max(maxDiff, std::abs(newBelief - belief[xt]));
		belief[xt] = newBelief;
	}

	return maxDiff;
}

static cv::Mat DecodeDisparityFromBeliefs(MCImg<float> &allBeliefs)
{
	int numRows = allBeliefs.h, numCols = allBeliefs.w, numDisps = allBeliefs.n;
	cv::Mat dispMap(numRows, numCols, CV_32FC1);
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			float *belief = allBeliefs.get(y, x);
			int bestIdx = 0;
			for (int k = 1; k < numDisps; k++) {
				if (belief[k] < belief[bestIdx]) {
					bestIdx = k;
				}
			}
			dispMap.at<float>(y, x) = bestIdx;
		}
	}
	return dispMap;
}

static cv::Mat RunLoopyBPOnGrideGraph(std::string rootFolder, MCImg<float> &unaryCost)
{
	int numRows = unaryCost.h, numCols = unaryCost.w, numDisps = unaryCost.n;
	std::vector<cv::Point2i> pixelList(numRows * numCols);
	for (int y = 0, id = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++, id++) {
			pixelList[id] = cv::Point2i(x, y);
		}
	}
	
	MCImg<float> oldMessages(numRows, numCols, 4 * numDisps);
	MCImg<float> newMessages(numRows, numCols, 4 * numDisps);
	MCImg<float> allBeliefs(numRows, numCols, numDisps);

	// Step 1 - Initialize messages
	memset(oldMessages.data, 0, 4 * numRows * numCols * numDisps * sizeof(float));
	memset(newMessages.data, 0, 4 * numRows * numCols * numDisps * sizeof(float));
	for (int i = 0; i < numRows * numCols * numDisps; i++) {
		allBeliefs.data[i] = 1.f / numDisps;
	}

	// Step 2 - Update messages
	float maxBeliefDiff = FLT_MAX;
	const int maxBPRound = 100;
	for (int round = 0; round < maxBPRound && maxBeliefDiff > 1e-2; round++) {

		if (round % 1 == 0) {
			cv::Mat dispMap = DecodeDisparityFromBeliefs(allBeliefs);
			std::vector<std::pair<std::string, void*>> auxParams;
			auxParams.push_back(std::pair<std::string, void*>("allMessages", &oldMessages));
			auxParams.push_back(std::pair<std::string, void*>("allBeliefs",  &allBeliefs));
			auxParams.push_back(std::pair<std::string, void*>("unaryCost",   &unaryCost));
			EvaluateDisparity(rootFolder, dispMap, 1.f, auxParams, "OnMouseLoopyBPOnGridGraph");
		}

		printf("Doing round %d ...\n", round);
		std::random_shuffle(pixelList.begin(), pixelList.end());
		printf("pixelList.size() = %d\n", pixelList.size());
		printf("numDisps = %d\n", numDisps);
		int numPixels = pixelList.size();

		// update messages
		#pragma omp parallel for
		for (int i = 0; i < numPixels; i++) {
			if (i % 1000 == 0) {
				//printf("i = %d\n", i);
			}

			cv::Point2i s = pixelList[i];
			for (int k = 0; k < 4; k++) {
				cv::Point2i t = s + dirDelta[k];
				UpdateMessage(s, t, oldMessages, newMessages, unaryCost);
			}
		}
		memcpy(oldMessages.data, newMessages.data, 4 * numRows * numCols * numDisps * sizeof(float));

		// update beliefs
		maxBeliefDiff = -1;
		for (int i = 0; i < pixelList.size(); i++) {
			cv::Point2i t = pixelList[i];
			float maxDiff = UpdateBelief(t, allBeliefs, oldMessages, unaryCost);
			maxBeliefDiff = std::max(maxBeliefDiff, maxDiff);
		}
		printf("maxBeliefDiff = %f\n", maxBeliefDiff);
	}


	// Step 3 - Collect Beliefs
	cv::Mat dispMap = DecodeDisparityFromBeliefs(allBeliefs);
	EvaluateDisparity(rootFolder, dispMap);
	return dispMap;
}

void RunLoopyBP(std::string rootFolder, cv::Mat &imL, cv::Mat &imR)
{
	int numRows = imL.rows, numCols = imL.cols;
	int numDisps, maxDisp, visualizeScale;
	SetupStereoParameters(rootFolder, numDisps, maxDisp, visualizeScale);

	if (gMatchingCostType == ADCENSUS) {
		gDsiL = ComputeAdCensusCostVolume(imL, imR, numDisps, -1, GRANULARITY);
		//gDsiR = ComputeAdCensusCostVolume(imR, imL, numDisps, +1, GRANULARITY);
	}
	else if (gMatchingCostType == ADGRADIENT) {
		gDsiL = ComputeAdGradientCostVolume(imL, imR, numDisps, -1, GRANULARITY);
		//gDsiR = ComputeAdGradientCostVolume(imR, imL, numDisps, +1, GRANULARITY);
	}

	std::vector<SimVector> simVecsStdL;
	std::vector<SimVector> simVecsStdR;

	if (gCostAggregationType == REGULAR_GRID) {
		bs::Timer::Tic("Precompute Similarity Weights");
		MCImg<float> PrecomputeSimilarityWeights(cv::Mat &img, int patchRadius, int simGamma);
		gSimWeightsL = PrecomputeSimilarityWeights(imL, PATCHRADIUS, SIMILARITYGAMMA);
		//gSimWeightsR = PrecomputeSimilarityWeights(imR, PATCHRADIUS, SIMILARITYGAMMA);
		bs::Timer::Toc();
	}
	else if (gCostAggregationType == TOP50) {
		bs::Timer::Tic("Begin SelfSimilarityPropagation");
		SelfSimilarityPropagation(imL, simVecsStdL);
		//SelfSimilarityPropagation(imR, simVecsStdR);
		InitSimVecWeights(imL, simVecsStdL);
		//InitSimVecWeights(imR, simVecsStdR);
		gSimVecsL = MCImg<SimVector>(numRows, numCols, 1, &simVecsStdL[0]);
		//gSimVecsR = MCImg<SimVector>(numRows, numCols, 1, &simVecsStdR[0]);
		bs::Timer::Toc();
	}

	MCImg<float> unaryCost(numRows, numCols, numDisps);
#if 0
	memcpy(unaryCost.data, gDsiL.data, numRows * numCols * numDisps * sizeof(float));
#else
	#pragma omp parallel for
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			for (int d = 0; d < numDisps; d++) {
				SlantedPlane sp = SlantedPlane::ConstructFromAbc(0.f, 0.f, d);
				unaryCost.get(y, x)[d] = PatchMatchSlantedPlaneCost(y, x, sp, -1);
			}
		}
	}
#endif

	cv::Mat dispL = RunLoopyBPOnGrideGraph(rootFolder, unaryCost);
	//cv::Mat dispL = WinnerTakesAll(unaryCost, GRANULARITY);
	EvaluateDisparity(rootFolder, dispL);
}



void TestLBPOnGridGraph()
{
	//cv::Point2i a(1, 1), b(1, 1);
	//int e = (a == b);
	//printf("%d\n", e);
	//return;

	cv::Mat imL = cv::imread("D:/data/stereo/tsukuba/im2.png");
	cv::Mat imR = cv::imread("D:/data/stereo/tsukuba/im6.png");
	cv::Mat dispL, dispR;

	RunLoopyBP("tsukuba", imL, imR);
}