#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "StereoAPI.h"
#include "Timer.h"




extern int					PATCHRADIUS;
extern int					PATCHWIDTH;
extern float				GRANULARITY;
extern float				SIMILARITY_GAMMA;
extern float				ISING_CUTOFF;
extern float				ISING_LAMBDA;

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


static const cv::Point2i dirDelta[4] = { cv::Point2i(0, -1), cv::Point2i(-1, 0), cv::Point2i(0, +1), cv::Point2i(+1, 0) };

static inline float PairwiseCost(int xs, int xt)
{
	return ISING_LAMBDA * std::min((int)ISING_CUTOFF, std::abs(xs - xt));
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

static void UpdateMessage(cv::Point2i &s, cv::Point2i &t, MCImg<float> &oldMessages, 
	MCImg<float> &newMessages, MCImg<float> &unaryCosts, cv::Mat &img)
{
	extern bool USE_CONVEX_BP;
	if (USE_CONVEX_BP) {
		extern float ISING_GAMMA;
		int numRows = unaryCosts.h, numCols = unaryCosts.w, numDisps = unaryCosts.n;
		if (!InBound(s, numRows, numCols) || !InBound(t, numRows, numCols)) {
			return;
		}
		float simWeight = exp(-L1Dist(img.at<cv::Vec3b>(s.y, s.x), img.at<cv::Vec3b>(t.y, t.x)) / ISING_GAMMA);

		float *s2tMsgNew = GetMessage(s, t, newMessages, numDisps);
		float *t2sMsgOld = GetMessage(t, s, oldMessages, numDisps);

		float minMsgVal = FLT_MAX;
		for (int xt = 0; xt < numDisps; xt++) {

			float bestCost = FLT_MAX;
			for (int xs = 0; xs < numDisps; xs++) {
				//float tmpCost = unaryCosts.get(s.y, s.x)[xs] + simWeight * PairwiseCost(xs, xt);
				float tmpCost = unaryCosts.get(s.y, s.x)[xs];
				float hat_c_s = 1.f;
				for (int k = 0; k < 4; k++) {
					cv::Point2i q = s + dirDelta[k];
					if (InBound(q, numRows, numCols)) {
						hat_c_s += 1.f;
						tmpCost += GetMessage(q, s, oldMessages, numDisps)[xs];
					}
				}
				tmpCost *= (1.f / hat_c_s);
				tmpCost += simWeight * PairwiseCost(xs, xt);
				tmpCost -= t2sMsgOld[xs];
				bestCost = std::min(bestCost, tmpCost);
			}

			s2tMsgNew[xt] = bestCost;
			minMsgVal = std::min(minMsgVal, bestCost);
		}

		// Normalize messages
		for (int xt = 0; xt < numDisps; xt++) {
			s2tMsgNew[xt] -= minMsgVal;
		}
	}
	else {
		extern float ISING_GAMMA;
		int numRows = unaryCosts.h, numCols = unaryCosts.w, numDisps = unaryCosts.n;
		if (!InBound(s, numRows, numCols) || !InBound(t, numRows, numCols)) {
			return;
		}
		float simWeight = exp(-L1Dist(img.at<cv::Vec3b>(s.y, s.x), img.at<cv::Vec3b>(t.y, t.x)) / ISING_GAMMA);

		float *s2tMsgNew = GetMessage(s, t, newMessages, numDisps);

		float minMsgVal = FLT_MAX;
		for (int xt = 0; xt < numDisps; xt++) {

			float bestCost = FLT_MAX;
			for (int xs = 0; xs < numDisps; xs++) {
				float tmpCost = unaryCosts.get(s.y, s.x)[xs] + simWeight * PairwiseCost(xs, xt);
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
}

static float UpdateBelief(cv::Point2i &t, MCImg<float> &allBeliefs, MCImg<float> &allMessages, MCImg<float> &unaryCosts)
{
	int numRows = unaryCosts.h, numCols = unaryCosts.w, numDisps = unaryCosts.n;
	float maxDiff = -1;
	float *s2tMsgs[4] = { 0 };
	float newBelief[1024];		// Use a large array in stack, to prevent dynamic allocation

	for (int k = 0; k < 4; k++) {
		cv::Point2i s = t + dirDelta[k];
		if (InBound(s, numRows, numCols)) {
			s2tMsgs[k] = GetMessage(s, t, allMessages, numDisps);
		}
	}

	float *belief = allBeliefs.get(t.y, t.x);
	float accSum = 0.f;
	for (int xt = 0; xt < numDisps; xt++) {
		newBelief[xt] = unaryCosts.get(t.y, t.x)[xt];
		for (int k = 0; k < 4; k++) {
			if (s2tMsgs[k]) {
				newBelief[xt] += s2tMsgs[k][xt];
			}
		}
		accSum += newBelief[xt];
	}

	for (int xt = 0; xt < numDisps; xt++) {
		newBelief[xt] /= accSum;
		maxDiff = std::max(maxDiff, std::abs(newBelief[xt] - belief[xt]));
		belief[xt] = newBelief[xt];
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

static cv::Mat RunLoopyBPOnGrideGraph(std::string rootFolder, MCImg<float> &unaryCosts, cv::Mat &imL)
{
	int numRows = unaryCosts.h, numCols = unaryCosts.w, numDisps = unaryCosts.n;
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
	for (int round = 0; round < maxBPRound && maxBeliefDiff > 1e-7; round++) {
		int tic = clock();
		if (round % 1 == -1) {
			cv::Mat dispMap = DecodeDisparityFromBeliefs(allBeliefs);
			std::vector<std::pair<std::string, void*>> auxParams;
			auxParams.push_back(std::pair<std::string, void*>("allMessages", &oldMessages));
			auxParams.push_back(std::pair<std::string, void*>("allBeliefs",  &allBeliefs));
			auxParams.push_back(std::pair<std::string, void*>("unaryCosts",   &unaryCosts));
			EvaluateDisparity(rootFolder, dispMap, 1.f, auxParams, "OnMouseLoopyBPOnGridGraph");
		}


		printf("Doing round %d ...\n", round);
		std::random_shuffle(pixelList.begin(), pixelList.end());
		//printf("pixelList.size() = %d\n", pixelList.size());
		//printf("numDisps = %d\n", numDisps);
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
				
				UpdateMessage(s, t, oldMessages, newMessages, unaryCosts, imL);
				
			}
		}
		memcpy(oldMessages.data, newMessages.data, 4 * numRows * numCols * numDisps * sizeof(float));

		// update beliefs
		maxBeliefDiff = -1;
		for (int i = 0; i < pixelList.size(); i++) {
			cv::Point2i t = pixelList[i];
			float maxDiff = UpdateBelief(t, allBeliefs, oldMessages, unaryCosts);
			maxBeliefDiff = std::max(maxBeliefDiff, maxDiff);
		}
		printf("maxBeliefDiff = %f\n", maxBeliefDiff);
		printf("%.2fs\n", (clock() - tic) / 1000.f);

		cv::Mat dispMap = DecodeDisparityFromBeliefs(allBeliefs);
		cv::cvtColor(dispMap, dispMap, CV_GRAY2BGR);
		int maxDisp, visualizeScale;
		SetupStereoParameters(rootFolder, numDisps, maxDisp, visualizeScale);
		dispMap.convertTo(dispMap, CV_8UC3, visualizeScale);
		char filePath[1024];
		sprintf(filePath, "d:/data/tmpResults/iter=%d.png", round);
		cv::imwrite(filePath, dispMap);
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
	InitGlobalDsiAndSimWeights(imL, imR, numDisps);

	MCImg<float> unaryCosts(numRows, numCols, numDisps);
#if 1
	memcpy(unaryCosts.data, gDsiL.data, numRows * numCols * numDisps * sizeof(float));
	for (int d = 0; d < 16; d++) {
		printf("%f\n", unaryCosts.get(116, 216)[d]);
	}
	//cv::Mat uselessCost(numRows, numCols, CV_8UC3);
	//uselessCost.setTo(cv::Vec3b(0, 0, 0));
	//extern float COLORGRADALPHA;
	//extern float COLORMAXDIFF;
	//extern float GRADMAXDIFF;
	//float cutoff = COLORGRADALPHA * COLORMAXDIFF + (1 - COLORGRADALPHA) * GRADMAXDIFF;
	//for (int y = 0; y < numRows; y++) {
	//	for (int x=  0; x < numCols; x++) {
	//		for (int d = 0; d < numDisps; d++) {
	//			if (unaryCosts.get(y, x)[d] < cutoff) {
	//				uselessCost.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
	//				break;
	//			}
	//		}
	//	}
	//}
	//cv::imshow("dark is useless", uselessCost);
	//cv::waitKey(0);
#else
	#pragma omp parallel for
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			for (int d = 0; d < numDisps; d++) {
				SlantedPlane sp = SlantedPlane::ConstructFromAbc(0.f, 0.f, d);
				unaryCosts.get(y, x)[d] = PatchMatchSlantedPlaneCost(y, x, sp, -1);
			}
		}
	}
#endif

	cv::Mat dispL = RunLoopyBPOnGrideGraph(rootFolder, unaryCosts, imL);
	//cv::Mat dispL = WinnerTakesAll(unaryCosts, GRANULARITY);
	EvaluateDisparity(rootFolder, dispL);
}



void TestLBPOnGridGraph()
{

	cv::Mat imL = cv::imread("D:/data/stereo/tsukuba/im2.png");
	cv::Mat imR = cv::imread("D:/data/stereo/tsukuba/im6.png");
	cv::Mat dispL, dispR;

	RunLoopyBP("tsukuba", imL, imR);
}