#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cassert>
#include <ctime>
#include <vector>
#include <algorithm>

#include "MCImg.h"
#include "StereoAPI.h"



#define		PAIRWISECUTOFF		5
#define		LAMBDA				0.001f

#define ASSERT(condition)								\
	if (!(condition)) {									\
		printf("ASSERT %s VIOLATED AT LINE %d, %s\n",	\
			#condition, __LINE__, __FILE__);			\
		exit(-1);										\
	}

typedef std::vector<float> Message;
typedef std::vector<float> Probs;

struct VarNode
{
	int						cardinality;
	std::vector<float>		pot;
	std::vector<int>		factorNbs;
	std::vector<Message>	msgNVarToFactor;
};

struct FactorNode
{
	int						cardinality;
	std::vector<int>		varNbs;
	std::vector<Message>	msgMFactorToVar;
	std::vector<int>		bases;
};

class BP
{
public:
	std::vector<VarNode>	varNodes;
	std::vector<FactorNode> factorNodes;
	std::vector<Probs>		allBeliefs;

public:

	void BP::Run(std::string rootFolder, std::vector<int> &outLabels, int maxIters = 2000, float tol = 1e-4);

	void UpdateMessageNVarToFactor(int i, int alpha);

	void UpdateMessageMFactorToVar(int alpha, int i);

	std::vector<float>& MsgRefN(int i, int alpha);

	std::vector<float>& MsgRefM(int alpha, int i);

	int LocalIdxInNbs(std::vector<int> &arr, int val);

	std::vector<int> LinearIdToConfig(int linearStateId, int factorId);

	void NormalizeMessage(Message &msg);

	void NormalizeBelief(Probs &p);

	float SmoothnessCost(std::vector<int> &varInds, std::vector<int> &config);

	void InitForGridGraph(MCImg<float> &unaryCosts);
};








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





void operator *=(Message &msg, float s)
{
	for (int i = 0; i < msg.size(); i++) {
		msg[i] *= s;
	}
}

void operator +=(Message &a, Message &b)
{
	ASSERT(a.size() == b.size())
	for (int i = 0; i < a.size(); i++) {
		a[i] += b[i];
	}
}

void operator -=(Message &a, Message &b)
{
	ASSERT(a.size() == b.size())
	for (int i = 0; i < a.size(); i++) {
		a[i] -= b[i];
	}
}




static cv::Mat DecodeDisparityFromBeliefs(int numRows, int numCols, std::vector<Probs> &allBeliefs)
{
	cv::Mat dispMap(numRows, numCols, CV_32FC1);
	for (int y = 0, id = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++, id++) {
			Probs &belief = allBeliefs[id];
			dispMap.at<float>(y, x) = std::min_element(belief.begin(), belief.end()) - belief.begin();
		}
	}
	return dispMap;
}

void BP::Run(std::string rootFolder, std::vector<int> &outLabels, int maxIters, float tol)
{
	cv::Mat imL = cv::imread("D:/data/stereo/tsukuba/im2.png");
	int numRows = imL.rows, numCols = imL.cols;
	int numDisps, maxDisp, visualizeScale;
	SetupStereoParameters("tsukuba", numDisps, maxDisp, visualizeScale);



	float maxBeliefDiff = FLT_MAX;
	for (int iter = 0; iter < maxIters && maxBeliefDiff > tol; iter++) {

		printf("BP::Run iter = %d...\n", iter);
		int tic = clock();

		// currently processing in sequential order, consider random shuffle later.
		printf("Updating Messages msgNVarToFactor...\n");
		#pragma omp parallel for
		for (int i = 0; i < varNodes.size(); i++) {
			for (int k = 0; k < varNodes[i].factorNbs.size(); k++) {
				int alpha = varNodes[i].factorNbs[k];
				//printf("updating N_%dto%d ...\n", i, alpha);
				UpdateMessageNVarToFactor(i, alpha);
			}
		}

		printf("Updating Messages msgMFactorToVar...\n");
		#pragma omp parallel for
		for (int alpha = 0; alpha < factorNodes.size(); alpha++) {
			for (int k = 0; k < factorNodes[alpha].varNbs.size(); k++) {
				int i = factorNodes[alpha].varNbs[k];
				//printf("updating M_%dto%d ...\n", alpha, i);
				UpdateMessageMFactorToVar(alpha, i);

			}
		}

		printf("Updating Beliefs...\n");
		maxBeliefDiff = -FLT_MAX;
		#pragma omp parallel for
		for (int i = 0; i < allBeliefs.size(); i++) {
			Probs  oldBelief = allBeliefs[i];
			Probs &newBelief = allBeliefs[i];

			newBelief = varNodes[i].pot;
			for (int k = 0; k < varNodes[i].factorNbs.size(); k++) {
				int alpha = varNodes[i].factorNbs[k];
				newBelief += MsgRefM(alpha, i);
			}

			NormalizeBelief(newBelief);
			for (int x_i = 0; x_i < newBelief.size(); x_i++) {
				maxBeliefDiff = std::max(maxBeliefDiff, std::abs(oldBelief[x_i] - newBelief[x_i]));
			}
		}
		printf("maxBeliefDiff = %lf\n", maxBeliefDiff);
		printf("%.2fs\n", (clock() - tic) / 1000.f);

		cv::Mat dispL = DecodeDisparityFromBeliefs(numRows, numCols, allBeliefs);
		EvaluateDisparity(rootFolder, dispL);
	}
}

void BP::UpdateMessageNVarToFactor(int i, int alpha)
{
	Message &nI2Alpha = MsgRefN(i, alpha);
	nI2Alpha = varNodes[i].pot;

	for (int k = 0; k < varNodes[i].factorNbs.size(); k++) {
		int beta = varNodes[i].factorNbs[k];
		if (beta != alpha) {
			nI2Alpha += MsgRefM(beta, i);
		}
	}

	NormalizeMessage(nI2Alpha);
}

void BP::UpdateMessageMFactorToVar(int alpha, int i)
{
	Message &mAlpha2i = MsgRefM(alpha, i);
	std::fill(mAlpha2i.begin(), mAlpha2i.end(), FLT_MAX);
	const int iLocalIdx = LocalIdxInNbs(factorNodes[alpha].varNbs, i);

	for (int id = 0; id < factorNodes[alpha].cardinality; id++) {
		std::vector<int> config = LinearIdToConfig(id, alpha);
		int xi = config[iLocalIdx];
		float newMAlpha2iAtxi = SmoothnessCost(factorNodes[alpha].varNbs, config);

		for (int k = 0; k < factorNodes[alpha].varNbs.size(); k++) {
			int j = factorNodes[alpha].varNbs[k];
			if (j != i) {	// or k != iLocalIdx
				newMAlpha2iAtxi += MsgRefN(j, alpha)[config[k]];
			}
		}

		mAlpha2i[xi] = std::min(mAlpha2i[xi], newMAlpha2iAtxi);
	}
}

std::vector<float>& BP::MsgRefN(int i, int alpha)
{
	int alphaLocalIdx = LocalIdxInNbs(varNodes[i].factorNbs, alpha);
	return varNodes[i].msgNVarToFactor[alphaLocalIdx];
}

std::vector<float>& BP::MsgRefM(int alpha, int i)
{
	int iLocalIdx = LocalIdxInNbs(factorNodes[alpha].varNbs, i);
	return factorNodes[alpha].msgMFactorToVar[iLocalIdx];
}

int BP::LocalIdxInNbs(std::vector<int> &arr, int val)
{
	for (int i = 0; i < arr.size(); i++) {
		if (arr[i] == val) {
			return i;
		}
	}
	printf("Cannot find a neighbor matching %d, this should never happen!\n", val);
	exit(-1);
	return -1;
}

std::vector<int> BP::LinearIdToConfig(int linearStateId, int factorId)
{
	std::vector<int> &bases = factorNodes[factorId].bases;
	std::vector<int> config(bases.size());
	for (int i = 0; i < bases.size(); i++){
		config[i] = linearStateId / bases[i];
		linearStateId %= bases[i];
	}
	return config;
}

void BP::NormalizeMessage(Message &msg)
{
	float minVal = *std::min_element(msg.begin(), msg.end());
	for (int i = 0; i < msg.size(); i++) {
		msg[i] -= minVal;
	}
}

void BP::NormalizeBelief(Probs &p)
{
	float sum = 0.0;
	for (int i = 0; i < p.size(); i++) {
		sum += p[i];
	}
	assert(std::abs(sum) > 1e-6);
	for (int i = 0; i < p.size(); i++) {
		p[i] /= sum;
	}
}

float BP::SmoothnessCost(std::vector<int> &varInds, std::vector<int> &config)
{
	// current doe not make use of varInds
	ASSERT(config.size() == 2)
	return LAMBDA * std::min(PAIRWISECUTOFF, std::abs(config[0] - config[1]));
}

enum FactorType { LIEONHORIZONTAL, LIEONVERTICAL };

static int GetFactorId(int y, int x, int numRows, int numCols, FactorType factorType)
{
	if (factorType == LIEONHORIZONTAL) {
		return y * (numCols - 1) + x;
	}
	else {
		return y * numCols + x
			+ numRows * (numCols - 1);
	}
}

void BP::InitForGridGraph(MCImg<float> &unaryCosts)
{
	// Factor node ID coding scheme: there are two types of factor nodes on grid graph.
	// Fist type lies on horizontal connection, second type lies on vertical connection.
	// The IDs of the first type go before the second type. there are R*(C-1) first type
	// factors, and C*(R-1) second type factors. 
	// The coordinates of first type node is (y, x), where y is the image row it lies in, 
	// and x is the column id of its left neighbor. 
	// The coordinates of first type node is (y, x), where x is the image col it lies in, 
	// and y is the row id of its top neighbor. 

	int numRows = unaryCosts.h, numCols = unaryCosts.w, numDisps = unaryCosts.n;

	// Step 1 - Initialize beliefs
	Probs tieBelief(numDisps, 1.f / numDisps);
	allBeliefs.resize(numRows * numCols);
	std::fill(allBeliefs.begin(), allBeliefs.end(), tieBelief);

	
	// Step 2 - Initialize vars
	printf("step 2 ...\n");
	varNodes.resize(numRows * numCols);
	for (int y = 0, id = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++, id++) {
			varNodes[id].cardinality = numDisps;
			varNodes[id].pot = std::vector<float>(unaryCosts.get(y, x), unaryCosts.get(y, x) + numDisps);

			if (x > 0)			 varNodes[id].factorNbs.push_back(GetFactorId(y, x - 1, numRows, numCols, LIEONHORIZONTAL));
			if (x < numCols - 1) varNodes[id].factorNbs.push_back(GetFactorId(y, x,		numRows, numCols, LIEONHORIZONTAL));
			if (y > 0)			 varNodes[id].factorNbs.push_back(GetFactorId(y - 1, x, numRows, numCols, LIEONVERTICAL));
			if (y < numRows - 1) varNodes[id].factorNbs.push_back(GetFactorId(y, x,		numRows, numCols, LIEONVERTICAL));

			// stores the neighboring factor messages in order LEFT, RIGHT, UP, DOWN
			varNodes[id].msgNVarToFactor = std::vector<Message>(4, Message(numDisps, 0.f));
		}
	}

	// Step 3 - Initialize factors
	printf("step 3 ...\n");
	factorNodes.resize(numRows * (numCols - 1) + numCols * (numRows - 1));
	printf("numRows = %d, numCols = %d\n", numRows, numCols);
	printf("size = %d\n", numRows * (numCols - 1) + numCols * (numRows - 1));
	std::vector<int> commonBases;
	commonBases.push_back(numDisps);
	commonBases.push_back(1);
	int id = 0;

	printf("step 3.1 ...\n");
	// Horizontal factors
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols - 1; x++, id++) {
			factorNodes[id].cardinality = numDisps * numDisps;
			factorNodes[id].bases = commonBases;
			factorNodes[id].varNbs.push_back(y * numCols + x);
			factorNodes[id].varNbs.push_back(y * numCols + x + 1);
			factorNodes[id].msgMFactorToVar = std::vector<Message>(2, Message(numDisps, 0.f));
		}
	}

	// Vertical factors
	printf("step 3.2 ...\n");
	for (int y = 0; y < numRows - 1; y++) {
		for (int x = 0; x < numCols; x++, id++) {
			factorNodes[id].cardinality = numDisps * numDisps;
			factorNodes[id].bases = commonBases;
			factorNodes[id].varNbs.push_back(      y * numCols + x);
			factorNodes[id].varNbs.push_back((y + 1) * numCols + x);
			factorNodes[id].msgMFactorToVar = std::vector<Message>(2, Message(numDisps, 0.f));
		}
		//printf("id = %d\n", id);
	}

	printf("done.\n");
}

void TestLBPOnFactorGraph()
{


	cv::Mat imL = cv::imread("D:/data/stereo/tsukuba/im2.png");
	cv::Mat imR = cv::imread("D:/data/stereo/tsukuba/im6.png");

	int numDisps, maxDisp, visualizeScale;
	SetupStereoParameters("tsukuba", numDisps, maxDisp, visualizeScale);

	//MCImg<float> unaryCosts = ComputeAdCensusCostVolume(imL, imR, numDisps, -1, 1);
	MCImg<float> unaryCosts = ComputeAdGradientCostVolume(imL, imR, numDisps, -1, 1);

#if 0
	cv::Mat dispL = WinnerTakesAll(unaryCosts, 1);
	EvaluateDisparity("tsukuba", dispL);
#else
	BP bp;
	std::vector<int> outLabels;
	bp.InitForGridGraph(unaryCosts);
	bp.Run("tsukuba", outLabels);
#endif
}