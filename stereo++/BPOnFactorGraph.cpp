#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cassert>
#include <ctime>
#include <vector>
#include <set>
#include <algorithm>

#include "MCImg.h"
#include "StereoAPI.h"
#include "BPOnFactorGraph.h"



#define		PAIRWISECUTOFF		5
#define		LAMBDA				0.001f

#define		BOOL_SPLIT			1
#define		BOOL_NOTSPLIT		0

#define		TAU1				0.01
#define		TAU2				0.1
#define		TAU3				0.1



#define ASSERT(condition)								\
	if (!(condition)) {									\
		printf("ASSERT %s VIOLATED AT LINE %d, %s\n",	\
			#condition, __LINE__, __FILE__);			\
		exit(-1);										\
	}




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

		/*cv::Mat dispL = DecodeDisparityFromBeliefs(numRows, numCols, allBeliefs);
		EvaluateDisparity(rootFolder, dispL);*/
		cv::Mat splitImg = DecodeSplittingImageFromBeliefs(numRows, numCols, allBeliefs);
		cv::imshow("slitImg", splitImg);
		cv::waitKey(0);
		//printf("enter any thing:");
		//int tmp;
		//scanf("%d", &tmp);
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

	for (int id = 0; id < factorNodes[alpha].numConfigs; id++) {
		std::vector<int> config = LinearIdToConfig(id, alpha);
		int xi = config[iLocalIdx];
		/*float newMAlpha2iAtxi = SmoothnessCost(factorNodes[alpha].varNbs, config);*/
		float newMAlpha2iAtxi = FactorPotential(factorNodes[alpha].varNbs, config);

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

float TangentPlaneDist(SlantedPlane &p, SlantedPlane &q, cv::Point2d &Xp, cv::Point2d &Xq)
{
	const float cutoff = 15.0;
	float Dp  = p.ToDisparity(Xp.y, Xp.x);
	float Dq  = q.ToDisparity(Xq.y, Xq.y);
	float Dpq = p.ToDisparity(Xq.y, Xq.y);
	float Dqp = q.ToDisparity(Xp.y, Xp.x);
	return 0.5 * (std::min(cutoff, std::abs(Dq - Dpq)) + std::min(cutoff, std::abs(Dp - Dqp)));
}

float BP::FactorPotential(std::vector<int> &varInds, std::vector<int> &config)
{
	//return 0;
	ASSERT(config.size() == 2 || config.size() == 3)
	const cv::Point2d halfOffset(0.5, 0.5);
	if (config.size() == 2) {
		int numPrivateVertices = candidateLabels.size();	// for clarity.
		if (varInds[0] < numPrivateVertices) {
			ASSERT(varInds[1] < numPrivateVertices)
			return TangentPlaneDist(candidateLabels[varInds[0]][config[0]], candidateLabels[varInds[1]][config[1]],
				varCoords[varInds[0]] - halfOffset, varCoords[varInds[1]] - halfOffset);
		}
		else {
			return TAU2 * (config[0] == config[1] ? 0 : 1);
		}
	}
	else {
		if (config[2] == BOOL_SPLIT) {
			return 0.f;
		}
		else {
			ASSERT(varInds[0] < varCoords.size())
			if (varInds[0] >= candidateLabels.size()) {
				printf("varInds[0] = %d\n", varInds[0]);
				printf("varInds[1] = %d\n", varInds[1]);
				printf("varInds[2] = %d\n", varInds[2]);
				printf("candidateLabels.size() = %d\n", candidateLabels.size());
			}
			ASSERT(varInds[0] < candidateLabels.size())
			ASSERT(varInds[1] < candidateLabels.size())
			cv::Point2d &p = varCoords[varInds[0]] - halfOffset;
			float d1 = candidateLabels[varInds[0]][config[0]].ToDisparity(p.y, p.x);
			float d2 = candidateLabels[varInds[1]][config[1]].ToDisparity(p.y, p.x);
			return TAU3 * (std::abs(d1 - d2) > 1e-4);
		}
	}
	ASSERT(0) // You would never reach here.
	return 0;
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

void BP::InitFromGridGraph(MCImg<float> &unaryCosts)
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
			varNodes[id].numLabels = numDisps;
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
			factorNodes[id].numConfigs = numDisps * numDisps;
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
			factorNodes[id].numConfigs = numDisps * numDisps;
			factorNodes[id].bases = commonBases;
			factorNodes[id].varNbs.push_back(      y * numCols + x);
			factorNodes[id].varNbs.push_back((y + 1) * numCols + x);
			factorNodes[id].msgMFactorToVar = std::vector<Message>(2, Message(numDisps, 0.f));
		}
		//printf("id = %d\n", id);
	}

	printf("done.\n");
}

void BP::InitFromTriangulation(int numRows, int numCols, int numDisps,
	std::vector<std::vector<SlantedPlane>> &candidateLabels, std::vector<std::vector<float>> &unaryCosts,
	std::vector<cv::Point2d> &vertexCoords, std::vector<std::vector<int>> &triVertexInds,
	std::vector<std::vector<cv::Point2i>> &triPixelList, cv::Mat &img)
{
	int numTriangles = triVertexInds.size();
	int numPrivateVertices = 3 * numTriangles;
	int numSplitNodes = vertexCoords.size();

	MCImg<std::vector<std::pair<int, int>>> triIndSets(numRows + 1, numCols + 1);
	for (int id = 0; id < numTriangles; id++) {
		for (int j = 0; j < 3; j++) {
			cv::Point2d &p = vertexCoords[triVertexInds[id][j]];
			triIndSets[p.y][(int)p.x].push_back(std::make_pair(id, j));
		}
	}

	std::vector<std::set<int>> splitNodeNbIndSets(numSplitNodes);
	for (int id = 0; id < numTriangles; id++) {
		std::vector<int> &vertexIds = triVertexInds[id];
		for (int i = 0; i < 3; i++) {
			for (int j = i + 1; j < 3; j++) {
				splitNodeNbIndSets[vertexIds[i]].insert(vertexIds[j]);
				splitNodeNbIndSets[vertexIds[j]].insert(vertexIds[i]);
			}
		}
	}
	std::vector<std::vector<int>> splitNodeNbInds(numSplitNodes);
	for (int i = 0; i < numSplitNodes; i++) {
		std::set<int> &nbSet = splitNodeNbIndSets[i];
		splitNodeNbInds[i] = std::vector<int>(nbSet.begin(), nbSet.end());
	}



	// Step 2 - Initialize var nodes and beliefs
	varNodes.resize(numPrivateVertices + numSplitNodes);
	printf("Step 2.1 - Prepare unary costs for varNodes ...\n");
	for (int id = 0; id < numTriangles; id++) {
		for (int j = 0; j < 3; j++) {
			varNodes[3 * id + j].numLabels = unaryCosts[3 * id + j].size();
			varNodes[3 * id + j].pot = unaryCosts[3 * id + j];
		}
	}

	std::vector<float> constSplitPot;
	constSplitPot.push_back(0.f);
	constSplitPot.push_back(TAU1);
	for (int i = 0; i < numSplitNodes; i++) {
		varNodes[numPrivateVertices + i].numLabels = 2;
		varNodes[numPrivateVertices + i].pot = constSplitPot;
	}

	allBeliefs.resize(varNodes.size());
	for (int i = 0; i < varNodes.size(); i++) {
		allBeliefs[i] = Probs(varNodes[i].numLabels, 1.f / varNodes[i].numLabels);
	}

	// Step 3 - Initialize factor nodes


	//FIXME: pre-calculate the number of factor nodes, and resize factorNodes.
	int numType1Factors = 3 * numTriangles;
	int numType2Factors = 0;
	for (int id = 0; id < splitNodeNbInds.size(); id++) {
		for (int k = 0; k < splitNodeNbInds[id].size(); k++) {
			int nbId = splitNodeNbInds[id][k];
			if (id < nbId) {
				numType2Factors++;
			}
		}
	}
	int numType3Factors = 0;
	for (int id = 0; id < vertexCoords.size(); id++) {
		int y = vertexCoords[id].y;
		int x = vertexCoords[id].x;
		std::vector<std::pair<int, int>> &triIds = triIndSets[y][x];
		int groupSize = triIds.size();
		if (groupSize >= 2) {
			numType3Factors += groupSize;
		}
	}
	factorNodes.resize(numType1Factors + numType2Factors + numType3Factors);



	printf("Step 3.1 - each pair of private vertex from the same triangle has a factor ...\n");
	int numPairwiseVertexFactors = 0;
	for (int id = 0; id < numTriangles; id++) {
		for (int j = 0; j < 3; j++) {
			int curFactorIdx = numPairwiseVertexFactors;
			int varIdx1 = 3 * id + j;
			int varIdx2 = 3 * id + (j + 1) % 3;
			int numLabels1 = varNodes[varIdx1].numLabels;
			int numLabels2 = varNodes[varIdx2].numLabels;

			factorNodes[curFactorIdx].numConfigs = numLabels1 * numLabels2;
			factorNodes[curFactorIdx].varNbs.push_back(varIdx1);
			factorNodes[curFactorIdx].varNbs.push_back(varIdx2);
			factorNodes[curFactorIdx].bases.push_back(numLabels2);
			factorNodes[curFactorIdx].bases.push_back(1);
			factorNodes[curFactorIdx].msgMFactorToVar.push_back(Message(numLabels1, 0.f));
			factorNodes[curFactorIdx].msgMFactorToVar.push_back(Message(numLabels2, 0.f));

			varNodes[varIdx1].factorNbs.push_back(curFactorIdx);
			varNodes[varIdx2].factorNbs.push_back(curFactorIdx);
			varNodes[varIdx1].msgNVarToFactor.push_back(Message(numLabels1, 0.f));
			varNodes[varIdx2].msgNVarToFactor.push_back(Message(numLabels2, 0.f));

			numPairwiseVertexFactors++;
		}
	}

	printf("Step 3.2 - each pair of neighboring split nodes has a factor ...\n");
	int numPairwiseSplitFactors = 0;
	for (int id = 0; id < splitNodeNbInds.size(); id++) {
		for (int k = 0; k < splitNodeNbInds[id].size(); k++) {
			int nbId = splitNodeNbInds[id][k];
			if (id < nbId) {
				int varIdx1 = numPrivateVertices + id;
				int varIdx2 = numPrivateVertices + nbId;
				int curFactorIdx = numPairwiseVertexFactors + numPairwiseSplitFactors;

				factorNodes[curFactorIdx].numConfigs = 4;
				factorNodes[curFactorIdx].varNbs.push_back(varIdx1);
				factorNodes[curFactorIdx].varNbs.push_back(varIdx2);
				factorNodes[curFactorIdx].bases.push_back(2);
				factorNodes[curFactorIdx].bases.push_back(1);
				factorNodes[curFactorIdx].msgMFactorToVar.push_back(Message(2, 0.f));
				factorNodes[curFactorIdx].msgMFactorToVar.push_back(Message(2, 0.f));

				varNodes[varIdx1].factorNbs.push_back(curFactorIdx);
				varNodes[varIdx2].factorNbs.push_back(curFactorIdx);
				varNodes[varIdx1].msgNVarToFactor.push_back(Message(2, 0.f));
				varNodes[varIdx2].msgNVarToFactor.push_back(Message(2, 0.f));

				numPairwiseSplitFactors++;
			}
		}
	}

	printf("Step 3.3 - each group indexed by the vertex position has N triple factor"
		"where N is the group size.\n");
	int numTripleCliqueFactors = 0;
	for (int id = 0; id < vertexCoords.size(); id++) {
		int y = vertexCoords[id].y;
		int x = vertexCoords[id].x;
		std::vector<std::pair<int, int>> &triIds = triIndSets[y][x];
		int groupSize = triIds.size();

		if (groupSize >= 2) {
			for (int k = 0; k < groupSize; k++) {
				int k1 = k, k2 = (k + 1) % groupSize;
				int varIdx1 = 3 * triIds[k1].first + triIds[k1].second;
				int varIdx2 = 3 * triIds[k2].first + triIds[k2].second;
				int varIdx3 = numPrivateVertices + id;
				int curFactorIdx = numPairwiseVertexFactors + numPairwiseSplitFactors + numTripleCliqueFactors;
				int numLabels1 = varNodes[varIdx1].numLabels;
				int numLabels2 = varNodes[varIdx2].numLabels;
				int numLabels3 = varNodes[varIdx3].numLabels;

				factorNodes[curFactorIdx].numConfigs = numLabels1 * numLabels2 * numLabels3;
				factorNodes[curFactorIdx].bases.push_back(numLabels2 * numLabels3);
				factorNodes[curFactorIdx].bases.push_back(numLabels3);
				factorNodes[curFactorIdx].bases.push_back(1);
				factorNodes[curFactorIdx].varNbs.push_back(varIdx1);
				factorNodes[curFactorIdx].varNbs.push_back(varIdx2);
				factorNodes[curFactorIdx].varNbs.push_back(varIdx3);
				factorNodes[curFactorIdx].msgMFactorToVar.push_back(Message(numLabels1, 0.f));
				factorNodes[curFactorIdx].msgMFactorToVar.push_back(Message(numLabels2, 0.f));
				factorNodes[curFactorIdx].msgMFactorToVar.push_back(Message(numLabels3, 0.f));

				varNodes[varIdx1].factorNbs.push_back(curFactorIdx);
				varNodes[varIdx2].factorNbs.push_back(curFactorIdx);
				varNodes[varIdx3].factorNbs.push_back(curFactorIdx);
				varNodes[varIdx1].msgNVarToFactor.push_back(Message(numLabels1, 0.f));
				varNodes[varIdx2].msgNVarToFactor.push_back(Message(numLabels2, 0.f));
				varNodes[varIdx3].msgNVarToFactor.push_back(Message(numLabels3, 0.f));

				numTripleCliqueFactors++;
			}
		}
	}

	printf("INIT: candidateLabels.size() = %d\n", candidateLabels.size());
	this->candidateLabels = candidateLabels;
	this->varCoords.resize(3 * numTriangles);
	for (int id = 0; id < numTriangles; id++) {
		varCoords[3 * id + 0] = vertexCoords[triVertexInds[id][0]];
		varCoords[3 * id + 1] = vertexCoords[triVertexInds[id][1]];
		varCoords[3 * id + 2] = vertexCoords[triVertexInds[id][2]];
	}
	varCoords.insert(varCoords.end(), vertexCoords.begin(), vertexCoords.end());
	
	/*for (int id = 0; id < numTriangles; id++) {
		std::vector<cv::Point2i> &coordList = triPixelList[id];
		float R = 0, G = 0, B = 0;
		for (int i = 0; i < coordList.size(); i++) {
			cv::Vec3b &c = img.at<cv::Vec3b>(coordList[i].y, coordList[i].x);
			B += c[0];
			G += c[1];
			R += c[2];
		}
		R /= coordList.size();
		G /= coordList.size();
		B /= coordList.size();
		cv::Vec3b meanColor(B, G, R);
		triMeanColors[3 * id + 0] = meanColor;
		triMeanColors[3 * id + 1] = meanColor;
		triMeanColors[3 * id + 2] = meanColor;
	}*/

}

cv::Mat BP::DecodeSplittingImageFromBeliefs(int numRows, int numCols, std::vector<Probs> &allBeliefs)
{
	cv::Mat canvas(numRows, numCols, CV_8UC3);
	canvas.setTo(cv::Vec3b(0, 0, 0));
	int numPrivateVertices = this->candidateLabels.size();
	for (int i = numPrivateVertices; i < varCoords.size(); i++) {
		if (allBeliefs[i][1] < allBeliefs[i][0]) {
			cv::Point2d p = varCoords[i];
			cv::circle(canvas, p - cv::Point2d(0.5, 0.5), 1.5, cv::Scalar(0, 0, 255), 2);
		}
	}
	return canvas;
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
	bp.InitFromGridGraph(unaryCosts);
	bp.Run("tsukuba", outLabels);
#endif
}