#pragma once
#ifndef __BPONFACTORGRAPH_H__
#define __BPONFACTORGRAPH_H__


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cassert>
#include <ctime>
#include <vector>
#include <set>
#include <algorithm>

#include "SlantedPlane.h"
#include "MCImg.h"


typedef std::vector<float> Message;
typedef std::vector<float> Probs;

struct VarNode
{
	int						numLabels;
	std::vector<float>		pot;
	std::vector<int>		factorNbs;
	std::vector<Message>	msgNVarToFactor;
};

struct FactorNode
{
	int						numConfigs;
	std::vector<int>		varNbs;
	std::vector<Message>	msgMFactorToVar;
	std::vector<int>		bases;
};

class BPOnFG
{
public:
	std::vector<VarNode>	varNodes;
	std::vector<FactorNode> factorNodes;
	std::vector<Probs>		allBeliefs;



public:
	float RunNextIteration();

	void UpdateMessageNVarToFactor(int i, int alpha);

	void UpdateMessageMFactorToVar(int alpha, int i);

	std::vector<float>& MsgRefN(int i, int alpha);

	std::vector<float>& MsgRefM(int alpha, int i);

	std::vector<int> LinearIdToConfig(int linearStateId, int factorId);

	int LocalIdxInNbs(std::vector<int> &arr, int val);

	void NormalizeMessage(Message &msg);

	void NormalizeBelief(Probs &p);

	virtual float FactorPotential(std::vector<int> &varInds, std::vector<int> &config) = 0;
};

class RegularGridBPOnFG : public BPOnFG
{
public:
	enum FactorType { LIEONHORIZONTAL, LIEONVERTICAL };

public:
	void InitFromGridGraph(MCImg<float> &unaryCosts);

	float FactorPotential(std::vector<int> &varInds, std::vector<int> &config);

	int GetFactorId(int y, int x, int numRows, int numCols, FactorType factorType);

	cv::Mat DecodeDisparityFromBeliefs(int numRows, int numCols, std::vector<Probs> &allBeliefs);

	void Run(std::string rootFolder, int maxIters = 200, float tol = 1e-4);
};

class MeshStereoBPOnFG : public BPOnFG
{
public:
	std::vector<std::vector<SlantedPlane>>	candidateLabels;
	std::vector<cv::Point2d>				varCoords;
	std::vector<cv::Vec3b>					triMeanColors;
	std::vector<cv::Point2d>				vertexCoords;
	std::vector<std::vector<int>>			triVertexInds;
	std::vector<std::vector<cv::Point2i>>   triPixelList;

public:
	void InitFromTriangulation(int numRows, int numCols, int numDisps,
		std::vector<std::vector<SlantedPlane>> &candidateLabels, std::vector<std::vector<float>> &unaryCosts,
		std::vector<cv::Point2d> &vertexCoords, std::vector<std::vector<int>> &triVertexInds,
		std::vector<std::vector<cv::Point2i>> &triPixelList, cv::Mat &img);

	float FactorPotential(std::vector<int> &varInds, std::vector<int> &config);

	cv::Mat DecodeSplitMapFromBeliefs(int numRows, int numCols, std::vector<Probs> &allBeliefs);

	cv::Mat DecodeSplittingImageFromBeliefs(int numRows, int numCols, std::vector<Probs> &allBeliefs);

	cv::Mat MeshStereoBPOnFG::DecodeDisparityMapFromBeliefs(int numRows, int numCols, std::vector<Probs> &allBeliefs,
		std::vector<std::vector<SlantedPlane>> &triVertexBestLabels, int subId);

	void Run(std::string rootFolder, int maxIters = 200, float tol = 1e-4);
};

#endif