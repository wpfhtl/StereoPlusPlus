#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <set>

#include "StereoAPI.h"
#include "SlantedPlane.h"
#include "Timer.h"


#define PATCHRADIUS				17
#define PATCHWIDTH				35
#define GRANULARITY				0.25
#define SIMILARITYGAMMA			10
#define MAXPATCHMATCHITERS		15

static enum CostAggregationType { REGULAR_GRID, TOP50 };
static enum MatchingCostType	{ ADGRADIENT, ADCENSUS };

static CostAggregationType	gCostAggregationType	= REGULAR_GRID;
static MatchingCostType		gMatchingCostType		= ADCENSUS;

static MCImg<float>			gDsiL;
static MCImg<float>			gDsiR;
static MCImg<float>			gSimWeightsL;
static MCImg<float>			gSimWeightsR;
static MCImg<SimVector>		gSimVecsL;
static MCImg<SimVector>		gSimVecsR;



struct SortByRowCoord {
	bool operator ()(const std::pair<cv::Point2d, int> &a, const std::pair<cv::Point2d, int> &b) const {
		return a.first.y < b.first.y;
	}
};

struct SortByColCoord {
	bool operator ()(const std::pair<cv::Point2d, int> &a, const std::pair<cv::Point2d, int> &b) const {
		return a.first.x < b.first.x;
	}
};

static bool InBound(int y, int x, int numRows, int numCols)
{
	return 0 <= y && y < numRows && 0 <= x && x < numCols;
}

float sign(cv::Point2d p1, cv::Point2d p2, cv::Point2d p3)
{
	return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

bool PointInTriangle(cv::Point2d pt, cv::Point2d v1, cv::Point2d v2, cv::Point2d v3)
{
	bool b1, b2, b3;

	b1 = sign(pt, v1, v2) < 0.0f;
	b2 = sign(pt, v2, v3) < 0.0f;
	b3 = sign(pt, v3, v1) < 0.0f;

	return ((b1 == b2) && (b2 == b3));
}

static float PatchMatchSlantedPlaneCost(int yc, int xc, SlantedPlane &slantedPlane, int sign)
{
	MCImg<float> &dsi = (sign == -1 ? gDsiL : gDsiR);
	MCImg<float> &simWeights = (sign == -1 ? gSimWeightsL : gSimWeightsR);
	MCImg<SimVector> &simVecs = (sign == -1 ? gSimVecsL : gSimVecsR);

	int numRows = dsi.h, numCols = dsi.w, maxDisp = dsi.n - 1;
	const int STRIDE = 1;
	float totalCost = 0.f;

	if (gCostAggregationType == REGULAR_GRID) {
		MCImg<float> w(PATCHWIDTH, PATCHWIDTH, 1, simWeights.line(yc * numCols + xc));
		for (int y = yc - PATCHRADIUS, id = 0; y <= yc + PATCHRADIUS; y += STRIDE) {
			for (int x = xc - PATCHRADIUS; x <= xc + PATCHRADIUS; x += STRIDE, id++) {
				if (InBound(y, x, numRows, numCols)) {
					id = (y - (yc - PATCHRADIUS)) * PATCHWIDTH + (x - (xc - PATCHRADIUS));
					float d = slantedPlane.ToDisparity(y, x);
					int level = 0.5 + d / GRANULARITY;
					level = std::max(0, std::min(maxDisp, level));
					totalCost += w.data[id] * dsi.get(y, x)[level];
				}
			}
		}
	}
	else if (gCostAggregationType == TOP50) {
		SimVector &simVec = simVecs[yc][xc];
		for (int i = 0; i < SIMVECTORSIZE; i++) {
			int y = simVec.pos[i].y;
			int x = simVec.pos[i].x;
			float d = slantedPlane.ToDisparity(y, x);
			int level = 0.5 + d / GRANULARITY;
			level = std::max(0, std::min(maxDisp, level));
			totalCost += simVec.w[i] * dsi.get(y, x)[level];
		}
	}

	return totalCost;
}

static void ImproveGuess(int y, int x, SlantedPlane &oldGuess, SlantedPlane &newGuess, float &bestCost, int sign)
{
	float newCost = PatchMatchSlantedPlaneCost(y, x, newGuess, sign);
	if (newCost < bestCost) {
		bestCost = newCost;
		oldGuess = newGuess;
	}
}

static void PropagateAndRandomSearch(int id, int sign, float maxDisp, cv::Point2d &srcPos,
	std::vector<SlantedPlane> &slantedPlanes, std::vector<float> &bestCosts, std::vector<std::vector<int>> &nbIndices)
{
	int y = srcPos.y + 0.5;
	int x = srcPos.x + 0.5;

	// Spatial propgation
	std::vector<int> &nbIds = nbIndices[id];
	for (int i = 0; i < nbIds.size(); i++) {
		SlantedPlane newGuess = slantedPlanes[nbIds[i]];
		ImproveGuess(y, x, slantedPlanes[id], newGuess, bestCosts[id], sign);
	}

	// Random search
	float zRadius = maxDisp / 2.f;
	float nRadius = 1.f;
	while (zRadius >= 0.1f) {
		SlantedPlane newGuess = SlantedPlane::ConstructFromRandomPertube(slantedPlanes[id], y, x, nRadius, zRadius);
		ImproveGuess(y, x, slantedPlanes[id], newGuess, bestCosts[id], sign);
		zRadius /= 2.f;
		nRadius /= 2.f;
	}

}

static void ConstructNeighboringGraph(int numRows, int numCols, std::vector<cv::Point2d> &vertexCoords, std::vector<std::vector<int>> &triVertexInds, 
	std::vector<cv::Point2d> &baryCenters, std::vector<std::vector<int>> &nbIndices)
{
	const int numTriangles	= triVertexInds.size();
	const int numVertices	= vertexCoords.size();

	printf("numVertices = %d\n", numVertices);
	printf("numTriangles = %d\n", numTriangles);


	// Step 1 - Determine an ordering of the triangles
	baryCenters.resize(numTriangles);
	for (int i = 0; i < baryCenters.size(); i++) {
		cv::Point2d a = vertexCoords[triVertexInds[i][0]];
		cv::Point2d b = vertexCoords[triVertexInds[i][1]];
		cv::Point2d c = vertexCoords[triVertexInds[i][2]];
		baryCenters[i] = cv::Point((a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0);
	}

	std::vector<std::pair<cv::Point2d, int>> centroidIdPairs(baryCenters.size());
	for (int i = 0; i < baryCenters.size(); i++) {
		centroidIdPairs[i] = std::pair<cv::Point2d, int>(baryCenters[i], i);
	}


	// Divide the canvas by horizontal stripes and then determine the ordering of each stripe accordingly
	std::sort(centroidIdPairs.begin(), centroidIdPairs.end(), SortByRowCoord());
	double rowMargin = std::sqrt(2.0 * (numRows * numCols) / (double)numTriangles);
	printf("rowMargin = %.2lf\n", rowMargin);
	int headIdx = 0;
	for (double y = 0; y <= numRows; y += rowMargin) {
		int idx = headIdx;
		while (idx < numTriangles && centroidIdPairs[idx].first.y < y + rowMargin) {
			idx++;
		}
		if (headIdx < numTriangles) {	// to ensure that we do not have access violation at headIdx
			std::sort(&centroidIdPairs[headIdx], &centroidIdPairs[0] + idx, SortByColCoord());
		}
		headIdx = idx;
	}

	std::vector<std::vector<int>> tmp(numTriangles);
	for (int i = 0; i < numTriangles; i++) {
		baryCenters[i] = centroidIdPairs[i].first;
		tmp[i] = triVertexInds[centroidIdPairs[i].second];
	}
	triVertexInds = tmp;


	// Visualize and verify the ordering
	cv::Mat canvas(numRows, numCols, CV_8UC3);
	cv::Point2d halfOffset(0.5, 0.5);
#if 0
	cv::Point2d oldEndPt(-0.5, -0.5);
	int stepSz = numTriangles / 20;
	for (int i = 0; i < numTriangles; i += stepSz) {
		for (int j = i; j < i + stepSz && j < numTriangles; j++) {
			cv::Point2d newEndPt = centroidIdPairs[j].first - halfOffset;
			cv::line(canvas, oldEndPt, newEndPt, cv::Scalar(0, 0, 255, 255), 1, CV_AA);
			oldEndPt = newEndPt;
		}
		cv::imshow("process", canvas);
		cv::waitKey(0);
	}
#endif


	// Step 2 - Construct a neighboring graph for the triangles
	std::vector<std::set<int>> triIndSets((numRows + 1) * (numCols + 1));
	for (int i = 0; i < numTriangles; i++) {
		for (int j = 0; j < 3; j++) {
			cv::Point2d &p = vertexCoords[triVertexInds[i][j]];
			triIndSets[p.y * (numCols + 1) + p.x].insert(i);
		}
	}

	nbIndices.resize(numTriangles);
	for (int i = 0; i < numTriangles; i++) {
		// Merge the neighbors of the three vertices into one
		std::set<int> idxSet;
		for (int j = 0; j < 3; j++) {
			cv::Point2d &p = vertexCoords[triVertexInds[i][j]];
			std::set<int> &tmp = triIndSets[p.y * (numCols + 1) + p.x];
			idxSet.insert(tmp.begin(), tmp.end());
		}
		for (std::set<int>::iterator it = idxSet.begin(); it != idxSet.end(); it++) {
			if (i != *it) {
				nbIndices[i].push_back(*it);
			}
		}
	}

	
	// Output some statistics of the neighboring graph
	int total = 0, cnt = 0;
	for (int i = 0; i < triIndSets.size(); i++) {
		if (triIndSets[i].size() > 0) {
			total += triIndSets[i].size();
			cnt++;
		}
	}
	printf("Averge degree: %.2lf\n", (double)total / cnt);

	total = 0, cnt = 0;
	for (int i = 0; i < nbIndices.size(); i++) {
		total += nbIndices[i].size();
		cnt++;
	}
	printf("Averge neighbors: %.2lf\n", (double)total / cnt);

	// Visualize and verify neighboring graph
#if 0
	for (int retry = 0; retry < 100; retry++) {
		canvas.setTo(cv::Scalar(0, 0, 0));
		int id = rand() % numTriangles;
		cv::Point2d A = baryCenters[id] - halfOffset;
		for (int j = 0; j < nbIndices[id].size(); j++) {
			cv::Point2d B = baryCenters[nbIndices[id][j]];
			cv::line(canvas, A, B, cv::Scalar(0, 0, 255), 1, CV_AA);
		}
		printf("number of neighbors: %d\n", nbIndices[id].size());
		cv::imshow("neighbor relations", canvas);
		cv::waitKey(0);
	}
#endif
}

static void DeterminePixelOwnership(int numRows, int numCols, std::vector<cv::Point2d> &vertexCoords,
	std::vector<std::vector<int>> &triVertexInds, std::vector<std::vector<cv::Point2d>> &triPixelLists)
{
	const cv::Point2d halfOffset(0.5, 0.5);
	int numTriangles = triVertexInds.size();
	triPixelLists.resize(numTriangles);
	
	for (int i = 0; i < numTriangles; i++) {
		cv::Point2d &a = vertexCoords[triVertexInds[i][0]];
		cv::Point2d &b = vertexCoords[triVertexInds[i][1]];
		cv::Point2d &c = vertexCoords[triVertexInds[i][2]];

		int xL = std::min(a.x, b.x); xL = std::min((double)xL, c.x);
		int xR = std::max(a.x, b.x); xR = std::max((double)xR, c.x);
		int yU = std::min(a.y, b.y); yU = std::min((double)yU, c.y);
		int yD = std::max(a.y, b.y); yD = std::max((double)yD, c.y);
	
		for (int y = std::max(0, yU); y < yD && y < numRows; y++) {
			for (int x = std::max(0, xL); x < xR && x < numCols; x++) {
				if (PointInTriangle(cv::Point2d(x, y) + halfOffset, a, b, c)) {
					triPixelLists[i].push_back(cv::Point2d(x, y));
				}
			}
		}
	}
	
#if 0
	// test and verify the ownership
	cv::Mat canvas(numRows, numCols, CV_8UC3);
	for (int retry = 0; retry < 100; retry++) {
		canvas.setTo(cv::Scalar(0, 0, 0));
		int id = rand() % numTriangles;
		std::vector<cv::Point2d> &pixelList = triPixelLists[id];
		for (int j = 0; j < pixelList.size(); j++) {
			int y = pixelList[j].y;
			int x = pixelList[j].x;
			canvas.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
		}
		cv::imshow("triangle", canvas);
		cv::waitKey(0);
	}
#endif
}

static cv::Mat TriangleLabelToDisparityMap(int numRows, int numCols, std::vector<SlantedPlane> &slantedPlanes, 
	std::vector<std::vector<cv::Point2d>> &triPixelLists)
{
	cv::Mat dispMap(numRows, numCols, CV_32FC1);
	for (int id = 0; id < triPixelLists.size(); id++) {
		std::vector<cv::Point2d> &pixelList = triPixelLists[id];
		for (int i = 0; i < pixelList.size(); i++) {
			int y = pixelList[i].y + 0.5;
			int x = pixelList[i].x + 0.5;
			dispMap.at<float>(y, x) = slantedPlanes[id].ToDisparity(y, x);
		}
	}
	return dispMap;
}

void RunPatchMatchOnTriangles(std::string rootFolder, cv::Mat &imL, cv::Mat &imR)
{
	int numRows = imL.rows, numCols = imL.cols;
	int numDisps, maxDisp, visualizeScale;
	SetupStereoParameters(rootFolder, numDisps, maxDisp, visualizeScale);

	std::vector<cv::Point2d> vertexCoords;
	std::vector<std::vector<int>> triVertexInds;
	Triangulate2DImage(imL, vertexCoords, triVertexInds);

	std::vector<cv::Point2d> baryCenters;
	std::vector<std::vector<int>> nbIndices;
	ConstructNeighboringGraph(numRows, numCols, vertexCoords, triVertexInds, baryCenters, nbIndices);

	std::vector<std::vector<cv::Point2d>> triPixelLists;
	DeterminePixelOwnership(numRows, numCols, vertexCoords, triVertexInds, triPixelLists);

	int numTriangles = baryCenters.size();
	std::vector<SlantedPlane> slantedPlanes(numTriangles);
	std::vector<float> bestCosts(numTriangles);

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

	for (int id = 0; id < numTriangles; id++) {
		slantedPlanes[id] = SlantedPlane::ConstructFromRandomInit(baryCenters[id].y, baryCenters[id].x, maxDisp);
		bestCosts[id] = PatchMatchSlantedPlaneCost(baryCenters[id].y + 0.5, baryCenters[id].x + 0.5, slantedPlanes[id], -1);
	}

	std::vector<int> idList(numTriangles);
	for (int i = 0; i < numTriangles; i++) {
		idList[i] = i;
	}

	

	for (int round = 0; round < MAXPATCHMATCHITERS; round++) {

		//#pragma omp parallel for
		for (int i = 0; i < numTriangles; i++) {
			int id = idList[i];
			PropagateAndRandomSearch(id, -1, maxDisp, baryCenters[id], slantedPlanes, bestCosts, nbIndices);
		}

		cv::Mat dispL = TriangleLabelToDisparityMap(numRows, numCols, slantedPlanes, triPixelLists);
		EvaluateDisparity(rootFolder, dispL, 0.5f);

		std::reverse(idList.begin(), idList.end());
	}

	
}

void TestPatchMatchOnTriangles()
{
	std::string rootFolder = "Baby1";

	cv::Mat imL = cv::imread("D:/data/stereo/" + rootFolder + "/im2.png");
	cv::Mat imR = cv::imread("D:/data/stereo/" + rootFolder + "/im6.png");

	RunPatchMatchOnTriangles(rootFolder, imL, imR);
}