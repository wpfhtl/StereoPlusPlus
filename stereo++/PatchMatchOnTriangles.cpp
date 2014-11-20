#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <set>
#include <algorithm>
#include <iostream>

#include "StereoAPI.h"
#include "SlantedPlane.h"
#include "Timer.h"
#include "BPOnFactorGraph.h"
#include "ReleaseAssert.h"
#include "PostProcess.h"

#define PROCESS_RIGHT_VIEW
#define WATER_TIGHT_MESH
#define SLIC_TRIANGULATION

extern int					PATCHRADIUS;
extern int					PATCHWIDTH;
extern float				GRANULARITY;
extern float				SIMILARITY_GAMMA;
extern int					MAX_PATCHMATCH_ITERS;
extern std::string			ROOTFOLDER;

extern enum CostAggregationType { GRID, TOP50 };
extern enum MatchingCostType	{ ADGRADIENT, ADCENSUS };

extern CostAggregationType	gCostAggregationType;
extern MatchingCostType		gMatchingCostType;

//extern MCImg<float>			gDsiL;
//extern MCImg<float>			gDsiR;
extern MCImg<float>			gSimWeightsL;
extern MCImg<float>			gSimWeightsR;
extern MCImg<SimVector>		gSimVecsL;
extern MCImg<SimVector>		gSimVecsR;



//void SaveStdVector(std::string filePath, std::vector<SlantedPlane> &slantedPlanes)
//{
//	int numPlanes = slantedPlanes.size();
//	FILE *fid = fopen(filePath.c_str(), "wb");
//	fwrite(&slantedPlanes[0], sizeof(SlantedPlane), numPlanes, fid);
//	fclose(fid);
//}

static struct SortByRowCoord {
	bool operator ()(const std::pair<cv::Point2f, int> &a, const std::pair<cv::Point2f, int> &b) const {
		return a.first.y < b.first.y;
	}
};

static struct SortByColCoord {
	bool operator ()(const std::pair<cv::Point2f, int> &a, const std::pair<cv::Point2f, int> &b) const {
		return a.first.x < b.first.x;
	}
};

static bool InBound(int y, int x, int numRows, int numCols)
{
	return 0 <= y && y < numRows && 0 <= x && x < numCols;
}

#if 1
float sign(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3)
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

bool PointInTriangleStrict(cv::Point2f pt, cv::Point2f v1, cv::Point2f v2, cv::Point2f v3)
{
	//bool b1, b2, b3;

	//b1 = sign(pt, v1, v2) < 0.0f;
	//b2 = sign(pt, v2, v3) < 0.0f;
	//b3 = sign(pt, v3, v1) < 0.0f;

	//return ((b1 == b2) && (b2 == b3));

	float c1 = sign(pt, v1, v2);
	float c2 = sign(pt, v2, v3);
	float c3 = sign(pt, v3, v1);
	bool b1 = c1 < 0.f;
	bool b2 = c2 < 0.f;
	bool b3 = c3 < 0.f;
	const float eps = 1e-8;
	return ((b1 == b2) && (b2 == b3) && (std::abs(c1) > eps && std::abs(c2) > eps && std::abs(c3) > eps));
}
#else
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
#endif

static void ImproveGuess(int y, int x, SlantedPlane &oldGuess, SlantedPlane &newGuess, float &bestCost, int sign)
{
	float newCost = PatchMatchSlantedPlaneCost(y, x, newGuess, sign);
	if (newCost < bestCost) {
		bestCost = newCost;
		oldGuess = newGuess;
	}
}

static void PropagateAndRandomSearch(int id, int sign, float maxDisp, cv::Point2f &srcPos,
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
	for (int retry = 0; retry < 1; retry++) {
		float zRadius = maxDisp / 2.f;
		float nRadius = 1.f;
		while (zRadius >= 0.1f) {
			SlantedPlane newGuess = SlantedPlane::ConstructFromRandomPertube(slantedPlanes[id], y, x, nRadius, zRadius);
			ImproveGuess(y, x, slantedPlanes[id], newGuess, bestCosts[id], sign);
			zRadius /= 2.f;
			nRadius /= 2.f;
		}
	}

}

void ConstructNeighboringTriangleGraph(int numRows, int numCols, std::vector<cv::Point2f> &vertexCoords, 
	std::vector<std::vector<int>> &triVertexInds, std::vector<cv::Point2f> &baryCenters, 
	std::vector<std::vector<int>> &nbIndices)
{
	const int numTriangles = triVertexInds.size();
	const int numVertices = vertexCoords.size();

	printf("numVertices = %d\n", numVertices);
	printf("numTriangles = %d\n", numTriangles);


	// Step 1 - Determine an ordering of the triangles
	baryCenters.resize(numTriangles);
	for (int i = 0; i < baryCenters.size(); i++) {
		cv::Point2f a = vertexCoords[triVertexInds[i][0]];
		cv::Point2f b = vertexCoords[triVertexInds[i][1]];
		cv::Point2f c = vertexCoords[triVertexInds[i][2]];
		baryCenters[i] = cv::Point((a.x + b.x + c.x) / 3.0 - 0.5, (a.y + b.y + c.y) / 3.0 - 0.5);
	}

	std::vector<std::pair<cv::Point2f, int>> centroidIdPairs(baryCenters.size());
	for (int i = 0; i < baryCenters.size(); i++) {
		centroidIdPairs[i] = std::pair<cv::Point2f, int>(baryCenters[i], i);
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
#if 0
	cv::Point2f oldEndPt(-0.5, -0.5);
	int stepSz = numTriangles / 20;
	for (int i = 0; i < numTriangles; i += stepSz) {
		for (int j = i; j < i + stepSz && j < numTriangles; j++) {
			cv::Point2f newEndPt = centroidIdPairs[j].first;
			cv::line(canvas, oldEndPt, newEndPt, cv::Scalar(0, 0, 255, 255), 1, CV_AA);
			oldEndPt = newEndPt;
		}
		cv::imshow("process", canvas);
		cv::waitKey(0);
	}
#endif


	// Step 2 - Construct a neighboring graph for the triangles
	printf("numRows = %d\n", numRows);
	printf("numCols = %d\n", numCols);
	std::vector<std::set<int>> triIndSets((numRows + 1) * (numCols + 1));
	for (int i = 0; i < numTriangles; i++) {
		for (int j = 0; j < 3; j++) {

			cv::Point2f &p = vertexCoords[triVertexInds[i][j]];
			if (p.y * (numCols + 1) + p.x >= triIndSets.size()) {
				std::cout << "index out of bound: p = " << p << "\n";
			}
			triIndSets[p.y * (numCols + 1) + p.x].insert(i);
		}
	}

	nbIndices.resize(numTriangles);
	for (int i = 0; i < numTriangles; i++) {
		// Merge the neighbors of the three vertices into one
		std::set<int> idxSet;
		for (int j = 0; j < 3; j++) {
			cv::Point2f &p = vertexCoords[triVertexInds[i][j]];
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
		cv::Point2f A = baryCenters[id];
		for (int j = 0; j < nbIndices[id].size(); j++) {
			cv::Point2f B = baryCenters[nbIndices[id][j]];
			cv::line(canvas, A, B, cv::Scalar(0, 0, 255), 1, CV_AA);
		}
		printf("number of neighbors: %d\n", nbIndices[id].size());
		cv::imshow("neighbor relations", canvas);
		cv::waitKey(0);
	}
#endif
}

void DeterminePixelOwnership(int numRows, int numCols, std::vector<cv::Point2f> &vertexCoords,
	std::vector<std::vector<int>> &triVertexInds, std::vector<std::vector<cv::Point2i>> &triPixelLists)
{
	const cv::Point2f halfOffset(0.5, 0.5);
	int numTriangles = triVertexInds.size();
	triPixelLists.resize(numTriangles);

	for (int i = 0; i < numTriangles; i++) {
		cv::Point2f &a = vertexCoords[triVertexInds[i][0]];
		cv::Point2f &b = vertexCoords[triVertexInds[i][1]];
		cv::Point2f &c = vertexCoords[triVertexInds[i][2]];

		int xL = std::min(a.x, b.x); xL = std::min((float)xL, c.x);
		int xR = std::max(a.x, b.x); xR = std::max((float)xR, c.x);
		int yU = std::min(a.y, b.y); yU = std::min((float)yU, c.y);
		int yD = std::max(a.y, b.y); yD = std::max((float)yD, c.y);

		for (int y = std::max(0, yU - 2); y < yD + 2 && y < numRows; y++) {
			for (int x = std::max(0, xL - 2); x < xR + 2 && x < numCols; x++) {
				if (PointInTriangle(cv::Point2f(x, y) + halfOffset, a, b, c)) {
					triPixelLists[i].push_back(cv::Point2i(x, y));
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
		std::vector<cv::Point2f> &pixelList = triPixelLists[id];
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

cv::Mat TriangleLabelToDisparityMap(int numRows, int numCols, std::vector<SlantedPlane> &slantedPlanes,
	std::vector<std::vector<cv::Point2i>> &triPixelLists)
{
	cv::Mat dispMap(numRows, numCols, CV_32FC1);
	for (int id = 0; id < triPixelLists.size(); id++) {
		std::vector<cv::Point2i> &pixelList = triPixelLists[id];
		for (int i = 0; i < pixelList.size(); i++) {
			int y = pixelList[i].y;
			int x = pixelList[i].x;
			dispMap.at<float>(y, x) = slantedPlanes[id].ToDisparity(y, x);
		}
	}
	return dispMap;
}

static void GenerateMeshStereoCandidateLabels(int numRows, int numCols, int numDisps, std::vector<cv::Point2f> &baryCenters,
	std::vector<SlantedPlane> &slantedPlanes, std::vector<cv::Point2f> &vertexCoords, std::vector<std::vector<int>> &triVertexInds,
	std::vector<std::vector<int>> &nbIndices, std::vector<std::vector<SlantedPlane>> &candidateLabels, std::vector<std::vector<float>> &unaryCosts)
{
	int numTriangles = baryCenters.size();
	int numPrivateVertices = 3 * numTriangles;
	candidateLabels.resize(numPrivateVertices);

	printf("Step 1.1 - Generate from spatial propagation ...\n");
	for (int id = 0; id < numTriangles; id++) {
		std::vector<SlantedPlane> labelSet;
		// Label from its own triangle
		labelSet.push_back(slantedPlanes[id]);
		// Labels from neighboring triangles
		for (int i = 0; i < nbIndices[id].size(); i++) {
			int nbId = nbIndices[id][i];
			labelSet.push_back(slantedPlanes[nbId]);
		}
		candidateLabels[3 * id + 0] = labelSet;
		candidateLabels[3 * id + 1] = labelSet;
		candidateLabels[3 * id + 2] = labelSet;
	}

	printf("Step 1.2 - Generate from intra group random search ...\n");
	MCImg<std::vector<std::pair<int, int>>> triIndSets(numRows + 1, numCols + 1);
	for (int id = 0; id < numTriangles; id++) {
		for (int j = 0; j < 3; j++) {
			cv::Point2f &p = vertexCoords[triVertexInds[id][j]];
			triIndSets[p.y][(int)p.x].push_back(std::make_pair(id, j));
		}
	}

	const int RAND_HALF = RAND_MAX / 2;
	for (int y = 0; y < numRows + 1; y++) {
		for (int x = 0; x < numCols + 1; x++) {

			std::vector<std::pair<int, int>> &triIds = triIndSets[y][x];
			if (!triIds.empty()) {
				// Vertices in the same group share the same depth
				float meanDisp = 0.f;
				for (int k = 0; k < triIds.size(); k++) {
					int id = triIds[k].first;
					meanDisp += slantedPlanes[id].ToDisparity(y - 0.5f, x - 0.5f);
				}
				meanDisp /= (float)triIds.size();

				// Label from its own normal and the shared disparity
				for (int k = 0; k < triIds.size(); k++) {
					int id = triIds[k].first;
					SlantedPlane &p = slantedPlanes[id];
					SlantedPlane candidate = SlantedPlane::ConstructFromNormalDepthAndCoord(
						p.nx, p.ny, p.nz, meanDisp, y - 0.5, x - 0.5);
					candidateLabels[3 * id + triIds[k].second].push_back(candidate);
				}

				float zRadius = (numDisps - 1) / 2.f;
				float nRadius = 1.f;
				while (zRadius >= 0.1f) {
					float zNew = meanDisp + zRadius * (((float)rand() - RAND_HALF) / RAND_HALF);
					for (int k = 0; k < triIds.size(); k++) {
						int id = triIds[k].first;
						float nx = slantedPlanes[id].nx + nRadius * (((float)rand() - RAND_HALF) / RAND_HALF);
						float ny = slantedPlanes[id].ny + nRadius * (((float)rand() - RAND_HALF) / RAND_HALF);
						float nz = slantedPlanes[id].nz + nRadius * (((float)rand() - RAND_HALF) / RAND_HALF);
						SlantedPlane candidate = SlantedPlane::ConstructFromNormalDepthAndCoord(
							nx, ny, nz, zNew, y - 0.5f, x - 0.5f);
						candidateLabels[3 * id + triIds[k].second].push_back(candidate);
					}
					zRadius /= 2.f;
					nRadius /= 2.f;
				}
			}
		}
	}

	// Store the cost of corresponding labels
	unaryCosts.resize(numPrivateVertices);
	for (int k = 0; k < numPrivateVertices; k++) {
		int id = k / 3;
		int yc = baryCenters[id].y + 0.5;
		int xc = baryCenters[id].x + 0.5;
		unaryCosts[k].resize(candidateLabels[k].size());
		for (int i = 0; i < candidateLabels[k].size(); i++) {
			unaryCosts[k][i] = PatchMatchSlantedPlaneCost(yc, xc, candidateLabels[k][i], -1);
		}
	}
}



void PatchMatchOnTrianglePostProcess(int numRows, int numCols,
	std::vector<SlantedPlane> &slantedPlanesL, std::vector<SlantedPlane> &slantedPlanesR,
	std::vector<cv::Point2f> &baryCentersL, std::vector<cv::Point2f> &baryCentersR,
	std::vector<std::vector<int>>& nbIndicesL, std::vector<std::vector<int>>& nbIndicesR,
	std::vector<std::vector<cv::Point2i>> &triPixelListsL, std::vector<std::vector<cv::Point2i>> &triPixelListsR,
	cv::Mat &dispL, cv::Mat &dispR)
{
	// Step 1 - CrossCheck
	dispL = TriangleLabelToDisparityMap(numRows, numCols, slantedPlanesL, triPixelListsL);
	dispR = TriangleLabelToDisparityMap(numRows, numCols, slantedPlanesR, triPixelListsR);

	cv::Mat validPixelMapL = CrossCheck(dispL, dispR, -1);
	cv::Mat validPixelMapR = CrossCheck(dispR, dispL, +1);

	std::vector<float> confidenceL = DetermineConfidence(validPixelMapL, triPixelListsL);
	std::vector<float> confidenceR = DetermineConfidence(validPixelMapR, triPixelListsR);

	cv::Mat confidenceImgL = DrawSegmentConfidenceMap(numRows, numCols, confidenceL, triPixelListsL);
	//cv::imshow("confidenceL", confidenceImgL);

	// Step 2 - Occlusion Filling
	// Replace the low-confidence triangles with their high-confidence neighbors
	SegmentOcclusionFilling(numRows, numCols, slantedPlanesL, baryCentersL, nbIndicesL, confidenceL, triPixelListsL);
	SegmentOcclusionFilling(numRows, numCols, slantedPlanesR, baryCentersR, nbIndicesR, confidenceR, triPixelListsR);

	// Step 3 - WMF
	// Finally, an optional pixelwise filtering
	// currently left empty.

	// Step 4 - Ouput disparity
	dispL = TriangleLabelToDisparityMap(numRows, numCols, slantedPlanesL, triPixelListsL);
	dispR = TriangleLabelToDisparityMap(numRows, numCols, slantedPlanesR, triPixelListsR);

	//std::vector<std::pair<std::string, void*>> auxParams;
	//auxParams.push_back(std::pair<std::string, void*>("triImg", &confidenceImgL));
	//EvaluateDisparity(ROOTFOLDER, dispL, 0.5f/*, auxParams*/);
}


void PatchMatchOnPixelPostProcess(std::vector<SlantedPlane> &slantedPlanesL, std::vector<SlantedPlane> &slantedPlanesR,
	std::vector<std::vector<cv::Point2i>> &triPixelListsL, std::vector<std::vector<cv::Point2i>> &triPixelListsR,
	cv::Mat &imL, cv::Mat &imR, cv::Mat &dispL, cv::Mat &dispR)
{
	int numRows = imL.rows, numCols = imL.cols;
	MCImg<SlantedPlane> pixelwiseSlantedPlanesL(numRows, numCols);
	MCImg<SlantedPlane> pixelwiseSlantedPlanesR(numRows, numCols);

	for (int id = 0; id < triPixelListsL.size(); id++) {
		std::vector<cv::Point2i> &pixelList = triPixelListsL[id];
		for (int i = 0; i < pixelList.size(); i++) {
			cv::Point2i &p = pixelList[i];
			pixelwiseSlantedPlanesL[p.y][p.x] = slantedPlanesL[id];
		}
	}
	for (int id = 0; id < triPixelListsR.size(); id++) {
		std::vector<cv::Point2i> &pixelList = triPixelListsR[id];
		for (int i = 0; i < pixelList.size(); i++) {
			cv::Point2i &p = pixelList[i];
			pixelwiseSlantedPlanesR[p.y][p.x] = slantedPlanesR[id];
		}
	}

	PatchMatchOnPixelPostProcess(pixelwiseSlantedPlanesL, pixelwiseSlantedPlanesR,
		imL, imR, dispL, dispR);
}

template<typename T>
void CostVolumeFromYamaguchi(std::string &leftFilePath, std::string &rightFilePath,
	MCImg<T> &dsiL, MCImg<T> &dsiR, int numDisps);

cv::Mat DeterminSplitMapByHeuristic(int numRows, int numCols,
	std::vector<SlantedPlane> &slantedPlanesL, std::vector<cv::Point2f> &vertexCoordsL,
	std::vector<std::vector<int>> &triVertexIndsL)
{
	cv::Mat splitMap = cv::Mat::zeros(numRows + 1, numCols + 1, CV_8UC1);
	std::vector<std::vector<float>> vertexDisps(vertexCoordsL.size());
	for (int id = 0; id < triVertexIndsL.size(); id++) {
		for (int j = 0; j < 3; j++) {
			int vertexInd = triVertexIndsL[id][j];
			cv::Point2f p = vertexCoordsL[vertexInd];
			float d = slantedPlanesL[id].ToDisparity(p.y - 0.5, p.x - 0.5);
			vertexDisps[vertexInd].push_back(d);
		}
	}

	for (int i = 0; i < vertexDisps.size(); i++) {
		float maxVal = -FLT_MAX;
		float minVal = FLT_MAX;
		for (int j = 0; j < vertexDisps[i].size(); j++) {
			maxVal = std::max(maxVal, vertexDisps[i][j]);
			minVal = std::min(minVal, vertexDisps[i][j]);
		}
		if (maxVal - minVal > 5.f) {
			splitMap.at<bool>(vertexCoordsL[i].y, vertexCoordsL[i].x) = true;
		}
	}

	return splitMap;
}

static cv::Point3f LinePlaneIntersection(cv::Point3f &M, cv::Point3f &N, cv::Point3f &A, cv::Point3f &B, cv::Point3f &C)
{
	// compute line plane instersection according to wikipedia:
	// http://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
	cv::Point3f l0 = M;
	cv::Point3f l = N - M;
	cv::Point3f p0 = A;
	cv::Point3f n = (A - B).cross(A - C);
	float d = (p0 - l0).dot(n) / l.dot(n);

	if (std::abs(l.dot(n)) < 1e-4) {
		printf("warning: line and plane are parallel, potential bugs!\n");
	}

	return l0 + d * l;
}

static void ProjectTextureOneView(float focalLen, float baselineLen, float alpha, int sign,
	std::vector<cv::Point3f> &vertexCoords, std::vector<std::vector<int>> &facetVertexIndsList,
	int numType1Factes,  // normal triangles are type1, corss depth triangles are type2
	cv::Mat &img, cv::Mat &imgRendered, cv::Mat &isProjected, cv::Mat &zBuffer, cv::Mat &isFromCrossDepthFacet)
{
	printf("entering........\n");
	int numRows = img.rows, numCols = img.cols;
	imgRendered = cv::Mat::zeros(numRows, numCols, CV_8UC3);
	isProjected = cv::Mat::zeros(numRows, numCols, CV_8UC1);
	zBuffer		= -1 * cv::Mat::ones(numRows, numCols, CV_32FC1);
	isFromCrossDepthFacet = cv::Mat::zeros(numRows, numCols, CV_8UC1);

	float XShift = (sign == -1 ? -alpha * baselineLen : (1 - alpha) * baselineLen);

	for (int i = 0; i < facetVertexIndsList.size(); i++) {
		for (int j = 0; j < 3; j++) {
			ASSERT(0 <= facetVertexIndsList[i][j] && facetVertexIndsList[i][j] < vertexCoords.size());
		}
		cv::Point3f A = vertexCoords[facetVertexIndsList[i][0]];
		cv::Point3f B = vertexCoords[facetVertexIndsList[i][1]];
		cv::Point3f C = vertexCoords[facetVertexIndsList[i][2]];

		// transform to the coordinate frame of the novel view.
		A.x += XShift;
		B.x += XShift;
		C.x += XShift;

		//printf("asdfasdfasdfasdf\n");
		// compute the projected 2d triangle abc in the novel view.
		cv::Point2f a(focalLen * A.x / A.z, focalLen * A.y / A.z);	// in texture coordinates
		cv::Point2f b(focalLen * B.x / B.z, focalLen * B.y / B.z);	// in texture coordinates
		cv::Point2f c(focalLen * C.x / C.z, focalLen * C.y / C.z);	// in texture coordinates

		// compute bounding box of triangle abc
		float xL = FLT_MAX, xR = -FLT_MAX;
		float yU = FLT_MAX, yD = -FLT_MAX;
		cv::Point2f abc[3] = { a, b, c };
		for (int j = 0; j < 3; j++) {
			xL = std::min(xL, abc[j].x);
			xR = std::max(xR, abc[j].x);
			yU = std::min(yU, abc[j].y);
			yD = std::max(yD, abc[j].y);
		}

		// compute the pixel correspondces in imL of pixel in triangle abc 
		for (int y = yU - 0.5; y <= yD + 0.5; y++) {
			for (int x = xL - 0.5; x <= xR + 0.5; x++) {
				if (InBound(y, x, numRows, numCols)
					&& PointInTriangle(cv::Point2f(x + 0.5, y + 0.5), a, b, c))
					//&& PointInTriangleStrict(cv::Point2f(x + 0.5, y + 0.5), a, b, c))
				{
					// P is the intersection of plane ABC and line (xZ/f, yZ/f, Z)
					cv::Point3f P = LinePlaneIntersection(
						cv::Point3f(0, 0, 0), cv::Point3f(x + 0.5, y + 0.5, focalLen), A, B, C);
					// transform back to the reference coordinate frame
					P.x -= XShift;
					float xRef = focalLen * P.x / P.z;   // xRef is in texture coordinate
					float yRef = focalLen * P.y / P.z;   // yRef is in texture coordinate
					ASSERT(std::abs((y + 0.5) - yRef) < 1e-4); // compare in texture coordinate


					if (i >= numType1Factes && zBuffer.at<float>(y, x) != -1.f) {
						// give low priority to cross depth facet.
						continue;
					}
					if (zBuffer.at<float>(y, x) == -1.f || P.z < zBuffer.at<float>(y, x)) {
						cv::Mat patch;
						/*ASSERT(InBound(y, xRef, numRows, numCols));*/
						if (InBound(y, xRef, numRows, numCols)) {
							cv::getRectSubPix(img, cv::Size(1, 1), cv::Point2f(xRef - 0.5, y), patch);
							//cv::getRectSubPix(img, cv::Size(1, 1), cv::Point2f(xRef - 0.5, y), patch);
							imgRendered.at<cv::Vec3b>(y, x) = patch.at<cv::Vec3b>(0, 0);
							isProjected.at<unsigned char>(y, x) += 1;
							zBuffer.at<float>(y, x) = P.z;

							if (i >= numType1Factes) {
								isFromCrossDepthFacet.at<unsigned char>(y, x) = 255;
							}
						}
						else {
							printf("not in bound: (y, xRef) = (%.2f, %.2f)\n", y, xRef);
						}
					}
				}
			}
		}
	}
	printf("leaving........\n");
}

static void ClusterDisparitiesOtsu(std::vector<float> &dispVals)
{
	// implement Otsu thresholding according to 
	// http://docs.opencv.org/trunk/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html

	// if disparity distribution is only one-mode, return only on cluster center
	std::sort(dispVals.begin(), dispVals.end());
	ASSERT(dispVals[dispVals.size() - 1] >= dispVals[0]);
	if (dispVals[dispVals.size() - 1] - dispVals[0] <= 5.f) {
		float mean = 0;
		for (int i = 0; i < dispVals.size(); i++) {
			mean += dispVals[i];
		}
		mean /= dispVals.size();
		dispVals.clear();
		dispVals.push_back(mean);
		return;
	}

	// otherwise return two cluster centers by otsu adaptive thresholding
	// by minimizing the weighted within-class variance
	const int N = 80;
	std::vector<float> P(N);
	for (int i = 0; i < dispVals.size(); i++) {
		int d = dispVals[i] + 0.5;

		if (!(0 <= d && d < N)) {
			printf("=================\n");
			for (int j = 0; j < dispVals.size(); j++) {
				printf("%.2f  ", dispVals[j]);
			}
			printf("\n=================\n");
		}
		d = std::max(0, std::min(d, N - 1));
		ASSERT(0 <= d && d < N);
		P[d] += 1;
	}

	std::vector<float> accSumP(N), accSumiP(N);
	for (int i = 0; i < N; i++) {
		P[i] /= dispVals.size();
		if (i == 0) {
			accSumP[0] = P[0];
			accSumiP[0] = 0 * P[0];

		}
		else {
			accSumP[i] = accSumP[i - 1] + P[i];
			accSumiP[i] = accSumiP[i - 1] + i * P[i];
		}
	}

	std::vector<float> q1(N), q2(N), mu1(N), mu2(N);
	for (int t = 0; t < N; t++) {
		q1[t] = accSumP[t];
		q2[t] = accSumP[N - 1] - accSumP[t];
		mu1[t] = accSumiP[t] / q1[t];
		mu2[t] = (accSumiP[N - 1] - accSumiP[t]) / q2[t];
	}


	float bestVariance = FLT_MAX;
	float threshold = 0;
	for (int t = 0; t < N - 1; t++) {
		float v1 = 0, v2 = 0;
		for (int i = 0; i <= t; i++) {
			v1 += (i - mu1[t]) * (i - mu1[t]) * P[i] / q1[t];
		}
		for (int i = t + 1; i < N; i++) {
			v2 += (i - mu2[t]) * (i - mu2[t]) * P[i] / q2[t];
		}
		float weightedWithinClassVariance = q1[t] * v1 + q2[t] * v2;
		if (weightedWithinClassVariance < bestVariance) {
			bestVariance = weightedWithinClassVariance;
			threshold = t;
		}
	}


	float loSum = 0, hiSum = 0, loCnt = 0, hiCnt = 0;
	for (int i = 0; i < dispVals.size(); i++) {
		if (dispVals[i] <= threshold + 0.5) {
			loSum += dispVals[i];
			loCnt += 1;
		}
		else {
			hiSum += dispVals[i];
			hiCnt += 1;
		}
	}

	if (!(loCnt > 0 && hiCnt > 0)) {
		printf("(loCnt > 0 && hiCnt > 0) VIOALTED!!!\n");
		printf("loCnt = %d\nhiCnt = %d\n", (int)loCnt, (int)hiCnt);
		printf("dispVals: ");
		for (int i = 0; i < dispVals.size(); i++) {
			printf("%.2f  ", dispVals[i]);
		}
		printf("\nthreshold = %.2f\n", threshold);
	}

	ASSERT(loCnt > 0 && hiCnt > 0);
	dispVals.clear();
	dispVals.push_back(loSum / loCnt);
	dispVals.push_back(hiSum / hiCnt);
}

static std::vector<std::vector<int>> ConstructNeighboring2DVertexGraph(int numRows, int numCols,
	std::vector<cv::Point2f> &vertexCoords, std::vector<std::vector<int>> &triVertexInds)
{
	std::vector<std::set<int>> nbGraph(vertexCoords.size());
	for (int id = 0; id < triVertexInds.size(); id++) {
		std::vector<int> &indices = triVertexInds[id];
		ASSERT(indices.size() == 3);
		for (int i = 0; i < 3; i++) {
			for (int j = i + 1; j < 3; j++) {
				int idI = indices[i];
				int idJ = indices[j];
				nbGraph[idI].insert(idJ);
				nbGraph[idJ].insert(idI);
			}
		}
	}

	std::vector<std::vector<int>> vertexNbGraph(vertexCoords.size());
	for (int i = 0; i < nbGraph.size(); i++) {
		vertexNbGraph[i] = std::vector<int>(nbGraph[i].begin(), nbGraph[i].end());
	}

	return vertexNbGraph;
}

static void BuildWaterTightMesh(int sign, int numRows, int numCols, float focalLen, float baselineLen,
	std::vector<cv::Point2f> &vertexCoords, std::vector<std::vector<int>> &triVertexInds, std::vector<SlantedPlane> &slantedPlanes,
	std::vector<cv::Point3f> &meshVertexCoords, std::vector<std::vector<int>> &facetVetexIndsList)
{
	// collect disparities at each vertex
	printf("collect disparities at each vertex ...\n");
	MCImg<std::vector<float>> anchorDisparitySets(numRows + 1, numCols + 1);
	for (int id = 0; id < triVertexInds.size(); id++) {
		for (int j = 0; j < 3; j++) {
			cv::Point2f p = vertexCoords[triVertexInds[id][j]];
			float d = slantedPlanes[id].ToDisparity(p.y - 0.5, p.x - 0.5);
			if (d > 1000) {
				std::cout << "d values too large:\n";
				std::cout << "d = " << d << "\n";
				std::cout << "p = " << p << "\n";
				printf("slantedPlane (a,b,c) = (%f,%f,%f)\n", slantedPlanes[id].a, slantedPlanes[id].b, slantedPlanes[id].c);
				printf("slantedPlane (nx,ny,nz) = (%f,%f,%f)\n", slantedPlanes[id].nx, slantedPlanes[id].ny, slantedPlanes[id].nz);

			}
			anchorDisparitySets[p.y][(int)p.x].push_back(d);
		}
	}

	// cluster each vertex's disparities into one or two clusters
	printf("cluster each vertex's disparities into one or two clusters...\n");
	meshVertexCoords.clear();
	MCImg<std::vector<std::pair<float, int>>> anchorDispIdPairSets(numRows + 1, numCols + 1);
	for (int y = 0; y <= numRows; y++) {
		for (int x = 0; x <= numCols; x++) {


			if (!anchorDisparitySets[y][x].empty()) {
				std::vector<float> tmpVec = anchorDisparitySets[y][x];
				ClusterDisparitiesOtsu(tmpVec);
				anchorDisparitySets[y][x] = tmpVec;
				/*for (int i = 0; i < anchorDisparitySets[y][x].size(); i++) {
					float d = anchorDisparitySets[y][x][i];
					meshVertexCoords.push_back(cv::Point3f(x, y, d));
					anchorDispIdPairSets[y][x].push_back(std::make_pair(d, meshVertexCoords.size() - 1));
				}*/
				for (int i = 0; i < tmpVec.size(); i++) {
					float d = tmpVec[i];
					meshVertexCoords.push_back(cv::Point3f(x, y, d));
					anchorDispIdPairSets[y][x].push_back(std::make_pair(d, meshVertexCoords.size() - 1));
				}
			}
		}
	}
	
	//return;
	// assigned each triangle's each vertex the new disparity value (i.e., the cluster center)
	printf("assigned each triangle's each vertex the new disparity value...\n");
	facetVetexIndsList.clear();
	for (int id = 0; id < triVertexInds.size(); id++) {

		facetVetexIndsList.push_back(std::vector<int>(3));
		for (int j = 0; j < 3; j++) {
			ASSERT(0 <= triVertexInds[id][j] && triVertexInds[id][j] < vertexCoords.size());
			cv::Point2f p = vertexCoords[triVertexInds[id][j]];
			cv::vector<std::pair<float, int>> &dispIdPairs = anchorDispIdPairSets[p.y][(int)p.x];
			ASSERT(1 <= dispIdPairs.size() && dispIdPairs.size() <= 2);

			float dispOriginal = slantedPlanes[id].ToDisparity(p.y - 0.5, p.x - 0.5);
			float bestDist = FLT_MAX;
			int bestId = -1;
			for (int i = 0; i < dispIdPairs.size(); i++) {
				float dist = std::abs(dispIdPairs[i].first - dispOriginal);
				if (dist < bestDist) {
					bestDist = dist;
					bestId = dispIdPairs[i].second; 
				}
			}
			if (bestId == -1) {
				printf("dispOriginal = %.2f\n", dispOriginal);
				printf("dispVals: ");
				for (int i = 0; i < dispIdPairs.size(); i++) {
					printf("%.2f  ", dispIdPairs[i].first);
				}
				printf("triangle id = %d\n", id);
				printf("(y, x) = (%.2f, %.2f)\n", p.y, p.x);
				printf("\n");
			}
			ASSERT(bestId != -1);

			facetVetexIndsList[facetVetexIndsList.size() - 1][j] = bestId;
			
		}
	}
	//return;

#ifdef WATER_TIGHT_MESH
	// for each edge of the triangulation, add one or two triangles accordingly 
	// if the edge is on depth-discontinuity.
	printf("add triangles at discontinuiteis. ....\n");
	std::vector<std::vector<int>> vertexNbGraph = ConstructNeighboring2DVertexGraph(numRows, numCols, vertexCoords, triVertexInds);
	for (int i = 0; i < vertexNbGraph.size(); i++) {
		int idI = i;
		for (int j = 0; j < vertexNbGraph[i].size(); j++) {
			int idJ = vertexNbGraph[i][j];
			if (idI < idJ) {
				cv::Point2i p = vertexCoords[idI];
				cv::Point2i q = vertexCoords[idJ];
				std::vector<std::pair<float, int>> &dispIdPairsI = anchorDispIdPairSets[p.y][p.x];
				std::vector<std::pair<float, int>> &dispIdPairsJ = anchorDispIdPairSets[q.y][q.x];
				if (dispIdPairsI.size() + dispIdPairsJ.size() == 3) {
					// one of the vertices is splitted into two, add one triangle
					facetVetexIndsList.push_back(std::vector<int>());
					for (int k = 0; k < dispIdPairsI.size(); k++) {
						facetVetexIndsList[facetVetexIndsList.size() - 1].push_back(dispIdPairsI[k].second);
					}
					for (int k = 0; k < dispIdPairsJ.size(); k++) {
						facetVetexIndsList[facetVetexIndsList.size() - 1].push_back(dispIdPairsJ[k].second);
					}
				}
				else if (dispIdPairsI.size() + dispIdPairsJ.size() == 4) {
					// both of the vertices is splitted into two, add two triangles
					facetVetexIndsList.push_back(std::vector<int>());
					facetVetexIndsList[facetVetexIndsList.size() - 1].push_back(dispIdPairsI[0].second);
					facetVetexIndsList[facetVetexIndsList.size() - 1].push_back(dispIdPairsI[1].second);
					facetVetexIndsList[facetVetexIndsList.size() - 1].push_back(dispIdPairsJ[0].second);

					facetVetexIndsList.push_back(std::vector<int>());
					facetVetexIndsList[facetVetexIndsList.size() - 1].push_back(dispIdPairsI[1].second);
					facetVetexIndsList[facetVetexIndsList.size() - 1].push_back(dispIdPairsJ[0].second);
					facetVetexIndsList[facetVetexIndsList.size() - 1].push_back(dispIdPairsJ[1].second);
				}
			}
		}
	}
#endif

	printf("meshVertexCoords is in (x,y,d) form, project them to (X, Y, Z) form. ....\n");
	// meshVertexCoords is in (x,y,d) form, project them to (X, Y, Z) form
	for (int i = 0; i < meshVertexCoords.size(); i++) {
		//meshVertexCoords[i].z += 240;
		meshVertexCoords[i].z = std::max(4.f, std::min(79.f, meshVertexCoords[i].z));
	}
	std::vector<cv::Point3f> meshVertexCoordsXyd = meshVertexCoords;
	for (int i = 0; i < meshVertexCoords.size(); i++) {
		cv::Point3f &p = meshVertexCoords[i];
		float Z = focalLen * baselineLen / p.z;
		float X = p.x * Z / focalLen;
		float Y = p.y * Z / focalLen;
		meshVertexCoords[i] = cv::Point3f(X, Y, Z);
	}
	
	// meshVertexCoordsXyd are in texture coordinates, serialize them.
	void SaveVectorVectorInt(std::string filePath, std::vector<std::vector<int>> &data, std::string mode);
	void SaveVectorPoint3f(std::string filePath, std::vector<cv::Point3f> &vertices, std::string mode);
	if (sign == -1) {
		SaveVectorPoint3f("d:/meshVertexCoordsXydL.txt", meshVertexCoordsXyd, "w");
		SaveVectorVectorInt("d:/facetVetexIndsListL.txt", facetVetexIndsList, "w");
	}
	else {
		SaveVectorPoint3f("d:/meshVertexCoordsXydR.txt", meshVertexCoordsXyd, "w");
		SaveVectorVectorInt("d:/facetVetexIndsListR.txt", facetVetexIndsList, "w");
	}
	



	printf("saving mesh to ply .....\n");
	void SaveMeshToPly(std::string plyFilePath, int numRows, int numCols, float focalLen, float baselineLen,
		std::vector<cv::Point3f> &meshVertexCoordsXyd, std::vector<std::vector<int>> &facetVetexIndsList,
		std::string textureFilePath, bool showInstantly = false);
	std::string filePathTextureImage;
	/*std::string filePathPly = "d:/data/stereo/" + ROOTFOLDER + "/waterTightMesh.ply";*/
	std::string filePathPly;
	std::string filePathSplittingMap;
	if (sign == -1) {
		filePathPly = "D:/data/Exp13_GenerateMesh/" + ROOTFOLDER + "_meshL.ply";
		filePathSplittingMap = "D:/data/Exp13_GenerateMesh/" + ROOTFOLDER + "_splittingL.png";
		filePathTextureImage = "D:/data/Midd2/ThirdSize/" + ROOTFOLDER + "/view1.png";
	}
	else {
		filePathPly = "D:/data/Exp13_GenerateMesh/" + ROOTFOLDER + "_meshR.ply";
		filePathSplittingMap = "D:/data/Exp13_GenerateMesh/" + ROOTFOLDER + "_splittingR.png";
		filePathTextureImage = "D:/data/Midd2/ThirdSize/" + ROOTFOLDER + "/view5.png";
	}
	
	SaveMeshToPly(filePathPly, numRows, numCols, focalLen, baselineLen, meshVertexCoordsXyd,
		 facetVetexIndsList, filePathTextureImage, false);

	cv::Mat splitMap = cv::Mat::zeros(numRows, numCols, CV_8UC3);
	cv::Mat textureImg = cv::imread(filePathTextureImage);
	cv::Mat DrawTriangleImage(int numRows, int numCols, std::vector<cv::Point2f> &vertexCoords, std::vector<std::vector<int>> &triVertexInds, cv::Mat &textureImg);
	//cv::Mat splitMap = DrawTriangleImage(numRows, numCols, vertexCoords, triVertexInds, textureImg);
	for (int i = 0; i < vertexCoords.size(); i++) {
		int y = vertexCoords[i].y;
		int x = vertexCoords[i].x;
		if (anchorDisparitySets[y][x].size() > 1) {
			cv::circle(splitMap, cv::Point2f(x - 0.5, y - 0.5), 3, cv::Scalar(0, 0, 255), 3, CV_AA);
		}
	}
	cv::imwrite(filePathSplittingMap, splitMap);
}


#if 1
static cv::Mat RenderNovelView(int numRows, int numCols, float focalLen, float baselineLen, float alpha,
	cv::Mat &imL, std::vector<cv::Point2f> &vertexCoordsL, std::vector<std::vector<int>> &triVertexIndsL, std::vector<SlantedPlane> &slantedPlanesL,
	cv::Mat &imR, std::vector<cv::Point2f> &vertexCoordsR, std::vector<std::vector<int>> &triVertexIndsR, std::vector<SlantedPlane> &slantedPlanesR)
{
	std::vector<cv::Point3f> meshVertexCoordsL, meshVertexCoordsR;
	std::vector<std::vector<int>> facetVertexIndsListL, facetVertexIndsListR;
	printf("Build water tight mesh left .......\n");
	BuildWaterTightMesh(-1, numRows, numCols, focalLen, baselineLen, vertexCoordsL, triVertexIndsL, slantedPlanesL, meshVertexCoordsL, facetVertexIndsListL);
	printf("Build water tight mesh right .......\n");
	BuildWaterTightMesh(+1, numRows, numCols, focalLen, baselineLen, vertexCoordsR, triVertexIndsR, slantedPlanesR, meshVertexCoordsR, facetVertexIndsListR);
	//exit(1);

	//return cv::Mat();

	//exit(1);

	for (float alpha = 0.01f; alpha < 1.f; alpha += 0.01f) {
		printf("*************************************\n");
		printf("*********** alpha = %.2f ************\n", alpha);
		printf("*************************************\n");


		cv::Mat imgRenderedL, imgRenderedR, isProjectedL, isProjectedR, zBufferL, zBufferR;
		cv::Mat isRenderedByCrossDepthFactL, isRenderedByCrossDepthFactR;
		ProjectTextureOneView(focalLen, baselineLen, alpha, -1,
			meshVertexCoordsL, facetVertexIndsListL, triVertexIndsL.size(),
			imL, imgRenderedL, isProjectedL, zBufferL, isRenderedByCrossDepthFactL);

		//cv::imshow("imgRenderedL", imgRenderedL);
		//cv::waitKey(0);
		//exit(1);

		ProjectTextureOneView(focalLen, baselineLen, alpha, +1,
			meshVertexCoordsR, facetVertexIndsListR, triVertexIndsR.size(),
			imR, imgRenderedR, isProjectedR, zBufferR, isRenderedByCrossDepthFactR);


		cv::Mat canvasRendered, canvasIsProjected, canvas;
		cv::hconcat(imgRenderedL, imgRenderedR, canvasRendered);
		cv::hconcat(isProjectedL, isProjectedR, canvasIsProjected);
		canvasIsProjected *= 127;
		cv::cvtColor(canvasIsProjected, canvasIsProjected, CV_GRAY2BGR);
		cv::vconcat(canvasRendered, canvasIsProjected, canvas);
		//cv::imshow("canvsa", canvas);
		//cv::waitKey(0);

		cv::Mat fB = focalLen * baselineLen * cv::Mat::ones(numRows, numCols, CV_32FC1);
		cv::Mat dBufferL, dBufferR;
		cv::divide(fB, zBufferL, dBufferL);
		cv::divide(fB, zBufferR, dBufferR);
		cv::Mat dBufferImgL, dBufferImgR;
		dBufferL.convertTo(dBufferImgL, CV_8UC1, 3);
		dBufferR.convertTo(dBufferImgR, CV_8UC1, 3);
		cv::Mat dBuffers;
		cv::hconcat(dBufferImgL, dBufferImgR, dBuffers);
		//cv::imshow("dBuffers", dBuffers);
		//void OnMouseDBuffers(int event, int x, int y, int flags, void *param);
		//cv::setMouseCallback("dBuffers", OnMouseDBuffers, &dBuffers);
		//cv::waitKey(0);

		cv::medianBlur(zBufferL, zBufferL, 5);
		cv::medianBlur(zBufferR, zBufferR, 5);

		cv::Mat blendedView = cv::Mat::zeros(numRows, numCols, CV_8UC3);
		for (int y = 0; y < numRows; y++) {
			for (int x = 0; x < numCols; x++) {
				float factorL = 1.f - alpha;
				float factorR = alpha;
				if (isProjectedL.at<unsigned char>(y, x) == 0
					|| isRenderedByCrossDepthFactL.at<unsigned char>(y, x)) {
					factorL = 0;
				}
				if (isProjectedR.at<unsigned char>(y, x) == 0
					|| isRenderedByCrossDepthFactR.at<unsigned char>(y, x)) {
					factorR = 0;
				}
				if (factorL == 0 && factorR == 0) {
					factorL = 1.f - alpha;
					//factorR = alpha;
				}
				if (factorL > 0 && factorR > 0) {
					float dL = dBufferL.at<float>(y, x);
					float dR = dBufferR.at<float>(y, x);
					/*			if (std::abs(dL - dR) > 3) {
					if (dL > dR) {
					factorR = 0;
					}
					else {
					factorL = 0;
					}
					}*/
					/*else {
					factorR = 0;
					}*/
					//// if both projected compare zBuffer
					//float zL = zBufferL.at<float>(y, x);
					//float zR = zBufferR.at<float>(y, x);
					//if (std::abs(zL - zR) > 10) {
					//	if (zL < zR) {
					//		factorR = 0;
					//	}
					//	else {
					//		factorL = 0;
					//	}
					//}
				}

				if (factorL != 0 || factorR != 0) {
					factorL /= (factorL + factorR);
					factorR /= (factorL + factorR);
				}
				cv::Vec3f blendedPixel =
					factorL * cv::Vec3f(imgRenderedL.at<cv::Vec3b>(y, x)) +
					factorR * cv::Vec3f(imgRenderedR.at<cv::Vec3b>(y, x));
				blendedView.at<cv::Vec3b>(y, x) = blendedPixel + cv::Vec3f(0.5, 0.5, 0.5);
			}
		}



		//cv::vconcat(imL, blendedView, canvas);
		//cv::vconcat(canvas, imR, canvas);
		//cv::imshow("blended view", canvas);
		//cv::waitKey(0);
		extern std::string filePathRenderOutput;
		//cv::imwrite(filePathRenderOutput, blendedView);
		char outfilepath[1025];
		sprintf(outfilepath, "%s_%.2f.png", filePathRenderOutput.c_str(), alpha);
		cv::imwrite(outfilepath, blendedView);
		printf("blended view saved to %s,   exiting ...\n", filePathRenderOutput.c_str());
	}

	
	exit(1);
	//return blendedView;



	return cv::Mat();	
}
#else 
static struct SortByDisparityVal {
	bool operator ()(const std::pair<float, int> &a, const std::pair<float, int> &b) const {
		return a.first < b.first;
	}
};

static void AverageValuesInsideEachConnectedComponent(std::vector<std::pair<float, int>> &dids)
{
	std::sort(dids.begin(), dids.end(), SortByDisparityVal());
	/*if (dids.size() > 1) {
		std::sort(dids.begin(), dids.end());
	}*/
	
	std::vector<int> clusterIds(dids.size());

	const float thresh = 2;
	int i = 0, startId = 0;
	float clusterSum = dids[0].first;

	while (i + 1 < dids.size()) {
		while (i + 1 < dids.size() && std::abs(dids[i + 1].first - dids[i].first) <= thresh) {
			clusterSum += dids[i + 1].first;
			i++;
		}
		i++;
		float avgVal = clusterSum / (i - startId);
		for (int j = startId; j < i; j++) {
			dids[j].first = avgVal;
		}
		startId = i;
		if (i < dids.size()) {
			clusterSum = dids[i].first;
		}
	}

}

static cv::Mat RenderNovelView(int numRows, int numCols, float focalLen, float baselineLen, float alpha,
	cv::Mat &imL, std::vector<cv::Point2f> &vertexCoordsL, std::vector<std::vector<int>> &triVertexIndsL, std::vector<SlantedPlane> &slantedPlanesL,
	cv::Mat &imR, std::vector<cv::Point2f> &vertexCoordsR, std::vector<std::vector<int>> &triVertexIndsR, std::vector<SlantedPlane> &slantedPlanesR)
{
	printf("11111111111111111111111\n");
	std::vector<cv::Point3f> vertexCoords3dL, vertexCoords3dR;
	std::vector<std::vector<int>> triVertexIndsList3dL(triVertexIndsL.size());
	std::vector<std::vector<int>> triVertexIndsList3dR(triVertexIndsR.size());


	MCImg<std::vector<std::pair<float, int>>> anchorDisparitySetsL(numRows + 1, numCols + 1);
	for (int id = 0; id < triVertexIndsL.size(); id++) {
		for (int j = 0; j < 3; j++) {
			cv::Point2f p = vertexCoordsL[triVertexIndsL[id][j]];
			float d = slantedPlanesL[id].ToDisparity(p.y - 0.5, p.x - 0.5);
			anchorDisparitySetsL[p.y][(int)p.x].push_back(std::make_pair(d, id));
		}
	}
	printf("11111111111111111111111\n");
	for (int y = 0; y <= numRows; y++) {
		for (int x = 0; x <= numCols; x++) {
			if (!anchorDisparitySetsL[y][x].empty()) {
				//printf("before Avg....\n");
				std::vector<std::pair<float, int>> tmpVec = anchorDisparitySetsL[y][x];
				//AverageValuesInsideEachConnectedComponent(tmpVec);
				anchorDisparitySetsL[y][x] = tmpVec;
				//printf("after Avg....\n");
			}
		}
	}


	printf("11111111111111111111111\n");
	for (int id = 0; id < triVertexIndsL.size(); id++) {
		for (int j = 0; j < 3; j++) {
			cv::Point2f p = vertexCoordsL[triVertexIndsL[id][j]];	// in texture coordinates
			float d = 0;
			std::vector<std::pair<float, int>> &dids = anchorDisparitySetsL[p.y][(int)p.x];
			int k = 0;
			for (k = 0; k < dids.size(); k++) {
				if (dids[k].second == id) {
					d = dids[k].first;
					break;
				}
			}
			ASSERT(k != dids.size());
			float Z = focalLen * baselineLen / d;
			/*float X = (p.x - 0.5) * Z / focalLen;
			float Y = (p.y - 0.5) * Z / focalLen;*/
			float X = (p.x ) * Z / focalLen;	// in texture coordinates
			float Y = (p.y ) * Z / focalLen;	// in texture coordinates
			vertexCoords3dL.push_back(cv::Point3f(X, Y, Z));
			triVertexIndsList3dL[id].push_back(vertexCoords3dL.size() - 1);
		}
	}



	printf("11111111111111111111111\n");
	MCImg<std::vector<std::pair<float, int>>> anchorDisparitySetsR(numRows + 1, numCols + 1);
	for (int id = 0; id < triVertexIndsR.size(); id++) {
		for (int j = 0; j < 3; j++) {
			cv::Point2f p = vertexCoordsR[triVertexIndsR[id][j]];
			float d = slantedPlanesR[id].ToDisparity(p.y - 0.5, p.x - 0.5);
			anchorDisparitySetsR[p.y][(int)p.x].push_back(std::make_pair(d, id));
		}
	}
	printf("11111111111111111111111\n");
	for (int y = 0; y <= numRows; y++) {
		for (int x = 0; x <= numCols; x++) {
			if (!anchorDisparitySetsR[y][x].empty()) {
				std::vector<std::pair<float, int>> tmpVec = anchorDisparitySetsR[y][x];
				AverageValuesInsideEachConnectedComponent(tmpVec);
				anchorDisparitySetsR[y][x] = tmpVec;
			}
		}
	}

	printf("11111111111111111111111\n");
	for (int id = 0; id < triVertexIndsR.size(); id++) {
		for (int j = 0; j < 3; j++) {
			cv::Point2f p = vertexCoordsR[triVertexIndsR[id][j]];
			float d = 0;
			std::vector<std::pair<float, int>> &dids = anchorDisparitySetsR[p.y][(int)p.x];
			int k = 0;
			for (k = 0; k < dids.size(); k++) {
				if (dids[k].second == id) {
					d = dids[k].first;
					break;
				}
			}
			ASSERT(k != dids.size());
			float Z = focalLen * baselineLen / d;
			float X = p.x * Z / focalLen;	// in texture coordinates
			float Y = p.y * Z / focalLen;	// in texture coordinates
			vertexCoords3dR.push_back(cv::Point3f(X, Y, Z));
			triVertexIndsList3dR[id].push_back(vertexCoords3dR.size() - 1);
		}
	}

	printf("333333333333333333\n");


#if 0
	cv::Mat projected = cv::Mat::zeros(numRows, numCols, CV_8UC1);
	for (int id = 0; id < triVertexIndsL.size(); id++) {
		cv::Point3f A = vertexCoords3dL[triVertexIndsList3dL[id][0]];
		cv::Point3f B = vertexCoords3dL[triVertexIndsList3dL[id][1]];
		cv::Point3f C = vertexCoords3dL[triVertexIndsList3dL[id][2]];
			
		cv::Point2f a(focalLen * A.x / A.z, focalLen * A.y / A.z);
		cv::Point2f b(focalLen * B.x / B.z, focalLen * B.y / B.z);
		cv::Point2f c(focalLen * C.x / C.z, focalLen * C.y / C.z);

		// compute bounding box of triangle abc
		float xL = FLT_MAX, xR = -FLT_MAX;
		float yU = FLT_MAX, yD = -FLT_MAX;
		cv::Point2f abc[3] = { a, b, c };
		for (int j = 0; j < 3; j++) {
			xL = std::min(xL, abc[j].x);
			xR = std::max(xR, abc[j].x);
			yU = std::min(yU, abc[j].y);
			yD = std::max(yD, abc[j].y);
		}

		// compute the pixel correspondces in imL of pixel in triangle abc 
		for (int y = yU - 2; y <= yD + 2; y++) {
			for (int x = xL - 2; x <= xR + 2; x++) {
				if (InBound(y, x, numRows, numCols)
					&& PointInTriangle(cv::Point2f(x, y) + cv::Point2f(0.5, 0.5), a, b, c))
				{
					projected.at<unsigned char>(y, x) += 1;
				}
			}
		}
	}
	cv::imshow("projected", 127 * projected);
	cv::waitKey(0);
#endif




	cv::Mat imgRenderedL, imgRenderedR, isProjectedL, isProjectedR, zBufferL, zBufferR;
	ProjectTextureOneView(focalLen, baselineLen, alpha, -1,
		vertexCoords3dL, triVertexIndsList3dL, imL, imgRenderedL, isProjectedL, zBufferL);
	
	ProjectTextureOneView(focalLen, baselineLen, alpha, +1,
		vertexCoords3dR, triVertexIndsList3dR, imR, imgRenderedR, isProjectedR, zBufferR);


	cv::Mat canvasRendered, canvasIsProjected, canvas;
	cv::hconcat(imgRenderedL, imgRenderedR, canvasRendered);
	cv::hconcat(isProjectedL, isProjectedR, canvasIsProjected);
	canvasIsProjected *= 127;
	cv::cvtColor(canvasIsProjected, canvasIsProjected, CV_GRAY2BGR);
	cv::vconcat(canvasRendered, canvasIsProjected, canvas);
	cv::imshow("canvsa", canvas);
	cv::waitKey(0);


	cv::Mat blendedView = cv::Mat::zeros(numRows, numCols, CV_8UC3);
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			float factorL = 1.f - alpha;
			float factorR = alpha;
			if (isProjectedL.at<unsigned char>(y, x) == 0) {
				factorL = 0;
			}
			if (isProjectedR.at<unsigned char>(y, x) == 0) {
				factorR = 0;
			}
			if (factorL > 0 && factorR > 0) {
				// if both projected compare zBuffer
				float zL = zBufferL.at<float>(y, x);
				float zR = zBufferR.at<float>(y, x);
				if (std::abs(zL - zR) > 10) {
					if (zL < zR) {
						factorR = 0;
					}
					else {
						factorL = 0;
					}
				}
			}

			if (factorL != 0 || factorR != 0) {
				factorL /= (factorL + factorR);
				factorR /= (factorL + factorR);
			}
			cv::Vec3f blendedPixel =
				factorL * cv::Vec3f(imgRenderedL.at<cv::Vec3b>(y, x)) +
				factorR * cv::Vec3f(imgRenderedR.at<cv::Vec3b>(y, x));
			blendedView.at<cv::Vec3b>(y, x) = blendedPixel + cv::Vec3f(0.5, 0.5, 0.5);
		}
	}
	cv::vconcat(imL, blendedView, canvas);
	cv::vconcat(canvas, imR, canvas);
	cv::imshow("blended view", canvas);
	cv::waitKey(0);

	return blendedView;

}
#endif

void RunPatchMatchOnTriangles(std::string rootFolder, cv::Mat &imL, cv::Mat &imR)
{




	int numRows = imL.rows, numCols = imL.cols;
	int numDisps, maxDisp, visualizeScale;
	SetupStereoParameters(rootFolder, numDisps, maxDisp, visualizeScale);
	//InitGlobalDsiAndSimWeights(imL, imR, numDisps);

	std::string leftFilePath  = "d:/data/stereo/" + rootFolder + "/im2.png";
	std::string rightFilePath = "d:/data/stereo/" + rootFolder + "/im6.png";
	extern MCImg<unsigned short> gDsiL, gDsiR;
	gDsiL = MCImg<unsigned short>(numRows, numCols, numDisps);
	gDsiR = MCImg<unsigned short>(numRows, numCols, numDisps);
	CostVolumeFromYamaguchi(leftFilePath, rightFilePath, gDsiL, gDsiR, numDisps);
	extern cv::Mat gImLabL, gImLabR;
	gImLabL = imL.clone();
	gImLabR = imR.clone();



	std::vector<cv::Point2f> vertexCoordsL, vertexCoordsR;
	std::vector<std::vector<int>> triVertexIndsL, triVertexIndsR;
#if 1
	Triangulate2DImage(imL, vertexCoordsL, triVertexIndsL);
	Triangulate2DImage(imR, vertexCoordsR, triVertexIndsR);
#else
	void ImageDomainTessellation(cv::Mat &img, std::vector<cv::Point2f> &vertexCoordList,
		std::vector<std::vector<int>> &triVertexIndsList);
	ImageDomainTessellation(imL, vertexCoordsL, triVertexIndsL);
	printf("left tesselation done.\n");
	ImageDomainTessellation(imR, vertexCoordsR, triVertexIndsR);
	printf("right tesselation done.\n");
#endif
	cv::Mat triImgL = DrawTriangleImage(numRows, numCols, vertexCoordsL, triVertexIndsL);
	//cv::imshow("triangulation", triImgL);
	//cv::waitKey(0);



	std::vector<cv::Point2f> baryCentersL, baryCentersR;
	std::vector<std::vector<int>> nbIndicesL, nbIndicesR; 
	ConstructNeighboringTriangleGraph(numRows, numCols, vertexCoordsL, triVertexIndsL, baryCentersL, nbIndicesL);
	ConstructNeighboringTriangleGraph(numRows, numCols, vertexCoordsR, triVertexIndsR, baryCentersR, nbIndicesR);


	std::vector<std::vector<cv::Point2i>> triPixelListsL, triPixelListsR;
	DeterminePixelOwnership(numRows, numCols, vertexCoordsL, triVertexIndsL, triPixelListsL);
	DeterminePixelOwnership(numRows, numCols, vertexCoordsR, triVertexIndsR, triPixelListsR);


	int numTrianglesL = baryCentersL.size();
	int numTrianglesR = baryCentersR.size();
	std::vector<SlantedPlane> slantedPlanesL(numTrianglesL), slantedPlanesR(numTrianglesR);
	std::vector<float> bestCostsL(numTrianglesL), bestCostsR(numTrianglesR);
	

	for (int id = 0; id < numTrianglesL; id++) {
		slantedPlanesL[id] = SlantedPlane::ConstructFromRandomInit(baryCentersL[id].y, baryCentersL[id].x, maxDisp);
		bestCostsL[id] = PatchMatchSlantedPlaneCost(baryCentersL[id].y + 0.5, baryCentersL[id].x + 0.5, slantedPlanesL[id], -1);
	}
	for (int id = 0; id < numTrianglesR; id++) {
		slantedPlanesR[id] = SlantedPlane::ConstructFromRandomInit(baryCentersR[id].y, baryCentersR[id].x, maxDisp);
		bestCostsR[id] = PatchMatchSlantedPlaneCost(baryCentersR[id].y + 0.5, baryCentersR[id].x + 0.5, slantedPlanesR[id], -1);
	}

	std::vector<int> idListL(numTrianglesL), idListR(numTrianglesR);
	for (int i = 0; i < numTrianglesL; i++) {
		idListL[i] = i;
	}
	for (int i = 0; i < numTrianglesR; i++) {
		idListR[i] = i;
	}
	 

	for (int round = 0; round < MAX_PATCHMATCH_ITERS; round++) {

		//#pragma omp parallel for  
		printf("PatchMatchOnTriangles round %d ...\n", round);
		for (int i = 0; i < numTrianglesL; i++) {
			int id = idListL[i];
			PropagateAndRandomSearch(id, -1, maxDisp, baryCentersL[id], slantedPlanesL, bestCostsL, nbIndicesL);
		}
		for (int i = 0; i < numTrianglesL; i++) {
			int id = idListR[i];
			PropagateAndRandomSearch(id, +1, maxDisp, baryCentersR[id], slantedPlanesR, bestCostsR, nbIndicesR);
		}

		//cv::Mat dispL = TriangleLabelToDisparityMap(numRows, numCols, slantedPlanesL, triPixelListsL);
		//std::vector<std::pair<std::string, void*>> aux Params;
		//auxParams.push_back(std::pair<std::string, void*>("triImg", &triImgL));
		////auxParams.push_back(std::pair<std::string, void*>("slantedPlanesL", &slantedPlanesL));
		//EvaluateDisparity(rootFolder, dispL, 0.5f, auxParams);

		printf("printf something...\n");
		std::reverse(idListL.begin(), idListL.end());
		std::reverse(idListR.begin(), idListR.end());
	}





	cv::Mat dispL, dispR;
	PatchMatchOnTrianglePostProcess(numRows, numCols, slantedPlanesL, slantedPlanesR,
		baryCentersL, baryCentersR, nbIndicesL, nbIndicesR, triPixelListsL, triPixelListsR, dispL, dispR);

	//cv::Mat dispL = TriangleLabelToDisparityMap(numRows, numCols, slantedPlanesL, triPixelListsL);
	//cv::Mat dispR = TriangleLabelToDisparityMap(numRows, numCols, slantedPlanesR, triPixelListsR);

	//std::vector<std::pair<std::string, void*>> auxParams;
	//auxParams.push_back(std::pair<std::string, void*>("triImg", &triImgL));
	//extern int DO_EVAL;
	//DO_EVAL = 1;
	//EvaluateDisparity(rootFolder, dispL, 0.5f/*, auxParams*/);

	cv::Mat dispCanvas;
	cv::hconcat(dispL, dispR, dispCanvas);
	dispCanvas.convertTo(dispCanvas, CV_8UC1, 255.f / maxDisp);
	cv::cvtColor(dispCanvas, dispCanvas, CV_GRAY2BGR);
	cv::hconcat(dispCanvas, triImgL, dispCanvas);
	cv::imshow("disparity", dispCanvas);
	cv::waitKey(0);

	printf("sdffffffffffffffff\n");

	extern float RENDER_ALPHA;

	RenderNovelView(numRows, numCols, 3740, 160, RENDER_ALPHA,
		imL, vertexCoordsL, triVertexIndsL, slantedPlanesL, 
		imR, vertexCoordsR, triVertexIndsR, slantedPlanesR);
	return;
















	void SaveMeshStereoResultToPly(cv::Mat &img, float maxDisp,
		std::string workingDir, std::string plyFilePath, std::string textureFilePath,
		std::vector<cv::Point2f> &vertexCoords, std::vector<std::vector<int>> &triVertexInds,
		std::vector<std::vector<SlantedPlane>> &triVertexBestLabels, cv::Mat &splitMap);

	std::vector<std::vector<SlantedPlane>> sp3(numTrianglesL);
	for (int i = 0; i < numTrianglesL; i++) {
		sp3[i].push_back(slantedPlanesL[i]);
		sp3[i].push_back(slantedPlanesL[i]);
		sp3[i].push_back(slantedPlanesL[i]);
	}
	
	//cv::Mat splitMap = DeterminSplitMapByHeuristic(numRows, numCols, 
	//	slantedPlanesL, vertexCoordsL, triVertexIndsL);
	//cv::Mat splitMap = cv::Mat::ones(numRows + 1, numCols + 1, CV_8UC1) * 255;
	cv::Mat splitMap = cv::Mat::zeros(numRows + 1, numCols + 1, CV_8UC1) * 255;
	SaveMeshStereoResultToPly(imL, maxDisp, "d:/data/stereo/" + rootFolder,
		"d:/data/stereo/" + rootFolder + "/PatchMatchOnTriangles.ply",
		"d:/data/stereo/" + rootFolder + "/im2.png",
		vertexCoordsL, triVertexIndsL, sp3, splitMap);

	return;


	/*PatchMatchOnPixelPostProcess(slantedPlanesL, slantedPlanesR, triPixelListsL, triPixelListsR,
	imL, imR, dispL, dispR);*/
	PatchMatchOnTrianglePostProcess(numRows, numCols, slantedPlanesL, slantedPlanesR,
		baryCentersL, baryCentersR, nbIndicesL, nbIndicesR, triPixelListsL, triPixelListsR, dispL, dispR);

	
	EvaluateDisparity(rootFolder, dispL, 0.5f/*, auxParams*/);

	std::vector<std::vector<SlantedPlane>> candidateLabels;
	std::vector<std::vector<float>> unaryCosts;
	GenerateMeshStereoCandidateLabels(numRows, numCols, numDisps,
		baryCentersL, slantedPlanesL, vertexCoordsL, triVertexIndsL, nbIndicesL, candidateLabels, unaryCosts);
	MeshStereoBPOnFG bp;
	bp.InitFromTriangulation(numRows, numCols, numDisps, candidateLabels, unaryCosts, vertexCoordsL, triVertexIndsL, triPixelListsL, imL);
	std::vector<int> outLabels;
	bp.Run(rootFolder);

}

static double ComputePSNR(cv::Mat &I1, cv::Mat &I2)
{
	std::cout << I1.size() << "\n";
	std::cout << I2.size() << "\n";
	std::string type2str(int type);
	std::cout << type2str(I1.type()) << "\n";
	std::cout << type2str(I2.type()) << "\n";
	printf("computing PSNR...\n");
	cv::Mat s1;
	cv::absdiff(I1, I2, s1);       // |I1 - I2|
	printf("1111111111\n");
	s1.convertTo(s1, CV_32FC3);  // cannot make a square on 8 bits
	printf("1111111111\n");
	s1 = s1.mul(s1);           // |I1 - I2|^2
	printf("1111111111\n");

	cv::Scalar s = cv::sum(s1);        // sum elements per channel
	std::cout << "scalar: " << s << "\n";

	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

	if (sse <= 1e-10) // for small values return zero
		return 0;
	else
	{
		double mse = sse / (double)(I1.channels() * I1.total());
		double psnr = 10.0 * log10((255 * 255) / mse);
		return psnr;
	}
}

void RansacPlaneFitOnTriangles(std::string rootFolder, cv::Mat &imL, cv::Mat &imR)
{
	//std::vector<float> ds;
	//ds.push_back(13.94);
	//ds.push_back(13.94);
	//ds.push_back(20.07);
	//ClusterDisparitiesOtsu(ds);
	//return;


	


	std::string leftFilePath = "D:/data/Midd2/ThirdSize/" + rootFolder + "/view1.png";
	std::string rightFilePath = "D:/data/Midd2/ThirdSize/" + rootFolder + "/view5.png";
	std::string filePathGTView = "D:/data/Midd2/ThirdSize/" + rootFolder + "/view3.png";
	cv::Mat gtVirtualVirew = cv::imread(filePathGTView);
	imL = cv::imread(leftFilePath);
	imR = cv::imread(rightFilePath);

	cv::Mat dispL, dispR;
	extern int USE_GT_DISP;
	bool FileExist(std::string filePath);
	if (USE_GT_DISP && FileExist("D:/data/Midd2/ThirdSize/" + rootFolder + "/disp1_backgorundFilled.png")) {
		std::string filePathDispL = "D:/data/Midd2/ThirdSize/" + rootFolder + "/disp1_backgorundFilled.png";
		std::string filePathDispR = "D:/data/Midd2/ThirdSize/" + rootFolder + "/disp5_backgorundFilled.png";
		
		
		dispL = cv::imread(filePathDispL);
		dispR = cv::imread(filePathDispR);
		//cv::imshow("dispL", dispL);
		//cv::waitKey(0);
		cv::cvtColor(dispL, dispL, CV_BGR2GRAY);
		cv::cvtColor(dispR, dispR, CV_BGR2GRAY);
		dispL.convertTo(dispL, CV_32FC1, 1.f / 3.f);
		dispR.convertTo(dispR, CV_32FC1, 1.f / 3.f);
	}
	else {
		//exit(-1);
		std::string filePathDispL = "D:/data/Exp8_TreeFiltering/Midd2/ThirdSize/" + rootFolder + "_treeFilter_dispL.png";
		std::string filePathDispR = "D:/data/Exp8_TreeFiltering/Midd2/ThirdSize/" + rootFolder + "_treeFilter_dispR.png";
		dispL = cv::imread(filePathDispL);
		dispR = cv::imread(filePathDispR);

		//cv::imshow("dispL", dispL);
		//cv::waitKey(0);

		//cv::imshow("dispL", dispL);
		//cv::waitKey(0);
		cv::cvtColor(dispL, dispL, CV_BGR2GRAY);
		cv::cvtColor(dispR, dispR, CV_BGR2GRAY);
		dispL.convertTo(dispL, CV_32FC1, 79.f / 255.f);
		dispR.convertTo(dispR, CV_32FC1, 79.f / 255.f);
	}

	
	//std::string leftFilePath = "D:/data/Herodion/camera7_138R.png";
	//std::string rightFilePath = "D:/data/Herodion/camera5_138R.png";
	//std::string filePathGTView = "D:/data/Herodion/camera6_138R.png";
	//cv::Mat gtVirtualVirew = cv::imread(filePathGTView);
	//imL = cv::imread(leftFilePath);
	//imR = cv::imread(rightFilePath);

	//cv::Mat dispL, dispR;
	//extern int USE_GT_DISP;
	//bool FileExist(std::string filePath);
	//
	//std::string filePathDispL = "D:/data/Exp8_TreeFiltering/camera7_138R_disp.png";
	//std::string filePathDispR = "D:/data/Exp8_TreeFiltering/camera5_138R_disp.png";
	//dispL = cv::imread(filePathDispL);
	//dispR = cv::imread(filePathDispR);
	////cv::imshow("dispL", dispL);
	////cv::waitKey(0);
	//cv::cvtColor(dispL, dispL, CV_BGR2GRAY);
	//cv::cvtColor(dispR, dispR, CV_BGR2GRAY);
	//dispL.convertTo(dispL, CV_32FC1, 59.f / 255.f);
	//dispR.convertTo(dispR, CV_32FC1, 59.f / 255.f);
	
	
	int numRows = imL.rows, numCols = imL.cols;
	int numDisps, maxDisp, visualizeScale;
	SetupStereoParameters(rootFolder, numDisps, maxDisp, visualizeScale);

	std::vector<cv::Point2f> vertexCoordsL, vertexCoordsR;
	std::vector<std::vector<int>> triVertexIndsL, triVertexIndsR;
#ifdef SLIC_TRIANGULATION
	Triangulate2DImage(imL, vertexCoordsL, triVertexIndsL);
	Triangulate2DImage(imR, vertexCoordsR, triVertexIndsR);
#else
	void ImageDomainTessellation(cv::Mat &img, std::vector<cv::Point2f> &vertexCoordList,
		std::vector<std::vector<int>> &triVertexIndsList);
	ImageDomainTessellation(imL, vertexCoordsL, triVertexIndsL);
	ImageDomainTessellation(imR, vertexCoordsR, triVertexIndsR);
#endif

	printf("11111111111111111\n");
	cv::Mat DrawTriangleImage(int numRows, int numCols, std::vector<cv::Point2f> &vertexCoords, std::vector<std::vector<int>> &triVertexInds, cv::Mat &textureImg);
	cv::Mat triImgL = DrawTriangleImage(numRows, numCols, vertexCoordsL, triVertexIndsL, imL);
	cv::Mat triImgR = DrawTriangleImage(numRows, numCols, vertexCoordsR, triVertexIndsR, imR);
	cv::Mat triImgs;
	cv::hconcat(triImgL, triImgR, triImgs);
	//cv::imshow("triangulation", triImgs);
	//cv::waitKey(0);

	std::string filePathTriangulationL = "D:/data/Exp13_GenerateMesh/" + ROOTFOLDER + "_triangulationL.png";
	std::string filePathTriangulationR = "D:/data/Exp13_GenerateMesh/" + ROOTFOLDER + "_triangulationR.png";
	cv::imwrite(filePathTriangulationL, triImgL);
	cv::imwrite(filePathTriangulationR, triImgR);

	printf("11111111111111111\n");
	std::vector<cv::Point2f> baryCentersL, baryCentersR;
	std::vector<std::vector<int>> nbIndicesL, nbIndicesR;
	ConstructNeighboringTriangleGraph(numRows, numCols, vertexCoordsL, triVertexIndsL, baryCentersL, nbIndicesL);
	ConstructNeighboringTriangleGraph(numRows, numCols, vertexCoordsR, triVertexIndsR, baryCentersR, nbIndicesR);

	printf("11111111111111111\n");
	std::vector<std::vector<cv::Point2i>> triPixelListsL, triPixelListsR;
	DeterminePixelOwnership(numRows, numCols, vertexCoordsL, triVertexIndsL, triPixelListsL);
	DeterminePixelOwnership(numRows, numCols, vertexCoordsR, triVertexIndsR, triPixelListsR);


	printf("11111111111111111\n");
	int numTrianglesL = baryCentersL.size();
	int numTrianglesR = baryCentersR.size();
	std::vector<SlantedPlane> slantedPlanesL(numTrianglesL);
	std::vector<SlantedPlane> slantedPlanesR(numTrianglesR);


	printf("11111111111111111\n");
	cv::Vec3f RansacPlaneFitting(std::vector<cv::Point3f> &pointList, float inlierThresh);
	for (int id = 0; id < numTrianglesL; id++) {
		std::vector<cv::Point3f> pointList(triPixelListsL[id].size());
		for (int i = 0; i < triPixelListsL[id].size(); i++) {
			int y = triPixelListsL[id][i].y;
			int x = triPixelListsL[id][i].x;
			pointList[i] = cv::Point3f(x, y, dispL.at<float>(y, x));
		}
		cv::Vec3f abc = RansacPlaneFitting(pointList, 1.f);
		//std::cout << abc << "\n";
		slantedPlanesL[id].SlefConstructFromAbc(abc[0], abc[1], abc[2]);
	}
	for (int id = 0; id < numTrianglesR; id++) {
		std::vector<cv::Point3f> pointList(triPixelListsR[id].size());
		for (int i = 0; i < triPixelListsR[id].size(); i++) {
			int y = triPixelListsR[id][i].y;
			int x = triPixelListsR[id][i].x;
			pointList[i] = cv::Point3f(x, y, dispR.at<float>(y, x));
		}
		cv::Vec3f abc = RansacPlaneFitting(pointList, 1.f);
		slantedPlanesR[id].SlefConstructFromAbc(abc[0], abc[1], abc[2]);
	}
	

	printf("11111111111111111\n");
	dispL = TriangleLabelToDisparityMap(numRows, numCols, slantedPlanesL, triPixelListsL);
	dispR = TriangleLabelToDisparityMap(numRows, numCols, slantedPlanesR, triPixelListsR);
	cv::Mat dispImgL, dispImgR, dispImg;
	dispL.convertTo(dispImgL, CV_8UC1, 255.f / 79.f);
	dispR.convertTo(dispImgR, CV_8UC1, 255.f / 79.f);
	cv::hconcat(dispImgL, dispImgR, dispImg);
	cv::imwrite("d:/dispL.png", dispL);
	cv::imwrite("d:/dispR.png", dispR);
	//cv::imshow("dispImgL", dispImg);
	//cv::waitKey(0);

	
	extern float RENDER_ALPHA;
	cv::Mat imgRendered = RenderNovelView(numRows, numCols, 3740, 160, RENDER_ALPHA,
		imL, vertexCoordsL, triVertexIndsL, slantedPlanesL,
		imR, vertexCoordsR, triVertexIndsR, slantedPlanesR);

	return;
//#ifndef WATER_TIGHT_MESH
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			if (imgRendered.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 0, 0)) {
				printf("filled\n");
				imgRendered.at<cv::Vec3b>(y, x) = gtVirtualVirew.at<cv::Vec3b>(y, x);
			}
		}
	}
//#endif

	//cv::medianBlur(imgRendered, imgRendered, 1);

	double PSNR = ComputePSNR(imgRendered, gtVirtualVirew);
	printf("**********************************\n");
	printf("PSNR = %.2lf\n", PSNR);
	printf("**********************************\n");

	
	cv::imshow("asdfsfasdf", imgRendered);
	cv::waitKey(0);
	extern std::string filePathRenderOutput;
	std::cout << filePathRenderOutput << "\n";
	cv::imwrite(filePathRenderOutput, imgRendered);
}

void TestPatchMatchOnTriangles()
{
	
	std::string rootFolder = ROOTFOLDER;


	
	cv::Mat imL = cv::imread("D:/data/Midd2/ThirdSize/" + rootFolder + "/view1.png");
	cv::Mat imR = cv::imread("D:/data/Midd2/ThirdSize/" + rootFolder + "/view5.png");

	RansacPlaneFitOnTriangles(rootFolder, imL, imR);
	return;
	RunPatchMatchOnTriangles(rootFolder, imL, imR);
}

void RenderByMyself()
{
	ROOTFOLDER = "Bowling2";
	std::string rootFolder = ROOTFOLDER;
	cv::Mat imL = cv::imread("D:/data/Midd2/ThirdSize/" + rootFolder + "/view1.png");
	cv::Mat imR = cv::imread("D:/data/Midd2/ThirdSize/" + rootFolder + "/view5.png");
	RansacPlaneFitOnTriangles(rootFolder, imL, imR);


	const float focalLen = 3740;
	const float baselineLen = 160;
	std::vector<cv::Point3f> LoadVectorPoint3f(std::string filePath, std::string mode);
	std::vector<std::vector<int>> LoadVectorVectorInt(std::string filePath, std::string mode);
	std::vector<cv::Point3f> meshVertexCoordsXyd = LoadVectorPoint3f("d:/meshVertexCoordsXydR.txt", "r");
	std::vector<std::vector<int>> facetVetexIndsList = LoadVectorVectorInt("d:/facetVetexIndsListR.txt", "r");

	std::vector<cv::Point3f> meshVertexCoords = meshVertexCoordsXyd;
	for (int i = 0; i < meshVertexCoords.size(); i++) {
		cv::Point3f &p = meshVertexCoords[i];
		float Z = focalLen * baselineLen / p.z;
		float X = p.x * Z / focalLen;
		float Y = p.y * Z / focalLen;
		meshVertexCoords[i] = cv::Point3f(X, Y, Z);
	}


	cv::Mat textureImg = cv::imread("d:/data/midd2/thirdSize/Bowling2/view1.png");
	cv::Mat imgRendered, isProjected, zBuffer, isFromCrossDepthFacet;



	//ProjectTextureOneView(focalLen, baselineLen, 0.5, -1, meshVertexCoords, facetVetexIndsList, facetVetexIndsList.size(),
	//	textureImg, imgRendered, isProjected, zBuffer, isFromCrossDepthFacet);
	extern float RENDER_ALPHA;
	ProjectTextureOneView(focalLen, baselineLen, RENDER_ALPHA, +1, meshVertexCoords, facetVetexIndsList, facetVetexIndsList.size(),
		imR, imgRendered, isProjected, zBuffer, isFromCrossDepthFacet);

	cv::imshow("imgRendered", imgRendered);
	cv::waitKey(0);

}