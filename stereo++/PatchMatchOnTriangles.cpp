#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <set>

#include "StereoAPI.h"
#include "SlantedPlane.h"
#include "Timer.h"
#include "BPOnFactorGraph.h"
#include "ReleaseAssert.h"
#include "PostProcess.h"

#define PROCESS_RIGHT_VIEW


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

extern MCImg<float>			gDsiL;
extern MCImg<float>			gDsiR;
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

float sign(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3)
{
	return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

bool PointInTriangle(cv::Point2f pt, cv::Point2f v1, cv::Point2f v2, cv::Point2f v3)
{
	bool b1, b2, b3;

	b1 = sign(pt, v1, v2) < 0.0f;
	b2 = sign(pt, v2, v3) < 0.0f;
	b3 = sign(pt, v3, v1) < 0.0f;

	return ((b1 == b2) && (b2 == b3));
}

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
	float zRadius = maxDisp / 2.f;
	float nRadius = 1.f;
	while (zRadius >= 0.1f) {
		SlantedPlane newGuess = SlantedPlane::ConstructFromRandomPertube(slantedPlanes[id], y, x, nRadius, zRadius);
		ImproveGuess(y, x, slantedPlanes[id], newGuess, bestCosts[id], sign);
		zRadius /= 2.f;
		nRadius /= 2.f;
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
	std::vector<std::set<int>> triIndSets((numRows + 1) * (numCols + 1));
	for (int i = 0; i < numTriangles; i++) {
		for (int j = 0; j < 3; j++) {
			cv::Point2f &p = vertexCoords[triVertexInds[i][j]];
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

		for (int y = std::max(0, yU); y < yD && y < numRows; y++) {
			for (int x = std::max(0, xL); x < xR && x < numCols; x++) {
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

	std::vector<std::pair<std::string, void*>> auxParams;
	auxParams.push_back(std::pair<std::string, void*>("triImg", &confidenceImgL));
	EvaluateDisparity(ROOTFOLDER, dispL, 0.5f, auxParams);
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

void RunPatchMatchOnTriangles(std::string rootFolder, cv::Mat &imL, cv::Mat &imR)
{
	int numRows = imL.rows, numCols = imL.cols;
	int numDisps, maxDisp, visualizeScale;
	SetupStereoParameters(rootFolder, numDisps, maxDisp, visualizeScale);
	InitGlobalDsiAndSimWeights(imL, imR, numDisps);

	std::vector<cv::Point2f> vertexCoordsL, vertexCoordsR;
	std::vector<std::vector<int>> triVertexIndsL, triVertexIndsR;
	Triangulate2DImage(imL, vertexCoordsL, triVertexIndsL);
	Triangulate2DImage(imR, vertexCoordsR, triVertexIndsR);
	cv::Mat triImgL = DrawTriangleImage(numRows, numCols, vertexCoordsL, triVertexIndsL);

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
	 


	for (int round = 0; round < 4/*MAX_PATCHMATCH_ITERS*/; round++) {

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

		std::reverse(idListL.begin(), idListL.end());
		std::reverse(idListR.begin(), idListR.end());
	}


	cv::Mat dispL = TriangleLabelToDisparityMap(numRows, numCols, slantedPlanesL, triPixelListsL);
	cv::Mat dispR = TriangleLabelToDisparityMap(numRows, numCols, slantedPlanesR, triPixelListsR);
	std::vector<std::pair<std::string, void*>> auxParams;
	auxParams.push_back(std::pair<std::string, void*>("triImg", &triImgL));
	EvaluateDisparity(rootFolder, dispL, 0.5f, auxParams);

	/*PatchMatchOnPixelPostProcess(slantedPlanesL, slantedPlanesR, triPixelListsL, triPixelListsR,
	imL, imR, dispL, dispR);*/
	PatchMatchOnTrianglePostProcess(numRows, numCols, slantedPlanesL, slantedPlanesR,
		baryCentersL, baryCentersR, nbIndicesL, nbIndicesR, triPixelListsL, triPixelListsR, dispL, dispR);

	return;
	EvaluateDisparity(rootFolder, dispL, 0.5f, auxParams);

	std::vector<std::vector<SlantedPlane>> candidateLabels;
	std::vector<std::vector<float>> unaryCosts;
	GenerateMeshStereoCandidateLabels(numRows, numCols, numDisps,
		baryCentersL, slantedPlanesL, vertexCoordsL, triVertexIndsL, nbIndicesL, candidateLabels, unaryCosts);
	MeshStereoBPOnFG bp;
	bp.InitFromTriangulation(numRows, numCols, numDisps, candidateLabels, unaryCosts, vertexCoordsL, triVertexIndsL, triPixelListsL, imL);
	std::vector<int> outLabels;
	bp.Run(rootFolder);

}

void TestPatchMatchOnTriangles()
{
	
	std::string rootFolder = ROOTFOLDER;

	cv::Mat imL = cv::imread("D:/data/stereo/" + rootFolder + "/im2.png");
	cv::Mat imR = cv::imread("D:/data/stereo/" + rootFolder + "/im6.png");

	RunPatchMatchOnTriangles(rootFolder, imL, imR);
}