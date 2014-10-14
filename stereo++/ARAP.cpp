#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <stack>
#include <set>
#include <string>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

#include "StereoAPI.h"
#include "SlantedPlane.h"
#include "PostProcess.h"
#include "ReleaseAssert.h"
#include "Timer.h"



struct ARAPEvalParams {
	int numRows, numCols;
	cv::Mat *labelMap;
	cv::Mat *confidenceImg;
	std::vector<cv::Point2f> *baryCenters;
	std::vector<cv::Vec3f> *n;
	std::vector<float> *u;
	std::vector<std::vector<int>> *nbGraph;
	std::vector<std::vector<cv::Point2i>> *segPixelLists;
	ARAPEvalParams() : numRows(0), numCols(0), labelMap(0), baryCenters(0),
		n(0), u(0), nbGraph(0), segPixelLists(0), confidenceImg(0) {}
};

static ARAPEvalParams evalParams;

template<typename T> static inline T mix(T a, T b, float alpha) { return a + (b - a) * alpha; }
template<typename T> static inline T mix(T a, T b, double alpha) { return a + (b - a) * alpha; }

template<typename T> static inline T clamp(T v, T min, T max) { return (v < min) ? min : (v > max) ? max : v; }

template<typename T> static inline T smoothstep(T edge0, T edge1, T x)
{
	T t = clamp<T>((x - edge0) / (edge1 - edge0), 0, 1);
	return t * t * (((T)3) - ((T)2) * t);
}

template<typename T> static inline T smoothstep(T x)
{
	T t = clamp<T>(x, 0, 1);
	return t * t * (((T)3) - ((T)2) * t);
}

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

static cv::Mat DrawSegmentImage(cv::Mat &labelMap)
{
	double minVal, maxVal;
	cv::minMaxIdx(labelMap, &minVal, &maxVal);
	int numSegs = maxVal;
	int numRows = labelMap.rows, numCols = labelMap.cols;

	std::vector<cv::Vec3b> colors(numSegs);
	for (int id = 0; id < numSegs; id++) {
		colors[id] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
	}

	cv::Mat segImg(numRows, numCols, CV_8UC3);
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			segImg.at<cv::Vec3b>(y, x) = colors[labelMap.at<int>(y, x)];
		}
	}
	return segImg;
}

static void ModifyGlobalSimilarityWeights(int sign, cv::Mat &labelMap, 
	std::vector<cv::Point2f> &baryCenters, std::vector<std::vector<cv::Point2i>> &segPixelLists)
{
	ASSERT(sign == -1 || sign == +1);
	extern MCImg<float>	gSimWeightsL;
	extern MCImg<float>	gSimWeightsR;
	MCImg<float> *simWeights = (sign == -1 ? &gSimWeightsL : &gSimWeightsR);

	//printf("gSimWeightsL.data = %p\n", gSimWeightsL.data);
	//printf("simWeights.data   = %p\n", simWeights.data);

	extern int PATCHRADIUS;
	extern int PATCHWIDTH;
	int numRows = labelMap.rows, numCols = labelMap.cols;
	int numSegs = baryCenters.size();

	bs::Timer::Tic("Modifying similarity weights");
	for (int id = 0; id < numSegs; id++) {
		std::vector<cv::Point2i> &pixelList = segPixelLists[id];
		for (int i = 0; i < pixelList.size(); i++) {
			for (int j = i + 1; j < pixelList.size(); j++) {
				int yI = pixelList[i].y;
				int xI = pixelList[i].x;
				int yJ = pixelList[j].y;
				int xJ = pixelList[j].x;

				float *wI = simWeights->line(yI * numCols + xI);
				float *wJ = simWeights->line(yJ * numCols + xJ);
				int idxJatI = (yJ - yI + PATCHRADIUS) * PATCHWIDTH + (xJ - xI + PATCHRADIUS);
				int idxIatJ = (yI - yJ + PATCHRADIUS) * PATCHWIDTH + (xI - xJ + PATCHRADIUS);
				wI[idxJatI] = 1.f;
				wJ[idxIatJ] = 1.f;
			}
		}
	}
	bs::Timer::Toc();
}

static void VisualizeSimilarityWegiths(cv::Mat &img, MCImg<float> &simWeights)
{
	cv::Mat canvas = img.clone();
	void *callbackParams[] = { &canvas, &simWeights };
	cv::imshow("OnMouseVisualizeSimilarityWegiths", canvas);
	void OnMouseVisualizeSimilarityWegiths(int event, int x, int y, int flags, void *param);
	cv::setMouseCallback("OnMouseVisualizeSimilarityWegiths", OnMouseVisualizeSimilarityWegiths, callbackParams);
	cv::waitKey(0);
}

static void ConstructBaryCentersAndPixelLists(int numSegs, cv::Mat &labelMap,
	std::vector<cv::Point2f> &baryCenters, std::vector<std::vector<cv::Point2i>> &segPixelLists)
{
	baryCenters = std::vector<cv::Point2f>(numSegs, cv::Point2f(0, 0));
	segPixelLists.resize(numSegs);

	int numRows = labelMap.rows, numCols = labelMap.cols;
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			int id = labelMap.at<int>(y, x);
			baryCenters[id] += cv::Point2f(x, y);
			segPixelLists[id].push_back(cv::Point2i(x, y));
		}
	}

	for (int id = 0; id < numSegs; id++) {
		baryCenters[id].x /= (float)segPixelLists[id].size();
		baryCenters[id].y /= (float)segPixelLists[id].size();
	}


	// Reorder the labeling such that the segments proceed in roughly scanline order.
	// Divide the canvas by horizontal stripes and then determine the ordering of each stripe accordingly
	std::vector<std::pair<cv::Point2f, int>> centroidIdPairs(baryCenters.size());
	for (int i = 0; i < baryCenters.size(); i++) {
		centroidIdPairs[i] = std::pair<cv::Point2f, int>(baryCenters[i], i);
	}
	
	std::sort(centroidIdPairs.begin(), centroidIdPairs.end(), SortByRowCoord());
	float rowMargin = sqrt((numRows * numCols) / (float)numSegs);	// avg segment side length.
	printf("rowMargin = %.2f\n", rowMargin);

	int headIdx = 0;
	for (double y = 0; y <= numRows; y += rowMargin) {
		int idx = headIdx;
		while (idx < numSegs && centroidIdPairs[idx].first.y < y + rowMargin) {
			idx++;
		}
		if (headIdx < numSegs) {	// to ensure that we do not have access violation at headIdx
			std::sort(&centroidIdPairs[headIdx], &centroidIdPairs[0] + idx, SortByColCoord());
		}
		headIdx = idx;
	}

	std::vector<std::vector<cv::Point2i>> tmpPixelLists(numSegs);
	for (int id = 0; id < numSegs; id++) {
		baryCenters[id] = centroidIdPairs[id].first;
		tmpPixelLists[id] = segPixelLists[centroidIdPairs[id].second];
		for (int k = 0; k < tmpPixelLists[id].size(); k++) {
			int y = tmpPixelLists[id][k].y;
			int x = tmpPixelLists[id][k].x;
			labelMap.at<int>(y, x) = id;
		}
	}
	segPixelLists = tmpPixelLists;



	cv::Mat canvas(numRows, numCols, CV_8UC3);
#if 0
	for (int id = 0; id < numSegs; id++) {
		canvas.setTo(cv::Vec3b(0, 0, 0));
		for (int k = 0; k < segPixelLists[id].size(); k++) {
			cv::Point2i p = segPixelLists[id][k];
			canvas.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(255, 255, 255);
		}
		cv::imshow("segment", canvas);
		cv::waitKey(0);
	}
#endif
#if 0
	cv::Point2f oldEndPt(0, 0);
	int stepSz = numCols / 8;
	for (int i = 0; i < numSegs; i += stepSz) {
		for (int j = i; j < i + stepSz && j < numSegs; j++) {
			cv::Point2f newEndPt = baryCenters[j];
			cv::line(canvas, oldEndPt, newEndPt, cv::Scalar(0, 0, 255, 255), 1, CV_AA);
			oldEndPt = newEndPt;
		}
		cv::imshow("process", canvas);
		cv::waitKey(0);
	}
#endif
}

static void ConstructNeighboringSegmentGraph(int numSegs, cv::Mat &labelMap, 
	std::vector<std::vector<int>> &nbGraph, bool useLargeNeighborhood = false)
{
	// This function has assumed all pixels are labeled.
	// And the label index starts from zero.

	std::vector<std::set<int>> nbIdxSets(numSegs);
	nbGraph.resize(numSegs);

	int numRows = labelMap.rows, numCols = labelMap.cols;
	for (int yc = 0; yc < numRows; yc++) {
		for (int xc = 0; xc < numCols; xc++) {			
			for (int y = yc - 1; y <= yc + 1; y++) {
				for (int x = xc - 1; x <= xc + 1; x++) {
					if (InBound(y, x, numRows, numCols)
						&& labelMap.at<int>(yc, xc) != labelMap.at<int>(y, x)) {
						int id1 = labelMap.at<int>(yc, xc);
						int id2 = labelMap.at<int>(y, x);
						nbIdxSets[id1].insert(id2);
						nbIdxSets[id2].insert(id1);
					}
				}
			}
		}
	}
	if (!useLargeNeighborhood) {
		for (int id = 0; id < numSegs; id++) {
			nbGraph[id] = std::vector<int>(nbIdxSets[id].begin(), nbIdxSets[id].end());
		}
	}
	else {
		std::vector<std::set<int>> extNbIdxSets(numSegs);
		for (int id = 0; id < numSegs; id++) {
			// Merge the neighbors of its neighbors to form bigger neighborhood.
			std::set<int> &nbs = nbIdxSets[id];
			extNbIdxSets[id] = nbs;
			for (std::set<int>::iterator it = nbs.begin(); it != nbs.end(); it++) {
				int nbId = *it;
				std::set<int> &newNbs = nbIdxSets[nbId];
				extNbIdxSets[id].insert(newNbs.begin(), newNbs.end());
			}
		}
		for (int id = 0; id < numSegs; id++) {
			nbGraph[id] = std::vector<int>(extNbIdxSets[id].begin(), extNbIdxSets[id].end());
		}
	}
	
}

static void ComputeSegmentSimilarityWeights(cv::Mat &img, std::vector<std::vector<int>> &nbGraph,
	std::vector<std::vector<cv::Point2i>> &segPixelLists, std::vector<std::vector<float>> &nbSimWeights)
{
	int numSegs = nbGraph.size();
	std::vector<cv::Vec3f> meanColors(numSegs, cv::Vec3f(0, 0, 0));

	for (int id = 0; id < numSegs; id++) {
		std::vector<cv::Point2i> &pixelList = segPixelLists[id];
		for (int k = 0; k < pixelList.size(); k++) {
			int y = pixelList[k].y;
			int x = pixelList[k].x;
			meanColors[id] += img.at<cv::Vec3b>(y, x);
		}
		meanColors[id][0] /= (float)pixelList.size();
		meanColors[id][1] /= (float)pixelList.size();
		meanColors[id][2] /= (float)pixelList.size();
	}

	nbSimWeights.resize(numSegs);
	for (int id = 0; id < numSegs; id++) {
		nbSimWeights[id].resize(nbGraph[id].size());
		for (int k = 0; k < nbGraph[id].size(); k++) {
			int nbId = nbGraph[id][k];
			float w = exp(-L1Dist(meanColors[id], meanColors[nbId]) / 30.f);
			nbSimWeights[id][k] = w;
			if (w < 0.3) { w = 0.f; }
			//printf("nbSimWeights: %f\n", w); 
		}
	}
}

static void SlantedPlanesToNormalDepth(std::vector<SlantedPlane> &slantedPlanes,
	std::vector<cv::Point2f> &baryCenters, std::vector<cv::Vec3f> &n, std::vector<float> &d)
{
	int numSegs = slantedPlanes.size();
	for (int id = 0; id < numSegs; id++) {
		SlantedPlane &p = slantedPlanes[id];
		n[id] = cv::Vec3f(p.nx, p.ny, p.nz);
		d[id] = p.ToDisparity(baryCenters[id].y, baryCenters[id].x);
	}
}

static std::vector<SlantedPlane> NormalDepthToSlantedPlanes(std::vector<cv::Vec3f> &n, 
	std::vector<float> &d, std::vector<cv::Point2f> &baryCenters)
{
	ASSERT(n.size() == d.size());
	std::vector<SlantedPlane> slantedPlanes(n.size());
	for (int id = 0; id < n.size(); id++) {
		float nx = n[id][0];
		float ny = n[id][1];
		float nz = n[id][2];
		float z = d[id];
		float x = baryCenters[id].x;
		float y = baryCenters[id].y;
		slantedPlanes[id] = SlantedPlane::ConstructFromNormalDepthAndCoord(nx, ny, nz, z, y, x);
	}
	return slantedPlanes;
}

static cv::Mat SegmentLabelToDisparityMap(int numRows, int numCols, 
	std::vector<SlantedPlane> &slantedPlanes, std::vector<std::vector<cv::Point2i>> &segPixelLists)
{
	cv::Mat dispMap(numRows, numCols, CV_32FC1);
	for (int id = 0; id < segPixelLists.size(); id++) {
		std::vector<cv::Point2i> &pixelList = segPixelLists[id];
		for (int i = 0; i < pixelList.size(); i++) {
			int y = pixelList[i].y;
			int x = pixelList[i].x;
			dispMap.at<float>(y, x) = slantedPlanes[id].ToDisparity(y, x);
		}
	}
	return dispMap;
}

static cv::Mat SegmentLabelToDisparityMap(int numRows, int numCols, std::vector<cv::Vec3f> &n, std::vector<float> &d,
	std::vector<cv::Point2f> &baryCenters, std::vector<std::vector<cv::Point2i>> &segPixelLists)
{
	std::vector<SlantedPlane> slantedPlanes = NormalDepthToSlantedPlanes(n, d, baryCenters);
	return SegmentLabelToDisparityMap(numRows, numCols, slantedPlanes, segPixelLists);
}

static float ConstrainedPatchMatchCost(float yc, float xc, SlantedPlane &newGuess, 
	cv::Vec3f &mL, float vL, float maxDisp, float theta, int sign)
{
	float dataCost = PatchMatchSlantedPlaneCost(yc + 0.5, xc + 0.5, newGuess, sign);
	cv::Vec3f nL(newGuess.nx, newGuess.ny, newGuess.nz);
	float uL = newGuess.ToDisparity(yc, xc) / maxDisp;
	vL /= maxDisp;
	//float smoothCost = (nL - mL).dot(nL - mL);// +(uL - vL) * (uL - vL);
	float nx = nL[0], ny = nL[1], mx = mL[0], my = mL[1];
	float smoothCost = (nx - mx) * (nx - mx) + (ny - my) * (ny - my);

	extern float ARAP_LAMBDA;
	extern float ARAP_SIGMA;
	return ARAP_LAMBDA * dataCost + 0.5 * ARAP_SIGMA * theta * smoothCost;
}

static void ImproveGuess(float y, float x, SlantedPlane &oldGuess, SlantedPlane &newGuess, 
	float &bestCost, cv::Vec3f &mL, float vL, float maxDisp, float theta, int sign)
{
	float newCost = ConstrainedPatchMatchCost(y, x, newGuess, mL, vL, maxDisp, theta, sign);
	if (newCost < bestCost) {
		bestCost = newCost;
		oldGuess = newGuess;
	}
}

static void PropagateAndRandomSearch(int id, int sign, float maxDisp, float theta, float gSmooth,
	cv::Point2f &srcPos, std::vector<SlantedPlane> &slantedPlanes, std::vector<float> &bestCosts, 
	std::vector<std::vector<int>> &nbGraph, std::vector<cv::Vec3f> &mL, cv::vector<float> &vL)
{
	float y = srcPos.y;
	float x = srcPos.x;

	// Spatial propgation
	std::vector<int> &nbIds = nbGraph[id];
	for (int i = 0; i < nbIds.size(); i++) {
		SlantedPlane newGuess = slantedPlanes[nbIds[i]];
		ImproveGuess(y, x, slantedPlanes[id], newGuess, bestCosts[id], 
			mL[id], vL[id], maxDisp, theta * gSmooth, sign);
	}

	// Random search

	if (1 && (y > 330)) {
		// Special treament for ground planes
		for (int retry = 0; retry < 10; retry++) {
			float curDisp = slantedPlanes[id].ToDisparity(y, x);
			SlantedPlane newGuess = SlantedPlane::ConstructFromGroundPlaneProposal(y, x, curDisp, /*zRadius*/0);
			ImproveGuess(y, x, slantedPlanes[id], newGuess, bestCosts[id],
				mL[id], vL[id], maxDisp, theta * gSmooth, sign);
		}
	}

	float zRadius = maxDisp / 2.f;
	float nRadius = 1.f;
	while (zRadius >= 0.1f) {
		SlantedPlane newGuess = SlantedPlane::ConstructFromRandomPertube(slantedPlanes[id], y, x, nRadius, zRadius);
		ImproveGuess(y, x, slantedPlanes[id], newGuess, bestCosts[id], 
			mL[id], vL[id], maxDisp, theta * gSmooth, sign);
	
		zRadius /= 2.f;
		nRadius /= 2.f;
	}

	
}

static void ConstrainedPatchMatchOnSegments(int sign, float theta, float maxDisp, int maxIters, bool doRandInit,
	std::vector<cv::Vec3f> &nL, std::vector<float> &uL, std::vector<cv::Vec3f> &mL, std::vector<float> &vL,
	std::vector<float>& gSmoothL, std::vector<cv::Point2f> &baryCentersL, std::vector<std::vector<int>> &nbGraphL,
	std::vector<std::vector<float>> &nbSimWeightsL, std::vector<std::vector<cv::Point2i>> &segPixelListsL)
{
	// Assemble the n, d to SlantedPlane
	// Random init if first iter
	int numSegsL = baryCentersL.size();
	std::vector<SlantedPlane> slantedPlanesL;
	if (doRandInit) {
		slantedPlanesL.resize(numSegsL);
		for (int id = 0; id < numSegsL; id++) {
			float x = baryCentersL[id].x;
			float y = baryCentersL[id].y;
			// FIXME: have to make sure that the nz is always positive.
			slantedPlanesL[id].SelfConstructFromRandomInit(y, x, maxDisp);
		}
	}
	else {
		slantedPlanesL = NormalDepthToSlantedPlanes(nL, uL, baryCentersL);
	}

	std::vector<float> bestCostsL(numSegsL);
	for (int id = 0; id < numSegsL; id++) {
		float x = baryCentersL[id].x;
		float y = baryCentersL[id].y;
		bestCostsL[id] = ConstrainedPatchMatchCost(y, x, slantedPlanesL[id], mL[id], vL[id], maxDisp, theta, sign);
	}
	

	// Propagation and Random Search
	// FIXME: you have to make sure that the ordering is roughly a scanline ordering.
	std::vector<int> idListL(numSegsL);
	for (int i = 0; i < numSegsL; i++) {
		idListL[i] = i;
	}

	for (int round = 0; round < maxIters; round++) {
		printf("ConstrainedPatchMatchOnSegments round %d ...\n", round);
		for (int i = 0; i < numSegsL; i++) {
			int id = idListL[i];
			PropagateAndRandomSearch(id, sign, maxDisp, theta, gSmoothL[id], 
				baryCentersL[id], slantedPlanesL, bestCostsL, nbGraphL, mL, vL);
		}
		std::reverse(idListL.begin(), idListL.end());
	}


	// Deassemble SlantedPlane to n, d
	for (int id = 0; id < numSegsL; id++) {
		SlantedPlane &p = slantedPlanesL[id]; 
		nL[id] = cv::Vec3f(p.nx, p.ny, p.nz);
		uL[id] = p.ToDisparity(baryCentersL[id].y, baryCentersL[id].x);
	}
}

static void SolveARAPSmoothness(float theta, std::vector<float> &confidence,
	std::vector<std::vector<int>> &nbGraph, std::vector<std::vector<float>> &nbSimWeights, 
	std::vector<cv::Vec3f> &n, std::vector<float> &u, std::vector<cv::Vec3f> &m, std::vector<float> &v)
{
	// This function solves the following sparse lienar system
	//   A_{NxN} * X = b_{Nx4}
	// where 
	//   A_ii = \theta_i + \lambda * \sum_j w_ij,
	//	 A_ij = -\lambda * w_ij
	//	 b_i  = \theta_i * n_i
	// N is the number of segments, w_ij is the similarity weights between neighboring segments.
	printf("SolveARAPSmoothness...\n");
	extern float ARAP_SIGMA;

	int numSegs = nbGraph.size();
	int numNonZeroEntries = 0;
	for (int id = 0; id < numSegs; id++) {
		numNonZeroEntries += 1 + nbGraph[id].size();
	}

	std::vector<float> g = confidence;
	// Give the data's confidence a 0.1 offset
	//for (int i = 0; i < g.size(); i++) {
	//	g[i] += 0.1f;
	//}
	std::vector<Eigen::Triplet<float>> entries;
	entries.reserve(numNonZeroEntries);
	for (int id = 0; id < numSegs; id++) {
		float wsum = 0.f;
		std::vector<int> &nbInds = nbGraph[id];
		std::vector<float> &nbWeights = nbSimWeights[id];

		for (int k = 0; k < nbInds.size(); k++) {
			int i = id;
			int j = nbInds[k];
			entries.push_back(Eigen::Triplet<float>(i, j, -nbWeights[k]));
			wsum += nbWeights[k];
		}
		entries.push_back(Eigen::Triplet<float>(id, id, ARAP_SIGMA * theta * g[id] + wsum));
	}


	// FIXME: you have to make sure the matrix is store in row-major order, to avoid any potential risk.
	// In defense: I think no matter what storing order is used, the accessing is still (y, x).
	Eigen::MatrixXf b(numSegs, 4);
	for (int id = 0; id < numSegs; id++) {
		b.coeffRef(id, 0) = ARAP_SIGMA * theta * g[id] * n[id][0];
		b.coeffRef(id, 1) = ARAP_SIGMA * theta * g[id] * n[id][1];
		b.coeffRef(id, 2) = ARAP_SIGMA * theta * g[id] * n[id][2];
		b.coeffRef(id, 3) = ARAP_SIGMA * theta * g[id] * u[id];
	}

	Eigen::SparseMatrix<float> A(numSegs, numSegs);
	A.setFromTriplets(entries.begin(), entries.end());
	Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>> chol(A);
	Eigen::MatrixXf M = chol.solve(b);

	for (int id = 0; id < numSegs; id++) {
		// The solution of the sparse system does not necessarily satisfy the 
		// unit-norm constraint. You should either normalize (nx,ny,nz) or Get nz by
		// sqrt(1-nx*nx-ny*ny)
		m[id][0] = M.coeffRef(id, 0);
		m[id][1] = M.coeffRef(id, 1);
		m[id][2] = M.coeffRef(id, 2);
		v[id]    = M.coeffRef(id, 3);
		m[id] = cv::normalize(m[id]);
		// Check for NaN values.
		if (isnan(m[id][0]) || isnan(m[id][1]) || isnan(m[id][2])) {
			printf("\nNaN value detected!!!!!!\n\n");
		}
	}
}

static void ARAPPostProcess(int numRows, int numCols, std::vector<cv::Vec3f> &nL, std::vector<cv::Vec3f> &nR, 
	std::vector<float> &uL, std::vector<float> &uR, std::vector<float> &gL, std::vector<float> &gR,
	std::vector<cv::Point2f> &baryCentersL, std::vector<cv::Point2f> &baryCentersR,
	std::vector<std::vector<int>> &nbGraphL, std::vector<std::vector<int>> &nbGraphR, 
	std::vector<std::vector<float>> &nbSimWeightsL, std::vector<std::vector<float>> &nbSimWeightsR, 
	std::vector<std::vector<cv::Point2i>> &segPixelListsL, std::vector<std::vector<cv::Point2i>> &segPixelListsR)
{	
	std::vector<SlantedPlane> slantedPlanesL = NormalDepthToSlantedPlanes(nL, uL, baryCentersL);
	std::vector<SlantedPlane> slantedPlanesR = NormalDepthToSlantedPlanes(nR, uR, baryCentersR);

	// Step 1 - CrossCheck
	cv::Mat dispL = SegmentLabelToDisparityMap(numRows, numCols, slantedPlanesL, segPixelListsL);
	cv::Mat dispR = SegmentLabelToDisparityMap(numRows, numCols, slantedPlanesR, segPixelListsR);

	cv::Mat validPixelMapL = CrossCheck(dispL, dispR, -1);
	cv::Mat validPixelMapR = CrossCheck(dispR, dispL, +1);

	gL = DetermineConfidence(validPixelMapL, segPixelListsL);
	gR = DetermineConfidence(validPixelMapR, segPixelListsR);

	// Step 2 - Occlusion Filling
	// Replace the low-confidence triangles with their high-confidence neighbors
	SegmentOcclusionFilling(numRows, numCols, slantedPlanesL, baryCentersL, nbGraphL, gL, segPixelListsL);
	SegmentOcclusionFilling(numRows, numCols, slantedPlanesR, baryCentersR, nbGraphR, gR, segPixelListsR);

	// Step 3 - WMF
	// Finally, an optional pixelwise filtering
	// currently left empty.

	// Step 4 - Ouput disparity and confidence
	dispL = SegmentLabelToDisparityMap(numRows, numCols, slantedPlanesL, segPixelListsL);
	dispR = SegmentLabelToDisparityMap(numRows, numCols, slantedPlanesR, segPixelListsR);

	validPixelMapL = CrossCheck(dispL, dispR, -1); 
	validPixelMapR = CrossCheck(dispR, dispL, +1);

	//gL = DetermineConfidence(validPixelMapL, segPixelListsL);
	//gR = DetermineConfidence(validPixelMapR, segPixelListsR);

	SlantedPlanesToNormalDepth(slantedPlanesL, baryCentersL, nL, uL);
	SlantedPlanesToNormalDepth(slantedPlanesR, baryCentersR, nR, uR);

	cv::Mat confidenceImgL = DrawSegmentConfidenceMap(numRows, numCols, gL, segPixelListsL);
	cv::Mat confidenceImgR = DrawSegmentConfidenceMap(numRows, numCols, gR, segPixelListsR);
	//cv::imshow("confidenceL", confidenceImgL);
#if 1
	extern std::string ROOTFOLDER;
	evalParams.confidenceImg = &confidenceImgL;
	evalParams.n = &nL;
	evalParams.u = &uL;
	EvaluateDisparity(ROOTFOLDER, dispL, 0.5f, &evalParams, "OnMouseTestARAP");
#endif
}

static void RunARAP(std::string rootFolder, cv::Mat &imL, cv::Mat &imR)
{
	int numRows = imL.rows, numCols = imL.cols;
	int numDisps, maxDisp, visualizeScale;
	SetupStereoParameters(rootFolder, numDisps, maxDisp, visualizeScale);
	extern int INTERP_ONLINE;
	if (INTERP_ONLINE) {
		InitGlobalColorGradientFeatures(imL, imR);
	}
	else {
		InitGlobalDsiAndSimWeights(imL, imR, numDisps);
	}

	// Segmentize/Triangulize both views
	cv::Mat labelMapL, labelMapR, contourImgL, contourImgR;
	extern int SEGMENT_LEN;
	//int numPreferedRegions = (numRows * numCols) / (SEGMENT_LEN * SEGMENT_LEN);
	//int numPreferedRegions = (numCols / 8 + 1) * (numRows / 8 + 1);
	int numPreferedRegions = 1000;
	float compactness = 20.f;
	int numSegsL = SLICSegmentation(imL, numPreferedRegions, compactness, labelMapL, contourImgL);
	int numSegsR = SLICSegmentation(imR, numPreferedRegions, compactness, labelMapR, contourImgR);

	std::vector<cv::Point2f> baryCentersL, baryCentersR;
	std::vector<std::vector<cv::Point2i>> segPixelListsL, segPixelListsR;
	ConstructBaryCentersAndPixelLists(numSegsL, labelMapL, baryCentersL, segPixelListsL);
	ConstructBaryCentersAndPixelLists(numSegsR, labelMapR, baryCentersR, segPixelListsR);

	std::vector<std::vector<int>> nbGraphL, nbGraphR;	
	ConstructNeighboringSegmentGraph(numSegsL, labelMapL, nbGraphL, 0);
	ConstructNeighboringSegmentGraph(numSegsR, labelMapR, nbGraphR, 0);

	std::vector<std::vector<float>> nbSimWeightsL, nbSimWeightsR;
	ComputeSegmentSimilarityWeights(imL, nbGraphL, segPixelListsL, nbSimWeightsL);
	ComputeSegmentSimilarityWeights(imR, nbGraphR, segPixelListsR, nbSimWeightsR);
	
	//srand(2466700234);
	//ModifyGlobalSimilarityWeights(-1, labelMapL, baryCentersL, segPixelListsL);
	//ModifyGlobalSimilarityWeights(+1, labelMapR, baryCentersR, segPixelListsR);
	//VisualizeSimilarityWegiths(segImgL, gSimWeightsL);


	// Variables being optimized:
	// n, m are normals; u, v are disparities; g are confidence.
	std::vector<cv::Vec3f>	nL(numSegsL), nR(numSegsR), mL(numSegsL), mR(numSegsR);
	std::vector<float>		uL(numSegsL), uR(numSegsR), vL(numSegsL), vR(numSegsR);
	std::vector<float>		gDataL(numSegsL, 0.f), gSmoothL(numSegsR, 0.f);  
	std::vector<float>		gDataR(numSegsL, 0.f), gSmoothR(numSegsR, 0.f);

	
	evalParams.numRows			= numRows;
	evalParams.numCols			= numCols;
	evalParams.labelMap			= &labelMapL;
	evalParams.baryCenters		= &baryCentersL;
	evalParams.nbGraph			= &nbGraphL;
	evalParams.segPixelLists	= &segPixelListsL;

	// Optimization starts
	ConstrainedPatchMatchOnSegments(-1, 0.f, maxDisp, 8, true,
		nL, uL, mL, vL, gSmoothL, baryCentersL, nbGraphL, nbSimWeightsL, segPixelListsL);
	//ConstrainedPatchMatchOnSegments(+1, 0.f, maxDisp, 4, true,
	//	nR, uR, mR, vR, gSmoothR, baryCentersR, nbGraphR, nbSimWeightsR, segPixelListsR);
	printf("Evaluating *** DispDataL ***\n");
	cv::Mat &disp = SegmentLabelToDisparityMap(numRows, numCols, nL, uL, baryCentersL, segPixelListsL);
	evalParams.n = &nL;
	evalParams.u = &uL;
	EvaluateDisparity(rootFolder, disp, 0.5f, &evalParams, "OnMouseTestARAP");


	extern int		ARAP_MAX_ITERS;
	extern float	ARAP_THETASCALE;
	extern float	ARAP_THETAOFFSET;

	for (int iter = 0; iter < ARAP_MAX_ITERS; iter++) {

		float theta = smoothstep<float>(((iter) / ((float)ARAP_MAX_ITERS))) * 1.0f;
		printf("\n\n=========== theta = %f ===========\n", theta); 
		
		//////////////////////////////////////////////////////////////////////////////////////////
		// Optimize E_DATA
		ConstrainedPatchMatchOnSegments(-1, theta, maxDisp, 2, false,
			nL, uL, mL, vL, gSmoothL, baryCentersL, nbGraphL, nbSimWeightsL, segPixelListsL);
		ConstrainedPatchMatchOnSegments(+1, theta, maxDisp, 2, false,
			nR, uR, mR, vR, gSmoothR, baryCentersR, nbGraphR, nbSimWeightsR, segPixelListsR);

		printf("Evaluating *** DispDataL ***\n");
		cv::Mat &dispDataL = SegmentLabelToDisparityMap(numRows, numCols, nL, uL, baryCentersL, segPixelListsL);
		evalParams.confidenceImg = NULL;
		evalParams.n = &nL;
		evalParams.u = &uL;
		EvaluateDisparity(rootFolder, dispDataL, 0.5f, &evalParams, "OnMouseTestARAP");

		// Post process data map robustness
		ARAPPostProcess(numRows, numCols, nL, nR, uL, uR, gDataL, gDataR, baryCentersL, baryCentersR,
			nbGraphL, nbGraphR, nbSimWeightsL, nbSimWeightsR, segPixelListsL, segPixelListsR);
		//////////////////////////////////////////////////////////////////////////////////////////


		//////////////////////////////////////////////////////////////////////////////////////////
		// Optimize E_SMOOTH
		SolveARAPSmoothness(theta + 0.1f, gDataL, nbGraphL, nbSimWeightsL, nL, uL, mL, vL);
		SolveARAPSmoothness(theta + 0.1f, gDataR, nbGraphR, nbSimWeightsR, nR, uR, mR, vR);

		printf("Evaluating *** DispSmoothL ***\n");
		cv::Mat &dispSmoothL = SegmentLabelToDisparityMap(numRows, numCols, mL, vL, baryCentersL, segPixelListsL);
		evalParams.confidenceImg = NULL;
		evalParams.n = &mL;
		evalParams.u = &vL;
		EvaluateDisparity(rootFolder, dispSmoothL, 0.5f, &evalParams, "OnMouseTestARAP");

		// Post process smooth map for robustness
		ARAPPostProcess(numRows, numCols, mL, mR, vL, vR, gSmoothL, gSmoothR, baryCentersL, baryCentersR,
			nbGraphL, nbGraphR, nbSimWeightsL, nbSimWeightsR, segPixelListsL, segPixelListsR);
		//////////////////////////////////////////////////////////////////////////////////////////
	}

	cv::Mat &dispDataL = SegmentLabelToDisparityMap(numRows, numCols, nL, uL, baryCentersL, segPixelListsL);
	std::string workingDir = "d:/data/stereo/" + rootFolder;
	std::string plyFilePath = workingDir + "/dispDataL.ply";
	SaveDisparityToPly(dispDataL, imL, maxDisp, workingDir, plyFilePath);
}

void TestARAP()
{
#if 0
	extern std::string ROOTFOLDER;
	std::string rootFolder = ROOTFOLDER;

	cv::Mat imL = cv::imread("D:/data/stereo/" + rootFolder + "/im2.png");
	cv::Mat imR = cv::imread("D:/data/stereo/" + rootFolder + "/im6.png");

#else
	cv::Mat imL, imR;
	extern std::string ROOTFOLDER;
	if (ROOTFOLDER == "KITTI") {
		extern bool useKitti;
		extern std::string kittiTestCaseId;

		imL = cv::imread("D:/data/KITTI/training/colored_0/" + kittiTestCaseId + ".png");
		imR = cv::imread("D:/data/KITTI/training/colored_1/" + kittiTestCaseId + ".png");
	}
	else {
		imL = cv::imread("D:/data/stereo/" + ROOTFOLDER + "/im2.png");
		imR = cv::imread("D:/data/stereo/" + ROOTFOLDER + "/im6.png");
	}
#endif

	RunARAP(ROOTFOLDER, imL, imR);
}