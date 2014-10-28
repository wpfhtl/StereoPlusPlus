#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <stack>
#include <set>
#include <string>

#include "Eigen/Dense"
#include "Eigen/SparseCore"
#include "Eigen/SparseCholesky"
#include "Eigen/SparseLU"
#include "Eigen/SparseQR"
#include "Eigen/IterativeLinearSolvers"
//#include "Eigen/Core"

#include "StereoAPI.h"
#include "SlantedPlane.h"
#include "PostProcess.h"
#include "ReleaseAssert.h"
#include "Timer.h"



#include "png++/png.hpp"
#include "spsstereo/SGMStereo.h"
#include "spsstereo/SPSStereo.h"
#include "spsstereo/defParameter.h"
#include "EvalParams.h"



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

cv::Mat DrawSegmentImage(cv::Mat &labelMap)
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

void ConstructBaryCentersAndPixelLists(int numSegs, cv::Mat &labelMap,
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
	
	

#if 0
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
#else 
	std::sort(centroidIdPairs.begin(), centroidIdPairs.end(), SortByColCoord());
	float rowMargin = sqrt((numRows * numCols) / (float)numSegs);	// avg segment side length.
	printf("rowMargin = %.2f\n", rowMargin);
	int headIdx = 0;
	for (double x = 0; x <= numCols; x += rowMargin) {
		int idx = headIdx;
		while (idx < numSegs && centroidIdPairs[idx].first.x < x + rowMargin) {
			idx++;
		}
		if (headIdx < numSegs) {	// to ensure that we do not have access violation at headIdx
			std::sort(&centroidIdPairs[headIdx], &centroidIdPairs[0] + idx, SortByRowCoord());
		}
		headIdx = idx;
	}
#endif

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
	/*int stepSz = numCols / 8;*/
	int stepSz = 1;
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
			float w = exp(-L1Dist(meanColors[id], meanColors[nbId]) / 15.f);
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

float ConstrainedPatchMatchCost(float yc, float xc, SlantedPlane &newGuess, 
	cv::Vec3f mL, float vL, float maxDisp, float theta, int sign)
{
	float dataCost = PatchMatchSlantedPlaneCost(yc + 0.5, xc + 0.5, newGuess, sign);

#if 0
	float oracleDiff = 0.f;
	extern cv::Mat gDispSGML, gDispSGMR;
	cv::Mat &dispSGM = (sign == -1 ? gDispSGML : gDispSGMR);
	
	if (!dispSGM.empty()) {
		if (dispSGM.at<float>(yc, xc) > 0) {
			oracleDiff = std::min(30.f, std::abs(dispSGM.at<float>(yc, xc) - newGuess.ToDisparity(yc, xc)));
			oracleDiff *= 25 * 2; // balance the scale of tensor matching cost and oracle matching cost.
		}
	}
	//dataCost = 0.7 * dataCost + 0.3f * oracleDiff;
#endif
	 
	

	extern float ARAP_NORMALDIFF_TRUNCATION;
	extern float ARAP_DISPDIFF_TRUNCATION;

	cv::Vec3f nL(newGuess.nx, newGuess.ny, newGuess.nz);
	mL = cv::normalize(mL);		// this line must precded the folloing line.
	float nx = nL[0], ny = nL[1], nz = nL[2], mx = mL[0], my = mL[1], mz = mL[2];
	
	double normalCouplingCost = 0.0;
	if (mz < 0.5 || (mx > 0.3 && my > 0.3)) {
		// in this case the smooth plane is invalid, coupling cost doesn't count
		normalCouplingCost = 0;
	}
	else {
		// in this case the smooth plane m is valid
		nz = std::max(nz, 1e-4f);
		mz = std::max(mz, 1e-4f);
		nx /= nz; ny /= nz;
		mx /= mz; my /= mz;
		double nxDiff = std::min(ARAP_NORMALDIFF_TRUNCATION, std::abs(nx - mx));
		double nyDiff = std::min(ARAP_NORMALDIFF_TRUNCATION, std::abs(ny - my));
		normalCouplingCost = (nxDiff * nxDiff + nyDiff * nyDiff) * maxDisp * maxDisp;
	}

	
	float uL = newGuess.ToDisparity(yc, xc);
	double dDiff = std::min(ARAP_DISPDIFF_TRUNCATION, std::abs(uL - vL));
	double dispCouplingCost = dDiff * dDiff;

	double totalCouplingCost = normalCouplingCost + dispCouplingCost;
	
	extern float ARAP_LAMBDA;
	extern float ARAP_SIGMA;
	return ARAP_LAMBDA * dataCost + 0.5 * ARAP_SIGMA * theta * totalCouplingCost;
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
	//if (gSmooth < 0.5) {
	//	gSmooth = 0;
	//}
	//if (gSmooth > 0.9) {
	//	gSmooth *= 10;
	//}

	// Spatial propgation
	std::vector<int> &nbIds = nbGraph[id];
	for (int i = 0; i < nbIds.size(); i++) {
		SlantedPlane newGuess = slantedPlanes[nbIds[i]];
		ImproveGuess(y, x, slantedPlanes[id], newGuess, bestCosts[id], 
			mL[id], vL[id], maxDisp, theta * gSmooth, sign);
	}

	// Random search

	if (0 && (y > 330)) {
		// Special treament for ground planes
		for (int retry = 0; retry < 10; retry++) {
			float curDisp = slantedPlanes[id].ToDisparity(y, x);
			SlantedPlane newGuess = SlantedPlane::ConstructFromGroundPlaneProposal(y, x, curDisp, /*zRadius*/0);
			ImproveGuess(y, x, slantedPlanes[id], newGuess, bestCosts[id],
				mL[id], vL[id], maxDisp, theta * gSmooth, sign);
		}
	}

	for (int retry = 0; retry < 5; retry++) {
		float zRadius = maxDisp / 2.f;
		float nRadius = 1.f;
		while (zRadius >= 0.1f) {
			for (int retry = 0; retry < 1; retry++) {
				SlantedPlane newGuess = SlantedPlane::ConstructFromRandomPertube(slantedPlanes[id], y, x, nRadius, zRadius);
				ImproveGuess(y, x, slantedPlanes[id], newGuess, bestCosts[id],
					mL[id], vL[id], maxDisp, theta * gSmooth, sign);
			}

			zRadius /= 2.f;
			nRadius /= 2.f;
		}
	}
	

	//for (int retry = 0; retry < 10; retry++) {
	//	SlantedPlane newGuess = SlantedPlane::ConstructFromRandomInit(y, x, maxDisp);
	//	ImproveGuess(y, x, slantedPlanes[id], newGuess, bestCosts[id],
	//		mL[id], vL[id], maxDisp, theta * gSmooth, sign);
	//}

	
}

static void ConstrainedPatchMatchOnSegments(int sign, float theta, float maxDisp, int maxIters, bool doRandInit,
	std::vector<cv::Vec3f> &nL, std::vector<float> &uL, std::vector<cv::Vec3f> &mL, std::vector<float> &vL,
	std::vector<float>& gSmoothL, std::vector<cv::Point2f> &baryCentersL, std::vector<std::vector<int>> &nbGraphL,
	std::vector<std::vector<float>> &nbSimWeightsL, std::vector<std::vector<cv::Point2i>> &segPixelListsL)
{
	printf("maxDisp = %f\n", maxDisp);
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

	bs::Timer::Tic();
	for (int round = 0; round < maxIters; round++) {
		printf("ConstrainedPatchMatchOnSegments round %d ...\n", round);
		#pragma omp parallel for
		for (int i = 0; i < numSegsL; i++) {
			int id = idListL[i];
			PropagateAndRandomSearch(id, sign, maxDisp, theta, gSmoothL[id], 
				baryCentersL[id], slantedPlanesL, bestCostsL, nbGraphL, mL, vL);
		}
		std::reverse(idListL.begin(), idListL.end());
	}
	bs::Timer::Toc();


	// Deassemble SlantedPlane to n, d
	for (int id = 0; id < numSegsL; id++) {
		SlantedPlane &p = slantedPlanesL[id]; 
		nL[id] = cv::Vec3f(p.nx, p.ny, p.nz);
		uL[id] = p.ToDisparity(baryCentersL[id].y, baryCentersL[id].x);
	}
}


#if 0
static void SolveARAPSmoothness(float theta, std::vector<float> &confidence, int maxDisp,
	std::vector<std::vector<int>> &nbGraph, std::vector<std::vector<float>> &nbSimWeights, 
	std::vector<cv::Vec3f> &n, std::vector<float> &u, std::vector<cv::Vec3f> &m, std::vector<float> &v,
	std::vector<cv::Point2f> &baryCenters)
{
	// This function minimize the following objective fucntioin w.r.t. m and v
	//     \min 0.5 * \sum_{i,j} w_ij*(v_i - v_j)^2
	//          + 0.5*\theta*\sigma * \sum_{i} G_i*(v_i - u_i)^2
	// by solving the following linear system
	//     A_{NxN} * X = b_{Nx4}
	// where 
	//     A_ii = G_i*\theta*\sigma + \sum_j w_ij,
	//	   A_ij = -w_ij
	//	   b_i  = G_i*\theta*\sigma*u_i
	// N is the number of segments, 
	// w_ij is the similarity weights between neighboring segments,
	// G_i is the confidence of the segment
	printf("SolveARAPSmoothness...\n");
	extern float ARAP_SIGMA;

	int numSegs = nbGraph.size();
	int numNonZeroEntries = 0;
	for (int id = 0; id < numSegs; id++) {
		numNonZeroEntries += 1 + nbGraph[id].size();
	}

	// Set low confidence value to zero, 
	// to eleminate the "bad" contribution of corresponding segments
	std::vector<float> G = confidence;
	for (int i = 0; i < G.size(); i++) {
		if (G[i] < 0.5) {
			//G[i] = 0; 
		}
	}

	std::vector<Eigen::Triplet<double>> entries;
	entries.reserve(numNonZeroEntries);
	for (int id = 0; id < numSegs; id++) {
		float wsum = 0.f;
		std::vector<int> &nbInds = nbGraph[id];
		std::vector<float> &nbWeights = nbSimWeights[id];

		for (int k = 0; k < nbInds.size(); k++) {
			int i = id;
			int j = nbInds[k];
			entries.push_back(Eigen::Triplet<double>(i, j, -nbWeights[k]));
			wsum += nbWeights[k];
		}
		entries.push_back(Eigen::Triplet<double>(id, id, G[id] * theta * ARAP_SIGMA + wsum));
	}


	// FIXME: you have to make sure the matrix is store in row-major order, to avoid any potential risk.
	// In defense: I think no matter what storing order is used, the accessing is still (y, x).
	Eigen::MatrixXd b(numSegs, 4);
	for (int id = 0; id < numSegs; id++) {
		b.coeffRef(id, 0) = ARAP_SIGMA * theta * G[id] * n[id][0];
		b.coeffRef(id, 1) = ARAP_SIGMA * theta * G[id] * n[id][1];
		b.coeffRef(id, 2) = ARAP_SIGMA * theta * G[id] * n[id][2];
		b.coeffRef(id, 3) = ARAP_SIGMA * theta * G[id] * (u[id] / maxDisp);
	}

	Eigen::SparseMatrix<double> A(numSegs, numSegs);
	A.setFromTriplets(entries.begin(), entries.end());
	A.makeCompressed();
	Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> chol(A);
	Eigen::MatrixXd M = chol.solve(b);



	//Eigen::SparseLU<Eigen::SparseMatrix<float>> lu(A);
	//Eigen::MatrixXf M = lu.solve(b);

	for (int id = 0; id < numSegs; id++) {
		// The solution of the sparse system does not necessarily satisfy the 
		// unit-norm constraint. You should either normalize (nx,ny,nz) or Get nz by
		// sqrt(1-nx*nx-ny*ny)
		m[id][0] = M.coeffRef(id, 0);
		m[id][1] = M.coeffRef(id, 1);
		m[id][2] = M.coeffRef(id, 2);
		v[id]    = M.coeffRef(id, 3) * maxDisp;
		m[id] = cv::normalize(m[id]);
		// Check for NaN values.
		if (isnan(m[id][0]) || isnan(m[id][1]) || isnan(m[id][2])) {
			printf("\nNaN value detected!!!!!!\n\n");
		}
	}
}
#else 
static cv::Vec3f NormalizeVec3f(cv::Vec3f &v)
{
	float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	return cv::Vec3f(v[0] / len, v[1] / len, v[2] / len);
}

static void SolveARAPSmoothness(float theta, std::vector<float> &confidence, int maxDisp,
	std::vector<std::vector<int>> &nbGraph, std::vector<std::vector<float>> &nbSimWeights,
	std::vector<cv::Vec3f> &n, std::vector<float> &u, std::vector<cv::Vec3f> &m, std::vector<float> &v,
	std::vector<cv::Point2f> &baryCenters)
{
	// This function minimize the following objective fucntioin w.r.t. m and v
	//     \min 0.5 * \sum_{i,j} w_ij*(v_i - v_j)^2
	//          + 0.5*\theta*\sigma * \sum_{i} G_i*(v_i - u_i)^2
	// by solving the following linear system
	//     A_{NxN} * X = b_{Nx4}
	// where 
	//     A_ii = G_i*\theta*\sigma + \sum_j w_ij,
	//	   A_ij = -w_ij
	//	   b_i  = G_i*\theta*\sigma*u_i
	// N is the number of segments, 
	// w_ij is the similarity weights between neighboring segments,
	// G_i is the confidence of the segment
	printf("SolveARAPSmoothness...\n");
	printf("THETA = %f\n", theta);
	extern float ARAP_SIGMA;

	int numSegs = nbGraph.size();
	int numNonZeroEntries = 0;
	float avgNumNbs = 0.f;
	for (int id = 0; id < numSegs; id++) {
		numNonZeroEntries += 3 + 16 * nbGraph[id].size();
		avgNumNbs += nbGraph[id].size();
	}
	avgNumNbs /= numSegs;
	printf("avgNumNbs = %.2f\n", avgNumNbs);

	// Set low confidence value to zero, 
	// to eleminate the "bad" contribution of corresponding segments
	std::vector<float> G = confidence;
	for (int i = 0; i < G.size(); i++) {
		if (G[i] < 0.5) {
			G[i] = 0;
		}
		//if (G[i] > 0.8) {
		//	G[i] *= 10;
		//}
	}

	/*
		In defense second order cost, derivatives calculated by MATLAB
	Run:
		diati = (-nxi/nzi)*xi + (-nyi/nzi)*yi + (nxi*xi + nyi*yi + nzi*di)/nzi;
		djati = (-nxj/nzj)*xi + (-nyj/nzj)*yi + (nxj*xj + nyj*yj + nzj*dj)/nzj;

		f1 = 0.5*(diati-djati)^2;

		df1 = sym('f1', [6 1]);
		df1(1) = simplify(diff(f1, 'nxi'));
		df1(2) = simplify(diff(f1, 'nyi'));
		df1(3) = simplify(diff(f1, 'di'));
		df1(4) = simplify(diff(f1, 'nxj'));
		df1(5) = simplify(diff(f1, 'nyj'));
		df1(6) = simplify(diff(f1, 'dj'));

	Result:
	df1 =
		0
		0
		di - dj + nxj*xi - nxj*xj + nyj*yi - nyj*yj
		(xi - xj)*(di - dj + nxj*xi - nxj*xj + nyj*yi - nyj*yj)
		(yi - yj)*(di - dj + nxj*xi - nxj*xj + nyj*yi - nyj*yj)
		dj - di - nxj*xi + nxj*xj - nyj*yi + nyj*yj
	*/
	const float maxDispSquared = maxDisp * maxDisp;
	std::vector<Eigen::Triplet<double>> entries;
	entries.reserve(numNonZeroEntries);
	for (int id = 0; id < numSegs; id++) {
		float wsum = 0.f;
		std::vector<int> &nbInds = nbGraph[id];
		std::vector<float> &nbWeights = nbSimWeights[id];

		for (int k = 0; k < nbInds.size(); k++) {
			//printf("sdfasfsadfsdf\n");
			int i = id;
			int j = nbInds[k];
			float xi = baryCenters[i].x;
			float yi = baryCenters[i].y;
			float xj = baryCenters[j].x;
			float yj = baryCenters[j].y;
			float wij = nbWeights[k]; 

			//*******************************************
			//wij = (double)wij / maxDispSquared;
			//*******************************************

			// \paritial f1 \partial di
			entries.push_back(Eigen::Triplet<double>(3 * i + 2, 3 * j + 0, (xi - xj) * wij));
			entries.push_back(Eigen::Triplet<double>(3 * i + 2, 3 * j + 1, (yi - yj) * wij));
			entries.push_back(Eigen::Triplet<double>(3 * i + 2, 3 * j + 2, -1 * wij));
			entries.push_back(Eigen::Triplet<double>(3 * i + 2, 3 * i + 2, +1 * wij));
			// \paritial f1 \partial nxj
			entries.push_back(Eigen::Triplet<double>(3 * j + 0, 3 * j + 0, (xi - xj)*(xi - xj) * wij));
			entries.push_back(Eigen::Triplet<double>(3 * j + 0, 3 * j + 1, (xi - xj)*(yi - yj) * wij));
			entries.push_back(Eigen::Triplet<double>(3 * j + 0, 3 * j + 2, (xi - xj)*(-1) * wij));
			entries.push_back(Eigen::Triplet<double>(3 * j + 0, 3 * i + 2, (xi - xj)*(+1) * wij));
			// \paritial f1 \partial nyj
			entries.push_back(Eigen::Triplet<double>(3 * j + 1, 3 * j + 0, (yi - yj)*(xi - xj) * wij));
			entries.push_back(Eigen::Triplet<double>(3 * j + 1, 3 * j + 1, (yi - yj)*(yi - yj) * wij));
			entries.push_back(Eigen::Triplet<double>(3 * j + 1, 3 * j + 2, (yi - yj)*(-1) * wij));
			entries.push_back(Eigen::Triplet<double>(3 * j + 1, 3 * i + 2, (yi - yj)*(+1) * wij));
			// \paritial f1 \partial dj
			entries.push_back(Eigen::Triplet<double>(3 * j + 2, 3 * j + 0, -(xi - xj) * wij));
			entries.push_back(Eigen::Triplet<double>(3 * j + 2, 3 * j + 1, -(yi - yj) * wij));
			entries.push_back(Eigen::Triplet<double>(3 * j + 2, 3 * j + 2, +1 * wij));
			entries.push_back(Eigen::Triplet<double>(3 * j + 2, 3 * i + 2, -1 * wij));

		}
		entries.push_back(Eigen::Triplet<double>(3 * id + 0, 3 * id + 0, avgNumNbs * G[id] * theta * ARAP_SIGMA * maxDispSquared));
		entries.push_back(Eigen::Triplet<double>(3 * id + 1, 3 * id + 1, avgNumNbs * G[id] * theta * ARAP_SIGMA * maxDispSquared));
		entries.push_back(Eigen::Triplet<double>(3 * id + 2, 3 * id + 2, avgNumNbs * G[id] * theta * ARAP_SIGMA /*/ maxDispSquared*/));
	}

	// Re-parameterize n to (nx, ny, 1) form
	for (int i = 0; i < n.size(); i++) {
		if (n[i][2] < 0.5) {
			printf("NORMAL nz component too small! Forget to ban such normal in PatchMatch?\n");
			exit(-1);
		}
		n[i][0] /= n[i][2];
		n[i][1] /= n[i][2];
		n[i][2] = 1;
	}

	// FIXME: you have to make sure the matrix is store in row-major order, to avoid any potential risk.
	// In defense: I think no matter what storing order is used, the accessing is still (y, x).
	Eigen::MatrixXd b(3 * numSegs, 1);
	for (int id = 0; id < numSegs; id++) {
		// Don't forget to deal with the disparity scale
		// Don't forget to deal with the re-parameterization of m and n
		b.coeffRef(3 * id + 0, 0) = avgNumNbs * ARAP_SIGMA * theta * G[id] * n[id][0] * maxDispSquared;
		b.coeffRef(3 * id + 1, 0) = avgNumNbs * ARAP_SIGMA * theta * G[id] * n[id][1] * maxDispSquared;
		b.coeffRef(3 * id + 2, 0) = avgNumNbs * ARAP_SIGMA * theta * G[id] * u[id] /*/ maxDispSquared*/;
	}

	Eigen::SparseMatrix<double> A(3 * numSegs, 3 * numSegs);
	A.setFromTriplets(entries.begin(), entries.end());
	A.makeCompressed();
	Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> chol(A);
	Eigen::MatrixXd M = chol.solve(b);
	//Eigen::SparseLU<Eigen::SparseMatrix<double>> lu(A);
	//Eigen::MatrixXd M = lu.solve(b);
	//Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> bi;
	//bi.compute(A);
	//Eigen::MatrixXd M = bi.solve(b);
	 

	for (int id = 0; id < numSegs; id++) {
		// The solution of the sparse system does not necessarily satisfy the 
		// unit-norm constraint. You should either normalize (nx,ny,nz) or Get nz by
		// sqrt(1-nx*nx-ny*ny)
		m[id][0] = M.coeffRef(3 * id + 0, 0);
		m[id][1] = M.coeffRef(3 * id + 1, 0);
		m[id][2] = 1.f;
		v[id] = M.coeffRef(3 * id + 2, 0);
		cv::Vec3f oldm = m[id];
		//m[id] = cv::normalize(m[id],);
		m[id] = NormalizeVec3f(m[id]);
		// n has been change to the (nx,ny,1) paramterization in this function, change it back.
		n[id] = NormalizeVec3f(n[id]);
		

		// Check for NaN values.
		if (isnan(m[id][0]) || isnan(m[id][1]) || isnan(m[id][2])) {
			printf("\nNaN value detected!!!!!!\n\n");
			std::cout << "oldm = " << oldm << "\n";
		}
	}
}
#endif


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

	cv::Mat validPixelMapL = CrossCheck(dispL, dispR, -1, 1.0);
	cv::Mat validPixelMapR = CrossCheck(dispR, dispL, +1, 1.0);

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

	validPixelMapL = CrossCheck(dispL, dispR, -1, 1.0); 
	validPixelMapR = CrossCheck(dispR, dispL, +1, 1.0);

	gL = DetermineConfidence(validPixelMapL, segPixelListsL);
	gR = DetermineConfidence(validPixelMapR, segPixelListsR);

	SlantedPlanesToNormalDepth(slantedPlanesL, baryCentersL, nL, uL);
	SlantedPlanesToNormalDepth(slantedPlanesR, baryCentersR, nR, uR);

	//cv::Mat confidenceImgL = DrawSegmentConfidenceMap(numRows, numCols, gL, segPixelListsL);
	//cv::Mat confidenceImgR = DrawSegmentConfidenceMap(numRows, numCols, gR, segPixelListsR);
	//cv::imshow("confidenceL", confidenceImgL);
#if 0
	extern std::string ROOTFOLDER;
	evalParams.confidenceImg = &confidenceImgL;
	evalParams.n = &nL;
	evalParams.u = &uL;
	EvaluateDisparity(ROOTFOLDER, dispL, 0.5f, &evalParams, "OnMouseTestARAP");
#endif
}

template<typename T>
void CostVolumeFromYamaguchi(std::string &leftFilePath, std::string &rightFilePath, 
	MCImg<T> &dsiL, MCImg<T> &dsiR, int numDisps)
{
	extern int SGMSTEREO_DEFAULT_DISPARITY_TOTAL;
	SGMSTEREO_DEFAULT_DISPARITY_TOTAL = numDisps;

	std::string leftImageFilename = leftFilePath;
	std::string rightImageFilename = rightFilePath;

	png::image<png::rgb_pixel> leftImage(leftImageFilename);
	png::image<png::rgb_pixel> rightImage(rightImageFilename);



	SGMStereo sgm;
	//sgm.compute(leftImage, rightImage, i);
	sgm.initialize(leftImage, rightImage);
	sgm.computeCostImage(leftImage, rightImage);
	int numRows = dsiL.h, numCols = dsiL.w; //numDisps = dsiL.n;
	int N = numRows * numCols * numDisps;

	#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		dsiL.data[i] = sgm.leftCostImage_[i];
	}
	#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		dsiR.data[i] = sgm.rightCostImage_[i];
	}

	sgm.freeDataBuffer();
}

std::vector<cv::Vec3b> ComupteSegmentMeanLabColor(cv::Mat &labImg, cv::Mat &labelMap, int numSegs)
{
	std::vector<cv::Vec3d> colors(numSegs, cv::Vec3d(0,0,0));
	std::vector<int> segSizes(numSegs, 0);
	for (int y = 0; y < labelMap.rows; y++) {
		for (int x = 0; x < labelMap.cols; x++) {
			int id = labelMap.at<int>(y, x);
			segSizes[id]++;
			colors[id] += labImg.at<cv::Vec3b>(y, x);
		}
	}
	
	std::vector<cv::Vec3b> results(numSegs);
	for (int i = 0; i < numSegs; i++) {
		colors[i] /= (double)segSizes[i];
		results[i] = colors[i];
	}

	return results;
}

cv::Mat SetInvalidDisparityToZeros(cv::Mat &dispL, cv::Mat &validPixelMapL)
{
	cv::Mat dispCrossCheckedL = dispL.clone();
	int numRows = dispL.rows, numCols = dispL.cols;
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			if (!validPixelMapL.at<bool>(y, x)) {
				dispCrossCheckedL.at<float>(y, x) = 0.f;
			}
		}
	}
	return dispCrossCheckedL;
}

static void VisualizeSegmentConfidence(cv::Mat &labelMap,
	std::vector<std::vector<cv::Point2i>> &segPixelLists, std::vector<float> &confidences)
{
	extern int VISUALIZE_EVAL;
	if (!VISUALIZE_EVAL) {
		return;
	}
	cv::Mat confidenceImg(labelMap.rows, labelMap.cols, CV_8UC3);
	for (int id = 0; id < confidences.size(); id++) {
		float confVal = confidences[id];
		cv::Vec3b color;
		if (confVal < 0.5) {
			color = cv::Vec3b(0, 0, 255);
		}
		else {
			unsigned char c = 255 * confVal + 0.5;
			color = cv::Vec3b(c, c, c);
		}
		std::vector<cv::Point2i> &pixelList = segPixelLists[id];
		for (int i = 0; i < pixelList.size(); i++) {
			int y = pixelList[i].y;
			int x = pixelList[i].x;
			confidenceImg.at<cv::Vec3b>(y, x) = color;
		}
	}
	cv::imshow("confidence image", confidenceImg);
	cv::waitKey(0);
}

static void VisualizeSegmengSimilarityWeights(cv::Mat &img, cv::Mat &labelMap,
	std::vector<std::vector<float>> &nbGraphL, std::vector<std::vector<cv::Point2i>> &segPixelList)
{
}

double ComputARAPDataCost(std::vector<cv::Vec3f> &n, std::vector<float> &d, std::vector<cv::Point2f> &baryCenters)
{
	double totalCost = 0.0;
	for (int i = 0; i < n.size(); i++) {

		SlantedPlane sp = SlantedPlane::ConstructFromNormalDepthAndCoord(
			n[i][0], n[i][1], n[i][2], d[i], baryCenters[i].y, baryCenters[i].x);
		int yc = baryCenters[i].y + 0.5;
		int xc = baryCenters[i].x + 0.5;
		double cost = PatchMatchSlantedPlaneCost(yc, xc, sp, -1);
		if (cost > 1e+6) {
			printf("cost at segment %d > 1e+6, costValue = %lf\n", i, cost);
			printf("(sp.nx, sp.ny, sp.nz) = (%f, %f, %f),    d = %f\n", sp.nx, sp.ny, sp.nz, d[i]);
		}
		totalCost += cost;
	}
	return totalCost;
}

double ComputeARAPSmoothCost(std::vector<cv::Vec3f> &nn, std::vector<float> &d, std::vector<cv::Point2f> &baryCenters,
	std::vector<std::vector<int>> &nbGraph, std::vector<std::vector<float>> &nbSimWeights)
{
	// By definition the cost are computed in the (nx,ny,1) parameterization.
	std::vector<cv::Vec3f> n(nn.size());
	for (int i = 0; i < nn.size(); i++) {
		float nx = nn[i][0];
		float ny = nn[i][1];
		float nz = nn[i][2];
		n[i] = cv::Vec3f(nx / nz, ny / nz, 1.f);
	}
	

	double totalCost = 0.0;
	double avgNumNbs = 0;
	for (int i = 0; i < nbGraph.size(); i++) {
		int idI = i;
		cv::Point2f centerI = baryCenters[idI];
		float dispIatI = d[idI];

		for (int j = 0; j < nbGraph[i].size(); j++) {
			int idJ = nbGraph[i][j];
			cv::Point2f centerJ = baryCenters[idJ];
			SlantedPlane planeJ = SlantedPlane::ConstructFromNormalDepthAndCoord(
				n[idJ][0], n[idJ][1], n[idJ][2], d[idJ], centerJ.y, centerJ.x);

			float wij = nbSimWeights[i][j];
			float dispJatI = planeJ.ToDisparity(centerI.y, centerI.x);
			totalCost += wij * (dispIatI - dispJatI) * (dispIatI - dispJatI);
		}

		avgNumNbs += nbGraph[i].size();
	}
	avgNumNbs /= nbGraph.size();
	return totalCost / avgNumNbs;
}

cv::Vec2d ComputeARAPCouplingCost(std::vector<cv::Vec3f> &n, std::vector<float> &u,
	std::vector<cv::Vec3f> &m, std::vector<float> &v, std::vector<float> &G, int maxDisp)
{

	double totalCouplingCost = 0.0;
	double totalCouplingCostWithConfidence = 0.0;
	for (int i = 0; i < n.size(); i++) {
		if (std::abs(n[i][2]) < 0.5 || std::abs(m[i][2]) < 0.5) {
			//printf("\nWARNING: normal nz < 0.5, potential bugs exist!!!!\n");
			//printf("at segment %d, (nx,ny,nz)=(%f,%f,%f),  (mx,my,mz)=(%f,%f,%f)\n",
			//	i, n[i][0], n[i][1], n[i][2], m[i][0], m[i][1], m[i][2]);
			
			// Don't count coupling cost for invalid planes.
			continue;
		}
		float nx = n[i][0] / n[i][2];
		float ny = n[i][1] / n[i][2];
		float mx = m[i][0] / m[i][2];
		float my = m[i][1] / m[i][2];
		double normalCouplingCost = (nx - mx) * (nx - mx) + (ny - my) * (ny - my);
		double dispCoupleCost = (u[i] - v[i]) * (u[i] - v[i]);
		double couplingCost = normalCouplingCost * maxDisp * maxDisp + dispCoupleCost;
		totalCouplingCost += couplingCost;
		totalCouplingCostWithConfidence += G[i] * couplingCost;
	}
	
	return cv::Vec2d(totalCouplingCost, totalCouplingCostWithConfidence);
}

static void RandomPrintCostVolume(MCImg<unsigned short> &costVolume, int numPrintElems)
{
	int numRows = costVolume.h, numCols = costVolume.w, numDisps = costVolume.n;
	for (int retry = 0; retry < numPrintElems; retry++) {
		int y = rand() % numRows;
		int x = rand() % numCols;
		int d = rand() % numDisps;
		float cost = costVolume.get(y, x)[d];
		printf("costVolume(%d,%d,%d) = %.2f\n", y, x, d, cost);
	}
}

void RunARAP(std::string rootFolder, cv::Mat &imL, cv::Mat &imR, cv::Mat &dispL, cv::Mat &dispR,
	std::string filePathImageL, std::string filePathImageR, std::string filePathOutImage)
{
	//int downFactor = 2;
	//cv::resize(imL, imL, cv::Size(imL.cols / downFactor, imL.rows / downFactor));
	//cv::resize(imR, imR, cv::Size(imR.cols / downFactor, imR.rows / downFactor));


	int numRows = imL.rows, numCols = imL.cols;
	int numDisps, maxDisp, visualizeScale;
	SetupStereoParameters(rootFolder, numDisps, maxDisp, visualizeScale);

	printf("rootFolder = %s\n", rootFolder.c_str());
	printf("numRows = %d\n", numRows);
	printf("numCols = %d\n", numCols);
	printf("numDisps = %d\n", numDisps);

	extern std::string gFilePathOracleL, gFilePathOracleR;
	extern cv::Mat gDispSGML, gDispSGMR;

	std::cout << "gFilePathOracleL:  " << gFilePathOracleL << "\n";
	std::cout << "gFilePathOracleR:  " << gFilePathOracleR << "\n";
	if (gFilePathOracleL != "") {
		gDispSGML = cv::imread(gFilePathOracleL, CV_LOAD_IMAGE_UNCHANGED);
		gDispSGMR = cv::imread(gFilePathOracleL, CV_LOAD_IMAGE_UNCHANGED);
		gDispSGML.convertTo(gDispSGML, CV_32FC1, 1.f / 256.f);
		gDispSGMR.convertTo(gDispSGMR, CV_32FC1, 1.f / 256.f);
	}


	extern cv::Mat gImLabL, gImLabR;
	gImLabL = imL;
	gImLabR = imR;
	//cv::cvtColor(imL, gImLabL, CV_BGR2Lab);
	//cv::cvtColor(imR, gImLabR, CV_BGR2Lab);

	bs::Timer::Tic("cost volume");

	double tensorSize = (double)numRows * numCols * numDisps * sizeof(unsigned short) / (double)(1024 * 1024);
	printf("(numRows, numCols) = (%d, %d)\n", numRows, numCols);
	printf("tensorSize: %lf\n", tensorSize);
	printf("numDisps: %d\n", numDisps);
	printf("visualizeScale: %d\n", visualizeScale);


	extern MCImg<unsigned short> gDsiL, gDsiR;
	if (tensorSize > 1200/*rootFolder == "Midd3"*/) {
		printf("*********************************************************\n");
		printf("WARNING: tensor size too large, using online calculation!");
		printf("*********************************************************\n");
		exit(-1);
		extern cv::Mat gSobelImgL, gSobelImgR, gCensusImgL, gCensusImgR;
		cv::Mat ComputeCappedSobelImage(cv::Mat &imgIn, int sobelCapValue);
		gSobelImgL  = ComputeCappedSobelImage(imL, 15);
		gSobelImgR  = ComputeCappedSobelImage(imR, 15);
		gCensusImgL = ComputeCensusImage(imL, 2, 2);
		gCensusImgR = ComputeCensusImage(imR, 2, 2);
	}
	else {
		gDsiL = MCImg<unsigned short>(numRows, numCols, numDisps);
		gDsiR = MCImg<unsigned short>(numRows, numCols, numDisps);
		std::cout << filePathImageL << "\n";
		std::cout << filePathImageR << "\n";
		CostVolumeFromYamaguchi<unsigned short>(filePathImageL, filePathImageR, gDsiL, gDsiR, numDisps);



		//MCImg<float> ComputeBirchfieldTomasiCostVolume(cv::Mat &imL, cv::Mat &imR, int numDisps, int sign);
		//MCImg<float> Compute5x5CensusCostVolume(cv::Mat &imL, cv::Mat &imR, int numDisps, int sign);
		//void CombineBirchfieldTomasiAnd5x5Census(MCImg<float> &dsiBT, MCImg<float> &dsiCensus);

		//MCImg<float> dsiBTL = ComputeBirchfieldTomasiCostVolume(imL, imR, numDisps, -1);
		//MCImg<float> dsiBTR = ComputeBirchfieldTomasiCostVolume(imR, imL, numDisps, +1);
		//gDsiL = Compute5x5CensusCostVolume(imL, imR, numDisps, -1);
		//gDsiR = Compute5x5CensusCostVolume(imR, imL, numDisps, +1);
		//CombineBirchfieldTomasiAnd5x5Census(dsiBTL, gDsiL);
		//CombineBirchfieldTomasiAnd5x5Census(dsiBTR, gDsiR);

		//gDsiL = ComputeAdGradientCostVolume(imL, imR, numDisps, -1, 1.f);
		//gDsiR = ComputeAdGradientCostVolume(imR, imL, numDisps, +1, 1.f);
		//gDsiL = ComputeAdCensusCostVolume(imL, imR, numDisps, -1, 1.f);
		//gDsiR = ComputeAdCensusCostVolume(imR, imL, numDisps, +1, 1.f);

		//for (int retry = 0; retry < 200; retry++) {
		//	int y = rand() % numRows;
		//	int x = rand() % numCols;
		//	int d = rand() % numDisps;
		//	printf("(%4d,%4d,%4d): %.2f\n", y, x, d, gDsiL.get(y, x)[d] / 25.f);
		//}
		//cv::waitKey(0);
	}

	bs::Timer::Toc();

	// Segmentize/Triangulize both views
	cv::Mat labelMapL, labelMapR, contourImgL, contourImgR;
	extern int SEGMENT_LEN;
	//int numPreferedRegions = (numRows * numCols) / (SEGMENT_LEN * SEGMENT_LEN);
	//int numPreferedRegions = (numCols / 8 + 1) * (numRows / 8 + 1);
	extern int NUM_PREFERED_REGIONS;
	int numPreferedRegions = NUM_PREFERED_REGIONS;
	//int numPreferedRegions = 8000;
	extern float SLIC_COMPACTNESS;
	float compactness = SLIC_COMPACTNESS;
#if 1
	int numSegsL = SLICSegmentation(imL, numPreferedRegions, compactness, labelMapL, contourImgL);
	int numSegsR = SLICSegmentation(imR, numPreferedRegions, compactness, labelMapR, contourImgR);
#else
	int MeanShiftSegmentation(cv::Mat &img, const float colorRadius, const float spatialRadius, const int minRegion, cv::Mat &labelMap);
	int numSegsL = MeanShiftSegmentation(imL, 3.f, 3.f, 300, labelMapL);
	int numSegsR = MeanShiftSegmentation(imR, 3.f, 3.f, 300, labelMapR);
	void VisualizeSegmentation(cv::Mat &img, cv::Mat &labelMap);
	VisualizeSegmentation(imL, labelMapL);
#endif

	printf("numRegionsL = %d\n", numSegsL);
	printf("numRegionsR = %d\n", numSegsR);

	extern cv::Mat gLabelMapL, gLabelMapR;
	gLabelMapL = labelMapL;
	gLabelMapR = labelMapR;

	extern std::vector<cv::Vec3b> gMeanLabColorsL, gMeanLabColorsR;
	gMeanLabColorsL = ComupteSegmentMeanLabColor(gImLabL, labelMapL, numSegsL);
	gMeanLabColorsR = ComupteSegmentMeanLabColor(gImLabR, labelMapR, numSegsR);

	//cv::imshow("contour", contourImgL);
	//cv::waitKey(0);

	std::vector<cv::Point2f> baryCentersL, baryCentersR;
	std::vector<std::vector<cv::Point2i>> segPixelListsL, segPixelListsR;
	ConstructBaryCentersAndPixelLists(numSegsL, labelMapL, baryCentersL, segPixelListsL);
	ConstructBaryCentersAndPixelLists(numSegsR, labelMapR, baryCentersR, segPixelListsR);

	extern std::vector<std::vector<cv::Point2i>> gSegPixelListsL, gSegPixelListsR;
	gSegPixelListsL = segPixelListsL;
	gSegPixelListsR = segPixelListsR;

	extern std::vector<std::vector<int>> segAnchorIndsL, segAnchorIndsR;
	segAnchorIndsL.resize(numSegsL);
	segAnchorIndsR.resize(numSegsR);
	extern int ARAP_NUM_ANCHORS;
	int numAnchors = ARAP_NUM_ANCHORS;
	for (int id = 0; id < numSegsL; id++) {
		segAnchorIndsL[id].resize(numAnchors);
		for (int j = 0; j < numAnchors; j++) {
			segAnchorIndsL[id][j] = rand() % segPixelListsL[id].size();
		}
	}
	for (int id = 0; id < numSegsR; id++) {
		segAnchorIndsR[id].resize(numAnchors);
		for (int j = 0; j < numAnchors; j++) {
			segAnchorIndsR[id][j] = rand() % segPixelListsR[id].size();
		}
	}

	std::vector<std::vector<int>> nbGraphL, nbGraphR;	
	ConstructNeighboringSegmentGraph(numSegsL, labelMapL, nbGraphL, 0);
	ConstructNeighboringSegmentGraph(numSegsR, labelMapR, nbGraphR, 0);

	std::vector<std::vector<float>> nbSimWeightsL, nbSimWeightsR;
	ComputeSegmentSimilarityWeights(imL, nbGraphL, segPixelListsL, nbSimWeightsL);
	ComputeSegmentSimilarityWeights(imR, nbGraphR, segPixelListsR, nbSimWeightsR);
	

	//VisualizeSegmengSimilarityWeights(imL, nbGraphL, segPixelListL);


	// Variables being optimized:
	// n, m are normals; u, v are disparities; g are confidence.
	std::vector<cv::Vec3f>	nL(numSegsL), nR(numSegsR), mL(numSegsL), mR(numSegsR);
	std::vector<float>		uL(numSegsL), uR(numSegsR), vL(numSegsL), vR(numSegsR);
	std::vector<float>		gDataL(numSegsL, 0.f), gSmoothL(numSegsR, 0.f);  
	std::vector<float>		gDataR(numSegsL, 0.f), gSmoothR(numSegsR, 0.f);

	
	evalParams.numRows			= numRows;
	evalParams.numCols			= numCols;
	evalParams.numDisps			= numDisps;
	evalParams.labelMap			= &labelMapL;
	evalParams.baryCenters		= &baryCentersL;
	evalParams.nbGraph			= &nbGraphL;
	evalParams.segPixelLists	= &segPixelListsL;
	evalParams.dsiL				= &gDsiL;

	// Optimization starts
	extern int ARAP_INIT_PATCHMATCH_ITERS;
	ConstrainedPatchMatchOnSegments(-1, 0.f, maxDisp, ARAP_INIT_PATCHMATCH_ITERS, true,
		nL, uL, mL, vL, gSmoothL, baryCentersL, nbGraphL, nbSimWeightsL, segPixelListsL);
	ConstrainedPatchMatchOnSegments(+1, 0.f, maxDisp, ARAP_INIT_PATCHMATCH_ITERS, true,
		nR, uR, mR, vR, gSmoothR, baryCentersR, nbGraphR, nbSimWeightsR, segPixelListsR);

	dispL = SegmentLabelToDisparityMap(numRows, numCols, nL, uL, baryCentersL, segPixelListsL);
	dispR = SegmentLabelToDisparityMap(numRows, numCols, nR, uR, baryCentersR, segPixelListsR);
	

	evalParams.n = &nL;
	evalParams.u = &uL;
	evalParams.dispL = &dispL;
	EvaluateDisparity(rootFolder, dispL, 1.f, &evalParams);
	cv::Mat dispPatchMatchOnlyL = dispL.clone();
	cv::Mat dispPatchMatchOnlyR = dispR.clone();

#if 0
	cv::Mat &validPixelMapL = CrossCheck(dispL, dispR, -1, 2.f);
	cv::Mat crossCheckedL = SetInvalidDisparityToZeros(dispL, validPixelMapL);
	cv::Mat wmfL = dispL.clone();
	void WeightedMedianFilterInvalidPixels(cv::Mat &disp, cv::Mat &validPixelMap, cv::Mat &img);
	bs::Timer::Tic("WMF");
	WeightedMedianFilterInvalidPixels(wmfL, validPixelMapL, gImLabL);
	bs::Timer::Toc();

	float scaleFactor = (rootFolder == "KITTI" ? 256 : 64);
	dispL.convertTo(dispL, CV_16UC1, scaleFactor);
	dispR.convertTo(dispR, CV_16UC1, scaleFactor);
	wmfL.convertTo(wmfL, CV_16UC1, scaleFactor);
	crossCheckedL.convertTo(crossCheckedL, CV_16UC1, scaleFactor);
	cv::imwrite(outImageFilePath + "_dispL.png", dispL);
	cv::imwrite(outImageFilePath + "_dispR.png", dispR);
	cv::imwrite(outImageFilePath + "_wmfL.png", wmfL);
	cv::imwrite(outImageFilePath + "_crossCheckedL.png", crossCheckedL);
	return;
#endif


	extern int ARAP_MAX_ITERS;
	

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


		//VisualizeSegmentConfidence(labelMapL, segPixelListsL, gDataL);


		//////////////////////////////////////////////////////////////////////////////////////////
		// Optimize E_SMOOTH
		SolveARAPSmoothness(theta + 0.1f, gDataL, maxDisp, nbGraphL, nbSimWeightsL, nL, uL, mL, vL, baryCentersL);
		SolveARAPSmoothness(theta + 0.1f, gDataR, maxDisp, nbGraphR, nbSimWeightsR, nR, uR, mR, vR, baryCentersR);

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

#if 1
	//RandomPrintCostVolume(gDsiL, 100);
	double dataCostL = ComputARAPDataCost(nL, uL, baryCentersL);
	double smoothCostL = ComputeARAPSmoothCost(nL, uL, baryCentersL, nbGraphL, nbSimWeightsL);
	cv::Vec2d couplingCostsL = ComputeARAPCouplingCost(nL, uL, mL, vL, gDataL, maxDisp);

	extern float ARAP_LAMBDA;
	extern float ARAP_SIGMA;

	printf("*************************************************\n");
	printf("total dataCostL							= %lf\n", dataCostL);
	printf("total smoothCostL						= %lf\n", smoothCostL);
	printf("total couplingCostWithConfidenceL		= %lf\n", couplingCostsL[1]);
	printf("total couplingCostL						= %lf\n\n", couplingCostsL[0]);
	printf("lambda*dataCostL						= %lf\n", ARAP_LAMBDA * dataCostL);
	printf("0.5*smoothCostL							= %lf\n", 0.5 * smoothCostL);
	printf("0.5*simga*couplingCostWithConfidenceL	= %lf\n", 0.5 * ARAP_SIGMA * couplingCostsL[1]);
	printf("*************************************************\n");

	FILE *fid = fopen((filePathOutImage + "_energies.txt").c_str(), "w");
	fprintf(fid, "total dataCostL						= %lf\n", dataCostL);
	fprintf(fid, "total smoothCostL						= %lf\n", smoothCostL);
	fprintf(fid, "total couplingCostWithConfidenceL		= %lf\n", couplingCostsL[1]);
	fprintf(fid, "total couplingCostL					= %lf\n\n", couplingCostsL[0]);
	fprintf(fid, "lambda*dataCostL						= %lf\n", ARAP_LAMBDA * dataCostL);
	fprintf(fid, "0.5*smoothCostL						= %lf\n", 0.5 * smoothCostL);
	fprintf(fid, "0.5*simga*couplingCostWithConfidenceL = %lf\n", 0.5 * ARAP_SIGMA * couplingCostsL[1]);
	fclose(fid);
#endif


	//////////////////////////////////////////////////////////////////////////////////////////
	//                                      Output
	//////////////////////////////////////////////////////////////////////////////////////////
	cv::Mat dispDataL   = SegmentLabelToDisparityMap(numRows, numCols, nL, uL, baryCentersL, segPixelListsL);
	cv::Mat dispDataR   = SegmentLabelToDisparityMap(numRows, numCols, nR, uR, baryCentersR, segPixelListsR);
	cv::Mat dispSmoothL = SegmentLabelToDisparityMap(numRows, numCols, mL, vL, baryCentersL, segPixelListsL);
	cv::Mat dispSmoothR = SegmentLabelToDisparityMap(numRows, numCols, mR, vR, baryCentersR, segPixelListsR);
	cv::Mat validPixelL = CrossCheck(dispDataL, dispDataR, -1, 1.f);
	cv::Mat validPixelR = CrossCheck(dispDataR, dispDataL, +1, 1.f);
	cv::Mat dispCrossCheckedL = SetInvalidDisparityToZeros(dispDataL, validPixelL);
	cv::Mat dispCrossCheckedR = SetInvalidDisparityToZeros(dispDataR, validPixelR);

	void WeightedMedianFilterInvalidPixels(cv::Mat &disp, cv::Mat &validPixelMap, cv::Mat &img);
	void FastWeightedMedianFilterInvalidPixels(cv::Mat &disp, cv::Mat &validPixelMap, cv::Mat &img, bool useValidPixelOnly);
	void PixelwiseOcclusionFilling(cv::Mat &disp, cv::Mat &validPixelMap);

	cv::Mat dispBackgroundFillL = dispDataL.clone();
	cv::Mat dispBackgroundFillR = dispDataR.clone();
	PixelwiseOcclusionFilling(dispBackgroundFillL, validPixelL);
	PixelwiseOcclusionFilling(dispBackgroundFillR, validPixelR);
	 
	cv::Mat dispWmfL = dispBackgroundFillL.clone();
	cv::Mat dispWmfR = dispBackgroundFillR.clone();
	validPixelL = CrossCheck(dispWmfL, dispWmfR, -1, 1.f);
	validPixelR = CrossCheck(dispWmfR, dispWmfL, +1, 1.f);
	 
	bs::Timer::Tic("WMF");
	//WeightedMedianFilterInvalidPixels(dispWmfL, validPixelL, gImLabL);
	FastWeightedMedianFilterInvalidPixels(dispWmfL, validPixelL, imL, 0);
	bs::Timer::Toc();
	EvaluateDisparity(rootFolder, dispWmfL, 0.5f, &evalParams, "OnMouseTestARAP");

	cv::Mat Float2ColorJet(cv::Mat &fimg, float dmin, float dmax);
	cv::imwrite(filePathOutImage + "_colorjet_dispCrossCheckedL.png", Float2ColorJet(dispCrossCheckedL, 0, numDisps));
	cv::imwrite(filePathOutImage + "_colorjet_dispCrossCheckedR.png", Float2ColorJet(dispCrossCheckedR, 0, numDisps));
	cv::imwrite(filePathOutImage + "_colorjet_dispDataL.png", Float2ColorJet(dispDataL, 0, numDisps));
	cv::imwrite(filePathOutImage + "_colorjet_dispDataR.png", Float2ColorJet(dispDataR, 0, numDisps));
	cv::imwrite(filePathOutImage + "_colorjet_dispSmoothL.png", Float2ColorJet(dispSmoothL, 0, numDisps));
	cv::imwrite(filePathOutImage + "_colorjet_dispSmoothR.png", Float2ColorJet(dispSmoothR, 0, numDisps));
	cv::imwrite(filePathOutImage + "_colorjet_dispBackgroundFillL.png", Float2ColorJet(dispBackgroundFillL, 0, numDisps));
	cv::imwrite(filePathOutImage + "_colorjet_dispBackgroundFillWmfUseAllL.png", Float2ColorJet(dispWmfL, 0, numDisps));
	cv::imwrite(filePathOutImage + "_colorjet_dispPatchMatchOnlyL.png", Float2ColorJet(dispPatchMatchOnlyL, 0, numDisps));
	cv::imwrite(filePathOutImage + "_colorjet_dispPatchMatchOnlyR.png", Float2ColorJet(dispPatchMatchOnlyR, 0, numDisps));

	float upFactor = (rootFolder == "KITTI" ? 256 : 64);
	dispDataL.convertTo(dispDataL, CV_16UC1, upFactor);
	dispDataR.convertTo(dispDataR, CV_16UC1, upFactor);
	dispSmoothL.convertTo(dispSmoothL, CV_16UC1, upFactor);
	dispSmoothR.convertTo(dispSmoothR, CV_16UC1, upFactor);
	dispCrossCheckedL.convertTo(dispCrossCheckedL, CV_16UC1, upFactor);
	dispCrossCheckedR.convertTo(dispCrossCheckedR, CV_16UC1, upFactor);
	dispBackgroundFillL.convertTo(dispBackgroundFillL, CV_16UC1, upFactor);
	dispWmfL.convertTo(dispWmfL, CV_16UC1, upFactor);
	dispPatchMatchOnlyL.convertTo(dispPatchMatchOnlyL, CV_16UC1, upFactor);
	dispPatchMatchOnlyR.convertTo(dispPatchMatchOnlyR, CV_16UC1, upFactor);

	cv::imwrite(filePathOutImage + "_dispCrossCheckedL.png", dispCrossCheckedL);
	cv::imwrite(filePathOutImage + "_dispCrossCheckedR.png", dispCrossCheckedR);
	cv::imwrite(filePathOutImage + "_dispDataL.png", dispDataL);
	cv::imwrite(filePathOutImage + "_dispDataR.png", dispDataR);
	cv::imwrite(filePathOutImage + "_dispSmoothL.png", dispSmoothL);
	cv::imwrite(filePathOutImage + "_dispSmoothR.png", dispSmoothR);
	cv::imwrite(filePathOutImage + "_dispBackgroundFillL.png", dispBackgroundFillL);
	cv::imwrite(filePathOutImage + "_dispBackgroundFillWmfUseAllL.png", dispWmfL);
	cv::imwrite(filePathOutImage + "_dispPatchMatchOnlyL.png", dispPatchMatchOnlyL);
	cv::imwrite(filePathOutImage + "_dispPatchMatchOnlyR.png", dispPatchMatchOnlyR);

	cv::Mat segImg = DrawSegmentImage(labelMapL);
	cv::hconcat(segImg, imL, segImg);
	cv::imwrite(filePathOutImage + "_SLIC.png", segImg);
	
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
	
	cv::Mat dispL, dispR;
	RunARAP(ROOTFOLDER, imL, imR, dispL, dispR,  "s", "s", "s");
}