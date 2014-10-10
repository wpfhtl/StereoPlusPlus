#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <stack>
#include <set>
#include <string>
#include <iostream>

#include "StereoAPI.h"
#include "ReleaseAssert.h"

//static const int HINGE		= 0;
//static const int COPLANAR	= 1;
//static const int LEFTOCC	= 2;
//static const int RIGHTOCC	= 3;

static enum {HINGE, COPLANAR, LEFTOCC, RIGHTOCC};


static bool IsBoundaryPixel(cv::Point2i &p, cv::Mat &labelMap)
{
	// Here the boundary refers to the segment boundary, not the image boundary.
	// Pixels that are adjacent to at least two segments are called boundary pixels.
	const cv::Point2i dirOffsets[] = {
		cv::Point2i(-1, 0), cv::Point2i(+1, 0),
		cv::Point2i(0, -1), cv::Point2i(0, +1)
	};

	for (int r = 0; r < 4; r++) {
		cv::Point2i q = p + dirOffsets[r];
		if (InBound(q, labelMap.rows, labelMap.cols)) {
			if (labelMap.at<int>(p.y, p.x) != labelMap.at<int>(q.y, q.x)) {
				return true;
			}
		}
	}
	return false;
}

static int SimpleFloodFill(int y, int x, cv::Mat &_3x3_)
{
	const cv::Point2i dirOffsets[] = {
		cv::Point2i(-1, 0), cv::Point2i(+1, 0),
		cv::Point2i(0, -1), cv::Point2i(0, +1)
	};
	int compVal = _3x3_.at<int>(y, x);
	int numPixelsInComp = 0;

	std::stack<cv::Point2i> stack;
	stack.push(cv::Point2i(x, y));
	_3x3_.at<int>(y, x) = -1;  // means visited.
	numPixelsInComp++;

	while (!stack.empty()) {
		cv::Point2i p = stack.top();
		stack.pop();
		for (int r = 0; r < 4; r++) {
			cv::Point2i q = p + dirOffsets[r];
			if (InBound(q, _3x3_.rows, _3x3_.cols)
				&& _3x3_.at<int>(q.y, q.x) == compVal) {
				_3x3_.at<int>(q.y, q.x) = -1;
				numPixelsInComp++;
				stack.push(q);
			}
		}
	}
	return numPixelsInComp;
}

static bool BreakConnectivity(cv::Point2i &p, cv::Mat &labelMap)
{
	const cv::Point2i dirOffsets[] = {
		cv::Point2i(-1, 0), cv::Point2i(+1, 0),
		cv::Point2i(0, -1), cv::Point2i(0, +1)
	};

	if (!IsBoundaryPixel(p, labelMap)) {
		// If p is inside the region, then flipping it will break connectivity
		return true;
	}

	//printf("here???????\n");

	// If flipping p will break connectivity of the original labels of p in its
	// 8-neighborhood, iff it will break connectivity of the whole segment.
	// The above is only my conjecture, i hopt it is true...
	cv::Mat _3x3_ = cv::Mat::zeros(3, 3, CV_32SC1);
	int N = 0;
	for (int y = p.y - 1, i = 0; y <= p.y + 1; y++, i++) {
		for (int x = p.x - 1, j = 0; x <= p.x + 1; x++, j++) {
			if (InBound(y, x, labelMap.rows, labelMap.cols)
				&& labelMap.at<int>(y, x) == labelMap.at<int>(p.y, p.x)) {
				_3x3_.at<int>(i, j) = 555;
				N++;
			}
		}
	}
	_3x3_.at<int>(1, 1) = 0;
	N--;

	int numFloodFilledPixels = 0;
	for (int i = 0; i < 9; i++) {
		int y = i / 3;
		int x = i % 3;
		if (_3x3_.at<int>(y, x) == 555) {
			numFloodFilledPixels = SimpleFloodFill(y, x, _3x3_);
			break;
		}
	}

	//printf("(p.y,p.x) = (%d, %d)\n", p.y, p.x);
	//std::cout << _3x3_ << "\n";
	//printf("N=%d\n", N);
	//printf("numFloodFilledPixels=%d\n", numFloodFilledPixels);

	//cv::Mat rand = cv::Mat(6,6, CV_8UC3);
	//cv::imshow("rand", rand);
	//cv::waitKey(0);

	if (N != 0 && numFloodFilledPixels != N) {
		// then the connectivity is break
		return true;
	}
	//printf("return false.\n");
	return false;
}

static inline double RunningMeanSquareCost(double x, double mu, double sqMu, int N, int sign)
{
	// Given "\mu = (\sum_{i} x_i) / N" and "\sqMu = (sum_{i} (x_i - \mu)^2) / N",
	// Calculate the new mean of \sqMu if we add (sign == -1) or remove (sign == +1) x. 
	double newMu = (mu * N + sign * x) * (1.f / (N + sign));
	double delta = newMu - mu;
	double newSqSum = N * sqMu + sign * (x - mu) * (x - mu)
		- 2 * delta * (N * mu + sign * x);
		//+ 2 * mu * delta + delta * delta;
	return newSqSum / (N + sign) + 2 * mu * delta + delta * delta;
}

static inline cv::Vec3d RunningMeanSquareCost(cv::Vec3d x, cv::Vec3d mu, cv::Vec3d sqMu, int N, int sign)
{
	cv::Vec3d res;
	res[0] = RunningMeanSquareCost(x[0], mu[0], sqMu[0], N, sign);
	res[1] = RunningMeanSquareCost(x[1], mu[1], sqMu[1], N, sign);
	res[2] = RunningMeanSquareCost(x[2], mu[2], sqMu[2], N, sign);
	return res;
}

static inline cv::Point2d RunningMeanSquareCost(cv::Point2d x, cv::Point2d mu, cv::Point2d sqMu, int N, int sign)
{
	cv::Point2d res;
	res.x = RunningMeanSquareCost(x.x, mu.x, sqMu.x, N, sign);
	res.y = RunningMeanSquareCost(x.y, mu.y, sqMu.y, N, sign);
	return res;
}

static void InitTPSByGrid(int numPreferedSegments, cv::Mat &img, cv::Mat &labelMap, std::vector<int> &segSizes,
	std::vector<cv::Vec3d> &meanColors, std::vector<cv::Point2d> &meanPositions,
	std::vector<cv::Vec3d> &meanColorCosts, std::vector<cv::Point2d> &meanPositionCosts)
{
	// Initialize Appearance Cost and Location Cost.
	int numRows = img.rows, numCols = img.cols;
	int segLen = std::sqrt((numRows * numCols * 1.f) / numPreferedSegments) + 0.5f;
	int M = numRows / segLen, N = numCols / segLen;
	M += (numRows % segLen > segLen / 3);
	N += (numCols % segLen > segLen / 3);
	segSizes			= std::vector<int>(M * N, 0);
	meanColors			= std::vector<cv::Vec3d>(M * N, cv::Vec3d(0, 0, 0));
	meanPositions		= std::vector<cv::Point2d>(M * N, cv::Point2d(0, 0));
	meanColorCosts		= std::vector<cv::Vec3d>(M * N, cv::Vec3d(0, 0, 0));
	meanPositionCosts	= std::vector<cv::Point2d>(M * N, cv::Point2d(0, 0));

	labelMap.create(numRows, numCols, CV_32SC1);
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			int i = std::min(y / segLen, M - 1);
			int j = std::min(x / segLen, N - 1);
			int id = i * N + j;
			labelMap.at<int>(y, x) = id;
			meanColors[id] += img.at<cv::Vec3b>(y, x);
			meanPositions[id] += cv::Point2d(x, y);
			segSizes[id]++;
		}
	}

	for (int id = 0; id < M * N; id++) {
		meanColors[id] /= segSizes[id];
		meanPositions[id] *= 1.f / segSizes[id];
	}

	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			int id = labelMap.at<int>(y, x);
			cv::Vec3d colorDiff = cv::Vec3d(img.at<cv::Vec3b>(y, x)) - meanColors[id];
			meanColorCosts[id] += colorDiff.mul(colorDiff);
			cv::Point2d posDiff = cv::Point2d(x, y) - meanPositions[id];
			meanPositionCosts[id] += cv::Point2d(posDiff.x * posDiff.x, posDiff.y * posDiff.y);
		}
	}

	for (int id = 0; id < M * N; id++) {
		meanColorCosts[id] /= segSizes[id];
		meanPositionCosts[id] *= 1.f / segSizes[id];
	}
}

static double LocalBoundaryLengthCost(cv::Point2i p, int label, cv::Mat &labelMap)
{
	int dist = 0;
	for (int y = p.y - 1; y <= p.y + 1; y++) {
		for (int x = p.x - 1; x <= p.x + 1; x++) {
			if (InBound(y, x, labelMap.rows, labelMap.cols)) {
				dist += (labelMap.at<int>(y, x) != label);
			}
		}
	}
	return dist;
}

static void VisualizeMeanPositionAndMeanColors(cv::Mat &labelMap, std::vector<int> &segSizes, 
	std::vector<cv::Vec3d> &meanColors, std::vector<cv::Point2d> &meanPositions)
{
	int numRows = labelMap.rows, numCols = labelMap.cols;
	int numSegs = segSizes.size();
	cv::Mat baryCenterMap(numRows, numCols, CV_8UC3);
	cv::Mat segColorMap(numRows, numCols, CV_8UC3);

	for (int id = 0; id < numSegs; id++) {
		cv::circle(baryCenterMap, meanPositions[id], 2, cv::Scalar(0, 0, 255), 2);
	}
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			int id = labelMap.at<int>(y, x);
			segColorMap.at<cv::Vec3b>(y, x) = meanColors[id];
		}
	}

	cv::Mat canvas;
	cv::hconcat(baryCenterMap, segColorMap, canvas);
	cv::imshow("baryCenterMap and segColorMap", canvas);
	cv::waitKey(0);
}

static void VisualizeSegmentation(cv::Mat &img, cv::Mat &labelMap)
{
	srand(12003);
	int numRows = img.rows, numCols = img.cols;
	double minVal, maxVal;
	cv::minMaxLoc(labelMap, &minVal, &maxVal);
	int numLabels = maxVal + 1;
	std::vector<cv::Vec3b> colors(numLabels);
	for (int i = 0; i < numLabels; i++) {
		colors[i][0] = rand() % 256;
		colors[i][1] = rand() % 256;
		colors[i][2] = rand() % 256;
	}

	cv::Mat segImg(numRows, numCols, CV_8UC3);
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			int id = labelMap.at<int>(y, x);
			segImg.at<cv::Vec3b>(y, x) = colors[id];
		}
	}

	cv::Mat canvas;
	cv::hconcat(img, segImg, canvas);
	cv::imshow("segmentation", canvas);
	cv::waitKey(0);
}

static void TopologyPreservingSegmentation(cv::Mat &img, int numPreferedSegments, 
	cv::Mat &labelMap, std::vector<int> &segSizes, 
	std::vector<cv::Vec3d> &meanColors, std::vector<cv::Point2d> &meanPositions,
	std::vector<cv::Vec3d> &meanColorCosts, std::vector<cv::Point2d> &meanPositionCosts)
{
	const double LAMBDA_POS = 200.;
	const double LAMBDA_BOU = 1000.0;

	const cv::Point2i dirOffsets[] = {
		cv::Point2i(-1, 0), cv::Point2i(+1, 0),
		cv::Point2i(0, -1), cv::Point2i(0, +1)
	};

	int numRows = img.rows, numCols = img.cols;

	InitTPSByGrid(numPreferedSegments, img, labelMap, segSizes, 
		meanColors, meanPositions, meanColorCosts, meanPositionCosts);

	//for (int i = 0; i < meanColorCosts.size(); i++) {
	//	for (int j = 0; j < 3; j++)
	//	printf("%10.1lf   ", meanColorCosts[i][j]);
	//}
	//for (int i = 0; i < meanPositionCosts.size(); i++) {
	//	printf("%10.1lf   ", meanPositionCosts[i].x);
	//	printf("%10.1lf   ", meanPositionCosts[i].y);
	//}
	//return;

	//VisualizeMeanPositionAndMeanColors(labelMap, segSizes, meanColors, meanPositions);


	// Initialize the stack to contain all boundary pixels
	printf("Initializing stack ...\n");
	std::stack<cv::Point2i> stack;
	cv::Mat boundaryPixelsMap(numRows, numCols, CV_8UC3);
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			if (IsBoundaryPixel(cv::Point2i(x, y), labelMap)) {
				cv::circle(boundaryPixelsMap, cv::Point2i(x, y), 1, cv::Scalar(0,0,255), 1);
				stack.push(cv::Point2i(x, y));
			}
		}
	}
	//cv::imshow("boundaryPixels", boundaryPixelsMap);
	//cv::waitKey(0);
	int dummyCounter = 0;
	// Adjust segmentation until stack is empty
	printf("Adjusting segmentation ...\n");
	while (!stack.empty()) {
		
		cv::Point2i p = stack.top();
		stack.pop();
		if (BreakConnectivity(p, labelMap)) {
			continue;
		}


		int idI = labelMap.at<int>(p.y, p.x);
		int bestId = idI;
		double bestCost = 0.f;
		cv::Vec3d pColor = cv::Vec3d(img.at<cv::Vec3b>(p.y, p.x));

		// AL stands for appearence and location
		double oldALCostI = meanColorCosts[idI].dot(cv::Vec3d(1, 1, 1))
			+ LAMBDA_POS * (meanPositionCosts[idI].dot(cv::Point2d(1, 1)));
		double newALCostI = cv::Vec3d(1, 1, 1).dot(RunningMeanSquareCost(pColor, meanColors[idI], meanColorCosts[idI], segSizes[idI], -1))
			+ LAMBDA_POS * cv::Point2d(1, 1).dot(RunningMeanSquareCost(cv::Point2d(p), meanPositions[idI], meanPositionCosts[idI], segSizes[idI], -1));
		oldALCostI *= segSizes[idI];
		newALCostI *= segSizes[idI] - 1;
		double oldBoundaryCost = LocalBoundaryLengthCost(p, idI, labelMap);

		std::set<int> candidateLabelSet;
		for (int r = 0; r < 4; r++) {
			cv::Point2i q = p + dirOffsets[r];
			if (InBound(q, numRows, numCols)) {
				candidateLabelSet.insert(labelMap.at<int>(q.y, q.x));
			}
		}	

		std::vector<int> candidateLabels(candidateLabelSet.begin(), candidateLabelSet.end());
		for (int i = 0; i < candidateLabels.size(); i++) {
			int idJ = candidateLabels[i]; 
			if (idJ == idI) {
				continue;
			}
			
			double oldALCostJ = meanColorCosts[idJ].dot(cv::Vec3d(1, 1, 1)) 
				+ LAMBDA_POS * (meanPositionCosts[idJ].dot(cv::Point2d(1, 1)));
			double newALCostJ = cv::Vec3d(1, 1, 1).dot(RunningMeanSquareCost(pColor, meanColors[idJ], meanColorCosts[idJ], segSizes[idJ], +1))
				+ LAMBDA_POS * cv::Point2d(1, 1).dot(RunningMeanSquareCost(cv::Point2d(p), meanPositions[idJ], meanPositionCosts[idJ], segSizes[idJ], +1));
			oldALCostJ *= segSizes[idJ];
			newALCostJ *= segSizes[idJ] + 1;
			double newBoundaryCost = LocalBoundaryLengthCost(p, idJ, labelMap);
			/*double newCost = newALCostI + newALCostJ + LAMBDA_BOU * newBoundaryCost
						  - oldALCostI - oldALCostJ - LAMBDA_BOU * oldBoundaryCost;*/
			double newCost = newALCostI + newALCostJ + LAMBDA_BOU * newBoundaryCost
				- oldALCostI - oldALCostJ - LAMBDA_BOU * oldBoundaryCost;

			if (newCost < bestCost) {
				bestId = idJ;
				bestCost = newCost;

				/*printf("*******************\n");
				printf("idI = %d\nidJ = %d\n", idI, idJ);
				printf("oldALCostI, oldALCostJ = %lf, %lf\n", oldALCostI, oldALCostJ);
				printf("newALCostI, newALCostJ = %lf, %lf\n", newALCostI, newALCostJ);
				printf("oldBouCost, newBouCost = %lf, %lf\n", oldBoundaryCost, newBoundaryCost);
				printf("meanColors[idI] = (%.1lf, %.1lf, %.1lf)\n", meanColors[idI][0], meanColors[idI][1], meanColors[idI][2]);
				printf("meanColors[idJ] = (%.1lf, %.1lf, %.1lf)\n", meanColors[idJ][0], meanColors[idJ][1], meanColors[idJ][2]);
				printf("meanColorCosts[idI] = (%.1lf, %.1lf, %.1lf)\n", meanColorCosts[idI][0], meanColorCosts[idI][1], meanColorCosts[idI][2]);
				printf("meanColorCosts[idJ] = (%.1lf, %.1lf, %.1lf)\n", meanColorCosts[idJ][0], meanColorCosts[idJ][1], meanColorCosts[idJ][2]);
				printf("meanPositions[idI] = (%.1lf, %.1lf)\n", meanPositions[idI].x, meanPositions[idI].y);
				printf("meanPositions[idJ] = (%.1lf, %.1lf)\n", meanPositions[idJ].x, meanPositions[idJ].y);
				printf("meanPositionCosts[idI] = (%.1lf, %.1lf)\n", meanPositionCosts[idI].x, meanPositionCosts[idI].y);
				printf("meanPositionCosts[idJ] = (%.1lf, %.1lf)\n", meanPositionCosts[idJ].x, meanPositionCosts[idJ].y);
				printf("*******************\n");*/
				
			}
		}

		// If segment label updated
		if (bestId != idI) {
			// update values.
			int idJ = bestId;
			meanColorCosts[idI]		= RunningMeanSquareCost(pColor, meanColors[idI], meanColorCosts[idI], segSizes[idI], -1);
			meanColorCosts[idJ]		= RunningMeanSquareCost(pColor, meanColors[idJ], meanColorCosts[idJ], segSizes[idJ], +1);
			meanPositionCosts[idI]	= RunningMeanSquareCost(cv::Point2d(p), meanPositions[idI], meanPositionCosts[idI], segSizes[idI], -1);
			meanPositionCosts[idJ]	= RunningMeanSquareCost(cv::Point2d(p), meanPositions[idJ], meanPositionCosts[idJ], segSizes[idJ], +1);
			meanColors[idI]		= (meanColors[idI] * segSizes[idI] - pColor) / (segSizes[idI] - 1.f);
			meanColors[idJ]		= (meanColors[idJ] * segSizes[idJ] + pColor) / (segSizes[idJ] + 1.f);
			meanPositions[idI]	= (meanPositions[idI] * segSizes[idI] - cv::Point2d(p)) * (1.f / (segSizes[idI] - 1));
			meanPositions[idJ]	= (meanPositions[idJ] * segSizes[idJ] + cv::Point2d(p)) * (1.f / (segSizes[idJ] + 1));
			segSizes[idI]--;
			segSizes[idJ]++;
			labelMap.at<int>(p.y, p.x) = bestId;

			// push boundary pixels onto the stack
			for (int r = 0; r < 4; r++) {
				cv::Point2i q = p + dirOffsets[r];
				if (InBound(q, numRows, numCols) && IsBoundaryPixel(q, labelMap)) {
					stack.push(q);
				}
			}

			//printf("dummyCounter = %d\n", ++dummyCounter);
			//VisualizeSegmentation(img, labelMap);
		}
		else {
			//printf("NOT CHANGED\n");
		}
	}

	//for (int i = 0; i < segSizes.size(); i++) {
	//	printf("%8d", segSizes[i]);
	//}
}

static void RandomPermute(std::vector<cv::Point3f> &pointList, int N)
{
	for (int i = 0; i < N; i++) {
		int j = rand() % pointList.size();
		std::swap(pointList[i], pointList[j]);
	}
}

static cv::Vec3f RansacPlanefit(std::vector<cv::Point3f> &pointList)
{
	const float inlierThresh = 3.f;

	if (pointList.size() < 5) {
		return cv::Vec3f(0, 0, 15);	// A frontal-parallel plane with constant disparity 15
	}

	int bestNumInliers = 0;
	cv::Vec3f bestAbc = cv::Vec3f(0, 0, 15);

	for (int retry = 0; retry < 200; retry++) {
		RandomPermute(pointList, 5);
		cv::Mat A(5, 3, CV_32FC1);
		cv::Mat boundarySizes(5, 1, CV_32FC1);
		cv::Mat X;
		for (int i = 0; i < 5; i++) {
			A.at<float>(i, 0) = pointList[i].x;
			A.at<float>(i, 1) = pointList[i].y;
			A.at<float>(i, 2) = 1.f;
			boundarySizes.at<float>(i, 0) = pointList[i].z;
		}

		cv::solve(A, boundarySizes, X, cv::DECOMP_QR);

		cv::Vec3f abc(X.at<float>(0, 0), X.at<float>(1, 0), X.at<float>(2, 0));
		int numInliers = 0;
		for (int i = 0; i < pointList.size(); i++) {
			cv::Point3f p = pointList[i];
			float d = abc.dot(cv::Vec3f(p.x, p.y, 1.f));
			if (std::abs(d - p.z) <= inlierThresh) {
				numInliers++;
			}
		}

		if (numInliers > bestNumInliers) {
			bestNumInliers = numInliers;
			bestAbc = abc;
		}
	}

	return bestAbc;
}

static void RansacPlanefit(cv::Mat &dispSgm, cv::Mat &validMap, cv::Mat &labelMap, cv::Mat &disp, std::vector<cv::Vec3f> &abc)
{
	double minVal, maxVal;
	cv::minMaxIdx(labelMap, &minVal, &maxVal);
	int numSegs = maxVal + 1;
	int numRows = labelMap.rows, numCols = labelMap.cols;

	abc.resize(numSegs);
	std::vector<std::vector<cv::Point3f>> xyd(numSegs);

	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++){
			int id = labelMap.at<int>(y, x);
			if (validMap.at<int>(y, x)) {
				xyd[id].push_back(cv::Point3f(x, y, dispSgm.at<float>(y, x)));
			}
		}
	}

	for (int id = 0; id < numSegs; id++) {
		abc[id] = RansacPlanefit(xyd[id]);
	}

	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++){
			int id = labelMap.at<int>(y, x);
			disp.at<float>(y, x) = abc[id].dot(cv::Vec3f(x, y, 1));
		}
	}
}

static inline bool BelongToBoundaryBij(cv::Point2i &p, cv::Mat &labelMap, int I, int J)
{
	const cv::Point2i offsetsN5[] = {
		cv::Point2i(-1, 0), cv::Point2i(+1, 0),
		cv::Point2i(0, -1), cv::Point2i(0, +1),
		cv::Point2i(0,  0)
	};
	int K = labelMap.at<int>(p.y, p.x);
	if (K != I && K != J) {
		return false;
	}
	bool isAdjToI = false;
	bool isAdjToJ = false;
	for (int r = 0; r < 5; r++) {
		cv::Point2i q = p + offsetsN5[r];
		if (InBound(q, labelMap.rows, labelMap.cols)) {
			if (labelMap.at<int>(q.y, q.x) == I) {
				isAdjToI = true;
			}
			if (labelMap.at<int>(q.y, q.x) == J) {
				isAdjToJ = true;
			}
		}
	}
	return isAdjToI && isAdjToJ;
}

static inline void UpdateSmoothCostChange(cv::Point2i &p, int i, int j, int I, int J, 
	cv::Mat &labelMapOld, cv::Mat &labelMapNew, cv::Vec3f &abcI, cv::Vec3f &abcJ,
	cv::Mat &unnormalizedCostInc, cv::Mat &boundarySizesInc, cv::Mat &segRelationMat)
{
	int belongToSijBefore = BelongToBoundaryBij(p, labelMapOld, I, J);
	int belongToSijAfter  = BelongToBoundaryBij(p, labelMapNew, I, J);

	// Only when the ownership of p change will it affect the cost
	if (belongToSijBefore ^ belongToSijAfter) {

		int sign = (belongToSijBefore ? -1 : +1);

		double dispDiff = abcI.dot(cv::Vec3f(p.x, p.y, 1.f)) - abcJ.dot(cv::Vec3f(p.x, p.y, 1.f));
		double contribution = dispDiff * dispDiff;
		int idxFront, idxBack;
		cv::Vec3f abcFront, abcBack;

		switch (segRelationMat.at<int>(I, J)) {
			
		case COPLANAR:
		case HINGE:
			unnormalizedCostInc.at<double>(i, j) = sign * contribution;
			unnormalizedCostInc.at<double>(j, i) = sign * contribution;
			break;

		case LEFTOCC:
			abcFront = abcI;
			abcBack  = abcJ;
			// no break here.
		case RIGHTOCC:
			abcFront = abcJ;
			abcBack  = abcI;
			// no break here.
		default:
			// In this case, smoothCost.at<float>(I,J) is the sum of the disparities of the 
			// foreground pixel minus the sum of the disparities of the background pixels at
			// the boundary region between I and J.
			float dispFront = abcFront.dot(cv::Vec3f(p.x, p.y, 1.f));
			float dispBack  =  abcBack.dot(cv::Vec3f(p.x, p.y, 1.f));
			unnormalizedCostInc.at<double>(i, j) += sign * (dispFront - dispBack);
			unnormalizedCostInc.at<double>(j, i) += sign * (dispFront - dispBack);

		}

		boundarySizesInc.at<int>(i, j) += sign;
		boundarySizesInc.at<int>(j, i) += sign;
	
	}
}

static std::pair<int, int> DetermineOptimalSegmentAndOcclusionLabel( 
	cv::Point2i &p, cv::Mat &img, int idI, int idJ, cv::Mat &labelMap, cv::Mat &labelMapNew,
	cv::Mat &dispSgm, cv::Mat &validMap, cv::Mat &segRelationMat, std::vector<int> &segSizes, 
	std::vector<cv::Vec3f> &abc, cv::Mat &smoothCosts, cv::Mat &boundarySizes,
	std::vector<cv::Vec3d> &meanColors, std::vector<cv::Point2d> &meanPositions,
	std::vector<cv::Vec3d> &meanColorCosts, std::vector<cv::Point2d> &meanPositionCosts)
{
	const double LAMBDA_POS		= 500;
	const double LAMBDA_DEPTH	= 2000;
	const double LAMBDA_SMO		= 400;
	const double LAMBDA_COM		= 400;
	const double LAMBDA_BOU		= 1000;
	const double LAMBDA_D		= 9;
	const double LAMBDA_OCC		= 15;
	const double LAMBDA_HINGE	= 5;
	const double LAMBDA_PEN		= 30;

	// This method determine best segment and occlusion label combinations (4 cases in total) of pixel p.
	// Changing the segment labels of p will affect E_col, E_pos, E_bou, E_smo, and E_depth.
	// The cost change of E_col, E_pos, E_bou is identical to that in TopologyPreservingSegmentation.
	// The cost change of E_depth is also trivial. Also changing the occlusion label only affect E_depth.
	// Determin the cost change of E_smo is the most difficult part. However and fortunately, 
	// the contributions to E_smo can be distributed to each individual pixel, and changing the label
	// of p can only affect the contributions of its 5-neighborhood. Thus we only need to compute
	// the contributions of the q\in N_5(p) to all possible E_smo(\theta_i, \theta_j, o_ij)

	int bestOldOccLabel, bestNewOccLabel;
	double oldDepthCost, newDepthCost;
	cv::Vec3f abcI = abc[idI], abcJ = abc[idJ];


	if (idI == idJ) {
		// The segment label stays the same, just update the occ label
		double depthSGM = dispSgm.at<float>(p.y, p.x);
		double depthPlaneJ = abcJ.dot(cv::Vec3f(p.x, p.y, 1.f));
		bestNewOccLabel = ((depthPlaneJ - depthSGM) * (depthPlaneJ - depthSGM) > LAMBDA_D ? 1 : 0);
		return std::make_pair(idI, bestOldOccLabel);
	}

	/////////////////////////////////////////////////////////////////////////////////
	// Step 1 - Compute cost change of E_col, E_pos and E_bou
	/////////////////////////////////////////////////////////////////////////////////
	// AL stands for appearence and location
	cv::Vec3d pColor = img.at<cv::Vec3b>(p.y, p.x);
	
	double oldALCostI = meanColorCosts[idI].dot(cv::Vec3d(1, 1, 1))
		+ LAMBDA_POS * (meanPositionCosts[idI].dot(cv::Point2d(1, 1)));
	double newALCostI = cv::Vec3d(1, 1, 1).dot(RunningMeanSquareCost(pColor, meanColors[idI], meanColorCosts[idI], segSizes[idI], -1))
		+ LAMBDA_POS * cv::Point2d(1, 1).dot(RunningMeanSquareCost(cv::Point2d(p), meanPositions[idI], meanPositionCosts[idI], segSizes[idI], -1));
	oldALCostI *= segSizes[idI];
	newALCostI *= segSizes[idI] - 1;
	
	double oldALCostJ = meanColorCosts[idJ].dot(cv::Vec3d(1, 1, 1))
		+ LAMBDA_POS * (meanPositionCosts[idJ].dot(cv::Point2d(1, 1)));
	double newALCostJ = cv::Vec3d(1, 1, 1).dot(RunningMeanSquareCost(pColor, meanColors[idJ], meanColorCosts[idJ], segSizes[idJ], +1))
		+ LAMBDA_POS * cv::Point2d(1, 1).dot(RunningMeanSquareCost(cv::Point2d(p), meanPositions[idJ], meanPositionCosts[idJ], segSizes[idJ], +1));
	oldALCostJ *= segSizes[idJ];
	newALCostJ *= segSizes[idJ] + 1;

	double oldALCost = oldALCostI + oldALCostJ;
	double newALCost = newALCostI + newALCostJ;

	double oldBoundaryCost = LocalBoundaryLengthCost(p, idI, labelMap);
	double newBoundaryCost = LocalBoundaryLengthCost(p, idJ, labelMap);



	/////////////////////////////////////////////////////////////////////////////////
	// Step 2 - Compute cost chagne of E_depth
	/////////////////////////////////////////////////////////////////////////////////
	if (validMap.at<int>(p.y, p.x) == false) {
		oldDepthCost = 0;
		newDepthCost = 0;
		bestOldOccLabel = 0;
		bestNewOccLabel = 0;
	}
	else {
		double depthSGM = dispSgm.at<float>(p.y, p.x);
		double depthPlaneI = abcI.dot(cv::Vec3f(p.x, p.y, 1.f));
		double depthPlaneJ = abcJ.dot(cv::Vec3f(p.x, p.y, 1.f));
		double dispDistCostI = (depthPlaneI - depthSGM) * (depthPlaneI - depthSGM);
		double dispDistCostJ = (depthPlaneJ - depthSGM) * (depthPlaneJ - depthSGM);
		bestOldOccLabel = (dispDistCostI > LAMBDA_D ? 1 : 0);
		bestNewOccLabel = (dispDistCostJ > LAMBDA_D ? 1 : 0);
		oldDepthCost = std::min(LAMBDA_D, dispDistCostI);
		newDepthCost = std::min(LAMBDA_D, dispDistCostJ);
	}



	/////////////////////////////////////////////////////////////////////////////////
	// Step 3 - Compute cost change of E_smo
	/////////////////////////////////////////////////////////////////////////////////
	// Compute each pixel's contribution to E_smo in N_{3x3}(p).
	int numRows = img.rows, numCols = img.cols;
	const cv::Point2i dirOffsets[] = {
		cv::Point2i(-1, 0), cv::Point2i(+1, 0),
		cv::Point2i(0, -1), cv::Point2i(0, +1)
	};

	// Foreach q\in N_{5}(p), we compute q's contribution to all possible combination
	// of E_smo(\theta_i, \theta_j, o_ij) and the change of boundary sizes B_ij, then
	// we collect the contributions to compute the cost change of each E_smo(i,j,o_ij) affected.

	int I = idI, J = idJ;
	std::set<int> tmp;
	tmp.insert(I);
	tmp.insert(J);
	for (int r = 0; r < 4; r++) {
		cv::Point2i q = p + dirOffsets[r];
		if (InBound(q, numRows, numCols)) {
			tmp.insert(labelMap.at<int>(q.y, q.x));
		}
	}
	std::vector<int> labelSet	= std::vector<int>(tmp.begin(), tmp.end());
	int numLabels				= labelSet.size();
	cv::Mat unnormalizedCostInc = cv::Mat::zeros(numLabels, numLabels, CV_64FC1);
	cv::Mat boundarySizesInc	= cv::Mat::zeros(numLabels, numLabels, CV_32SC1);

	// Compute each pixel's contribution to muInc and boundarySizesInc
	const cv::Point2i offsetsN5[] = {
		cv::Point2i(-1, 0), cv::Point2i(+1, 0),
		cv::Point2i(0, -1), cv::Point2i(0, +1),
		cv::Point2i(0,  0)
	};
	
	for (int r = 0; r < 5; r++) {
		cv::Point2i q = p + offsetsN5[r];
		if (InBound(q, numRows, numCols)) {

			for (int i = 0; i < numLabels; i++) {
				for (int j = i + 1; j < numLabels; j++) {
					int idxI = labelSet[i];
					int idxJ = labelSet[j];
					UpdateSmoothCostChange(p, i, j, idxI, idxJ, labelMap, labelMapNew,
						abc[idxI], abc[idxJ], unnormalizedCostInc, boundarySizesInc, segRelationMat);
				}
			}
		}
	}



	
	// Pay the bill - collect the contributions
	double smoothCostChanged = 0;
	cv::Mat smoothCostsInc = cv::Mat::zeros(numLabels, numLabels, CV_64FC1);
	for (int i = 0; i < numLabels; i++) {
		for (int j = i + 1; j < numLabels; j++) {

			int idxI = labelSet[i];
			int idxJ = labelSet[j];
			int oldBoundarySize = boundarySizes.at<int>(idxI, idxJ);
			int newBoundarySize = oldBoundarySize + boundarySizesInc.at<int>(i, j);
			double oldCost = smoothCosts.at<double>(idxI, idxJ);
			double newCost = oldCost;
			int oij = segRelationMat.at<int>(idxI, idxJ);

			int newSegSizeI, newSegSizeJ;
			double oldVal, newVal;

			switch (oij) {
			case HINGE:
				newCost = (oldCost * oldBoundarySize + unnormalizedCostInc.at<double>(idxI, idxJ)) / newBoundarySize;
				smoothCostChanged += newCost - oldCost;
				smoothCostsInc.at<double>(i, j) = newCost - oldCost;
				smoothCostsInc.at<double>(j, i) = newCost - oldCost;
				break;

			case COPLANAR:
				newSegSizeI = segSizes[idxI];
				newSegSizeJ = segSizes[idxJ];
				newSegSizeI -= (idxI == idI);
				newSegSizeJ -= (idxJ == idI);
				newSegSizeI += (idxI == idJ);
				newSegSizeJ += (idxJ == idJ);
				newCost = (oldCost * (segSizes[idxI] + segSizes[idxJ]) 
					+ unnormalizedCostInc.at<double>(idxI, idxJ)) / (newSegSizeI + newSegSizeJ);
				smoothCostChanged += newCost - oldCost;
				smoothCostsInc.at<double>(i, j) = newCost - oldCost;
				smoothCostsInc.at<double>(j, i) = newCost - oldCost;
				break;

			case LEFTOCC:
			case RIGHTOCC:
				oldVal = smoothCosts.at<double>(idxI, idxJ);
				newVal = oldVal + unnormalizedCostInc.at<double>(idxI, idxJ);
				if (oldVal < 0 && newVal >= 0) {
					smoothCostChanged -= LAMBDA_PEN;
				}
				else if (oldVal >= 0 && newVal < 0) {
					smoothCostChanged += LAMBDA_PEN;
				}
				smoothCostsInc.at<double>(i, j) = newVal - oldVal;
				smoothCostsInc.at<double>(j, i) = newVal - oldVal;
				break;
			}
		}
	}




	/////////////////////////////////////////////////////////////////////////////////
	// Step 4 - UpdateLabels
	/////////////////////////////////////////////////////////////////////////////////
	double allCostChanged  = (newALCost - oldALCost)
		+ LAMBDA_BOU   * (newBoundaryCost - oldBoundaryCost)
		+ LAMBDA_DEPTH * (newDepthCost - oldDepthCost)
		+ LAMBDA_SMO   * (smoothCostChanged);

	if (allCostChanged < 0.0) {
		meanColorCosts[idI]		= RunningMeanSquareCost(pColor, meanColors[idI], meanColorCosts[idI], segSizes[idI], -1);
		meanColorCosts[idJ]		= RunningMeanSquareCost(pColor, meanColors[idJ], meanColorCosts[idJ], segSizes[idJ], +1);
		meanPositionCosts[idI]	= RunningMeanSquareCost(cv::Point2d(p), meanPositions[idI], meanPositionCosts[idI], segSizes[idI], -1);
		meanPositionCosts[idJ]	= RunningMeanSquareCost(cv::Point2d(p), meanPositions[idJ], meanPositionCosts[idJ], segSizes[idJ], +1);
		meanColors[idI]		= (meanColors[idI] * segSizes[idI] - pColor) / (segSizes[idI] - 1.f);
		meanColors[idJ]		= (meanColors[idJ] * segSizes[idJ] + pColor) / (segSizes[idJ] + 1.f);
		meanPositions[idI]	= (meanPositions[idI] * segSizes[idI] - cv::Point2d(p)) * (1.f / (segSizes[idI] - 1));
		meanPositions[idJ]	= (meanPositions[idJ] * segSizes[idJ] + cv::Point2d(p)) * (1.f / (segSizes[idJ] + 1));
		segSizes[idI]--;
		segSizes[idJ]++;
		labelMap.at<int>(p.y, p.x) = idJ;

		// Update smoothCosts and boundarySizes
		for (int i = 0; i < numLabels; i++) {
			for (int j = i + 1; j < numLabels; j++) {
				int idxI = labelSet[i];
				int idxJ = labelSet[j];
				boundarySizes.at<int>(idxI, idxJ)  += boundarySizesInc.at<int>(i, j);
				boundarySizes.at<int>(idxJ, idxI)  += boundarySizesInc.at<int>(j, i);
				smoothCosts.at<double>(idxI, idxJ) += smoothCostsInc.at<double>(i, j);
				smoothCosts.at<double>(idxJ, idxI) += smoothCostsInc.at<double>(j, i);
			}
		}

		return std::make_pair(idJ, bestNewOccLabel);
	}
	else {
		return std::make_pair(idI, bestOldOccLabel);
	}
}

static void InitializeSmoothCostsAndBoundarySizes(int numSegs, cv::Mat &labelMap, cv::Mat &segRelationMat, 
	std::vector<cv::Vec3f> &abc, cv::Mat &smoothCosts, cv::Mat &boundarySizes)
{
	const cv::Point2i offsetsN4[] = {
		cv::Point2i(-1, 0), cv::Point2i(+1, 0),
		cv::Point2i(0, -1), cv::Point2i(0, +1),
	};
	int numRows = labelMap.rows, numCols = labelMap.cols;

	smoothCosts		= cv::Mat::zeros(numSegs, numSegs, CV_64FC1);
	boundarySizes	= cv::Mat::zeros(numSegs, numSegs, CV_32SC1);
	std::vector<bool> visited(numSegs, false);

	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {

			int idI = labelMap.at<int>(y, x);
			for (int r = 0; r < 4; r++) {
				cv::Point2i q = cv::Point2i(x, y) + offsetsN4[r];
				if (InBound(q, numRows, numCols)) {
					int idJ = labelMap.at<int>(q.y, q.x);

					if (!visited[idJ]) {
						visited[idJ] = true;
						boundarySizes.at<int>(idI, idJ) += 1;
						boundarySizes.at<int>(idJ, idI) += 1;

						int idxFront, idxBack;
						double dispFront, dispBack, dispDiff;
						cv::Vec3f &abcI = abc[idI];
						cv::Vec3f &abcJ = abc[idJ];

						switch (segRelationMat.at<int>(idI, idJ)) {
						case COPLANAR:
						case HINGE:
							dispDiff = abcI.dot(cv::Vec3f(q.x, q.y, 1.f)) - abcJ.dot(cv::Vec3f(q.x, q.y, 1.f));
							smoothCosts.at<double>(idI, idJ) += dispDiff * dispDiff;
							smoothCosts.at<double>(idJ, idI) += dispDiff * dispDiff;
							break;

						case LEFTOCC:
							idxFront = idI;
							idxBack = idJ;
							// No break here
						case RIGHTOCC:
							idxFront = idJ;
							idxBack = idI;
							// No break here	
						
							dispFront = abc[idxFront].dot(cv::Vec3f(q.x, q.y, 1.f));
							dispBack  = abc[idxBack].dot(cv::Vec3f(q.x, q.y, 1.f));
							smoothCosts.at<double>(idI, idJ) += dispFront - dispBack;
							smoothCosts.at<double>(idJ, idI) += dispFront - dispBack;
							break;
						}
					}
				}
			}

			for (int r = 0; r < 4; r++) {
				cv::Point2i q = cv::Point2i(x, y) + offsetsN4[r];
				if (InBound(q, numRows, numCols)) {
					visited[labelMap.at<int>(q.y, q.x)] = false;
				}
			}
		}
	}
}

static void ExtendeTPS(cv::Mat &img, cv::Mat &dispSgm, cv::Mat &validMap, 
	cv::Mat &labelMap, cv::Mat & occMap, cv::Mat &segRelationMat, cv::Mat &boundarySizes,
	std::vector<int> &segSizes, std::vector<cv::Vec3f> abc, 
	std::vector<cv::Vec3d> &meanColors, std::vector<cv::Point2d> &meanPositions,
	std::vector<cv::Vec3d> &meanColorCosts, std::vector<cv::Point2d> &meanPositionCosts)
{
	const cv::Point2i dirOffsets[] = {
		cv::Point2i(-1, 0), cv::Point2i(+1, 0),
		cv::Point2i(0, -1), cv::Point2i(0, +1)
	};
	int numRows = labelMap.rows, numCols = labelMap.cols;
	cv::Mat labelMapNew = labelMap.clone();


	//////////////////////////////////////////////////////////////////////////////
	// Step 1 - Prepare intial E_smo cost, and B_ij
	//////////////////////////////////////////////////////////////////////////////
	cv::Mat smoothCosts;
	InitializeSmoothCostsAndBoundarySizes(segSizes.size(), 
		labelMap, segRelationMat, abc, smoothCosts, boundarySizes);


	//////////////////////////////////////////////////////////////////////////////
	// Step 2 - Adjust the ownerships of boundary pixels
	//////////////////////////////////////////////////////////////////////////////
	std::stack<cv::Point2i> stack;
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			if (IsBoundaryPixel(cv::Point2i(x, y), labelMap)) {
				stack.push(cv::Point2i(x, y));
			}
		}
	}

	while (!stack.empty()) {
		cv::Point2i p = stack.top();
		stack.pop();
		int oldSegId = labelMap.at<int>(p.y, p.x);

		if (BreakConnectivity(p, labelMap)) {
			continue;
		}

		std::set<int> candidateLabels;
		for (int r = 0; r < 4; r++) {
			cv::Point2i q = p + dirOffsets[r];
			if (InBound(q, numRows, numCols)) {
				candidateLabels.insert(labelMap.at<int>(q.y, q.x));
			}
		}

		std::pair<int, int> newSegOccLabelPair = std::pair<int, int>(oldSegId, 0);
		for (std::set<int>::iterator it = candidateLabels.begin(); it != candidateLabels.end(); it++) {
			int idI = labelMap.at<int>(p.y, p.x);
			int idJ = *it;
			labelMapNew.at<int>(p.y, p.x) = idJ;

			newSegOccLabelPair = DetermineOptimalSegmentAndOcclusionLabel(p, img, idI, idJ, labelMap, labelMapNew,
				dispSgm, validMap, segRelationMat, segSizes, abc, smoothCosts, boundarySizes,
				meanColors, meanPositions, meanColorCosts, meanPositionCosts);

			labelMap.at<int>(p.y, p.x)		= newSegOccLabelPair.first;
			occMap.at<int>(p.y, p.x)		= newSegOccLabelPair.second;
			// Recover labelMapNew (set it identical to labelMap for next use)
			labelMapNew.at<int>(p.y, p.x)	= labelMap.at<int>(p.y, p.x);  
		}

		
		if (newSegOccLabelPair.first != oldSegId) {
			for (int r = 0; r < 4; r++) {
				cv::Point2i q = p + dirOffsets[r];
				if (IsBoundaryPixel(q, labelMap)) {
					stack.push(q);
				}
			}
		}
	}
}

static void UpdateBoundaryLabeling(cv::Mat &segRelationMat, cv::Mat &boundarySizes,
	cv::Mat &labelMap, std::vector<cv::Vec3f> &abc, std::vector<int> &segSizes, 
	std::vector<std::vector<int>> &adjGraph)
{
	const double LAMBDA_PEN		= 30;
	const double LAMBDA_HINGE	= 5;
	const double LAMBDA_OCC		= 15;

	int numSegs = segSizes.size();
	int numRows = labelMap.rows, numCols = labelMap.cols;
	cv::Mat smoothCosts = cv::Mat::zeros(numRows, numCols, CV_64FC4);


	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {

			// Contribute itselft to all the related oij
			int idI = labelMap.at<int>(y, x);
			cv::Vec3f &abcI = abc[idI];

			for (int k = 0; k < adjGraph[idI].size(); k++) {
				int idJ = adjGraph[idI][k];
				cv::Vec3f &abcJ = abc[idJ];

				// COPLANR
				double dispDiff = abcI.dot(cv::Vec3f(x, y, 1)) - abcJ.dot(cv::Vec3f(x, y, 1));
				smoothCosts.at<cv::Vec4d>(idI, idJ)[COPLANAR] += dispDiff * dispDiff;
				smoothCosts.at<cv::Vec4d>(idJ, idI)[COPLANAR] += dispDiff * dispDiff;
			
				if (BelongToBoundaryBij(cv::Point2i(x, y), labelMap, idI, idJ)) {
					// HINGE
					smoothCosts.at<cv::Vec4d>(idI, idJ)[HINGE] += dispDiff * dispDiff;
					smoothCosts.at<cv::Vec4d>(idJ, idI)[HINGE] += dispDiff * dispDiff;

					// LEFTOCC, RIGHTOCC
					double dispI = abcI.dot(cv::Vec3f(x, y, 1));
					double dispJ = abcJ.dot(cv::Vec3f(x, y, 1));
					smoothCosts.at<cv::Vec4d>(idI, idJ)[LEFTOCC ] += dispI - dispJ;
					smoothCosts.at<cv::Vec4d>(idI, idJ)[RIGHTOCC] += dispJ - dispI;
					smoothCosts.at<cv::Vec4d>(idJ, idI)[LEFTOCC ] += dispJ - dispI;
					smoothCosts.at<cv::Vec4d>(idJ, idI)[RIGHTOCC] += dispI - dispJ;
				}
			}
		}
	}

	for (int i = 0; i < numSegs; i++){
		for (int j = 0; j < numSegs; j++){
			if (i == j) {
				continue;
			}
			if (boundarySizes.at<int>(i, j) > 0) {
				smoothCosts.at<cv::Vec4d>(i, j)[COPLANAR] /= (segSizes[i] + segSizes[j]);
				smoothCosts.at<cv::Vec4d>(i, j)[HINGE]    /= boundarySizes.at<int>(i, j);
				smoothCosts.at<cv::Vec4d>(i, j)[LEFTOCC]   = (smoothCosts.at<cv::Vec4d>(i, j)[LEFTOCC]  < 0 ? LAMBDA_PEN : 0);
				smoothCosts.at<cv::Vec4d>(i, j)[RIGHTOCC]  = (smoothCosts.at<cv::Vec4d>(i, j)[RIGHTOCC] < 0 ? LAMBDA_PEN : 0);

				smoothCosts.at<cv::Vec4d>(i, j)[HINGE]		+= LAMBDA_HINGE;
				smoothCosts.at<cv::Vec4d>(i, j)[LEFTOCC]	+= LAMBDA_OCC;
				smoothCosts.at<cv::Vec4d>(i, j)[RIGHTOCC]	+= LAMBDA_OCC;
			}

			cv::Vec4d costs = smoothCosts.at<cv::Vec4d>(i, j);
			int bestIdx = 0;
			for (int k = 1; k < 4; k++) {
				if (costs[k] < costs[bestIdx]) {
					bestIdx = k;
				}
			}
			segRelationMat.at<int>(i, j) = bestIdx;
		}
	}
}

static void UpdatePlaneEquations(std::vector<int> &segSizes, std::vector<cv::Vec3f> &abc, 
	cv::Mat &dispSgm,  cv::Mat &labelMap, std::vector<std::vector<int>> &adjGraph, 
	cv::Mat &segRelationMat, cv::Mat &boundarySizes, 
	std::vector<std::vector<cv::Point2i>> &segPixelLists,
	std::vector<std::vector<cv::Point2i>> &segValidPixelLists,
	std::vector<std::vector<cv::Point2i>> &segBoundaryPixelLists)
{
	const double LAMBDA_DEPTH	= 2000;
	const double LAMBDA_SMO		= 400;
	int numSegs = segSizes.size();

	for (int idI = 0; idI < numSegs; idI++) {
		
		int numConstraints = segValidPixelLists[idI].size();
		for (int k = 0; k < adjGraph[idI].size(); k++) {
			int idJ = adjGraph[idI][k];
			int oij = segRelationMat.at<int>(idI, idJ);

			switch (oij) {
			case COPLANAR:
				numConstraints += segSizes[idI] + segSizes[idJ];
				break;

			case HINGE:
				numConstraints += boundarySizes.at<int>(idI, idJ);
				break;

			case LEFTOCC:
			case RIGHTOCC:
				// In the plane equation update, they are ignored.
				break;
			}
		}

		cv::Mat A(numConstraints, 3, CV_64FC1);
		cv::Mat B(numConstraints, 1, CV_64FC1);

		int cnt = 0;
		///////////////////////////////////////////////////////////////////
		// Constraints for E_depth
		///////////////////////////////////////////////////////////////////
		std::vector<cv::Point2i> &pixelList = segValidPixelLists[idI];
		double lambda = LAMBDA_DEPTH / LAMBDA_SMO;
		double sqrtLambda = std::sqrt(lambda);

		for (int k = 0; k < pixelList.size(); k++) {
			double x = pixelList[k].x;
			double y = pixelList[k].x;
			A.at<double>(cnt, 0) = sqrtLambda * x;
			A.at<double>(cnt, 1) = sqrtLambda * y;
			A.at<double>(cnt, 2) = sqrtLambda * 1.0;
			B.at<double>(cnt, 0) = sqrtLambda * dispSgm.at<float>(y, x);
			cnt++;
		}

		///////////////////////////////////////////////////////////////////
		// Constraints for COPLANAR pairs
		///////////////////////////////////////////////////////////////////
		for (int k = 0; k < adjGraph[idI].size(); k++) {
			int idJ = adjGraph[idI][k];
			int oij = segRelationMat.at<int>(idI, idJ);

			std::vector<cv::Point2i> mergedList = segPixelLists[idI];
			mergedList.insert(mergedList.end(), segPixelLists[idJ].begin(), segPixelLists[idJ].end());

			if (oij == COPLANAR) {
				///////////////////////////////////////////////////////////////////
				// Constraints for COPLANAR pairs
				///////////////////////////////////////////////////////////////////
				double w = 1.0 / (segSizes[idI] + segSizes[idJ]);
				double sqrtW = std::sqrt(w);

				for (int k = 0; k < mergedList.size(); k++) {
					double x = mergedList[k].x;
					double y = mergedList[k].x;
					A.at<double>(cnt, 0) = sqrtW * x;
					A.at<double>(cnt, 1) = sqrtW * y;
					A.at<double>(cnt, 2) = sqrtW * 1.0;
					B.at<double>(cnt, 0) = sqrtW * abc[idJ].dot(cv::Vec3f(x, y, 1));
					cnt++;
				}
			}
			else if (oij == HINGE) {
				///////////////////////////////////////////////////////////////////
				// Constraints for HINGE pairs
				///////////////////////////////////////////////////////////////////
				std::vector<cv::Point2i> bijPixelList;
				std::vector<cv::Point2i> boundaryPixelsI = segBoundaryPixelLists[idI];
				std::vector<cv::Point2i> boundaryPixelsJ = segBoundaryPixelLists[idJ];

				for (int k = 0; k < boundaryPixelsI.size(); k++) {
					if (BelongToBoundaryBij(boundaryPixelsI[k], labelMap, idI, idJ)) {
						bijPixelList.push_back(boundaryPixelsI[k]);
					}
				}
				for (int k = 0; k < boundaryPixelsJ.size(); k++) {
					if (BelongToBoundaryBij(boundaryPixelsJ[k], labelMap, idI, idJ)) {
						bijPixelList.push_back(boundaryPixelsJ[k]);
					}
				}

				ASSERT(bijPixelList.size() == boundarySizes.at<int>(idI, idJ));
				double w = 1.0 / (bijPixelList.size());
				double sqrtW = std::sqrt(w);

				for (int k = 0; k < mergedList.size(); k++) {
					double x = mergedList[k].x;
					double y = mergedList[k].x;
					A.at<double>(cnt, 0) = sqrtW * x;
					A.at<double>(cnt, 1) = sqrtW * y;
					A.at<double>(cnt, 2) = sqrtW * 1.0;
					B.at<double>(cnt, 0) = sqrtW * abc[idJ].dot(cv::Vec3f(x, y, 1));
					cnt++;
				}
			}
			else {
				///////////////////////////////////////////////////////////////////
				// Constraints for LEFT/RIGHTOCC pairs are not used
				///////////////////////////////////////////////////////////////////
			}
		}

		
		// Update plane equation
		cv::Mat X;
		cv::solve(A, B, X, cv::DECOMP_QR);
		abc[idI][0] = X.at<double>(0, 0);
		abc[idI][1] = X.at<double>(1, 0);
		abc[idI][2] = X.at<double>(2, 0);
	}
}

static void PrepareSegmentValidPixelLists(
	std::vector<int> &segSizes, cv::Mat &labelMap, cv::Mat &validMap, cv::Mat &boundarySizes,
	std::vector<std::vector<cv::Point2i>> &segPixelLists,
	std::vector<std::vector<cv::Point2i>> &segValidPixelLists, 
	std::vector<std::vector<cv::Point2i>> &segBoundaryPixelLists,
	std::vector<std::vector<int>> &adjGraph)
{
	int numRows = labelMap.rows, numCols = labelMap.cols;
	int numSegs = segSizes.size();
	segPixelLists.resize(numSegs);
	segValidPixelLists.resize(numSegs);
	segBoundaryPixelLists.resize(numSegs);
	
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			int id = labelMap.at<int>(y, x);

			segPixelLists[id].push_back(cv::Point2i(x, y));

			if (validMap.at<int>(y, x)) {
				segValidPixelLists[id].push_back(cv::Point2i(x, y));
			}

			if (IsBoundaryPixel(cv::Point2i(x, y), labelMap)) {
				segBoundaryPixelLists[id].push_back(cv::Point2i(x, y));
			}
		}
	}

	adjGraph.resize(numSegs);
	for (int i = 0; i < numSegs; i++){
		for (int j = i + 1; j < numSegs; j++){
			if (boundarySizes.at<int>(i, j) > 0) {
				adjGraph[i].push_back(j);
				adjGraph[j].push_back(i);
			}
		}
	}
}

void RunSPSS(std::string rootFolder, cv::Mat &imL, cv::Mat &imR, 
	cv::Mat &dispL, cv::Mat &dispR, cv::Mat &validMapL, cv::Mat &validMapR)
{
	int numRows = imL.rows, numCols = imL.cols;
	int numDisps, maxDisp, visualizeScale;
	SetupStereoParameters(rootFolder, numDisps, maxDisp, visualizeScale);

	int numPreferedRegions = 500;
	cv::Mat labelMapL;
	std::vector<int> segSizes;
	std::vector<cv::Vec3d> meanColors, meanColorCosts;
	std::vector<cv::Point2d> meanPositions, meanPositionCosts;


	// Step 1 - Topology Preserving Segmentation (TPS)
	TopologyPreservingSegmentation(imL, numPreferedRegions, labelMapL, segSizes,
		meanColors, meanPositions, meanColorCosts, meanPositionCosts);

	// Visualize segmentation.
	VisualizeSegmentation(imL, labelMapL);

	// Step 2 - Initialize \theta_i by RANSAC
	cv::Mat dispSgmL = dispL.clone();
	cv::Mat dispSgmR = dispR.clone();
	std::vector<cv::Vec3f> abc;
	RansacPlanefit(dispSgmL, validMapL, labelMapL, dispL, abc);

	

	// Visualize initial disparity map
	/*cv::Mat dispCmp;
	cv::hconcat(dispSgmL, dispL, dispCmp);
	cv::Mat dispImg(numRows, numCols, CV_8UC3);*/
	EvaluateDisparity(rootFolder, dispL, 0.5f);



	cv::Mat boundarySizes;
	cv::Mat occMap = cv::Mat::zeros(numRows, numCols, CV_32SC1);
	cv::Mat segRelationMat = COPLANAR * cv::Mat::ones(numRows, numCols, CV_32SC1);


	// Step 3 - Alternated Optimization
	const int maxOuterIters = 10;
	const int maxInnerIters = 10;
	for (int outerIters = 0; outerIters < maxOuterIters; outerIters++) {


		// Step 3.1 - Update segmentation and outlier labels by ETPS
		ExtendeTPS(imL, dispSgmL, validMapL, labelMapL, occMap, segRelationMat, boundarySizes, segSizes,
			abc, meanColors, meanPositions, meanColorCosts, meanPositionCosts);

		std::vector<std::vector<int>> adjGraph;
		std::vector<std::vector<cv::Point2i>> segPixelLists;
		std::vector<std::vector<cv::Point2i>> segValidPixelLists;
		std::vector<std::vector<cv::Point2i>> segBoundaryPixelLists;
		PrepareSegmentValidPixelLists(segSizes, labelMapL, validMapL, boundarySizes, 
			segPixelLists, segValidPixelLists, segBoundaryPixelLists, adjGraph);
		
		for (int innerIters = 0; innerIters < maxInnerIters; innerIters++) {
			// Step 3.2 - Update boudary labeling by brute force enumeration.
			UpdateBoundaryLabeling(segRelationMat, boundarySizes, labelMapL, abc, segSizes, adjGraph);

			// Step 3.3 - Update plane parameters by solving quadratic systems.
			UpdatePlaneEquations(segSizes, abc, dispSgmL, labelMapL, adjGraph, segRelationMat,
				boundarySizes, segPixelLists, segValidPixelLists, segBoundaryPixelLists);
		}
	}
}

void RunSlantedPlaneSmoothingStereoFlow(std::string rootFolder, cv::Mat &imL, cv::Mat &imR)
{
	cv::Mat dispL, dispR;
	cv::Mat validMapL, validMapR;
	void RunCSGM(std::string rootFolder, cv::Mat &imL, cv::Mat &imR,
		cv::Mat &dispL, cv::Mat &dispR, cv::Mat &validPixelMapL, cv::Mat &validPixelMapR);
	RunCSGM(rootFolder, imL, imR, dispL, dispR, validMapL, validMapR);
	RunSPSS(rootFolder, imL, imR, dispL, dispR, validMapL, validMapR);
}

void TestStereoFlow()
{
	extern std::string ROOTFOLDER;
	cv::Mat imL = cv::imread("D:/data/stereo/" + ROOTFOLDER + "/im2.png");
	cv::Mat imR = cv::imread("D:/data/stereo/" + ROOTFOLDER + "/im6.png");
	RunSlantedPlaneSmoothingStereoFlow(ROOTFOLDER, imL, imR);
}