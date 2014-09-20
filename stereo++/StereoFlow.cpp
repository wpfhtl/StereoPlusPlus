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
		- 2 * delta * (N * mu + sign * x)
		+ 2 * mu * delta + delta * delta;
	return newSqSum / (N + sign);
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
	const double LAMBDA_POS = 5000.f;
	const double LAMBDA_BOU = 1000.f;

	const cv::Point2i dirOffsets[] = {
		cv::Point2i(-1, 0), cv::Point2i(+1, 0),
		cv::Point2i(0, -1), cv::Point2i(0, +1)
	};

	int numRows = img.rows, numCols = img.cols;

	InitTPSByGrid(numPreferedSegments, img, labelMap, segSizes, 
		meanColors, meanPositions, meanColorCosts, meanPositionCosts);

	for (int i = 0; i < meanColorCosts.size(); i++) {
		printf("%10.1f   ", meanColorCosts[i]);
	}
	for (int i = 0; i < meanPositionCosts.size(); i++) {
		printf("%10.1f   ", meanPositionCosts[i]);
	}
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

			printf("dummyCounter = %d\n", ++dummyCounter);
			VisualizeSegmentation(img, labelMap);
		}
		else {
			printf("NOT CHANGED\n");
		}
	}

	for (int i = 0; i < segSizes.size(); i++) {
		printf("%8d", segSizes[i]);
	}
}

void RunSPSS(std::string rootFolder, cv::Mat &imL, cv::Mat &imR, cv::Mat &dispL, cv::Mat &dispR)
{
	int numRows = imL.rows, numCols = imL.cols;
	int numDisps, maxDisp, visualizeScale;
	SetupStereoParameters(rootFolder, numDisps, maxDisp, visualizeScale);

	int numPreferedRegions = 500;
	cv::Mat labelMap;
	std::vector<int> segSizes;
	std::vector<cv::Vec3d> meanColors, meanColorCosts;
	std::vector<cv::Point2d> meanPositions, meanPositionCosts;


	// Step 2 - Topology Preserving Segmentation (TPS)
	TopologyPreservingSegmentation(imL, numPreferedRegions, labelMap, segSizes,
		meanColors, meanPositions, meanColorCosts, meanPositionCosts);

	// Visualize segmentation.
	VisualizeSegmentation(imL, labelMap);


	// Step 3 - Alternated Optimization

	// Step 3.1 - Update segmentation and outlier labels by ETPS

	// Step 3.2 - Update boudary labeling by brute force enumeration.

	// Step 3.3 - Update plane parameters by solving quadratic systems.
}

void RunSlantedPlaneSmoothingStereoFlow(std::string rootFolder, cv::Mat &imL, cv::Mat &imR)
{
	cv::Mat dispL, dispR;
	//void RunCSGM(std::string rootFolder, cv::Mat &imL, cv::Mat &imR, cv::Mat &dispL, cv::Mat &dispR);
	//RunCSGM(rootFolder, imL, imR, dispL, dispR);
	RunSPSS(rootFolder, imL, imR, dispL, dispR);
}

void TestStereoFlow()
{
	extern std::string ROOTFOLDER;
	cv::Mat imL = cv::imread("D:/data/stereo/" + ROOTFOLDER + "/im2.png");
	cv::Mat imR = cv::imread("D:/data/stereo/" + ROOTFOLDER + "/im6.png");
	RunSlantedPlaneSmoothingStereoFlow(ROOTFOLDER, imL, imR);
}