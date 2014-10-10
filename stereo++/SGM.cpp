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
#include "Timer.h"


MCImg<float> SemiGlobalCostAggregation(MCImg<float> &dsi, cv::Mat &img)
{
	// Only assume 1.0 granularity
	int numRows = dsi.h, numCols = dsi.w, numDisps = dsi.n;
	MCImg<float> dsiAcc(numRows, numCols, numDisps);
	MCImg<float> L(numRows, numCols, numDisps);
	memset(dsiAcc.data, 0, numRows * numCols * numDisps * sizeof(float));

	const cv::Point2i dirOffsets[] = { 
		cv::Point2i(-1, -1), cv::Point2i(+1, +1),
		cv::Point2i(-1, +1), cv::Point2i(+1, -1),
		cv::Point2i( 0, -1), cv::Point2i( 0, +1), 
		cv::Point2i(-1,  0), cv::Point2i(+1,  0)
	};

	std::vector<cv::Point2i> LBorder(numRows), RBorder(numRows);
	std::vector<cv::Point2i> UBorder(numCols), DBorder(numCols);
	for (int y = 0; y < numRows; y++) {
		LBorder[y] = cv::Point2i(0, y);
		RBorder[y] = cv::Point2i(numCols - 1, y);
	}
	for (int x = 0; x < numCols; x++) {
		UBorder[x] = cv::Point2i(x, 0);
		DBorder[x] = cv::Point2i(x, numRows - 1);
	}


	for (int r = 0; r < 8; r++) {
		
		// Determine starting locations
		cv::Point2i offset = dirOffsets[r];
		std::vector<cv::Point2i> startLocations;

		if (offset.x == +1) {
			startLocations.insert(startLocations.end(), LBorder.begin(), LBorder.end());
		}
		if (offset.x == -1) {
			startLocations.insert(startLocations.end(), RBorder.begin(), RBorder.end());
		}
		if (offset.y == +1) {
			startLocations.insert(startLocations.end(), UBorder.begin(), UBorder.end());
		}
		if (offset.y == -1) {
			startLocations.insert(startLocations.end(), DBorder.begin(), DBorder.end());
		}

		// Init starting locations
		std::vector<cv::Point2i> curLocations = startLocations;
		std::vector<float> lastMinCosts(curLocations.size());
		for (int id = 0; id < curLocations.size(); id++) {
			lastMinCosts[id] = FLT_MAX;
			cv::Point2i pos = curLocations[id];
			for (int d = 0; d < numDisps; d++) {
				L.get(pos.y, pos.x)[d] = dsi.get(pos.y, pos.x)[d];
				lastMinCosts[id] = std::min(lastMinCosts[id], L.get(pos.y, pos.x)[d]);
			}
			curLocations[id] = curLocations[id] + offset;
		}

		// for AdGradient features
		//const float P1 = 0.03f;
		//const float P2 = 0.06f;

		//const float P1 = 28;
		//const float P2min = 30;
		//const float alpha = 0.3;
		//const float gamma = 63;

		const float P1 = 7;
		const float P2min = 17;
		const float alpha = 0.f;
		const float gamma = 100;

		cv::Mat gray;
		cv::cvtColor(img, gray, CV_BGR2GRAY);
		gray.convertTo(gray, CV_32FC1);

		// Sweeping
		int numSweepingLines = curLocations.size();
		#pragma omp parallel for
		for (int id = 0; id < /*curLocations.size()*/numSweepingLines; id++) {
			while (InBound(curLocations[id], numRows, numCols)) {
				cv::Point2i pos = curLocations[id];
				cv::Point2i lastPos = pos - offset;
				float &I_cur  = gray.at<float>(pos.y, pos.x);
				float &I_last = gray.at<float>(lastPos.y, lastPos.x);
				float P2 = -alpha * std::abs(I_cur - I_last) + gamma;
				P2 = std::max(P2, P2min);
				float minCost = FLT_MAX;

				for (int d = 0; d < numDisps; d++) {
					float cost = std::min(
						L.get(lastPos.y, lastPos.x)[std::max(d - 1, 0)],
						L.get(lastPos.y, lastPos.x)[std::min(d + 1, numDisps - 1)]
						) + P1;
					cost = std::min(cost, L.get(lastPos.y, lastPos.x)[d]);
					cost = std::min(cost, lastMinCosts[id] + P2);
					cost += dsi.get(pos.y, pos.x)[d] - lastMinCosts[id];
					L.get(pos.y, pos.x)[d] = cost;
					minCost = std::min(minCost, cost);
				}

				lastMinCosts[id] = minCost;
				curLocations[id] = curLocations[id] + offset;
			}
		}

		// Update accumulated cost
		#pragma omp parallel for
		for (int i = 0; i < numRows * numCols * numDisps; i++) {
			dsiAcc.data[i] += L.data[i];
		}
	}

	return dsiAcc;
}

cv::Mat QuadraticInterpDisp(cv::Mat &disp, MCImg<float> &dsi, float granularity)
{
	// Currently only deal with 1.0 granularity
	ASSERT(granularity == 1.f);
	int numRows = disp.rows, numCols = disp.cols;
	cv::Mat dispOut(numRows, numCols, CV_32FC1);
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			int x2  = disp.at<float>(y, x);
			int x1 = std::max(0, x2 - 1);
			int x3 = std::min(dsi.n - 1, x2 + 1);
			float y1 = 100 * dsi.get(y, x)[x1];
			float y2 = 100 * dsi.get(y, x)[x2];
			float y3 = 100 * dsi.get(y, x)[x3];
			float xv = (x1*x1*y2 - x2*x2*y1 - x1*x1*y3 + x3*x3*y1 + x2*x2*y3 - x3*x3*y2) / (2 * (x1*y2 - x2*y1 - x1*y3 + x3*y1 + x2*y3 - x3*y2));
			dispOut.at<float>(y, x) = std::max(x1 - 0.5f, std::min(x3 + 0.5f, xv));
		}
	}
	return dispOut;
}

// Implement the algorithm Consistent Semi Global Matching (CSGM)
// PAMI'08 Stereo Processing by Semiglobal Matching and Mutual Information
void RunCSGM(std::string rootFolder, cv::Mat &imL, cv::Mat &imR,
	cv::Mat &dispL, cv::Mat &dispR, cv::Mat &validPixelMapL, cv::Mat &validPixelMapR)
{
	bs::Timer::Tic("SGM");

	const float GRANULARITY = 1.f;
	int numDisps, maxDisp, visualizeScale;
	SetupStereoParameters(rootFolder, numDisps, maxDisp, visualizeScale);

	// Step 1 - Building Cost Volume
	//MCImg<float> dsiL = ComputeAdGradientCostVolume(imL, imR, numDisps, -1, GRANULARITY);
	//MCImg<float> dsiR = ComputeAdGradientCostVolume(imR, imL, numDisps, +1, GRANULARITY);
	MCImg<float> dsiL = Compute9x7CensusCostVolume(imL, imR, numDisps, -1, 1.f);
	//MCImg<float> dsiR = Compute9x7CensusCostVolume(imR, imL, numDisps, +1, 1.f);


	int numRows = imL.rows, numCols = imL.cols;
	for (int retry = 0; retry < 100; retry++) {
		int y = rand() % numRows;
		int x = rand() % numCols;
		int d = rand() % numDisps;
		//printf("%f\n", dsiL.get(y, x)[d]);
	}


	// Step 2 - Aggregate cost volume at 8 different directions
	MCImg<float> aggrDsiL = SemiGlobalCostAggregation(dsiL, imL);
	//MCImg<float> aggrDsiR = SemiGlobalCostAggregation(dsiR, imR);


	// Step 3 - Subpixel WTA using quadratic interpolation, followed by 3x3 median filtering
	dispL = WinnerTakesAll(aggrDsiL, GRANULARITY);
	//dispR = WinnerTakesAll(aggrDsiR, GRANULARITY);
	cv::medianBlur(dispL, dispL, 3);
	//cv::medianBlur(dispR, dispR, 3);
	//dispL = QuadraticInterpDisp(dispL, aggrDsiL, GRANULARITY);
	//dispR = QuadraticInterpDisp(dispR, aggrDsiR, GRANULARITY);


	// Step 4 - Consistency Check
	//validPixelMapL = CrossCheck(dispL, dispR, -1, 0.5f);
	//validPixelMapR = CrossCheck(dispR, dispL, +1, 0.5f);
	//cv::imshow("consitency map", validPixelMapL);
	//cv::waitKey(0);

	// Step 5 - Remove of peaks (optional).
	//          This step segments the disparity iamge by allowing disparities inside segment
	//          to vary by only one pixel, then remove segment below cetern size. This is 
	//          not a reasonable assumption for teddy's ground plane, where ground truth
	//          disparities do vary larger than one pixel. Therefore  it is left optional.


	// Step 6 - Special treament for textureless region (Intensity Consistent Disparity Selection)


	// Step 7 - Filled occluded and mismatched pixels (Discontinuity Preserving Interpolation)
	//          If you're just using SGM to feed the ECCV'14 Joint Stereo Flow Algorihtm, then
	//          this step is not neccesary. Joint Stereo Flow only requires a semi-dense input.
	//          So it's better to leave the filled pixels empty, since they are low-confident.


	// Step 8 - Another final consistency checked will be great before you feed the result
	//          into Joint Stereo Flow.
	bs::Timer::Toc();
	EvaluateDisparity(rootFolder, dispL, 0.5f);
}

std::string kittiTestCaseId = "000000_10";

void TestSemiGlobalMatching()
{
	cv::Mat imL, imR;
	extern std::string ROOTFOLDER;
	if (ROOTFOLDER == "KITTI") {
		extern bool useKitti;
		extern std::string kittiTestCaseId;
		
		imL = cv::imread("D:/data/KITTI/training/colored_0/" + kittiTestCaseId + ".png");
		imR = cv::imread("D:/data/KITTI/training/colored_1/" + kittiTestCaseId + ".png");

		//cv::Mat GT_NOC = cv::imread("D:/data/KITTI/training/disp_noc/" + kittiTestCaseId + ".png", CV_LOAD_IMAGE_UNCHANGED);
		//cv::imshow("tmp", GT_NOC);
		//cv::waitKey(0);
		//GT_NOC.convertTo(GT_NOC, CV_32FC1, 1.f / 255.f);
		//GT_NOC.convertTo(GT_NOC, CV_8UC1, 3);
		//cv::imshow("tmp", GT_NOC);
		//cv::waitKey(0);
		//return;

		//cv::Mat &disp = cv::imread("D:/code/rSGM/bin/Release/mydisp.png", CV_LOAD_IMAGE_UNCHANGED);
		//disp.convertTo(disp, CV_32FC1, 1.f / 256.f);
		//EvaluateDisparity(ROOTFOLDER, disp);
		//return;
	}
	else {
		imL = cv::imread("D:/data/stereo/" + ROOTFOLDER + "/im2.png");
		imR = cv::imread("D:/data/stereo/" + ROOTFOLDER + "/im6.png");
	}
	

	cv::Mat dispL, dispR;
	cv::Mat validMapL, validMapR;
	RunCSGM(ROOTFOLDER, imL, imR, dispL, dispR, validMapL, validMapR);
}