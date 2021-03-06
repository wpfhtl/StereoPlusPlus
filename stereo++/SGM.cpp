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

		//const float P1 = 7;
		//const float P2min = 17;
		//const float alpha = 0.f;
		//const float gamma = 100;
		//const float P12 = 15;

		const int P1 = 100 /25;
		const int P2 = 1600/25;

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
			/*	float &I_cur  = gray.at<float>(pos.y, pos.x);
				float &I_last = gray.at<float>(lastPos.y, lastPos.x);
				float P2 = -alpha * std::abs(I_cur - I_last) + gamma;
				P2 = std::max(P2, P2min);*/
				float minCost = FLT_MAX;

				for (int d = 0; d < numDisps; d++) {
					float cost = std::min(
						L.get(lastPos.y, lastPos.x)[std::max(d - 1, 0)],
						L.get(lastPos.y, lastPos.x)[std::min(d + 1, numDisps - 1)]
						) + P1;
					cost = std::min(cost, L.get(lastPos.y, lastPos.x)[d]);

					//cost = std::min(cost, P12 + L.get(lastPos.y, lastPos.x)[std::max(d - 2, 0)]);
					//cost = std::min(cost, P12 + L.get(lastPos.y, lastPos.x)[std::min(d + 2, numDisps - 1)]);

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

cv::Mat ComputeCappedSobelImage(cv::Mat &imgIn, int sobelCapValue)
{

	cv::Mat img;
	if (imgIn.channels() == 3) {
		cv::cvtColor(imgIn, img, CV_BGR2GRAY);
	}
	else {
		img = imgIn.clone();
	}
	//cv::imshow("img", img);
	//cv::waitKey(0);

	int numRows = img.rows, numCols = img.cols;
	cv::Mat sobelImage(numRows, numCols, CV_8UC1);
	ASSERT(sobelImage.isContinuous());
	memset(sobelImage.data, sobelCapValue, numRows * numCols * sizeof(unsigned char));

	for (int y = 1; y < numRows - 1; ++y) {
		for (int x = 1; x < numCols - 1; ++x) {
			int sobelValue
				= (1.f * img.at<unsigned char>(y - 1, x + 1) + 2.f * img.at<unsigned char>(y, x + 1) + 1.f * img.at<unsigned char>(y + 1, x + 1))
				- (1.f * img.at<unsigned char>(y - 1, x - 1) + 2.f * img.at<unsigned char>(y, x - 1) + 1.f * img.at<unsigned char>(y + 1, x - 1));
			if (sobelValue > sobelCapValue) sobelValue = 2 * sobelCapValue;
			else if (sobelValue < -sobelCapValue) sobelValue = 0;
			else sobelValue += sobelCapValue;
			sobelImage.at<unsigned char>(y, x) = sobelValue;
		}
	}



	//unsigned char *image = img.data;
	//for (int y = 1; y < numRows - 1; ++y) {
	//	for (int x = 1; x < numCols - 1; ++x) {
	//		int sobelValue = (image[numCols*(y - 1) + x + 1] + 2 * image[numCols*y + x + 1] + image[numCols*(y + 1) + x + 1])
	//			- (image[numCols*(y - 1) + x - 1] + 2 * image[numCols*y + x - 1] + image[numCols*(y + 1) + x - 1]);
	//		if (sobelValue > sobelCapValue) sobelValue = 2 * sobelCapValue;
	//		else if (sobelValue < -sobelCapValue) sobelValue = 0;
	//		else sobelValue += sobelCapValue;
	//		sobelImage.data[numCols*y + x] = sobelValue;
	//	}
	//}

	return sobelImage;
}

MCImg<float> ComputeBirchfieldTomasiCostVolume(cv::Mat &imL, cv::Mat &imR, int numDisps, int sign)
{
	cv::Mat sobelL = ComputeCappedSobelImage(imL, 15);
	cv::Mat sobelR = ComputeCappedSobelImage(imR, 15);

	//cv::imshow("sobelL", sobelL);
	//cv::waitKey(0);
	//cv::imshow("sobelR", sobelR);
	//cv::waitKey(0);

	int numRows = sobelL.rows, numCols = sobelR.cols;
	cv::Mat leftHalfImgL(numRows, numCols, CV_8UC1), rightHalfImgL(numRows, numCols, CV_8UC1);
	cv::Mat leftHalfImgR(numRows, numCols, CV_8UC1), rightHalfImgR(numRows, numCols, CV_8UC1);

	for (int y = 0; y < numRows; y++) {
		leftHalfImgL.at<unsigned char>(y, 0) = sobelL.at<unsigned char>(y, 0);
		for (int x = 1; x < numCols; x++) {
			leftHalfImgL.at<unsigned char>(y, x) = 0.5f 
				* ((unsigned short)sobelL.at<unsigned char>(y, x - 1) + sobelL.at<unsigned char>(y, x));
		}
		for (int x = 0; x < numCols - 1; x++) {
			rightHalfImgL.at<unsigned char>(y, x) = 0.5f
				* ((unsigned short)sobelL.at<unsigned char>(y, x + 1) + sobelL.at<unsigned char>(y, x));
		}
		rightHalfImgL.at<unsigned char>(y, numCols - 1) = sobelL.at<unsigned char>(y, numCols - 1);
	}

	for (int y = 0; y < numRows; y++) {
		leftHalfImgR.at<unsigned char>(y, 0) = sobelR.at<unsigned char>(y, 0);
		for (int x = 1; x < numCols; x++) {
			leftHalfImgR.at<unsigned char>(y, x) = 0.5f
				* ((unsigned short)sobelR.at<unsigned char>(y, x - 1) + sobelR.at<unsigned char>(y, x));
		}
		for (int x = 0; x < numCols - 1; x++) {
			rightHalfImgR.at<unsigned char>(y, x) = 0.5f
				* ((unsigned short)sobelR.at<unsigned char>(y, x + 1) + sobelR.at<unsigned char>(y, x));
		}
		rightHalfImgR.at<unsigned char>(y, numCols - 1) = sobelR.at<unsigned char>(y, numCols - 1);
	}

	cv::Mat IminL, ImaxL;
	cv::Mat IminR, ImaxR;
	cv::min(leftHalfImgL, rightHalfImgL, IminL);	cv::min(sobelL, IminL, IminL);
	cv::max(leftHalfImgL, rightHalfImgL, ImaxL);	cv::max(sobelL, ImaxL, ImaxL);
	cv::min(leftHalfImgR, rightHalfImgR, IminR);	cv::min(sobelR, IminR, IminR);
	cv::max(leftHalfImgR, rightHalfImgR, ImaxR);	cv::max(sobelR, ImaxR, ImaxR);

	MCImg<float> dsiL(numRows, numCols, numDisps);

	#pragma omp parallel for
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			for (int d = 0; d < numDisps; d++) {
				float xm = x + sign * d;

				// FIXME: has implicitly assumed "ndisps <= numCols", it's not safe.
				if (xm < 0)				xm += numCols;
				if (xm > numCols - 1)	xm -= numCols;

				float costLtoR = std::max(
					(float)sobelL.at<unsigned char>(y, x) - ImaxR.at<unsigned char>(y, xm),
					(float)IminR.at<unsigned char>(y, xm) - sobelL.at<unsigned char>(y, x));
				costLtoR = std::max(0.f, costLtoR);

				float costRtoL = std::max(
					(float)sobelR.at<unsigned char>(y, xm) - ImaxL.at<unsigned char>(y, x),
					(float)IminL.at<unsigned char>(y, x) - sobelR.at<unsigned char>(y, xm));
				costRtoL = std::max(0.f, costRtoL);

				dsiL.get(y, x)[d] = std::min(costLtoR, costRtoL);
			}
		}
	}

	return dsiL;
}

MCImg<float> Compute5x5CensusCostVolume(cv::Mat &imL, cv::Mat &imR, int numDisps, int sign)
{
	float granularity = 1.f;
	int numRows = imL.rows, numCols = imL.cols;
	int numLevels = numDisps / granularity;
	MCImg<float> dsiL(numRows, numCols, numLevels);

	cv::Mat censusImgL = ComputeCensusImage(imL, 2, 2);
	cv::Mat censusImgR = ComputeCensusImage(imR, 2, 2);

	#pragma omp parallel for
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			for (int level = 0; level < numLevels; level++) {

				float d = level * granularity;
				float xm = x + sign * d;

				// FIXME: has implicitly assumed "ndisps <= numCols", it's not safe.
				if (xm < 0)				xm += numCols;
				if (xm > numCols - 1)	xm -= numCols;

				long long censusL = censusImgL.at<long long>(y, x);
				long long censusR = censusImgR.at<long long>(y, xm + 0.5);

				float censusDiff = HammingDist(censusL, censusR);
				dsiL.get(y, x)[level] = censusDiff;
			}
		}
	}

	return dsiL;
}

void CombineBirchfieldTomasiAnd5x5Census(MCImg<float> &dsiBT, MCImg<float> &dsiCensus)
{
	const float censusWeight = 1.0 / 0.5;
	ASSERT(dsiBT.w == dsiCensus.w && dsiBT.h == dsiCensus.h && dsiBT.n == dsiCensus.n);
	int N = dsiBT.w * dsiBT.h * dsiBT.n;

	#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		dsiCensus.data[i] = dsiBT.data[i] + censusWeight * dsiCensus.data[i];
	}
}

void BoxFilterCostVolume(MCImg<float> &dsi, int radius)
{
	int numRows = dsi.h, numCols = dsi.w, numDisps = dsi.n;
	cv::Size kernelSize = cv::Size(2 * radius + 1, 2 * radius + 1);

	cv::Mat population = cv::Mat::ones(numRows, numCols, CV_32FC1);
	cv::boxFilter(population, population, population.type(), kernelSize, cv::Point(-1, -1), false, cv::BORDER_CONSTANT);

	#pragma omp parallel for
	for (int d = 0; d < numDisps; d++) {
		cv::Mat costSlice(numRows, numCols, CV_32FC1);
		for (int y = 0; y < numRows; y++) {
			for (int x = 0; x < numCols; x++) {
				costSlice.at<float>(y, x) = dsi.get(y, x)[d];
			}
		}
		cv::boxFilter(costSlice, costSlice, costSlice.type(), kernelSize, cv::Point(-1, -1), false, cv::BORDER_CONSTANT);
		cv::divide(costSlice, population, costSlice);
		for (int y = 0; y < numRows; y++) {
			for (int x = 0; x < numCols; x++) {
				dsi.get(y, x)[d] = costSlice.at<float>(y, x);
			}
		}
	}
}

// Implement the algorithm Consistent Semi Global Matching (CSGM)
// PAMI'08 Stereo Processing by Semiglobal Matching and Mutual Information
void RunCSGM(std::string rootFolder, cv::Mat &imL, cv::Mat &imR,
	cv::Mat &dispL, cv::Mat &dispR, cv::Mat &validPixelMapL, cv::Mat &validPixelMapR)
{

	const float GRANULARITY = 1.f;
	int numDisps, maxDisp, visualizeScale;
	SetupStereoParameters(rootFolder, numDisps, maxDisp, visualizeScale);
	int numRows = imL.rows, numCols = imL.cols;

	//cv::imshow("imL", imL);
	//cv::waitKey(0);


	/*MCImg<float> dsiCensus = Compute5x5CensusCostVolume(imL, imR, numDisps, -1);
	cv::Mat dispCensus = WinnerTakesAll(dsiCensus, 1.f);
	dispCensus.convertTo(dispCensus, CV_8UC1);
	cv::imshow("dispCensus", dispCensus);
	cv::waitKey(0);*/

	//MCImg<float> dsiBT = ComputeBirchfieldTomasiCostVolume(imL, imR, numDisps, -1);
	//cv::Mat dispBT = WinnerTakesAll(dsiBT, 1.f);
	//dispBT.convertTo(dispBT, CV_8UC1);
	//cv::imshow("dispBT", dispBT);
	//cv::waitKey(0);
	//return;


	// Step 1 - Building Cost Volume
	//MCImg<float> dsiL = ComputeAdGradientCostVolume(imL, imR, numDisps, -1, GRANULARITY);
	//MCImg<float> dsiR = ComputeAdGradientCostVolume(imR, imL, numDisps, +1, GRANULARITY);
	/*MCImg<float> dsiL = Compute9x7CensusCostVolume(imL, imR, numDisps, -1, 1.f);
	MCImg<float> dsiR = Compute9x7CensusCostVolume(imR, imL, numDisps, +1, 1.f);*/
	//MCImg<float> dsiL = ComputeBirchfieldTomasiCostVolume(imL, imR, numDisps, -1);
	//MCImg<float> dsiR = ComputeBirchfieldTomasiCostVolume(imR, imL, numDisps, +1);

	MCImg<float> dsiBTL = ComputeBirchfieldTomasiCostVolume(imL, imR, numDisps, -1);
	MCImg<float> dsiBTR = ComputeBirchfieldTomasiCostVolume(imR, imL, numDisps, +1);
	BoxFilterCostVolume(dsiBTL, 2);
	BoxFilterCostVolume(dsiBTR, 2);
	MCImg<float> dsiL = Compute5x5CensusCostVolume(imL, imR, numDisps, -1);
	MCImg<float> dsiR = Compute5x5CensusCostVolume(imR, imL, numDisps, +1);
	BoxFilterCostVolume(dsiL, 2);
	BoxFilterCostVolume(dsiR, 2);
	CombineBirchfieldTomasiAnd5x5Census(dsiBTL, dsiL);
	CombineBirchfieldTomasiAnd5x5Census(dsiBTR, dsiR);

	//MCImg<unsigned short> yamaguchiCostVolume(numRows, numCols, numDisps);
	//yamaguchiCostVolume.LoadFromBinaryFile("D:\\data\\KITTI\\myresults\\Yamaguchi_CostVolume.bin");
	//int N = numRows * numCols * numDisps;
	//#pragma omp parallel for
	//for (int i = 0; i < N; i++) {
	//	dsiL.data[i] = yamaguchiCostVolume.data[i];
	//}



	
	for (int retry = 0; retry < 100; retry++) {
		int y = rand() % numRows;
		int x = rand() % numCols;
		int d = rand() % numDisps;
		//printf("%f\n", dsiL.get(y, x)[d]);
	}




	//cv::Mat dispRaw = WinnerTakesAll(dsiL, 1.f);
	//dispRaw.convertTo(dispRaw, CV_8UC1);
	//cv::imshow("dispRaw", dispRaw);
	//cv::waitKey(0);


	// Step 2 - Aggregate cost volume at 8 different directions
	MCImg<float> aggrDsiL = SemiGlobalCostAggregation(dsiL, imL);
	MCImg<float> aggrDsiR = SemiGlobalCostAggregation(dsiR, imR);


	// Step 3 - Subpixel WTA using quadratic interpolation, followed by 3x3 median filtering
	dispL = WinnerTakesAll(aggrDsiL, GRANULARITY);
	dispR = WinnerTakesAll(aggrDsiR, GRANULARITY);
	//cv::medianBlur(dispL, dispL, 3);
	//cv::medianBlur(dispR, dispR, 3);
	//dispL = QuadraticInterpDisp(dispL, aggrDsiL, GRANULARITY);
	//dispR = QuadraticInterpDisp(dispR, aggrDsiR, GRANULARITY);


	// Step 4 - Consistency Check
	validPixelMapL = CrossCheck(dispL, dispR, -1, 1.f);
	//for (int y = 0; y < numRows; y++) {
	//	for (int x = 0; x < numCols; x++) {
	//		if (!validPixelMapL.at<bool>(y, x)) {
	//			dispL.at<float>(y, x) = 0.f; 
	//		}
	//	}
	//}

	//cv::Mat dispImg;
	//dispL.convertTo(dispImg, CV_8UC1);
	//cv::imshow("dispOut", dispImg);
	//cv::waitKey(0);

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

	EvaluateDisparity(rootFolder, dispL, 0.5f);
}



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