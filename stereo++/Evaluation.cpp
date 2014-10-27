#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ReleaseAssert.h"
#include "EvalParams.h"


void SetupStereoParameters(std::string rootFolder, int &numDisps, int &maxDisp, int &visualizeScale)
{
	if (rootFolder == "Midd3") {
		numDisps = 280;
		visualizeScale = 1;
		extern int gNumDisps;
		if (gNumDisps > 0) {
			numDisps = gNumDisps;
		}
		if (numDisps % 16 != 0) {
			numDisps += (16 - numDisps % 16);
		}
	}
	else if (rootFolder == "KITTI") {
		numDisps = 224;
		visualizeScale = 1;
	}
	else if (rootFolder == "tsukuba") {
		numDisps = 16;
		visualizeScale = 16;
	}
	else if (rootFolder == "venus") {
		numDisps = 20;
		visualizeScale = 8;
	}
	else if (rootFolder == "teddy" || rootFolder == "cones") {
		numDisps = 60;
		visualizeScale = 4;
	}
	else if (rootFolder == "face"  || rootFolder == "face1") {
		numDisps = 80;
		visualizeScale = 3;
	}
	else {
		numDisps = 70;
		visualizeScale = 3;
	}
	if (numDisps % 16 != 0) {
		numDisps += (16 - numDisps % 16);
	}
	maxDisp = numDisps - 1;
}

void RegisterMouseCallbacks(std::string mouseCallbackName, void *callbackParams)
{
	if (mouseCallbackName == "OnMouseEvaluateDisparity") {
		void OnMouseEvaluateDisparity(int event, int x, int y, int flags, void *param);
		cv::setMouseCallback(mouseCallbackName, OnMouseEvaluateDisparity, callbackParams);
		return;
	}
	if (mouseCallbackName == "OnMousePatchMatchOnPixels") {
		void OnMousePatchMatchOnPixels(int event, int x, int y, int flags, void *param);
		cv::setMouseCallback(mouseCallbackName, OnMousePatchMatchOnPixels, callbackParams);
		return;
	}
	//if (mouseCallbackName == "OnMousePatchMatchOnTriangles") {
	//	cv::setMouseCallback(mouseCallbackName, OnMousePatchMatchOnTriangles, callbackParams);
	//	return;
	//}
	if (mouseCallbackName == "OnMouseLoopyBPOnGridGraph") {
		void OnMouseLoopyBPOnGridGraph(int event, int x, int y, int flags, void *param);
		cv::setMouseCallback(mouseCallbackName, OnMouseLoopyBPOnGridGraph, callbackParams);
		return;
	}
	if (mouseCallbackName == "OnMouseMeshStereoOnFactorGraph") {
		void OnMouseMeshStereoOnFactorGraph(int event, int x, int y, int flags, void *param);
		cv::setMouseCallback(mouseCallbackName, OnMouseMeshStereoOnFactorGraph, callbackParams);
		return;
	}
	if (mouseCallbackName == "OnMouseTestARAP") {
		void OnMouseTestARAP(int event, int x, int y, int flags, void *param);
		cv::setMouseCallback(mouseCallbackName, OnMouseTestARAP, callbackParams);
		return;
	}
	ASSERT(0)
}

static void EvaluateDisparityKITTI(cv::Mat &disp, void *ptrEvalParams)
{
	printf("EvaluateDisparityKITTI...\n");
	//disp *= 3;


	extern std::string kittiTestCaseId;
	cv::Mat &GT_NOC = cv::imread("D:/data/KITTI/training/disp_noc/" + kittiTestCaseId + ".png", CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat &GT_ALL = cv::imread("D:/data/KITTI/training/disp_occ/" + kittiTestCaseId + ".png", CV_LOAD_IMAGE_UNCHANGED);

	GT_NOC.convertTo(GT_NOC, CV_32FC1, 1.f / 256.f);
	GT_ALL.convertTo(GT_ALL, CV_32FC1, 1.f / 256.f);


	cv::Mat absDiffNoc, absDiffAll;
	cv::absdiff(disp, GT_NOC, absDiffNoc);
	cv::absdiff(disp, GT_ALL, absDiffAll);

	std::vector<float> errRatesNoc(6, 0), errRatesAll(6, 0);
	int numGtPixelsNoc = 0, numGtPixelsAll = 0;
	int numRows = disp.rows, numCols = disp.cols;


	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			// NONOCC
			if (GT_NOC.at<float>(y, x) > 0.f) {
				numGtPixelsNoc++;
				for (int e = 0; e <= 5; e++) {
					if (absDiffNoc.at<float>(y, x) > e) {
						errRatesNoc[e] += 1.f;
					}
				}
			}
			// ALL
			if (GT_ALL.at<float>(y, x) > 0.f) {
				numGtPixelsAll++;
				for (int e = 0; e <= 5; e++) {
					if (absDiffAll.at<float>(y, x) > e) {
						errRatesAll[e] += 1.f;
					}
				}
			}
		}
	}

	for (int e = 0; e <= 5; e++) {
		errRatesNoc[e] /= numGtPixelsNoc;
		errRatesAll[e] /= numGtPixelsAll;
		errRatesNoc[e] *= 100;
		errRatesAll[e] *= 100;
	}

	printf("        %6s %6s %6s %6s\n", "2px", "3px", "4px", "5px");
	printf("errNOC  %6.2f %6.2f %6.2f %6.2f\n",
		errRatesNoc[2], errRatesNoc[3], errRatesNoc[4], errRatesNoc[5]);
	printf("errALL  %6.2f %6.2f %6.2f %6.2f\n\n",
		errRatesAll[2], errRatesAll[3], errRatesAll[4], errRatesAll[5]);

	//cv::Mat disp16bitDetph;
	//disp.convertTo(disp16bitDetph, CV_16UC1, 256);
	//cv::imwrite("D:/code/rSGM/bin/Release/mydisp.png", disp16bitDetph);

	extern int VISUALIZE_EVAL;
	if (VISUALIZE_EVAL) {
		cv::Mat cmpImg;
		cv::vconcat(disp, GT_ALL, cmpImg);
		cmpImg.convertTo(cmpImg, CV_8UC1, 1.0);
		cv::cvtColor(cmpImg, cmpImg, CV_GRAY2BGR);
#if 1
		cv::imshow("cmpImg", cmpImg);

		void* callbackParams[] = { ptrEvalParams, &cmpImg };
		void OnMouseTestARAPKITTI(int event, int x, int y, int flags, void *param);
		if (ptrEvalParams != NULL) {
			cv::setMouseCallback("cmpImg", OnMouseTestARAPKITTI, callbackParams);
		}
		cv::waitKey(0);
#endif
	}
}

static void EvaluateDisparityMidd3(cv::Mat &disp, void *ptrEvalParams)
{
	printf("EvaluateDisparityMidd3...\n");

	extern std::string midd3TestCaseId;
	extern std::string midd3Resolution;

	bool FileExist(std::string filePath);
	if (!FileExist("D:/data/MiddEval3/" + midd3Resolution + "/" + midd3TestCaseId + "/disp0GT.pfm")) {
		extern int VISUALIZE_EVAL;
		if (VISUALIZE_EVAL) {
			ARAPEvalParams *evalParams = (ARAPEvalParams*)(ptrEvalParams);
			float numDisps = evalParams->numDisps;

#if 1
			MCImg<SlantedPlane> slantedPlanesL(disp.rows, disp.cols);
			MCImg<float> bestCostsL(disp.rows, disp.cols);
			slantedPlanesL.LoadFromBinaryFile("d:/pixelwiseSlantedPlanesL.bin");
			bestCostsL.LoadFromBinaryFile("d:/pixelwiseBestCostsL.bin");
			evalParams->pixelwiseSlantedPlanesL = &slantedPlanesL;
			evalParams->pixelwiseBestCostsL = &bestCostsL;
#endif
			
			cv::Mat Float2ColorJet(cv::Mat &fimg, float dmin, float dmax);
			cv::Mat dispImg = Float2ColorJet(disp, 0, numDisps);
			cv::imshow("dispImg", dispImg);
			evalParams->canvas = &dispImg;
			void OnMouseComparePixelwiseAndSegmentwisePM(int event, int x, int y, int flags, void *param);
			cv::setMouseCallback("dispImg", OnMouseComparePixelwiseAndSegmentwisePM, ptrEvalParams);
			cv::waitKey(0);
			
		}
		return;
	}


	cv::Mat ReadFilePFM(std::string filePath);
	cv::Mat GT   = ReadFilePFM("D:/data/MiddEval3/" + midd3Resolution + "/" + midd3TestCaseId + "/disp0GT.pfm");
	cv::Mat MASK = cv::imread( "D:/data/MiddEval3/" + midd3Resolution + "/" + midd3TestCaseId + "/mask0nocc.png", CV_LOAD_IMAGE_UNCHANGED);



	cv::Mat absDiff;
	cv::absdiff(disp, GT, absDiff);

	std::vector<float> errRatesNoc(6, 0), errRatesAll(6, 0);
	int numGtPixelsNoc = 0, numGtPixelsAll = 0;
	int numRows = disp.rows, numCols = disp.cols;

	const float errThresholds[5] = { 0.5, 1, 2, 3, 4 };

	// I has assumed the invalid values in the GT png file have been set to zero.
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			// NONOCC
			if (MASK.at<unsigned char>(y, x) == 255) {
				numGtPixelsNoc++;
				for (int e = 0; e <= 4; e++) {
					if (absDiff.at<float>(y, x) > errThresholds[e]) {
						errRatesNoc[e] += 1.f;
					}
				}
			}
			// ALL
			if (MASK.at<unsigned char>(y, x) >= 128) {
				numGtPixelsAll++;
				for (int e = 0; e <= 4; e++) {
					if (absDiff.at<float>(y, x) > errThresholds[e]) {
						errRatesAll[e] += 1.f;
					}
				}
			}
		}
	}

	for (int e = 0; e <= 4; e++) {
		errRatesNoc[e] /= numGtPixelsNoc;
		errRatesAll[e] /= numGtPixelsAll;
		errRatesNoc[e] *= 100;
		errRatesAll[e] *= 100;
	}

	printf("        %6s %6s %6s %6s\n", "0.5px", "1px", "2px", "4px");
	printf("errNOC  %6.2f %6.2f %6.2f %6.2f\n",
		errRatesNoc[0], errRatesNoc[1], errRatesNoc[2], errRatesNoc[4]);
	printf("errALL  %6.2f %6.2f %6.2f %6.2f\n\n",
		errRatesAll[0], errRatesAll[1], errRatesAll[2], errRatesAll[4]);


	extern int VISUALIZE_EVAL;
	if (VISUALIZE_EVAL) {
		cv::Mat errImg(numRows, numCols, CV_8UC1);
		for (int y = 0; y < numRows; y++) {
			for (int x = 0; x < numCols; x++) {
				if (absDiff.at<float>(y, x) > 1.0) {
					errImg.at<unsigned char>(y, x) = 0;
				}
				else {
					errImg.at<unsigned char>(y, x) = 255;
				}
			}
		}



		ARAPEvalParams *evalParams = (ARAPEvalParams*)(ptrEvalParams);
		evalParams->GT = &GT;
		evalParams->dispL = &disp;


		cv::Mat dispVisualize, canvas;
		int maxDisp = evalParams->numDisps - 1;
		float visualizeFactor = 255.f / maxDisp;
		disp.convertTo(dispVisualize, CV_8UC1, visualizeFactor);
		cv::hconcat(dispVisualize, errImg, canvas);

		while (canvas.rows > 700) {
			cv::resize(canvas, canvas, cv::Size(canvas.cols / 2, canvas.rows / 2));
		}
		evalParams->canvas = &canvas;
		cv::imshow("cmpImg", canvas);
		void OnMouseTestPlanefitMidd3(int event, int x, int y, int flags, void *param);
		if (ptrEvalParams != NULL) {
			cv::setMouseCallback("cmpImg", OnMouseTestPlanefitMidd3, ptrEvalParams);
		}
		cv::waitKey(0);
	}
	
}

void EvaluateDisparity(std::string rootFolder, cv::Mat &dispL, float eps = 1.f,
	void *auxParamsPtr = NULL, std::string mouseCallbackName = "OnMouseEvaluateDisparity")
{
	extern int DO_EVAL;
	if (!DO_EVAL) {
		return;
	}
	if (rootFolder == "KITTI") {
		EvaluateDisparityKITTI(dispL, auxParamsPtr);
		return;
	}
	if (rootFolder == "Midd3") {
		EvaluateDisparityMidd3(dispL, auxParamsPtr);
		return;
	}

	// Step 1 - Load images, parepare parameters
	std::string folderPrefix = "D:/data/stereo/";
	std::string workingDir = folderPrefix + rootFolder;

	cv::Mat imL		= cv::imread(workingDir + "/im2.png");
	cv::Mat imR		= cv::imread(workingDir + "/im6.png");
	cv::Mat NONOCC	= cv::imread(workingDir + "/nonocc.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat ALL		= cv::imread(workingDir + "/all.png",	 CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat DISC	= cv::imread(workingDir + "/disc.png",	 CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat GT		= cv::imread(workingDir + "/disp2.png",	 CV_LOAD_IMAGE_GRAYSCALE);
	
	int numDisps, maxDisp, visualizeScale;
	SetupStereoParameters(rootFolder, numDisps, maxDisp, visualizeScale);


	// Step 2 - Statistics
	cv::Mat absDiff, badPixelMap, dispTmp; 
	GT.convertTo(GT, CV_32FC1);
	dispL.convertTo(dispTmp, CV_8UC1, visualizeScale);
	dispTmp.convertTo(dispTmp, CV_32FC1);
	cv::absdiff(dispTmp, GT, absDiff);
	cv::compare(absDiff, eps * visualizeScale * cv::Mat::ones(GT.size(), CV_32FC1), badPixelMap, cv::CMP_GT);

	cv::Mat badOnNonocc(GT.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat badOnAll   (GT.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat badOnDisc  (GT.size(), CV_8UC3, cv::Scalar(255, 255, 255));

	const cv::Vec3b bgrBlue  = cv::Vec3b(255, 0, 0);
	const cv::Vec3b bgrGreen = cv::Vec3b(0, 255, 0);
	const cv::Vec3b bgrRed   = cv::Vec3b(0, 0, 255);

	float badRateOnNonocc	= 0.f,  numPixelsOnNonocc	= 0.f;
	float badRateOnAll		= 0.f,	numPixelsOnAll		= 0.f;
	float badRateOnDisc		= 0.f,	numPixelsOnDisc		= 0.f;

	int numRows = GT.rows, numCols = GT.cols;
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {

			bool isBadPixel	= (255 == badPixelMap.at<unsigned char>(y, x));
			numPixelsOnNonocc += (255 == NONOCC.at<unsigned char>(y, x));
			numPixelsOnAll	  += (255 == ALL.at<unsigned char>(y, x));
			numPixelsOnDisc	  += (255 == DISC.at<unsigned char>(y, x));

			if (isBadPixel && NONOCC.at<unsigned char>(y, x) == 255) {
				badOnNonocc.at<cv::Vec3b>(y, x) = bgrRed;
				badRateOnNonocc += 1.f;
			}
			else if (NONOCC.at<unsigned char>(y, x) != 255) {
				badOnNonocc.at<cv::Vec3b>(y, x) = bgrBlue;
			}

			if (isBadPixel && ALL.at<unsigned char>(y, x) == 255) {
				badOnAll.at<cv::Vec3b>(y, x) = bgrRed;
				badRateOnAll += 1.f;
			}
			else if (ALL.at<unsigned char>(y, x) != 255) {
				badOnAll.at<cv::Vec3b>(y, x) = bgrBlue;
			}

			if (isBadPixel && DISC.at<unsigned char>(y, x) == 255) {
				badOnDisc.at<cv::Vec3b>(y, x) = bgrRed;
				badRateOnDisc += 1.f;
			}
		}
	}

	badRateOnNonocc /= numPixelsOnNonocc;
	badRateOnAll    /= numPixelsOnAll;
	badRateOnDisc	/= numPixelsOnDisc;
	printf("%10.6f %10.6f %10.6f\n",
		100 * badRateOnNonocc, 100 * badRateOnAll, 100 * badRateOnDisc);
	
	// Step 3 - Prepare images for canvas
	cv::Mat canvas, topRow, bottomRow, gtImg, dispImg;
	dispL.convertTo(dispImg, CV_8UC1, visualizeScale);
	cv::cvtColor(dispImg, dispImg, CV_GRAY2BGR);
	GT.convertTo(gtImg, CV_8UC1);
	cv::cvtColor(gtImg, gtImg, CV_GRAY2BGR);
	GT.convertTo(GT, CV_32FC1, 1.f / visualizeScale);

	cv::hconcat(gtImg, dispImg, topRow);
	cv::hconcat(topRow, imL, topRow);
	cv::hconcat(badOnAll, badOnNonocc, bottomRow);
	cv::hconcat(bottomRow, imR, bottomRow);
	cv::vconcat(topRow, bottomRow, canvas);

	/*cv::Mat childWindowBL = canvas(cv::Rect(0, numRows, numCols, numRows));
	if (!auxParams.empty()) {
		if (auxParams[0].first == "triImg" || auxParams[0].first == "segImg") {
			(*(cv::Mat*)auxParams[0].second).copyTo(childWindowBL);
			auxParams.erase(auxParams.begin());
		}
	}*/

	// step 4 - Invoke mouse callbacks
	void *callbackParams[] = { 
		auxParamsPtr,
		&canvas, &dispL, &GT, &imL, &imR,
		&badRateOnNonocc, &badRateOnAll, &badRateOnDisc, 
		&numDisps, &visualizeScale, &workingDir
	};
	cv::imshow(mouseCallbackName, canvas);
	RegisterMouseCallbacks(mouseCallbackName, callbackParams);
	cv::waitKey(0);
}