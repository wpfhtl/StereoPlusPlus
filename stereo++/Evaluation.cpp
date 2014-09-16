#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ReleaseAssert.h"



void SetupStereoParameters(std::string rootFolder, int &numDisps, int &maxDisp, int &visualizeScale)
{
	if (rootFolder == "tsukuba") {
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

void EvaluateDisparity(std::string rootFolder, cv::Mat &dispL, float eps = 1.f, 
	void *auxParamsPtr = NULL, std::string mouseCallbackName = "OnMouseEvaluateDisparity")
{
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