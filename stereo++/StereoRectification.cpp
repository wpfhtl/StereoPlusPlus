#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <fstream>
#define _USE_MATH_DEFINES
#include <math.h>


#include <GL/glut.h>
#include <stdio.h>
#include <gl\gl.h>
#include <gl\glu.h>
#include <stdlib.h>

#include "util.h"
#include "StereoAPI.h"
#include "ReleaseAssert.h"
#include "spsstereo/SGMStereo.h"
#include "Serialize.h"
















static void ComputeRelativePoseFromAbsolute(cv::Mat &R1, cv::Mat &R2, 
	cv::Mat &t1, cv::Mat &t2, cv::Mat &R12, cv::Mat &t12)
{
	R12 = R2 * R1.t();
	//cv::Mat C1 = -R1.t() * t1;
	//cv::Mat C2 = -R2.t() * t2;
	cv::Mat C1 = t1;
	cv::Mat C2 = t2;
	t12 = R2 * (C1 - C2);

	std::cout << "\n\nR2 * (C1 - C2) = " << t12 << "\n";
	std::cout << "t2 - R2 * R1.inv() * t1 = " << t2 - R2 * R1.inv() * t1 << "\n\n\n";
}

static cv::Mat ReadProjMatrixFromTxt(std::string filePath)
{
	cv::Mat P(3, 4, CV_64FC1);
	std::string dummy;

	std::ifstream inFile;
	inFile.open(filePath);
	inFile >> dummy;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 4; j++) {
			inFile >> P.at<double>(i, j);
		}
	}
	inFile.close();

	return P;
}

static void RectifyStereoPair(std::string filePathImageL, std::string filePathImageR,
	std::string filePathProjectionMatrixL, std::string filePathProjectionMatrixR)
{
	cv::Mat imL = cv::imread(filePathImageL);
	cv::Mat imR = cv::imread(filePathImageR);
	cv::Mat P1 = ReadProjMatrixFromTxt(filePathProjectionMatrixL);
	cv::Mat P2 = ReadProjMatrixFromTxt(filePathProjectionMatrixR);

	std::cout << "P1 = " << P1 << "\n";
	std::cout << "P2 = " << P2 << "\n";

	cv::Mat K1, R1, t1, C1;
	cv::Mat K2, R2, t2, C2;
	cv::decomposeProjectionMatrix(P1, K1, R1, C1);
	cv::decomposeProjectionMatrix(P2, K2, R2, C2);
	C1 = C1(cv::Rect(0, 0, 1, 3)).clone() / C1.at<double>(3, 0);
	C2 = C2(cv::Rect(0, 0, 1, 3)).clone() / C2.at<double>(3, 0);
	t1 = -R1 * C1;
	t2 = -R2 * C2;

	cv::Mat R12, t12;
	R12 = R2 * R1.t();
	t12 = t2 - R2 * R1.t() * t1;

	std::cout << "image size " << imL.size() << "\n";
	std::cout << "===============================================================\n";
	std::cout << "K1 = " << K1 << "\n\nR1=" << R1 << "\n\nt1=" << t1 << "\n\n";
	std::cout << "===============================================================\n";
	std::cout << "K2 = " << K2 << "\n\nR2=" << R2 << "\n\nt1=" << t2 << "\n\n";
	std::cout << "===============================================================\n";
	std::cout << "R12 = " << R12 << "\n\nt12 = " << t12 << "\n";
	std::cout << "===============================================================\n";


	cv::Size imgSize = imL.size();
	cv::Rect roi1, roi2;
	cv::Mat  Q;
	cv::Mat  distortCoeffs1 = cv::Mat::zeros(5, 1, CV_64FC1);
	cv::Mat  distortCoeffs2 = cv::Mat::zeros(5, 1, CV_64FC1);
	cv::stereoRectify(K1, distortCoeffs1, K2, distortCoeffs2, imgSize,
		R12, t12, R1, R2, P1, P2, Q, 
		0,	// don't align principal point
		-1, imgSize, &roi1, &roi2);

	std::cout << "\n\n";
	std::cout << "===============================================================\n";
	std::cout << "R1 = " << R1 << "\n\nP1 = " << P1 << "\n\n";
	std::cout << "===============================================================\n";
	std::cout << "R2 = " << R2 << "\n\nP2 = " << P2 << "\n\n";
	std::cout << "===============================================================\n";
	std::cout << "Q = " << Q << "\n\n";
	std::cout << "===============================================================\n";


	cv::Mat map1x, map1y, map2x, map2y;
	cv::initUndistortRectifyMap(K1, distortCoeffs1, R1, P1, imgSize, CV_32FC1, map1x, map1y);
	cv::initUndistortRectifyMap(K2, distortCoeffs2, R2, P2, imgSize, CV_32FC1, map2x, map2y);

	std::cout << "initUndistortRectifyMap done.\n";
	std::cout << map1x.size() << "   " << map1y.size() << "\n";
	std::cout << map2x.size() << "   " << map2y.size() << "\n";

	cv::Mat imgRectifiedL, imgRectifiedR;
	cv::remap(imL, imgRectifiedL, map1x, map1y, cv::INTER_LINEAR);
	cv::remap(imR, imgRectifiedR, map2x, map2y, cv::INTER_LINEAR);

	float ratio = 0.25;
	cv::Size newSize(imL.cols * ratio, imL.rows * ratio);
	cv::resize(imgRectifiedL, imgRectifiedL, newSize);
	cv::resize(imgRectifiedR, imgRectifiedR, newSize);
	cv::Mat canvas;
	cv::hconcat(imgRectifiedL, imgRectifiedR, canvas);
	cv::imshow("Rectified Pair", canvas);
	cv::waitKey(0);
}

static void EnhancedImShow(std::string title, cv::Mat &canvas, cv::Size tileSize = cv::Size(0, 0))
{
	cv::imshow(title, canvas);
	void OnMouseDisplayPixelValue(int event, int x, int y, int flags, void *param);
	if (tileSize == cv::Size(0, 0)) {
		tileSize = canvas.size();
	}
	void* callbackParams[] = { &title, &canvas, &tileSize };
	cv::setMouseCallback(title, OnMouseDisplayPixelValue, callbackParams);
	cv::waitKey(0);
	// destroy window to avoid runtime error. because when you call a new EnhancedImShow,
	// the auxiliary data of the previous EnhanceImshow invoke will be overwritten, 
	// therefore the previous window is using the wrong data and will probably crash.
	cv::destroyWindow(title);
}

static cv::Mat disp2XYZ(cv::Mat &disp, cv::Mat &P1, cv::Mat &P2)
{
	cv::Mat K1, K2, R1, R2, C1, C2;
	cv::decomposeProjectionMatrix(P1, K1, R1, C1);
	cv::decomposeProjectionMatrix(P2, K2, R2, C2);
	C1 = C1(cv::Rect(0, 0, 1, 3)).clone() / C1.at<double>(3, 0);
	C2 = C2(cv::Rect(0, 0, 1, 3)).clone() / C2.at<double>(3, 0);


#if 0
	std::cout << "C1 = " << C1 << "\n";
	std::cout << "C2 = " << C2 << "\n";
	std::cout << "K1 = " << K1 << "\n";
	std::cout << "K2 = " << K2 << "\n";
	std::cout << "R1 = " << R1 << "\n";
	std::cout << "R2 = " << R2 << "\n";
#endif


	double B = C2.at<double>(0, 0) - C1.at<double>(0, 0);	// baseline, may be negative?
	B = std::abs(B);
	double f = K1.at<double>(0, 0);							// focal length

	std::cout << "f = " << f << "\n";
	std::cout << "B = " << B << "\n";

	
	cv::Mat XYZ(disp.rows, disp.cols, CV_32FC3);
	for (int y = 0; y < disp.rows; y++) {
		for (int x = 0; x < disp.cols; x++) {
			if (isnan(disp.at<float>(y, x))) {	// do not process invalid pixels
				std::cout << "isnan detected at disp" << cv::Point2f(x, y)
					<< " = " << disp.at<float>(y, x) << "\n";
				XYZ.at<cv::Vec3f>(y, x) = cv::Vec3f(0, 0, 0);
				continue;
			}
			//disp.at<float>(y, x) += 270;
			if (disp.at<float>(y, x) == 0) {	// do not process invalid pixels
				XYZ.at<cv::Vec3f>(y, x) = cv::Vec3f(0, 0, 0);
				continue;
			}
			if (disp.at<float>(y, x) < 20) {
				XYZ.at<cv::Vec3f>(y, x) = cv::Vec3f(0, 0, 0);
				continue;
			}

			disp.at<float>(y, x) = std::min(disp.at<float>(y, x), 144.f);
			disp.at<float>(y, x) = std::max(disp.at<float>(y, x), 20.f);

			double Z = f * B / disp.at<float>(y, x);
			double X = x * Z / f;
			double Y = y * Z / f;
			XYZ.at<cv::Vec3f>(y, x) = cv::Vec3f(X, Y, Z);
		}
	}

	return XYZ;
}

static void VerifyQ(cv::Mat &R1, cv::Mat &R2, cv::Mat &P1, cv::Mat &P2, cv::Mat &Q,
	cv::Mat &R12, cv::Mat &t12)
{
	
	std::cout << "oldR1 = " << R1 << "\n";
	std::cout << "oldR2 = " << R2 << "\n\n";

	cv::Mat oldR1 = R1.clone();
	cv::Mat oldR2 = R2.clone();;

	cv::Mat A4x4 = cv::Mat::zeros(4, 4, CV_64FC1);
	cv::Mat dummpy, K1, K2, C1, C2;
	cv::decomposeProjectionMatrix(P1, K1, R1, C1);
	cv::decomposeProjectionMatrix(P2, K2, R2, C2);
	double X1 = C1.at<double>(0, 0) / C1.at<double>(3, 0);
	double X2 = C2.at<double>(0, 0) / C2.at<double>(3, 0);
	double B = X2 - X1;
	double f = K2.at<double>(0, 0);
	double u = K2.at<double>(0, 2);
	double v = K2.at<double>(1, 2);

	C1 = C1(cv::Rect(0, 0, 1, 3)) / C1.at<double>(3, 0);
	C2 = C2(cv::Rect(0, 0, 1, 3)) / C2.at<double>(3, 0);


	std::cout << "P1 = " << P1 << "\n";
	std::cout << "P2 = " << P2 << "\n\n";
	std::cout << "K1 = " << K1 << "\n";
	std::cout << "K2 = " << K2 << "\n\n";
	std::cout << "R1 = " << R1 << "\n";
	std::cout << "R2 = " << R2 << "\n\n";
	std::cout << "C1 = " << C1 << "\n";
	std::cout << "C2 = " << C2 << "\n\n";

	A4x4.at<double>(0, 0) = B;
	A4x4.at<double>(1, 1) = B;
	A4x4.at<double>(2, 3) = f*B;
	A4x4.at<double>(3, 2) = 1;
	A4x4.at<double>(0, 3) = -u*B;
	A4x4.at<double>(1, 3) = -v*B;

	
	std::cout << "---------------------------------------------\n";
	std::cout << "R2*R12 = " << oldR2 * R12 << "\n";
	std::cout << "R1 = " << oldR1 << "\n";
	std::cout << "R2*t12 = " << oldR2 * t12 << "\n";
	std::cout << "-C2 = " << -C2 << "\n";
	std::cout << "---------------------------------------------\n";

	//std::cout << "R1 = " << R1 << "\n";
	//std::cout << "R2 = " << R2 << "\n";
	//std::cout << "K1 = " << K1 << "\n";
	//std::cout << "K2 = " << K2 << "\n\n\n";

	cv::Mat R = cv::Mat::eye(4, 4, CV_64FC1);
	R2.copyTo(R(cv::Rect(0, 0, 3, 3)));
	cv::Mat myInvTransform = R.t() * A4x4;
	
	//std::cout << "my = " << myInvTransform << "\n";
	//std::cout << "Q = " << Q << "\n";
	//std::cout << "ratio = " << myInvTransform / Q << "----\n";
}

static void StereoReconstruct(std::string filePathImageL, std::string filePathImageR,
	std::string filePathProjectionMatrixL, std::string filePathProjectionMatrixR)
{
	const int maxRowSize = 500;
	const int maxColSize = 700;

	cv::Mat imL = cv::imread(filePathImageL);
	cv::Mat imR = cv::imread(filePathImageR);
	cv::Mat P1 = ReadProjMatrixFromTxt(filePathProjectionMatrixL);
	cv::Mat P2 = ReadProjMatrixFromTxt(filePathProjectionMatrixR);


	// Compute relative pose to call stereoRectify.
	cv::Mat K1, R1, t1, C1;
	cv::Mat K2, R2, t2, C2;
	cv::Mat R12, t12;
	cv::decomposeProjectionMatrix(P1, K1, R1, C1);
	cv::decomposeProjectionMatrix(P2, K2, R2, C2);
	C1 = C1(cv::Rect(0, 0, 1, 3)).clone() / C1.at<double>(3, 0);
	C2 = C2(cv::Rect(0, 0, 1, 3)).clone() / C2.at<double>(3, 0);
	t1 = -R1 * C1;
	t2 = -R2 * C2;
	R12 = R2 * R1.t();
	//t12 = t2 - R2 * R1.t() * t1;
	t12 = R2 * (C1 - C2);


	// Use acceptable image size for pixelwise stereo matching.
	float k = 1.f;
	for (k = 1.f; k <= 10.f; k += 1.f) {
		if (imL.rows / k <= maxRowSize && imL.cols / k <= maxColSize) {
			break;
		}
	}
	int oldNumRows = imL.rows, oldNumCols = imL.cols;
	int numRows = oldNumRows / k + 0.5;
	int numCols = oldNumCols / k + 0.5;
	cv::resize(imL, imL, cv::Size(numCols, numRows));
	cv::resize(imR, imR, cv::Size(numCols, numRows));


	// Compensate the K matrices for the image resizing.
	double ratioX = (double)numCols / oldNumCols;
	double ratioY = (double)numRows / oldNumRows;
	K1.at<double>(0, 0) *= ratioX;	K2.at<double>(0, 0) *= ratioX;
	K1.at<double>(0, 2) *= ratioX;	K2.at<double>(0, 2) *= ratioX;
	K1.at<double>(1, 1) *= ratioY;	K2.at<double>(1, 1) *= ratioY;
	K1.at<double>(1, 2) *= ratioY;	K2.at<double>(1, 2) *= ratioY;

	
	// Perform Rectification.
	cv::Mat		distortCoeffs1 = cv::Mat::zeros(1, 5, CV_64FC1);
	cv::Mat		distortCoeffs2 = cv::Mat::zeros(1, 5, CV_64FC1);
	cv::Mat		Q;
	cv::Size	imgSize = imL.size();
	cv::Rect	roi1, roi2;
	int			flagsAlignPrincipalPoints = CV_CALIB_ZERO_DISPARITY;
	cv::stereoRectify(K1, distortCoeffs1, K2, distortCoeffs2, imgSize,
		R12, t12,
		R1,		// rot matrix that transforms cam1's original frame to cam1's rectified frame
		R2,		// rot matrix that transforms cam2's original frame to cam2's rectified frame
		P1,		// projection matrix of the rectified cam1 in cam1's rectified coordinate frame
		P2,		// projection matrix of the rectified cam2 in cam1's rectified coordinate frame
		Q,
		flagsAlignPrincipalPoints,
		-1,		// detect cropped region automatically
		imgSize, &roi1, &roi2);

	//VerifyQ(R1, R2, P1, P2, Q);
	//return;

	cv::Mat map1x, map1y, map2x, map2y;
	cv::initUndistortRectifyMap(K1, distortCoeffs1, R1, P1, imgSize, CV_32FC1, map1x, map1y);
	cv::initUndistortRectifyMap(K2, distortCoeffs1, R2, P2, imgSize, CV_32FC1, map2x, map2y);

	cv::Mat imgRectifiedL, imgRectifiedR;
	cv::remap(imL, imgRectifiedL, map1x, map1y, cv::INTER_CUBIC);
	cv::remap(imR, imgRectifiedR, map2x, map2y, cv::INTER_CUBIC);

	if (flagsAlignPrincipalPoints) {
		cv::imwrite("d:/L_imgRectified_ppAligned.png", imgRectifiedL);
		cv::imwrite("d:/R_imgRectified_ppAligned.png", imgRectifiedR);
	}
	else {
		cv::imwrite("d:/L_imgRectified_ppNoAligned.png", imgRectifiedL);
		cv::imwrite("d:/R_imgRectified_ppNoAligned.png", imgRectifiedR);
	}

	cv::Mat canvasColor;
	cv::hconcat(imgRectifiedL, imgRectifiedR, canvasColor);
	EnhancedImShow("Rectified Pair", canvasColor, imL.size());

	//cv::Mat canvas2;
	//cv::hconcat(imL, imR, canvas2);
	//EnhancedImShow("canvas2", canvas2, imL.size());

#if 0
	// Perform stereo matching.
	int sadWindowSize = 15;
	int numDisps = 128;
	cv::StereoSGBM sgm(
		0,		// min disparity
		numDisps,
		sadWindowSize,								// SAD window size
		8 * 3 * sadWindowSize * sadWindowSize,		// P1
		24 * 3 * sadWindowSize * sadWindowSize,		// P2
		2,		// error threshold for cross check
		15,		// trucation value on x-derivative
		5,		// margin in percentage by which the best (minimum) computed cost function value should ¡°win¡± the second best value 
		0,		// disable speckle filtering
		0,		// max disp variation in each connected component, not used
		false	// only perform single pass.
		);

	cv::Mat disp;
	sgm(imL, imR, disp);
	disp.convertTo(disp, CV_8UC1, 255.f / (16 * numDisps));
	cv::imshow("disp", disp);
	cv::waitKey(0);
#else
	int numDisps = 144;
	numDisps += 15 - (numDisps - 1) % 16;
	extern int SGMSTEREO_DEFAULT_DISPARITY_TOTAL;
	SGMSTEREO_DEFAULT_DISPARITY_TOTAL = numDisps;

	cv::Mat dispL(numRows, numCols, CV_32FC1);
	cv::Mat dispR(numRows, numCols, CV_32FC1);
	ASSERT(dispL.isContinuous());
	ASSERT(dispR.isContinuous());

	std::string pathL = "d:/pngInputForYamaguchiL.png";
	std::string pathR = "d:/pngInputForYamaguchiR.png";
	cv::imwrite(pathL, imgRectifiedL);
	cv::imwrite(pathR, imgRectifiedR);
	png::image<png::rgb_pixel> pngImageL(pathL);
	png::image<png::rgb_pixel> pngImageR(pathR);

	SGMStereo sgm;
	sgm.compute(pngImageL, pngImageR, (float*)dispL.data, (float*)dispR.data);
	dispL = cv::max(dispL, 20.f);
	//dispR = cv::max(dispR, 20.f);
	cv::Mat SetInvalidDisparityToZeros(cv::Mat &dispL, cv::Mat &validPixelMapL);
	cv::Mat validPixelsL = CrossCheck(dispL, dispR, -1);
	cv::Mat validPixelsR = CrossCheck(dispR, dispL, +1);
	dispL = SetInvalidDisparityToZeros(dispL, validPixelsL);
	dispR = SetInvalidDisparityToZeros(dispR, validPixelsR);

	dispL.convertTo(dispL, CV_8UC1);
	dispR.convertTo(dispR, CV_8UC1);
	
	cv::Mat canvasDisp;
	cv::hconcat(dispL, dispR, canvasDisp);
	EnhancedImShow("disparities", canvasDisp, dispL.size());

	dispL.convertTo(dispL, CV_8UC1, 255.0 / numDisps);
	dispR.convertTo(dispR, CV_8UC1, 255.0 / numDisps);
	cv::hconcat(dispL, dispR, canvasDisp);
	cv::cvtColor(canvasDisp, canvasDisp, CV_GRAY2BGR);
	cv::vconcat(canvasColor, canvasDisp, canvasDisp);
	cv::imwrite("d:/color+disp.png", canvasDisp);
#endif


	cv::Mat XYZL, XYZR;
	
	cv::reprojectImageTo3D(dispL, XYZL, Q, false);
	cv::reprojectImageTo3D(dispR, XYZR, Q, false);
	//XYZL = disp2XYZ(dispL, P1, P2);
	//XYZR = disp2XYZ(dispR, P1, P2);

	void SavePointCloudToPly(std::string filePathPly, cv::Mat &XYZ, cv::Mat &img, cv::Mat &validPixels = cv::Mat());
	SavePointCloudToPly("d:/pointCloudL.ply", XYZL, imgRectifiedL);
	SavePointCloudToPly("d:/pointCloudR.ply", XYZR, imgRectifiedR);
	std::cout << "point clouds saved.\n";


	void BuildMesh(cv::Mat &imL, cv::Mat &imR, cv::Mat &dispL, cv::Mat &dispR,
		double focalLen, double baselineLen, cv::Mat &Q, cv::Mat &P1, cv::Mat &P2);
	double focalLen = 539.36;
	double baselineLen = 0.24; // not sure, never mind.
	BuildMesh(imgRectifiedL, imgRectifiedR, dispL, dispR, focalLen, baselineLen, Q, P1, P2);


	//RegisterViewingTrack();

}

void BuildMesh(cv::Mat &imL, cv::Mat &imR, cv::Mat &dispL, cv::Mat &dispR,
	double focalLen, double baselineLen, cv::Mat &Q, cv::Mat &P1, cv::Mat &P2)
{
	void ConstructNeighboringTriangleGraph(int numRows, int numCols, std::vector<cv::Point2f> &vertexCoords,
		std::vector<std::vector<int>> &triVertexInds, std::vector<cv::Point2f> &baryCenters,
		std::vector<std::vector<int>> &nbIndices);
	void DeterminePixelOwnership(int numRows, int numCols, std::vector<cv::Point2f> &vertexCoords,
		std::vector<std::vector<int>> &triVertexInds, std::vector<std::vector<cv::Point2i>> &triPixelLists);
	cv::Mat TriangleLabelToDisparityMap(int numRows, int numCols, std::vector<SlantedPlane> &slantedPlanes,
		std::vector<std::vector<cv::Point2i>> &triPixelLists);




	int numRows = imL.rows, numCols = imL.cols;
	int numDisps = 144;
	int maxDisp = numDisps - 1;

	std::vector<cv::Point2f> vertexCoordsL, vertexCoordsR;
	std::vector<std::vector<int>> triVertexIndsL, triVertexIndsR;
	//#define SLIC_TRIANGULATION
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
	cv::imwrite("d:/triangulation.png", triImgs);
	//cv::imshow("triangulation", triImgs);
	//cv::waitKey(0);

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
	dispL.convertTo(dispImgL, CV_8UC1, 255.f / maxDisp);
	dispR.convertTo(dispImgR, CV_8UC1, 255.f / maxDisp);
	cv::hconcat(dispImgL, dispImgR, dispImg);
	cv::imwrite("d:/dispL.png", dispL);
	cv::imwrite("d:/dispR.png", dispR);
	//cv::imshow("dispImgL", dispImg);
	//cv::waitKey(0);
	cv::Mat XYZL, XYZR;
	dispL = cv::max(20.f, cv::min(144.f, dispL));
	dispR = cv::max(20.f, cv::min(144.f, dispR));
	//cv::reprojectImageTo3D(dispL, XYZL, Q, false);
	//cv::reprojectImageTo3D(dispR, XYZR, Q, false);
	XYZL = disp2XYZ(dispL, P1, P2);
	XYZR = disp2XYZ(dispR, P1, P2);
	void SavePointCloudToPly(std::string filePathPly, cv::Mat &XYZ, cv::Mat &img, cv::Mat &validPixels = cv::Mat());
	SavePointCloudToPly("d:/triangleMeshL.ply", XYZL, imL);
	SavePointCloudToPly("d:/triangleMeshR.ply", XYZR, imR);
	std::cout << "point clouds saved.\n";




	void BuildWaterTightMesh(int sign, int numRows, int numCols, float focalLen, float baselineLen,
		std::vector<cv::Point2f> &vertexCoords, std::vector<std::vector<int>> &triVertexInds, std::vector<SlantedPlane> &slantedPlanes,
		std::vector<cv::Point3f> &meshVertexCoords, std::vector<std::vector<int>> &facetVetexIndsList);
	std::vector<cv::Point3f> meshVertexCoordsL, meshVertexCoordsR;
	std::vector<std::vector<int>> facetVertexIndsListL, facetVertexIndsListR;

	printf("Build water left mesh right .......\n");
	BuildWaterTightMesh(-1, numRows, numCols, focalLen, baselineLen, vertexCoordsL, triVertexIndsL, slantedPlanesL, meshVertexCoordsL, facetVertexIndsListL);
	printf("Build water tight mesh right .......\n");
	BuildWaterTightMesh(+1, numRows, numCols, focalLen, baselineLen, vertexCoordsR, triVertexIndsR, slantedPlanesR, meshVertexCoordsR, facetVertexIndsListR);
	
}

static void RegisterViewTrack(std::vector<cv::Mat> &projectionMatrices, cv::Point3f &objectCenter)
{
	int numCameras = projectionMatrices.size();
	std::vector<cv::Mat> intrinsicMatrices(numCameras), rotationMatrices(numCameras);
	std::vector<cv::Point3f> cameraCenters(numCameras), translationVectors(numCameras);

	for (int i = 0; i < projectionMatrices.size(); i++) {
		cv::Mat P = projectionMatrices[i];
		cv::Mat K, R, C, t;
		cv::decomposeProjectionMatrix(P, K, R, C);
		C = C(cv::Rect(0, 0, 1, 3)) / C.at<double>(3, 0);
		t = -R.t() * C;
		intrinsicMatrices[i] = K;
		rotationMatrices[i] = R;
		cameraCenters[i] = cv::Point3f(C.at<double>(0, 0), C.at<double>(1, 0), C.at<double>(2, 0));
		translationVectors[i] = cv::Point3f(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0));
	}

	for (int i = 0; i < projectionMatrices.size(); i++) {
		std::cout << "R" << i << " = " << rotationMatrices[i] << "\n";
	}
	for (int i = 0; i < projectionMatrices.size(); i++) {
		std::cout << "C" << i << " = " << cameraCenters[i] << "\n";
	}
}

void TestStereoRectification()
{
	int numCameras = 28;
	std::vector<cv::Mat> projectionMatrices;
	cv::Point3f objectCenter;
	for (int i = 0; i < numCameras; i++) {
		char buf[1024];
		sprintf(buf, "D:/data/Lumia800/CanSequence/CanSequence.nvm.cmvs/00/txt/%08d.txt", i);
		std::string filePathProjectionMatrix = buf;
		std::cout << "reading " << i << "\n";
		cv::Mat P = ReadProjMatrixFromTxt(filePathProjectionMatrix);
		projectionMatrices.push_back(P);
	}
	std::cout << " sdfsfsdf\n";
	RegisterViewTrack(projectionMatrices, objectCenter);
	return;


	std::string filePathImageL = "D:/data/Lumia800/CanSequence/CanSequence.nvm.cmvs/00/visualize/00000003.jpg";
	std::string filePathImageR = "D:/data/Lumia800/CanSequence/CanSequence.nvm.cmvs/00/visualize/00000002.jpg";
	std::string filePathProjectionMatrixL = "D:/data/Lumia800/CanSequence/CanSequence.nvm.cmvs/00/txt/00000003.txt";
	std::string filePathProjectionMatrixR = "D:/data/Lumia800/CanSequence/CanSequence.nvm.cmvs/00/txt/00000002.txt";


	//std::string filePathImageL = "D:/data/Lumia800/mvi_shoe/visualize/00000001.png";
	//std::string filePathImageR = "D:/data/Lumia800/mvi_shoe/visualize/00000000.png";
	//std::string filePathProjectionMatrixL = "D:/data/Lumia800/mvi_shoe/txt/00000001.txt";
	//std::string filePathProjectionMatrixR = "D:/data/Lumia800/mvi_shoe/txt/00000000.txt";


	//std::string filePathImageL = "D:/data/Lumia800/buddha/visualize/00000010.jpg";
	//std::string filePathImageR = "D:/data/Lumia800/buddha/visualize/00000015.jpg";
	//std::string filePathProjectionMatrixL = "D:/data/Lumia800/buddha/txt/00000010.txt";
	//std::string filePathProjectionMatrixR = "D:/data/Lumia800/buddha/txt/00000015.txt";

	StereoReconstruct(filePathImageL, filePathImageR, filePathProjectionMatrixL, filePathProjectionMatrixR);
}

























#include <sys\stat.h>
#include <sys\types.h>
#include <direct.h>



struct Mesh
{
	std::vector<cv::Point3f>		vertexCoords;
	std::vector<std::vector<int>>	facetVertexIndsList;
	std::vector<cv::Point2f>		textureCoords;
	cv::Mat							textureImg;
	std::string						texturePath;
	void SaveToFolder(std::string folderPath) {
		struct stat info;
		if (stat(folderPath.c_str(), &info) != 0) {
			// not exist
			_mkdir(folderPath.c_str());
		}

		SaveVectorPoint3f(folderPath + "/vertexCoords.txt", vertexCoords);
		SaveVectorPoint2f(folderPath + "/textureCoords.txt", textureCoords);
		SaveVectorVectorInt(folderPath + "/facetVertexIndsList.txt", facetVertexIndsList);
		cv::imwrite(folderPath + "/textureImg.png", textureImg);
		texturePath = folderPath + "/textureImg.png";
	}

	void LoadFromFolder(std::string folderPath)
	{
		struct stat info;
		if (stat(folderPath.c_str(), &info) != 0 
			|| !(info.st_mode & S_IFDIR)) {
			// not exist
			std::cerr << "folder " << folderPath << " not exitst, mesh NOT loaded!\n";
			return;
		}
		vertexCoords		= LoadVectorPoint3f(folderPath + "/vertexCoords.txt");
		textureCoords		= LoadVectorPoint2f(folderPath + "/textureCoords.txt");
		facetVertexIndsList = LoadVectorVectorInt(folderPath + "/facetVertexIndsList.txt");
		textureImg			= cv::imread(folderPath + "/textureImg.png");
		texturePath = folderPath + "/textureImg.png";
	}
};

struct RenderData
{
	GLuint texture[2];
	Mesh mesh1, mesh2;
	cv::Mat K1, K2, R1, R2, C1, C2, t1, t2;
	cv::Mat viewingPos, viewingDirection;
	double alpha;
	cv::Mat renderedImgL, renderedImgR, renderedImgBlended;
};
RenderData gRenderData;



static void RegisterViewingTrack(std::vector<cv::Mat> &projMats, std::vector<int> &orderedCamIdList)
{
	int numCameras = projMats.size();
	std::vector<cv::Mat> K(numCameras), 
		R(numCameras),
		C(numCameras), 
		t(numCameras);

	std::vector<cv::Point3f> cameraCenters(numCameras);
	for (int i = 0; i < numCameras; i++) {
		cv::decomposeProjectionMatrix(projMats[i], K[i], R[i], C[i]);
		C[i] = C[i](cv::Rect(0, 0, 1, 3)) / C[i].at<double>(3, 0);
		cameraCenters[i] = cv::Point3f(
			C[i].at<double>(0, 0),
			C[i].at<double>(1, 0),
			C[i].at<double>(2, 0));
		//std::cout << "C" << i << " = " << cameraCenters[i] << "\n";
		//std::cout << "K" << i << " = " << K[i] << "\n";
	}




	//////////////////////////////////////////////////////////////////////////////////////
	// determine the three axis of the ellipse by PCA
	//////////////////////////////////////////////////////////////////////////////////////
	cv::Mat A(3, cameraCenters.size(), CV_64FC1);
	for (int i = 0; i < cameraCenters.size(); i++) {
		A.at<double>(0, i) = cameraCenters[i].x;
		A.at<double>(1, i) = cameraCenters[i].y;
		A.at<double>(2, i) = cameraCenters[i].z;
	}

	cv::Mat covar, mean;
	cv::calcCovarMatrix(A, covar, mean, CV_COVAR_NORMAL | CV_COVAR_COLS);
	std::cout << "covar = " << covar << "\n";
	std::cout << "mean = " << mean << "\n";

	cv::Mat singularVals, u, vt;
	cv::SVD::compute(covar, singularVals, u, vt, cv::SVD::FULL_UV);
	std::cout << "u = " << u << "\n";
	std::cout << "vt = " << vt << "\n";
	std::cout << "lambdas = " << singularVals << "\n";

	cv::Mat v1 = vt.row(0).t();		// v1 is the long axis
	cv::Mat v2 = vt.row(1).t();		// v2 is the short axis
	cv::Mat v3 = vt.row(2).t();		// v3 is the normal of the common plane
	std::cout << "v1 = " << v1 << "\n";
	std::cout << "v2 = " << v2 << "\n";
	std::cout << "v3 = " << v3 << "\n\n\n";


	//for (int i = 0; i < cameraCenters.size(); i++) {
	//	std::cout << cameraCenters[i] << "\n";
	//}


	//////////////////////////////////////////////////////////////////////////////////////
	// determine the ordering of the cameras by shooting 360 rays from the center
	//////////////////////////////////////////////////////////////////////////////////////
	// Project camera centers to their common planes.
	std::vector<cv::Point3f> projCamCanters(cameraCenters.size());
	cv::Mat mu = mean, n = v3;
	for (int i = 0; i < cameraCenters.size(); i++) {
		cv::Mat x0(3, 1, CV_64FC1);
		x0.at<double>(0, 0) = cameraCenters[i].x;
		x0.at<double>(1, 0) = cameraCenters[i].y;
		x0.at<double>(2, 0) = cameraCenters[i].z;
		cv::Mat lambda = mu.t() * n - x0.t() * n;
		cv::Mat x = x0 + lambda.at<double>(0, 0) * n;
		projCamCanters[i] = cv::Point3f(x.at<double>(0, 0), x.at<double>(1, 0), x.at<double>(2, 0));
	}


	// Determine camera ordering
	std::vector<cv::Point3f> rotPoints;
	for (float theta = 0.f; theta <= 360.f; theta += 1.f) {
		// Rotatioin against arbitrary axis
		// http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/
		double a = mean.at<double>(0, 0);
		double b = mean.at<double>(1, 0);
		double c = mean.at<double>(2, 0);
		double u = v3.at<double>(0, 0);
		double v = v3.at<double>(1, 0);
		double w = v3.at<double>(2, 0);
		double x = cv::Mat(mean + v1).at<double>(0, 0);
		double y = cv::Mat(mean + v1).at<double>(1, 0);
		double z = cv::Mat(mean + v1).at<double>(2, 0);

		double cos0 = std::cos(theta / 180.f * M_PI);
		double sin0 = std::sin(theta / 180.f * M_PI);
		double dotprod = u*x + v*y + w*z;
		double newx = (a*(v*v + w*w) - u*(b*v + c*w - dotprod))*(1 - cos0)
			+ x*cos0 + (-c*v + b*w - w*y + v*z)*sin0;
		double newy = (b*(u*u + w*w) - v*(a*u + c*w - dotprod))*(1 - cos0)
			+ y*cos0 + (+c*u - a*w + w*x - u*z)*sin0;
		double newz = (c*(u*u + v*v) - w*(a*u + b*v - dotprod))*(1 - cos0)
			+ z*cos0 + (-b*u + a*v - v*x + u*y)*sin0;
		
		rotPoints.push_back(cv::Point3f(newx, newy, newz));
	}


	// cameras centers closest (in normalized distance) to the current ray owns the ray
	//std::vector<int> orderedCamIdList;
	for (int i = 0; i < rotPoints.size(); i++) {
		double bestDist = DBL_MAX;
		int bestCamId = 0;

		for (int j = 0; j < projCamCanters.size(); j++) {
			cv::Point3f mu(
				mean.at<double>(0, 0),
				mean.at<double>(1, 0),
				mean.at<double>(2, 0));
			cv::Point3f r = rotPoints[i] - mu;
			cv::Point3f x = projCamCanters[j];
			double lambda = r.dot(x) - r.dot(mu);

			if (lambda > 0) {
				cv::Point3f y = mu + lambda * r;	// perpendicular intersection
				double dist = cv::norm(x - y);
				dist /= lambda;		// normalized for fiarness in comparison
				if (dist < bestDist) {
					bestDist = dist;
					bestCamId = j;
				}
			}
		}

		orderedCamIdList.push_back(bestCamId);
	}


	// unique the 360 vector
	std::vector<int>::iterator itEnd = std::unique(orderedCamIdList.begin(), orderedCamIdList.end());
	orderedCamIdList.erase(itEnd, orderedCamIdList.end());
	if (orderedCamIdList[0] == orderedCamIdList[orderedCamIdList.size() - 1]) {
		orderedCamIdList.erase(orderedCamIdList.begin());
	}

	
	// erase cameras when adjacent angles are too small
	int i = 0;
	while (i < orderedCamIdList.size()) {
		int id1 = orderedCamIdList[i];
		int id2 = orderedCamIdList[(i + 1) % orderedCamIdList.size()];
		cv::Point3f mu(mean.at<double>(0, 0), mean.at<double>(1, 0), mean.at<double>(2, 0));
		cv::Point3f r1 = projCamCanters[id1] - mu;
		cv::Point3f r2 = projCamCanters[id2] - mu;
		float cosTheta = r1.dot(r2) / (cv::norm(r1) * cv::norm(r2));
		float angle = 180 * std::acos(cosTheta);
		if (angle < 9.f) {	// then merge
			orderedCamIdList.erase(orderedCamIdList.begin() + ((i + 1) % orderedCamIdList.size()));
			// don't increase i
		}
		else {
			i++;
		}
	}
	

	// output new ordering
	std::cout << "size = " << orderedCamIdList.size() << "\n";
	for (int i = 0; i < orderedCamIdList.size(); i++) {
		std::cout << orderedCamIdList[i] << "\n";
	}
	



	//////////////////////////////////////////////////////////////////////////////////////
	// Output axis and camera centers
	//////////////////////////////////////////////////////////////////////////////////////
	std::vector<cv::Point3f> axis;
	std::vector<cv::Vec3b> colors;
	// axis
	for (float s = 0.f; s <= 2.f; s += 0.01f) {
		cv::Mat x = mean + s * v1;
		cv::Mat y = mean + s * v2;
		cv::Mat z = mean + s * v3;
		axis.push_back(cv::Point3f(x.at<double>(0, 0), x.at<double>(1, 0), x.at<double>(2, 0)));
		axis.push_back(cv::Point3f(y.at<double>(0, 0), y.at<double>(1, 0), y.at<double>(2, 0)));
		axis.push_back(cv::Point3f(z.at<double>(0, 0), z.at<double>(1, 0), z.at<double>(2, 0)));
		colors.push_back(cv::Vec3b(255, 0, 0));
		colors.push_back(cv::Vec3b(0, 255, 0));
		colors.push_back(cv::Vec3b(0, 0, 255));
	}
	// camera centers
	for (int i = 0; i < orderedCamIdList.size(); i++) {
		int id = orderedCamIdList[i];
		axis.push_back(projCamCanters[id]);
		if (i % 2 == 0) {
			colors.push_back(cv::Vec3b(255, 0, 0));
		}
		else {
			colors.push_back(cv::Vec3b(0, 255, 0));
		}
	}
	// circle
	for (int i = 0; i < rotPoints.size(); i++) {
		axis.push_back(rotPoints[i]);
		colors.push_back(cv::Vec3b(255, 255, 0));
	}

	void SavePointCloudToPly(std::string filePathPly, std::vector<cv::Point3f> &points, std::vector<cv::Vec3b> &colors = std::vector<cv::Vec3b>());
	SavePointCloudToPly("d:/ellipseAxises.ply", axis, colors);


	exit(-1);
}

void RenderInit()
{
	glClearColor(1, 0, 0, 0);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	cv::cvtColor(gRenderData.mesh1.textureImg, gRenderData.mesh1.textureImg, CV_BGR2RGB);
	cv::cvtColor(gRenderData.mesh2.textureImg, gRenderData.mesh2.textureImg, CV_BGR2RGB);
	cv::Size imgSize = gRenderData.mesh1.textureImg.size();
	gRenderData.renderedImgL		= cv::Mat::zeros(imgSize, CV_8UC3);
	gRenderData.renderedImgR		= cv::Mat::zeros(imgSize, CV_8UC3);
	gRenderData.renderedImgBlended	= 128 * cv::Mat::ones(imgSize, CV_8UC3);
	gRenderData.renderedImgBlended.setTo(cv::Vec3b(0, 255, 0));
	//gRenderData.mesh2.textureImg.setTo(cv::Vec3b(0, 255, 0));

	// Create Texture
	glEnable(GL_TEXTURE_2D);
	glGenTextures(2, gRenderData.texture);

	glBindTexture(GL_TEXTURE_2D, gRenderData.texture[0]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); //scale linearly when image bigger than texture
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); //scale linearly when image smalled than texture
	cv::Mat &textureImg = gRenderData.mesh1.textureImg;
	glTexImage2D(GL_TEXTURE_2D, 0, 3, textureImg.cols, textureImg.rows, 0,
		GL_RGB, GL_UNSIGNED_BYTE, textureImg.data);
	glBindTexture(GL_TEXTURE_2D, 0);

	glBindTexture(GL_TEXTURE_2D, gRenderData.texture[1]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); //scale linearly when image bigger than texture
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); //scale linearly when image smalled than texture
	textureImg = gRenderData.mesh2.textureImg;
	glTexImage2D(GL_TEXTURE_2D, 0, 3, textureImg.cols, textureImg.rows, 0,
		GL_RGB, GL_UNSIGNED_BYTE, textureImg.data);
	glBindTexture(GL_TEXTURE_2D, 0);

	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	glShadeModel(GL_FLAT);

	//glEnable(GL_BLEND);
	//glBlendFunc(GL_ZERO, GL_ONE);

}
  
void RenderDisplay(void){

	std::vector<cv::Point3f>		&meshVertexCoordsL	 = gRenderData.mesh1.vertexCoords;
	std::vector<cv::Point2f>		&textureCoordsL		 = gRenderData.mesh1.textureCoords;
	std::vector<std::vector<int>>	&faceVertexIndsListL = gRenderData.mesh1.facetVertexIndsList;

	std::vector<cv::Point3f>		&meshVertexCoordsR	 = gRenderData.mesh2.vertexCoords;
	std::vector<cv::Point2f>		&textureCoordsR		 = gRenderData.mesh2.textureCoords;
	std::vector<std::vector<int>>	&faceVertexIndsListR = gRenderData.mesh2.facetVertexIndsList;

	
	

	int numRows = gRenderData.mesh1.textureImg.rows;
	int numCols = gRenderData.mesh1.textureImg.cols;

#if 1
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0, 0, numCols, numRows);
	glBindTexture(GL_TEXTURE_2D, gRenderData.texture[0]);
	glBegin(GL_TRIANGLES);
	for (int id = 0; id < faceVertexIndsListL.size(); id++) {
		for (int j = 0; j < 3; j++) {
			cv::Point3f pCoord = meshVertexCoordsL[faceVertexIndsListL[id][j]];
			cv::Point2f tCoord = textureCoordsL[faceVertexIndsListL[id][j]];
			glTexCoord2f(tCoord.x, tCoord.y);
			//std::cout << tCoord << "\n";
			glVertex3f(pCoord.x, pCoord.y, pCoord.z);
			//std::cout << pCoord << "\n\n";
		}
	}
	glEnd();
	//glPixelStorei(GL_PACK_ALIGNMENT, 1);
	//glReadPixels(0, 0, numCols, numRows, GL_RGB, GL_UNSIGNED_BYTE, gRenderData.renderedImgL.data);
	//cv::imshow("L", gRenderData.renderedImgL);
	//cv::waitKey(0);

	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(numCols, 0, numCols, numRows);
	glBindTexture(GL_TEXTURE_2D, gRenderData.texture[1]);
	glBegin(GL_TRIANGLES);
	for (int id = 0; id < faceVertexIndsListR.size(); id++) {
		for (int j = 0; j < 3; j++) {
			cv::Point3f pCoord = meshVertexCoordsR[faceVertexIndsListR[id][j]];
			cv::Point2f tCoord = textureCoordsR[faceVertexIndsListR[id][j]];
			glTexCoord2f(tCoord.x, tCoord.y);
			glVertex3f(pCoord.x, pCoord.y, pCoord.z);
		}
	}
	glEnd();
	//glPixelStorei(GL_PACK_ALIGNMENT, 1);
	//glReadPixels(0, 0, numCols, numRows, GL_RGB, GL_UNSIGNED_BYTE, gRenderData.renderedImgR.data);
	//cv::imshow("R", gRenderData.renderedImgR);
	//cv::waitKey(0);

	//gRenderData.renderedImgBlended = 0.5 * gRenderData.renderedImgL + 0.5 * gRenderData.renderedImgR;
	//cv::imshow("M", gRenderData.renderedImgBlended);
	//cv::waitKey(0);

	//glPixelStorei(GL_PACK_ALIGNMENT, 1);
	//glDrawPixels(numCols, numRows, GL_RGB, GL_UNSIGNED_BYTE, gRenderData.renderedImgBlended.data);
#endif
	//glPixelStorei(GL_PACK_ALIGNMENT, 1);
	//glDrawPixels(numCols, numRows, GL_RGB, GL_UNSIGNED_BYTE, gRenderData.mesh1.textureImg.data);



	
	glutSwapBuffers();
	
}

void SetIntrinsicAndPose()
{


	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	float f1 = (gRenderData.K1.at<double>(0, 0) + gRenderData.K1.at<double>(1, 1)) / 2.0;
	float f2 = (gRenderData.K2.at<double>(0, 0) + gRenderData.K2.at<double>(1, 1)) / 2.0;
	float f = (1.f - gRenderData.alpha) * f1 + (gRenderData.alpha) * f2;
	std::cout << "f1 = " << f1 << "\nf2 = " << f2 << "\nf = " << f << "\n";

	float cx1 = gRenderData.K1.at<double>(0, 2);
	float cy1 = gRenderData.K1.at<double>(1, 2);
	float cx2 = gRenderData.K2.at<double>(0, 2);
	float cy2 = gRenderData.K2.at<double>(1, 2);
	float alpha = gRenderData.alpha;
	float cx = (1.f - alpha) * cx1 + alpha * cx2;
	float cy = (1.f - alpha) * cy1 + alpha * cy2;
	std::cout << "alpha = " << alpha << "\n";
	std::cout << "(cx1, cy1) = " << cv::Point2f(cx1, cy1) << "\n";
	std::cout << "(cx2, cy2) = " << cv::Point2f(cx2, cy2) << "\n";
	std::cout << "(cx, cy) = " << cv::Point2f(cx, cy) << "\n";



	int numRows = gRenderData.mesh1.textureImg.rows;
	int numCols = gRenderData.mesh1.textureImg.cols;
	const float Znear = 0.1;	//FIXME
	const float Zfar = 70.0;	//FIXME
	glFrustum(
		(0 - cx) * Znear / f, (numCols - cx) * Znear / f,
		(0 - cy) * Znear / f, (numRows - cy) * Znear / f,
		Znear, Zfar);
	//gluPerspective(60, (double)w / h, 0.1, 70);



	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();


	cv::Mat &C1 = gRenderData.C1;
	cv::Mat &C2 = gRenderData.C2;
	cv::Mat &R1 = gRenderData.R1;
	cv::Mat &R2 = gRenderData.R2;
	cv::Mat &t1 = gRenderData.t1;
	cv::Mat &t2 = gRenderData.t2;

	cv::Mat viewingPosition = (1.f - alpha) * C1 + alpha * C2;
	cv::Mat _001_3x1 = cv::Mat::zeros(3, 1, CV_64FC1);
	_001_3x1.at<double>(2, 0) = 1.0;
	cv::Mat viewingDirection1 = R1.t() * (_001_3x1 - t1) - C1;	// viewing direction in world frame
	cv::Mat viewingDirection2 = R2.t() * (_001_3x1 - t2) - C2;	// viewing direction in world frame
	cv::Mat viewingDirection = (1.f - alpha) * viewingDirection1 + alpha * viewingDirection2;

	cv::Mat eyePos = R1 * viewingPosition + t1;
	cv::Mat centerPos = R1 * (viewingPosition + viewingDirection) + t1;

	//cv::Mat eyePos = gRenderData.viewingPos;
	//cv::Mat centerPos = gRenderData.viewingPos + gRenderData.viewingDirection;
	//eyePos = gRenderData.R1 * eyePos + gRenderData.t1;
	//centerPos = gRenderData.R1 * centerPos + gRenderData.t1;

	std::cout << "R1 = " << gRenderData.R1 << "\n";
	std::cout << "C1 = " << gRenderData.C1 << "\n";
	std::cout << "t1 = " << gRenderData.t1 << "\n";
	std::cout << "eyePos = " << eyePos << "\n";
	std::cout << "centerPos = " << centerPos << "\n";

	gluLookAt(
		eyePos.at<double>(0, 0),
		eyePos.at<double>(1, 0),
		eyePos.at<double>(2, 0),
		centerPos.at<double>(0, 0),
		centerPos.at<double>(1, 0),
		centerPos.at<double>(2, 0),
		0, -1, 0);	//TODO: y direction can also be blended
}

void RenderReshape(int w, int h)
{
	cv::Mat &textureImg = gRenderData.mesh1.textureImg;
	if (!textureImg.empty()) {
		w = textureImg.cols;
		h = textureImg.rows;
	}
	printf("myReshape w = %d,  h = %d\n", w, h);
	int numRows = textureImg.rows, numCols = textureImg.cols;



	//glViewport(0, 0, w, h);

	SetIntrinsicAndPose();
}

void RenderOnAsciiKeys(unsigned char key, int x, int y)
{
	switch (key) {
	case 27: // ¡°esc¡± on keyboard
		exit(0);
		break;

	default: // ¡°a¡± on keyboard
		break;
	}
}

void RenderOnSpecialKeys(int key, int xx, int yy)
{

	float fraction = 0.1f;

	switch (key) {
	case GLUT_KEY_LEFT:
		gRenderData.alpha -= 0.02;
		gRenderData.alpha = std::max(0.0, std::min(1.0, gRenderData.alpha));
		SetIntrinsicAndPose();
		RenderDisplay();
		break;
	case GLUT_KEY_RIGHT:
		gRenderData.alpha += 0.02;
		gRenderData.alpha = std::max(0.0, std::min(1.0, gRenderData.alpha));
		SetIntrinsicAndPose();
		RenderDisplay();
		break;
	}
}

static void RenderIntermediateViews(Mesh &meshL, Mesh &meshR, cv::Mat &P1, cv::Mat &P2)
{
	cv::Mat K1, K2, R1, R2, C1, C2, t1, t2;
	cv::decomposeProjectionMatrix(P1, K1, R1, C1);
	cv::decomposeProjectionMatrix(P2, K2, R2, C2);
	C1 = C1(cv::Rect(0, 0, 1, 3)) / C1.at<double>(3, 0);
	C2 = C2(cv::Rect(0, 0, 1, 3)) / C2.at<double>(3, 0);
	t1 = -R1 * C1;
	t2 = -R2 * C2;

	// prepare rendering data for OpenGL
	gRenderData.mesh1 = meshL;
	gRenderData.mesh2 = meshR;
	gRenderData.K1 = K1;
	gRenderData.R1 = R1;
	gRenderData.C1 = C1;
	gRenderData.t1 = t1;
	gRenderData.K2 = K2;
	gRenderData.R2 = R2;
	gRenderData.C2 = C2;
	gRenderData.t2 = t2;


	gRenderData.alpha = 0.0;

	// start OpenGL
	int argc	= 1;
	char *dummy = "sdsfsdf";
	char **argv = &dummy;
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	cv::Mat &textureImg = gRenderData.mesh1.textureImg;
	glutInitWindowSize(2 * textureImg.cols, textureImg.rows);
	glutCreateWindow("Texture Mapping - Programming Techniques");

	RenderInit();
	glutReshapeFunc(RenderReshape);
	glutDisplayFunc(RenderDisplay);
	glutKeyboardFunc(RenderOnAsciiKeys);
	glutSpecialFunc(RenderOnSpecialKeys);
	glutMainLoop();

}

void ConvertFacetSetToMesh(int sign, int numRows, int numCols,
	float focalLen, float baselineLen, float minDisp, float maxDisp,
	std::vector<cv::Point2f> &vertexCoords, std::vector<std::vector<int>> &triVertexInds, std::vector<SlantedPlane> &slantedPlanes,
	std::vector<cv::Point3f> &meshVertexCoords, std::vector<std::vector<int>> &facetVetexIndsList,
	std::vector<cv::Point3f> &meshVertexCoords_xyd)
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
				void ClusterDisparitiesOtsu(std::vector<float> &dispVals);
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
		//meshVertexCoords[i].z = std::max(4.f, std::min(79.f, meshVertexCoords[i].z));
		meshVertexCoords[i].z = std::max(minDisp, std::min(maxDisp, meshVertexCoords[i].z));
	}
	////////////////////////////
	meshVertexCoords_xyd = meshVertexCoords;
	////////////////////////////
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
	std::string filePathPly;
	std::string filePathSplittingMap;
	if (sign == -1) {
		filePathPly = "d:/meshL.ply";
		filePathSplittingMap = "d:/splittingL.png";
		filePathTextureImage = "d:/pngInputForYamaguchiL.png";
	}
	else {
		filePathPly = "d:/meshR.ply";
		filePathSplittingMap = "d:/splittingR.png";
		filePathTextureImage = "d:/pngInputForYamaguchiR.png";
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

void TriangulateAndLiftToMesh(cv::Mat &imL, cv::Mat &imR, cv::Mat &dispL, cv::Mat &dispR,
	double focalLen, double baselineLen, int minDisp, int numDisps,
	cv::Mat &Q, cv::Mat &R1, cv::Mat &R2, cv::Mat &P1, cv::Mat &P2,
	Mesh &mesh1, Mesh &mesh2)
{
	void ConstructNeighboringTriangleGraph(int numRows, int numCols, std::vector<cv::Point2f> &vertexCoords,
		std::vector<std::vector<int>> &triVertexInds, std::vector<cv::Point2f> &baryCenters,
		std::vector<std::vector<int>> &nbIndices);
	void DeterminePixelOwnership(int numRows, int numCols, std::vector<cv::Point2f> &vertexCoords,
		std::vector<std::vector<int>> &triVertexInds, std::vector<std::vector<cv::Point2i>> &triPixelLists);
	cv::Mat TriangleLabelToDisparityMap(int numRows, int numCols, std::vector<SlantedPlane> &slantedPlanes,
		std::vector<std::vector<cv::Point2i>> &triPixelLists);


	int numRows = imL.rows, numCols = imL.cols;
	int maxDisp = numDisps - 1;

	std::vector<cv::Point2f> vertexCoordsL, vertexCoordsR;
	std::vector<std::vector<int>> triVertexIndsL, triVertexIndsR;
	//#define SLIC_TRIANGULATION
#ifdef SLIC_TRIANGULATION
	Triangulate2DImage(imL, vertexCoordsL, triVertexIndsL);
	Triangulate2DImage(imR, vertexCoordsR, triVertexIndsR);
#else
	void ImageDomainTessellation(cv::Mat &img, std::vector<cv::Point2f> &vertexCoordList,
		std::vector<std::vector<int>> &triVertexIndsList);
	ImageDomainTessellation(imL, vertexCoordsL, triVertexIndsL);
	ImageDomainTessellation(imR, vertexCoordsR, triVertexIndsR);
#endif

	LOGLINE();
	cv::Mat DrawTriangleImage(int numRows, int numCols, std::vector<cv::Point2f> &vertexCoords, std::vector<std::vector<int>> &triVertexInds, cv::Mat &textureImg);
	cv::Mat triImgL = DrawTriangleImage(numRows, numCols, vertexCoordsL, triVertexIndsL, imL);
	cv::Mat triImgR = DrawTriangleImage(numRows, numCols, vertexCoordsR, triVertexIndsR, imR);
	cv::Mat triImgs;
	cv::hconcat(triImgL, triImgR, triImgs);
	cv::imwrite("d:/triangulation.png", triImgs);
	//cv::imshow("triangulation", triImgs);
	//cv::waitKey(0);

	LOGLINE();
	std::vector<cv::Point2f> baryCentersL, baryCentersR;
	std::vector<std::vector<int>> nbIndicesL, nbIndicesR;
	ConstructNeighboringTriangleGraph(numRows, numCols, vertexCoordsL, triVertexIndsL, baryCentersL, nbIndicesL);
	ConstructNeighboringTriangleGraph(numRows, numCols, vertexCoordsR, triVertexIndsR, baryCentersR, nbIndicesR);

	LOGLINE();
	std::vector<std::vector<cv::Point2i>> triPixelListsL, triPixelListsR;
	DeterminePixelOwnership(numRows, numCols, vertexCoordsL, triVertexIndsL, triPixelListsL);
	DeterminePixelOwnership(numRows, numCols, vertexCoordsR, triVertexIndsR, triPixelListsR);


	LOGLINE();
	int numTrianglesL = baryCentersL.size();
	int numTrianglesR = baryCentersR.size();
	std::vector<SlantedPlane> slantedPlanesL(numTrianglesL);
	std::vector<SlantedPlane> slantedPlanesR(numTrianglesR);


	LOGLINE();
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


	LOGLINE();
	dispL = TriangleLabelToDisparityMap(numRows, numCols, slantedPlanesL, triPixelListsL);
	dispR = TriangleLabelToDisparityMap(numRows, numCols, slantedPlanesR, triPixelListsR);
	cv::Mat dispImgL, dispImgR, dispImg;
	dispL.convertTo(dispImgL, CV_8UC1, 255.f / maxDisp);
	dispR.convertTo(dispImgR, CV_8UC1, 255.f / maxDisp);
	cv::hconcat(dispImgL, dispImgR, dispImg);
	cv::imwrite("d:/meshDispL.png", dispL);
	cv::imwrite("d:/meshDispR.png", dispR);

	cv::Mat XYZL, XYZR;
	dispL = cv::max(20.f, cv::min(144.f, dispL));
	dispR = cv::max(20.f, cv::min(144.f, dispR));
	//cv::reprojectImageTo3D(dispL, XYZL, Q, false);
	//cv::reprojectImageTo3D(dispR, XYZR, Q, false);
	XYZL = disp2XYZ(dispL, P1, P2);
	XYZR = disp2XYZ(dispR, P1, P2);
	void SavePointCloudToPly(std::string filePathPly, cv::Mat &XYZ, cv::Mat &img, cv::Mat &validPixels = cv::Mat());
	SavePointCloudToPly("d:/trianglePointCloudL.ply", XYZL, imL);
	SavePointCloudToPly("d:/trianglePointCloudR.ply", XYZR, imR);
	std::cout << "point clouds saved.\n";




	std::vector<cv::Point3f> meshVertexCoordsL, meshVertexCoordsR, meshVertexCoords_xydL, meshVertexCoords_xydR;
	std::vector<std::vector<int>> facetVertexIndsListL, facetVertexIndsListR;
	printf("Build water tight mesh left  .......\n");
	ConvertFacetSetToMesh(-1, numRows, numCols, focalLen, baselineLen, minDisp, maxDisp, 
		vertexCoordsL, triVertexIndsL, slantedPlanesL, 
		meshVertexCoordsL, facetVertexIndsListL, meshVertexCoords_xydL);
	printf("Build water tight mesh right .......\n");
	ConvertFacetSetToMesh(+1, numRows, numCols, focalLen, baselineLen, minDisp, maxDisp,
		vertexCoordsR, triVertexIndsR, slantedPlanesR, 
		meshVertexCoordsR, facetVertexIndsListR, meshVertexCoords_xydR);



	std::vector<cv::Point2f> textureCoordsL(meshVertexCoords_xydL.size());
	std::vector<cv::Point2f> textureCoordsR(meshVertexCoords_xydR.size());
	for (int i = 0; i < meshVertexCoords_xydL.size(); i++) {
		cv::Point3f p = meshVertexCoords_xydL[i];
		textureCoordsL[i] = cv::Point2f(p.x / numCols, p.y / numRows);
	}
	for (int i = 0; i < meshVertexCoords_xydR.size(); i++) {
		cv::Point3f p = meshVertexCoords_xydR[i];
		textureCoordsR[i] = cv::Point2f(p.x / numCols, p.y / numRows);
	}

	for (int i = 0; i < meshVertexCoordsL.size(); i++) {
		cv::Point3f p = meshVertexCoords_xydL[i];
		double data[] = { p.x, p.y, p.z, 1};
		cv::Mat A(4, 1, CV_64FC1, data);
		cv::Mat B = Q * A;
		B = B(cv::Rect(0, 0, 1, 3)).clone() / B.at<double>(3, 0);
		meshVertexCoordsL[i] = cv::Point3f(B.at<double>(0, 0), B.at<double>(1, 0), B.at<double>(2, 0));
	}
	for (int i = 0; i < meshVertexCoordsR.size(); i++) {
		cv::Point3f p = meshVertexCoords_xydR[i];
		double data[] = { p.x + p.z, p.y, p.z, 1 };		// use cam1's rectified coordinate frame
		cv::Mat A(4, 1, CV_64FC1, data);
		cv::Mat B = Q * A;
		B = B(cv::Rect(0, 0, 1, 3)).clone() / B.at<double>(3, 0);
		meshVertexCoordsR[i] = cv::Point3f(B.at<double>(0, 0), B.at<double>(1, 0), B.at<double>(2, 0));
	}


	void SaveMeshToPly(std::string plyFilePath,
		std::vector<cv::Point3f> &meshVertexCoordsXYZ, std::vector<std::vector<int>> &facetVetexIndsList,
		std::vector<cv::Point2f> &textureCoords,
		std::string textureFilePath, bool showInstantly = false);
	SaveMeshToPly("d:/meshNoRectifiedL.ply", meshVertexCoordsL, facetVertexIndsListL,
		textureCoordsL, "d:/pngInputForYamaguchiL.png");
	SaveMeshToPly("d:/meshNoRectifiedR.ply", meshVertexCoordsR, facetVertexIndsListR,
		textureCoordsR, "d:/pngInputForYamaguchiR.png");
	std::cout << "non-rectifid mesh saved.\n";
	//exit(-1);

	ASSERT(textureCoordsL.size() == meshVertexCoordsL.size());
	mesh1.textureCoords = textureCoordsL;
	mesh1.vertexCoords = meshVertexCoordsL;
	mesh1.facetVertexIndsList = facetVertexIndsListL;
	mesh1.textureImg = imL.clone();
	mesh1.texturePath = "d:/pngInputForYamaguchiL.png";

	ASSERT(textureCoordsR.size() == meshVertexCoordsR.size());
	mesh2.textureCoords = textureCoordsR;
	mesh2.vertexCoords = meshVertexCoordsR;
	mesh2.facetVertexIndsList = facetVertexIndsListR;
	mesh2.textureImg = imR.clone();
	mesh1.texturePath = "d:/pngInputForYamaguchiR.png";
}

static void ReconstructMeshes(cv::Mat &imL, cv::Mat &imR, cv::Mat &P1, cv::Mat &P2, Mesh &meshL, Mesh &meshR)
{

	// Compute relative pose to call stereoRectify.
	cv::Mat K1, R1, t1, C1;
	cv::Mat K2, R2, t2, C2;
	cv::Mat R12, t12;
	cv::decomposeProjectionMatrix(P1, K1, R1, C1);
	cv::decomposeProjectionMatrix(P2, K2, R2, C2);
	C1 = C1(cv::Rect(0, 0, 1, 3)).clone() / C1.at<double>(3, 0);
	C2 = C2(cv::Rect(0, 0, 1, 3)).clone() / C2.at<double>(3, 0);
	t1 = -R1 * C1;
	t2 = -R2 * C2;
	R12 = R2 * R1.t();
	t12 = R2 * (C1 - C2);



	// Perform Rectification.
	cv::Mat		distortCoeffs1 = cv::Mat::zeros(1, 5, CV_64FC1);
	cv::Mat		distortCoeffs2 = cv::Mat::zeros(1, 5, CV_64FC1);
	cv::Mat		Q, newP1, newP2, newR1, newR2;
	cv::Size	imgSize = imL.size();
	cv::Rect	roi1, roi2;
	int			flagsAlignPrincipalPoints = CV_CALIB_ZERO_DISPARITY;
	cv::stereoRectify(K1, distortCoeffs1, K2, distortCoeffs2, imgSize,
		R12, t12,
		newR1, newR2,
		newP1,	// P1 is the projection matrix of the 1st camera in the new rectified coordinates, 
				// the upper-left 3x3 block of which is just the new K1.
		newP2,	// P2 is the projection matrix of the 2nd camera in the new rectified coordinates, 
				// the upper-left 3x3 block of which is just the new K2.
		Q,
		flagsAlignPrincipalPoints,
		-1,		// detect cropped region automatically
		imgSize, &roi1, &roi2);




	//VerifyQ(newR1, newR2, newP1, newP2, Q, R12, t12);
	//exit(-1);
	//return;

	cv::Mat map1x, map1y, map2x, map2y;
	cv::initUndistortRectifyMap(K1, distortCoeffs1, newR1, newP1, imgSize, CV_32FC1, map1x, map1y);
	cv::initUndistortRectifyMap(K2, distortCoeffs1, newR2, newP2, imgSize, CV_32FC1, map2x, map2y);

	cv::Mat imgRectifiedL, imgRectifiedR;
	cv::remap(imL, imgRectifiedL, map1x, map1y, cv::INTER_CUBIC);
	cv::remap(imR, imgRectifiedR, map2x, map2y, cv::INTER_CUBIC);

	cv::Mat canvasColor;
	cv::hconcat(imgRectifiedL, imgRectifiedR, canvasColor);
	//EnhancedImShow("Rectified Pair", canvasColor, imL.size());



	// SGM stereo from Yamaguchi
	int numDisps = 144;
	numDisps += 15 - (numDisps - 1) % 16;
	int maxDisp = numDisps - 1;
	extern int SGMSTEREO_DEFAULT_DISPARITY_TOTAL;
	SGMSTEREO_DEFAULT_DISPARITY_TOTAL = numDisps;

	int numRows = imL.rows, numCols = imL.cols;
	cv::Mat dispL(numRows, numCols, CV_32FC1);
	cv::Mat dispR(numRows, numCols, CV_32FC1);
	ASSERT(dispL.isContinuous());
	ASSERT(dispR.isContinuous());

	std::string pathL = "d:/pngInputForYamaguchiL.png";
	std::string pathR = "d:/pngInputForYamaguchiR.png";
	cv::imwrite(pathL, imgRectifiedL);
	cv::imwrite(pathR, imgRectifiedR);
	png::image<png::rgb_pixel> pngImageL(pathL);
	png::image<png::rgb_pixel> pngImageR(pathR);

	SGMStereo sgm;
	sgm.compute(pngImageL, pngImageR, (float*)dispL.data, (float*)dispR.data);

	cv::Mat SetInvalidDisparityToZeros(cv::Mat &dispL, cv::Mat &validPixelMapL);
	cv::Mat validPixelsL = CrossCheck(dispL, dispR, -1);
	cv::Mat validPixelsR = CrossCheck(dispR, dispL, +1);
	dispL = SetInvalidDisparityToZeros(dispL, validPixelsL);
	dispR = SetInvalidDisparityToZeros(dispR, validPixelsR);


#if 0
	cv::Mat canvasDisp;
	cv::hconcat(dispL, dispR, canvasDisp);
	canvasDisp.convertTo(canvasDisp, CV_8UC1);
	EnhancedImShow("disparities", canvasDisp, dispL.size());

	cv::cvtColor(canvasDisp, canvasDisp, CV_GRAY2BGR);
	cv::vconcat(canvasColor, canvasDisp, canvasDisp);
	cv::imwrite("d:/color+disp.png", canvasDisp);
#endif


	// Output point clouds
	cv::compare(dispL, 20.f, validPixelsL, cv::CMP_GE);	// only show ponits with disp > 20
	cv::compare(dispR, 20.f, validPixelsR, cv::CMP_GE);	// only show ponits with disp > 20
	cv::Mat XYZL, XYZR;
	cv::reprojectImageTo3D(dispL, XYZL, Q, false);		// this will cause the wall artifact, since all invalid pixels have the same depth
	cv::reprojectImageTo3D(dispR, XYZR, Q, false);		// this will cause the wall artifact, since all invalid pixels have the same depth


	void SavePointCloudToPly(std::string filePathPly, cv::Mat &XYZ, cv::Mat &img, cv::Mat &validPixels = cv::Mat());
	SavePointCloudToPly("d:/pointCloudL.ply", XYZL, imgRectifiedL, validPixelsL);
	SavePointCloudToPly("d:/pointCloudR.ply", XYZR, imgRectifiedR, validPixelsR);
	std::cout << "Point clouds saved.\n";


	double focalLen = newP1.at<double>(0, 0);	// newP1 is the projection matrix in the rectified frame
	cv::Mat dist = (C2 - C1).t() * (C2 - C1);
	double baselineLen = std::sqrt(dist.at<double>(0, 0));
	TriangulateAndLiftToMesh(imgRectifiedL, imgRectifiedR, dispL, dispR, 
		focalLen, baselineLen, 
		20.f, numDisps, 
		Q, R1, R2, newP1, newP2, 
		meshL, meshR);

}

static void RunAutoPlayRenderingSystem(std::string filePathCmvsOutput, int numCameras)
{
	// Step 1 - Reading all cameras' images and projection matrices
	std::vector<cv::Mat> imgs(numCameras);
	std::vector<cv::Mat> projMats(numCameras);

	for (int i = 2; i <= 3; i++) {
		char buf[1024];
		sprintf(buf, "%s/visualize/%08d.jpg", filePathCmvsOutput.c_str(), i);	// FIXME: may also be .png
		imgs[i] = cv::imread(buf);
	}
	for (int i = 0; i < numCameras; i++) {
		char buf[1024];
		sprintf(buf, "%s/txt/%08d.txt", filePathCmvsOutput.c_str(), i);
		projMats[i] = ReadProjMatrixFromTxt(buf);
	}

	const int maxRowSize = 500;
	const int maxColSize = 700;
	cv::Mat imL = imgs[3];
	cv::Mat imR = imgs[2];

	float downSampleFactor = 1.f;
	for (downSampleFactor = 1.f; downSampleFactor <= 10.f; downSampleFactor += 1.f) {
		if (imL.rows / downSampleFactor <= maxRowSize && imL.cols / downSampleFactor <= maxColSize) {
			break;
		}
	}
	int oldNumRows = imL.rows, oldNumCols = imL.cols;
	int numRows = oldNumRows / downSampleFactor + 0.5;
	int numCols = oldNumCols / downSampleFactor + 0.5;

	// Compensate the K matrices for the image resizing.
	double ratioX = (double)numCols / oldNumCols;
	double ratioY = (double)numRows / oldNumRows;
	for (int i = 0; i < numCameras; i++) {
		cv::Mat K, R, C, t, Rt;
		cv::decomposeProjectionMatrix(projMats[i], K, R, C);
		K.at<double>(0, 0) *= ratioX;
		K.at<double>(0, 2) *= ratioX;
		K.at<double>(1, 1) *= ratioY;
		K.at<double>(1, 2) *= ratioY;
		C = C(cv::Rect(0, 0, 1, 3)).clone() / C.at<double>(3, 0);
		t = -R * C;
		cv::hconcat(R, t, Rt);
		projMats[i] = K * Rt;
	}

	// Resize images, currently only the two under consideration to save power
	cv::resize(imL, imL, cv::Size(numCols, numRows));
	cv::resize(imR, imR, cv::Size(numCols, numRows));




	// Step 2 - Register a viewing track for rendering, whhich includes computing the 
	// determine the ordering of the cameras, remove redundant (or difficult) cameras.
#if 1
	std::vector<int> orderedCamIdList;
	RegisterViewingTrack(projMats, orderedCamIdList);
#endif



	// Step 3 -
	// for all consecutive image pairs along the track
	//	   build meshes.
	for (int i = 0; i < orderedCamIdList.size(); i++) {

	}



	Mesh meshL, meshR;
	cv::Mat P1 = projMats[3], P2 = projMats[2];
#if 0
	ReconstructMeshes(imL, imR, P1, P2, meshL, meshR);
	meshL.SaveToFolder("d:/meshL_dir");
	meshR.SaveToFolder("d:/meshR_dir");
#else 
	meshL.LoadFromFolder("d:/meshL_dir");
	meshR.LoadFromFolder("d:/meshR_dir");
	void SaveMeshToPly(std::string plyFilePath,
		std::vector<cv::Point3f> &meshVertexCoordsXYZ, std::vector<std::vector<int>> &facetVetexIndsList,
		std::vector<cv::Point2f> &textureCoords,
		std::string textureFilePath, bool showInstantly = false);
	SaveMeshToPly("d:/meshNoRectifiedL.ply", meshL.vertexCoords, meshL.facetVertexIndsList,
		meshL.textureCoords, meshL.texturePath);
	SaveMeshToPly("d:/meshNoRectifiedR.ply", meshR.vertexCoords, meshR.facetVertexIndsList,
		meshR.textureCoords, meshR.texturePath);

	std::cout << meshL.textureImg.size() << "\n";
	std::cout << meshR.textureImg.size() << "\n";
#endif


	// Step 4 -
	// for all consecutive image pairs along the track
	//	   for t in [0, 1] do 
	//		   compute viewing position and direction
	//		   transform cooridnates using OpenGL for rendering (using fixed frstum?)
	RenderIntermediateViews(meshL, meshR, P1, P2);
}

int main(int argc, char **argv)
{
	std::string filePathImageL = "D:/data/Lumia800/CanSequence/CanSequence.nvm.cmvs/00/visualize/00000003.jpg";
	std::string filePathImageR = "D:/data/Lumia800/CanSequence/CanSequence.nvm.cmvs/00/visualize/00000002.jpg";
	std::string filePathProjectionMatrixL = "D:/data/Lumia800/CanSequence/CanSequence.nvm.cmvs/00/txt/00000003.txt";
	std::string filePathProjectionMatrixR = "D:/data/Lumia800/CanSequence/CanSequence.nvm.cmvs/00/txt/00000002.txt";


	//std::string filePathImageL = "D:/data/Lumia800/mvi_shoe/visualize/00000001.png";
	//std::string filePathImageR = "D:/data/Lumia800/mvi_shoe/visualize/00000000.png";
	//std::string filePathProjectionMatrixL = "D:/data/Lumia800/mvi_shoe/txt/00000001.txt";
	//std::string filePathProjectionMatrixR = "D:/data/Lumia800/mvi_shoe/txt/00000000.txt";


	//std::string filePathImageL = "D:/data/Lumia800/buddha/visualize/00000010.jpg";
	//std::string filePathImageR = "D:/data/Lumia800/buddha/visualize/00000015.jpg";
	//std::string filePathProjectionMatrixL = "D:/data/Lumia800/buddha/txt/00000010.txt";
	//std::string filePathProjectionMatrixR = "D:/data/Lumia800/buddha/txt/00000015.txt";


	int opt = 0;
	if (argc > 1) {
		opt = atoi(argv[1]);
	}
	
	if (opt == 0) {
		StereoReconstruct(filePathImageL, filePathImageR, filePathProjectionMatrixL, filePathProjectionMatrixR);
	}
	else {
		RunAutoPlayRenderingSystem("D:/data/Lumia800/CanSequence/CanSequence.nvm.cmvs/00", 28);
	}
	
	return 0;
}
