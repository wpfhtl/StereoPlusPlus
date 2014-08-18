#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "StereoAPI.h"
#include "BPOnFactorGraph.h"
#include "ReleaseAssert.h"



/* Note: each OnMouse callback function should take a fixed list of parameters.
 * It is the caller's responsibility to prepare different parameter lists for
 * different OnMouse fucntions.
 */

void OnMouseEvaluateDisparityDefaultDrawing(int event, int x, int y, int flags, void *param, cv::Mat &tmp)
{
	//cv::Mat &canvas = *(cv::Mat*)((void**)param)[0];
	cv::Mat &dispL	= *(cv::Mat*)((void**)param)[1];
	cv::Mat &GT		= *(cv::Mat*)((void**)param)[2];
	cv::Mat &imL	= *(cv::Mat*)((void**)param)[3];
	cv::Mat &imR	= *(cv::Mat*)((void**)param)[4];

	float badRateOnNonocc	= *(float*)((void**)param)[5];
	float badRateOnAll		= *(float*)((void**)param)[6];
	float badRateOnDisc		= *(float*)((void**)param)[7];

	int numDisps			= *(int*)((void**)param)[8];
	int visualizeScale		= *(int*)((void**)param)[9];
	int maxDisp				= numDisps - 1;

	std::string workingDir	= *(std::string*)((void**)param)[10];


	int numRows = GT.rows, numCols = GT.cols;
	x %= numCols;
	y %= numRows;
	//cv::Mat tmp = canvas.clone();

	if (event == CV_EVENT_MOUSEMOVE)
	{
		cv::Point CC(x + 1 * numCols, y);
		cv::Point LL(x + 0 * numCols, y);
		cv::Point RR(x + 2 * numCols, y);

		cv::line(tmp, LL, RR, cv::Scalar(255, 0, 0));
		cv::line(tmp, LL + cv::Point(0, numRows), RR + cv::Point(0, numRows), cv::Scalar(255, 0, 0));
		cv::circle(tmp, CC, 1, cv::Scalar(0, 0, 255), 2, CV_AA);
		cv::circle(tmp, CC + cv::Point(0, numRows), 1, cv::Scalar(0, 0, 255), 2, CV_AA);

		float dGT = GT.at<float>(y, x);
		float dMY = dispL.at<float>(y, x);
		char text[1024];
		sprintf(text, "(%d, %d)  GT: %.2f  MINE: %.2f", y, x, dGT, dMY);
		cv::putText(tmp, std::string(text), cv::Point2d(20, 50), 0, 0.6, cv::Scalar(0, 0, 255, 1), 2);

		cv::Point UU(x + 2 * numCols, y);
		cv::Point DD(x + 2 * numCols - dMY, y + numRows);
		cv::line(tmp, UU, DD, cv::Scalar(0, 0, 255), 1, CV_AA);
	}

	if (event == CV_EVENT_RBUTTONDOWN)
	{
		char text[1024];
		sprintf(text, "BadPixelRate  %.2f%%  %.2f%%  %.2f%%", 100.f * badRateOnNonocc, 100.f * badRateOnAll, 100.f * badRateOnDisc);
		cv::putText(tmp, std::string(text), cv::Point2d(20, 50), 0, 0.6, cv::Scalar(0, 0, 255, 1), 2);
	}

	if (event == CV_EVENT_LBUTTONDBLCLK)
	{
		std::string plyFilePath = workingDir + "/OnMouseEvaluateDisprity.ply";
		std::string cmdInvokeMeshlab = "meshlab " + plyFilePath;
		void SaveDisparityToPly(cv::Mat &disp, cv::Mat& img, float maxDisp,
			std::string workingDir, std::string plyFilePath, cv::Mat &validPixelMap = cv::Mat());
		SaveDisparityToPly(dispL, imL, maxDisp, workingDir, plyFilePath);
		system(cmdInvokeMeshlab.c_str());
	}
}

void OnMouseEvaluateDisparity(int event, int x, int y, int flags, void *param)
{
	/*void *callbackParams[] = {
		&auxParams,
		&canvas, &dispL, &GT, &imL, &imR,
		&badRateOnNonocc, &badRateOnAll, &badRateOnDisc,
		&numDisps, &visualizeScale, &workingDir
	};*/
	cv::Mat &canvas = *(cv::Mat*)((void**)param)[1];
	cv::Mat tmp = canvas.clone();
	OnMouseEvaluateDisparityDefaultDrawing(event, x, y, flags, (void*)((void**)param + 1), tmp);
	cv::imshow("OnMouseEvaluateDisparity", tmp);
}

void OnMousePatchMatchOnPixels(int event, int x, int y, int flags, void *param)
{
	cv::Mat tmpCanvas = (*(cv::Mat*)((void**)param)[1]).clone();
	OnMouseEvaluateDisparityDefaultDrawing(event, x, y, flags, (void*)((void**)param + 1), tmpCanvas);

	if (event == CV_EVENT_LBUTTONDOWN) {
		std::vector<std::pair<std::string, void*>> &auxParams
			= *(std::vector<std::pair<std::string, void*>>*)((void**)param)[0];

		ASSERT(auxParams[0].first == "slantedPlanesL")
		ASSERT(auxParams[1].first == "bestCostsL")
		MCImg<SlantedPlane> &slantedPlanesL = *(MCImg<SlantedPlane>*)auxParams[0].second;
		cv::Mat &bestCostsL					= *(cv::Mat*)auxParams[1].second;

		int numRows = bestCostsL.rows, numCols = bestCostsL.cols;
		y %= numRows;
		x %= numCols;
		SlantedPlane &sp = slantedPlanesL[y][x];

		/*printf("( a,  b,  c) = (% 11.6f, % 11.6f, % 11.6f)\n", sp.a,  sp.b,  sp.c);
		printf("(nx, ny, nz) = (% 11.6f, % 11.6f, % 11.6f)\n", sp.nx, sp.ny, sp.nz);
		printf("bestCost(%3d, %3d) = %f\n\n", y, x, bestCostsL.at<float>(y, x));*/

		char textBuf[1024];
		sprintf(textBuf, "( a,  b,  c) = (% 11.6f, % 11.6f, % 11.6f)", sp.a, sp.b, sp.c);
		cv::putText(tmpCanvas, std::string(textBuf), cv::Point2d(20, 50),  CV_FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
		sprintf(textBuf, "(nx, ny, nz) = (% 11.6f, % 11.6f, % 11.6f)", sp.nx, sp.ny, sp.nz);
		cv::putText(tmpCanvas, std::string(textBuf), cv::Point2d(20, 80),  CV_FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
		sprintf(textBuf, "bestCost(%3d, %3d) = %f", y, x, bestCostsL.at<float>(y, x));
		cv::putText(tmpCanvas, std::string(textBuf), cv::Point2d(20, 110), CV_FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
	}
	
	cv::imshow("OnMousePatchMatchOnPixels", tmpCanvas);
}

void OnMouseLoopyBPOnGridGraph(int event, int x, int y, int flags, void *param)
{
	cv::Mat tmpCanvas = (*(cv::Mat*)((void**)param)[1]).clone();
	OnMouseEvaluateDisparityDefaultDrawing(event, x, y, flags, (void*)((void**)param + 1), tmpCanvas);

	if (event == CV_EVENT_LBUTTONDOWN) {
		std::vector<std::pair<std::string, void*>> &auxParams
			= *(std::vector<std::pair<std::string, void*>>*)((void**)param)[0];

		ASSERT(auxParams[0].first == "allMessages")
		ASSERT(auxParams[1].first == "allBeliefs")
		ASSERT(auxParams[2].first == "unaryCosts")
		MCImg<float> &allMessages = *(MCImg<float>*)auxParams[0].second;
		MCImg<float> &allBeliefs  = *(MCImg<float>*)auxParams[1].second;
		MCImg<float> &unaryCost   = *(MCImg<float>*)auxParams[2].second;

		int numRows = allBeliefs.h, numCols = allBeliefs.w, numDisps = allBeliefs.n;
		y %= numRows; x %= numCols;
		
		printf("\n\n(y, x) = (%d, %d)\n", y, x);
		printf("===================== Beliefs | unaryCost | Messages =====================\n");
		for (int d = 0; d < numDisps; d++) {
			printf("% 10.6f | % 10.6f   %2d  "
				   "% 10.6f   % 10.6f   %2d  % 10.6f   % 10.6f\n", 
				   allBeliefs.get(y, x)[d], unaryCost.get(y, x)[d], d,
				   allMessages.get(y, x)[d + 0 * numDisps],
				   allMessages.get(y, x)[d + 1 * numDisps], d,
				   allMessages.get(y, x)[d + 2 * numDisps],
				   allMessages.get(y, x)[d + 3 * numDisps]);
		}

		cv::Mat &imL = *(cv::Mat*)((void**)param)[3 + 1];
		const cv::Point2i dirDelta[4] = { cv::Point2i(0, -1), cv::Point2i(-1, 0), cv::Point2i(0, +1), cv::Point2i(+1, 0) };
		cv::Point2i s(x, y);
		extern float ISING_GAMMA;
		for (int k = 0; k < 4; k++) {
			cv::Point2i t = s + dirDelta[k];
			if (InBound(t, numRows, numCols)) {
				float simWeight = exp(-L1Dist(imL.at<cv::Vec3b>(s.y, s.x), imL.at<cv::Vec3b>(t.y, t.x)) / ISING_GAMMA);
				printf("(%d, %d) simWeight = %f\n", t.y, t.x, simWeight);
			}
		}
	}

	cv::imshow("OnMouseLoopyBPOnGridGraph", tmpCanvas);
}

void OnMouseTestSelfSimilarityPropagation(int event, int x, int y, int flags, void *param)
{
	cv::Mat canvas = *(cv::Mat*)((void**)param)[0];
	cv::Mat tmp = canvas.clone();

	int nrows = canvas.rows, ncols = canvas.cols / 2;
	x %= ncols;
	y %= nrows;

	if (event == CV_EVENT_MOUSEMOVE)
	{
		cv::Point AA(x, y), BB(x + ncols, y);
		cv::line(tmp, AA, BB, cv::Scalar(255, 0, 0, 1));
	}

	if (event == CV_EVENT_LBUTTONDOWN) {
		std::vector<SimVector> &simVecs = *(std::vector<SimVector>*)((void**)param)[1];
		cv::Mat nbImg = tmp(cv::Rect(0, 0, ncols, nrows));
		cv::Mat img = tmp(cv::Rect(ncols, 0, ncols, nrows));
		nbImg.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
		int id = y * ncols + x;
		for (int i = 0; i < SIMVECTORSIZE; i++) {
			cv::Point2i nb = simVecs[id].pos[i];
			nbImg.at<cv::Vec3b>(nb.y, nb.x) = cv::Vec3b(255, 255, 255);
			img.at<cv::Vec3b>(nb.y, nb.x) = cv::Vec3b(255, 255, 255);
		}
	}

	cv::imshow("TestSelfSimilarityPropagation", tmp);
}

void OnMouseMeshStereoOnFactorGraph(int event, int x, int y, int flags, void *param)
{
	std::vector<std::pair<std::string, void*>> &auxParams
		= *(std::vector<std::pair<std::string, void*>>*)((void**)param)[0];
	cv::Mat &imL			= *(cv::Mat*)((void**)param)[3 + 1];
	int numDisps			= *(int*)((void**)param)[8 + 1];
	int maxDisp				= numDisps - 1;
	std::string workingDir	= *(std::string*)((void**)param)[10 + 1];

	ASSERT(auxParams[0].first == "splitMap")
	ASSERT(auxParams[1].first == "MeshStereoBPOnFGObject")

	cv::Mat									&splitMap = *(cv::Mat*)auxParams[0].second;
	MeshStereoBPOnFG						&bp = *(MeshStereoBPOnFG*)auxParams[1].second;
	std::vector<cv::Point2d>				&vertexCoords = bp.vertexCoords;
	std::vector<std::vector<int>>			&triVertexInds = bp.triVertexInds;
	std::vector<std::vector<SlantedPlane>>	&triVertexBestLabels = bp.triVertexBestLabels;


	if (event == CV_EVENT_LBUTTONDOWN)
	{
		y %= imL.rows; x %= imL.cols;
		int bestIdx = 0;
		for (int i = 0; i < vertexCoords.size(); i++) {
			cv::Point2d &p = vertexCoords[i];
			cv::Point2d &q = vertexCoords[bestIdx];
			if ((y - p.y) * (y - p.y) + (x - p.x) * (x - p.x) 
				< (y - q.y) * (y - q.y) + (x - q.x) * (x - q.x)) {
				bestIdx = i;
			}
		}
		y = vertexCoords[bestIdx].y;
		x = vertexCoords[bestIdx].x;


		std::vector<std::pair<int, int>> triIds;
		for (int id = 0; id < triVertexInds.size(); id++) {
			for (int j = 0; j < 3; j++) {
				if (triVertexInds[id][j] == bestIdx) {
					triIds.push_back(std::make_pair(id, j));
				}
			}
		}

		printf("\n\nCurrent vextex (%d, %d) has %d owners.\n", y, x, triIds.size());
		std::vector<Probs> beliefs(triIds.size());
		int maxNumLabels = 0;
		for (int i = 0; i < triIds.size(); i++) {
			int id = triIds[i].first;
			int subId = triIds[i].second;
			beliefs[i] = bp.allBeliefs[3 * id + subId];
			maxNumLabels = std::max(maxNumLabels, (int)beliefs[i].size());
		}
		printf("maxNumLabels = %d\n", maxNumLabels);

		printf("\n=========================== Beliefs ==============================\n");
		for (int rowId = 0; rowId < maxNumLabels; rowId++) {
			for (int i = 0; i < beliefs.size(); i++) {
				if (rowId < beliefs[i].size()) {
					printf("%10.4f  ", beliefs[i][rowId]);
				}
				else {
					printf("**********  ");
				}
			}
			printf("\n");
		}

		printf("\n\n=========================== Messages ==============================\n");

		return;
	}

	if (event == CV_EVENT_LBUTTONDBLCLK) {
		std::string plyFilePath = workingDir + "/OnMouseMeshStereoMesh.ply";
		std::string cmdInvokeMeshlab = "meshlab " + plyFilePath;
		std::string textureFilePath = workingDir + "/im2.png";

		void SaveMeshStereoResultToPly(cv::Mat &img, float maxDisp,
			std::string workingDir, std::string plyFilePath, std::string textureFilePath,
			std::vector<cv::Point2d> &vertexCoords, std::vector<std::vector<int>> &triVertexInds,
			std::vector<std::vector<SlantedPlane>> &triVertexBestLabels, cv::Mat &splitMap);
		SaveMeshStereoResultToPly(imL, maxDisp, workingDir, plyFilePath, textureFilePath,
			vertexCoords, triVertexInds, triVertexBestLabels, splitMap);
		system(cmdInvokeMeshlab.c_str());
		return;
	}


	cv::Mat tmpCanvas = (*(cv::Mat*)((void**)param)[1]).clone();
	OnMouseEvaluateDisparityDefaultDrawing(event, x, y, flags, (void*)((void**)param + 1), tmpCanvas);
	cv::imshow("OnMouseMeshStereoOnFactorGraph", tmpCanvas);
}

void OnMouseGroundTruthPlaneStatistics(int event, int x, int y, int flags, void *param)
{
	struct GTStatParams {
		cv::Mat *labelMap;
		std::vector<SlantedPlane> *gtPlanes;
	};
	GTStatParams &gtStatParams = *(GTStatParams*)param;

	cv::Mat &labelMap = *gtStatParams.labelMap;
	std::vector<SlantedPlane> &gtPlanes = *gtStatParams.gtPlanes;

	if (event == CV_EVENT_LBUTTONDOWN) 
	{
		y %= labelMap.rows;
		x %= labelMap.cols;

		int segId = labelMap.at<int>(y, x);
		if (segId > -1) {
			SlantedPlane p = gtPlanes[segId];
			printf("(nx, ny, nz) = (%f, %f, %f)\n", p.nx, p.ny, p.nz);
		}
	}
}