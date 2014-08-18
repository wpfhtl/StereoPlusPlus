#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <string>

#include "StereoAPI.h"
#include "ReleaseAssert.h"
#include "Timer.h"
#include "gco-v3.0/GCoptimization.h"

void MRFStereoByGraphCut(std::string rootFolder)
{

	cv::Mat imL = cv::imread("D:/data/stereo/" + rootFolder + "/im2.png");
	cv::Mat imR = cv::imread("D:/data/stereo/" + rootFolder + "/im6.png");
	int numRows = imL.rows, numCols = imL.cols;
	int numDisps, maxDisp, visualizeScale;
	SetupStereoParameters(rootFolder, numDisps, maxDisp, visualizeScale);


	// first set up the array for data costs
	MCImg<float> dsiL = ComputeAdGradientCostVolume(imL, imR, numDisps, -1, 1.f);
	ASSERT(dsiL.n == numDisps);
	float *data = new float[numRows * numCols * numDisps];
	memcpy(data, dsiL.data, numRows * numCols * numDisps * sizeof(float));

	// next set up the array for smooth costs
	extern float ISING_CUTOFF;
	extern float ISING_LAMBDA;
	float *smooth = new float[numDisps * numDisps];
	for (int l1 = 0; l1 < numDisps; l1++) {
		for (int l2 = 0; l2 < numDisps; l2++) {
			smooth[l1 + l2*numDisps] = ISING_LAMBDA * std::min((int)ISING_CUTOFF, std::abs(l1 - l2));
		}
	}

	try{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(numCols, numRows, numDisps);
		gc->setDataCost(data);
		gc->setSmoothCost(smooth);


		printf("\nBefore optimization energy is %f", gc->compute_energy());
		bs::Timer::Tic("Alpha expansion");
		gc->expansion(20);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		bs::Timer::Toc();
		printf("\nAfter optimization energy is %f", gc->compute_energy());

		cv::Mat dispL(numRows, numCols, CV_32FC1);
		for (int y = 0, id = 0; y < numRows; y++) {
			for (int x = 0; x < numCols; x++, id++) {
				dispL.at<float>(y, x) = gc->whatLabel(id);
			}
		}
		delete gc;


		EvaluateDisparity(rootFolder, dispL);
	}
	catch (GCException e){
		e.Report();
	}

	delete[] smooth;
	delete[] data;
}

void TestMRFStereoByGraphCut()
{
	MRFStereoByGraphCut("tsukuba");
}