
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <set>

#include "StereoAPI.h"
#include "Timer.h"

#ifdef _DEBUG
#pragma comment(lib, "opencv_core248d.lib")
#pragma comment(lib, "opencv_highgui248d.lib")
#pragma comment(lib, "opencv_imgproc248d.lib")
#pragma comment(lib, "opencv_features2d248d.lib")
#pragma comment(lib, "opencv_calib3d248d.lib")
#pragma comment(lib, "opencv_video248d.lib")
#pragma comment(lib, "opencv_flann248d.lib")
#pragma comment(lib, "opencv_nonfree248d.lib")
#else
#pragma comment(lib, "opencv_core248.lib")
#pragma comment(lib, "opencv_highgui248.lib")
#pragma comment(lib, "opencv_imgproc248.lib")
#pragma comment(lib, "opencv_features2d248.lib")
#pragma comment(lib, "opencv_calib3d248.lib")
#pragma comment(lib, "opencv_video248.lib")
#pragma comment(lib, "opencv_flann248.lib")
#pragma comment(lib, "opencv_nonfree248.lib")
#endif



#if 0
// wmf
int main(int argc, char **argv)
{
	if (argc != 6) {
		printf("usage: %s filePathImgL filePathDispL filePathDispR filePathDispOut useValidPixelOnly [visualizeScale=64]", argv[0]);
		exit(-1);
	}


	std::string filePathImgL	= std::string(argv[1]);
	std::string filePathDispL	= std::string(argv[2]);
	std::string filePathDispR	= std::string(argv[3]);
	std::string filePathDispOut = std::string(argv[4]);
	int useValidPixelOnly		= atoi(argv[5]);
	float visualizeScale = 64;
	if (argc == 7) {
		visualizeScale = atof(argv[6]);
	}

	cv::Mat img		= cv::imread(filePathImgL);
	cv::Mat dispL	= cv::imread(filePathDispL, CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat dispR	= cv::imread(filePathDispR, CV_LOAD_IMAGE_UNCHANGED);
	dispL.convertTo(dispL, CV_32FC1, 1.f / visualizeScale);
	dispR.convertTo(dispR, CV_32FC1, 1.f / visualizeScale);
	cv::Mat validPixelMapL = CrossCheck(dispL, dispR, -1, 1.f);
	cv::Mat validPixelMapR = CrossCheck(dispR, dispL, +1, 1.f);

	void PixelwiseOcclusionFilling(cv::Mat &disp, cv::Mat &validPixelMap);
	PixelwiseOcclusionFilling(dispL, validPixelMapL);
	PixelwiseOcclusionFilling(dispR, validPixelMapR);

	
	validPixelMapL = CrossCheck(dispL, dispR, -1, 1.f);
	validPixelMapR = CrossCheck(dispR, dispL, +1, 1.f);


	void FastWeightedMedianFilterInvalidPixels(cv::Mat &disp,
		cv::Mat &validPixelMap, cv::Mat &img, bool useValidPixelOnly);
	cv::Mat dispWmfL = dispL.clone();
	bs::Timer::Tic("wmf");
	FastWeightedMedianFilterInvalidPixels(dispWmfL, validPixelMapL, img, useValidPixelOnly);
	bs::Timer::Toc();

	double minVal, maxVal;
	cv::minMaxIdx(dispL, &minVal, &maxVal);
	cv::Mat Float2ColorJet(cv::Mat &fimg, float dmin, float dmax);
	cv::imwrite(filePathDispOut + "_colorjet_wmfL.png", Float2ColorJet(dispL, minVal, maxVal));

	dispWmfL.convertTo(dispWmfL, CV_16UC1, visualizeScale);
	cv::imwrite(filePathDispOut + "_wmfL.png", dispWmfL);
}
#endif
#if 0
// disparity .png or .pfm to colorjet mpa.
int main(int argc, char **argv)
{
	if (argc < 5) {
		fprintf(stderr, "usage: %s.exe [-calib calibfile.txt] [-vminmax vmin vmax] dispIn.pfm/.png dispOut.png [visualizeScale=64]", argv[0]);
		exit(-1);
	}

	char *filePathCalib = NULL;
	float dmin = 0;
	float dmax = 0;
	int optIdx = 0;
	float visualizeScale = 64.f;

	if (strcmp(argv[1], "-calib") == 0) {
		void readVizRange(char *calibfile, float& dmin, float& dmax);
		filePathCalib = argv[2];
		readVizRange(filePathCalib, dmin, dmax);
		optIdx = 3;
	}
	else if (strcmp(argv[1], "-vminmax") == 0) {
		dmin = atof(argv[2]);
		dmax = atof(argv[3]);
		optIdx = 4;
	}
	if (dmin == dmax) {
		fprintf(stderr, "error, dmin == dmax. dmin = %f, dmax = %f\n", dmin, dmax);
		exit(-1);
	}
	std::string filePathDisp		= argv[optIdx++];
	std::string filePathColorJet	= argv[optIdx++];
	if (optIdx < argc) {
		visualizeScale = atof(argv[optIdx]);
	}

	printf("dmin = %f\n", dmin);
	printf("dmax = %f\n", dmax);
	printf("visualizeScale = %f\n", visualizeScale);
	printf("infilePath = %s\n", filePathDisp.c_str());
	printf("outfilePath = %s\n", filePathColorJet.c_str());

	std::cout << filePathDisp.substr(filePathDisp.length() - 4, 4) << "\n";

	cv::Mat disp, colorJetImg;
	if (filePathDisp.substr(filePathDisp.length() - 4, 4) == ".pfm") {
		printf("reading PFM file...\n");
		cv::Mat ReadFilePFM(std::string filePath);
		disp = ReadFilePFM(filePathDisp);
	}
	else {
		printf("reading PNG file...\n");
		disp = cv::imread(filePathDisp, CV_LOAD_IMAGE_UNCHANGED);
		disp.convertTo(disp, CV_32FC1, 1.f / visualizeScale);
	}

	/*disp.convertTo(disp, CV_8UC1, 64);
	cv::imshow("disp[", disp);
	cv::waitKey(0);*/

	cv::Mat Float2ColorJet(cv::Mat &fimg, float dmin, float dmax);
	colorJetImg = Float2ColorJet(disp, dmin, dmax);
	cv::imwrite(filePathColorJet, colorJetImg); 

	return 0;
}

#endif

#if 0

struct ParamsCombine {
	cv::Mat *labelMap;
	cv::Mat *dispPM;
	cv::Mat *dispARAP;
	cv::Mat *dispCombined;
	cv::Mat *canvas;
	std::vector<std::vector<cv::Point2i>> *segPixelLists;
	int numDisps;
	ParamsCombine() : labelMap(0), dispPM(0), dispARAP(0), 
		dispCombined(0), canvas(0), segPixelLists(0), numDisps(0) {}
};

void OnMouseCombineDisparities(int event, int x, int y, int flags, void *param)
{

	if (event != CV_EVENT_LBUTTONDOWN && event != CV_EVENT_RBUTTONDOWN
		&& event != CV_EVENT_LBUTTONDBLCLK && event != CV_EVENT_RBUTTONDBLCLK) {
		return;
	}

	ParamsCombine *params = (ParamsCombine*)param;
	int numRows = params->dispARAP->rows;
	int numCols = params->dispARAP->cols;

	cv::Mat &canvas = *params->canvas;
	float zoomFactor = (float)numRows / canvas.rows;
	x *= zoomFactor;
	y *= zoomFactor;
	y %= numRows;
	x %= numCols;

	cv::Mat &dispPM = *params->dispPM;
	cv::Mat &dispARAP = *params->dispARAP;
	cv::Mat &dispCombined = *params->dispCombined;
	std::vector<std::vector<cv::Point2i>> &segPixelLists = *params->segPixelLists;
	cv::Mat &labelMap = *params->labelMap;

	int id = labelMap.at<int>(y, x);
	cv::Mat &dispSrc = cv::Mat();

	if (event == CV_EVENT_LBUTTONDOWN || event == CV_EVENT_LBUTTONDBLCLK) {
		// choose PM
		dispSrc = dispPM;
	}
	if (event == CV_EVENT_RBUTTONDOWN || event == CV_EVENT_RBUTTONDBLCLK) {
		// choose ARAP
		dispSrc = dispARAP;
	}

	if (event == CV_EVENT_LBUTTONDOWN || event == CV_EVENT_RBUTTONDOWN) {
		std::vector<cv::Point2i> &pointList = segPixelLists[id];
		for (int i = 0; i < pointList.size(); i++) {
			cv::Point2i &p = pointList[i];
			dispCombined.at<float>(p.y, p.x) = dispSrc.at<float>(p.y, p.x);
			int r = 0;
			int g = 0;
			int b = 0;
			float scale = 1.0 / (params->numDisps - 0);
			void jet(float x, int& r, int& g, int& b);
			float val = scale * (dispCombined.at<float>(p.y, p.x) - 0);
			jet(val, r, g, b);

			canvas.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(b, g, r);
		}
	}
	if (event == CV_EVENT_LBUTTONDBLCLK || event == CV_EVENT_RBUTTONDBLCLK) {
		std::set<int> idSet;
		const int STRIDE = 12;
		for (int yy = y - 100; yy <= y + 100; yy += 12)  {
			for (int xx = x - 100; xx <= x + 100; xx += 12) {
				if (InBound(yy, xx, numRows, numCols)) {
					idSet.insert(labelMap.at<int>(yy, xx));
				}
			}
		}

		std::vector<int> segIds = std::vector<int>(idSet.begin(), idSet.end());
		for (int k = 0; k < segIds.size(); k++) {
			std::vector<cv::Point2i> &pointList = segPixelLists[segIds[k]];
			for (int i = 0; i < pointList.size(); i++) {
				cv::Point2i &p = pointList[i];
				dispCombined.at<float>(p.y, p.x) = dispSrc.at<float>(p.y, p.x);
				int r = 0;
				int g = 0;
				int b = 0;
				float scale = 1.0 / (params->numDisps - 0);
				void jet(float x, int& r, int& g, int& b);
				float val = scale * (dispCombined.at<float>(p.y, p.x) - 0);
				jet(val, r, g, b);

				canvas.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(b, g, r);
			}
		}
		
	}
	

	//cv::Mat concat;
	//cv::hconcat(dispCombined, dispARAP, concat);
	//cv::Mat Float2ColorJet(cv::Mat &fimg, float dmin, float dmax);
	//canvas = Float2ColorJet(concat, 0, params->numDisps);
	//cv::resize(canvas, canvas, cv::Size(canvas.cols / 2, canvas.rows / 2));
	//printf("fuck.\n");
	cv::imshow("canvas", canvas);
}

int main(int argc, char **argv)
{
	if (argc != 6) {
		fprintf(stderr, "usage: %s.exe filePathImgL filePathDispPML filePathDispARAPL filePathDispCombined ndisp\n", argv[0]);
		exit(-1);
	}

	std::string filePathImage			= argv[1];
	std::string filePathDispPM			= argv[2];
	std::string filePathDispARAP		= argv[3];
	std::string filePathDispCombined	= argv[4];
	int numDisps = atoi(argv[5]);

	cv::Mat img			= cv::imread(filePathImage);
	cv::Mat dispPM		= cv::imread(filePathDispPM, CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat dispARAP	= cv::imread(filePathDispARAP, CV_LOAD_IMAGE_UNCHANGED);
	
	dispPM.convertTo(dispPM, CV_32FC1, 1.f / 64.f);
	dispARAP.convertTo(dispARAP, CV_32FC1, 1.f / 64.f);
	cv::Mat dispCombined = dispPM.clone();

	int numRegions = 8000;
	float compactness = 25.f;
	int SLICSegmentation(const cv::Mat &img, const int numPreferedRegions, const int compactness, cv::Mat& labelMap, cv::Mat& contourImg);
	cv::Mat labelMap, contourImg;
	int numSegsL = SLICSegmentation(img, numRegions, compactness, labelMap, contourImg);

	void ConstructBaryCentersAndPixelLists(int numSegs, cv::Mat &labelMap,
		std::vector<cv::Point2f> &baryCenters, std::vector<std::vector<cv::Point2i>> &segPixelLists);
	std::vector<cv::Point2f> baryCenters;
	std::vector<std::vector<cv::Point2i>> segPixelLists;
	ConstructBaryCentersAndPixelLists(numSegsL, labelMap, baryCenters, segPixelLists);

	cv::Mat Float2ColorJet(cv::Mat &fimg, float dmin, float dmax);
	cv::Mat jetPM = Float2ColorJet(dispPM, 0, numDisps);
	cv::Mat jetARAP = Float2ColorJet(dispARAP, 0, numDisps);
	cv::Mat canvas;
	cv::hconcat(jetPM, jetARAP, canvas);
	//cv::resize(canvas, canvas, cv::Size(canvas.cols / 2, canvas.rows / 2));

	ParamsCombine params;
	params.dispARAP			= &dispARAP;
	params.dispPM			= &dispPM;
	params.dispCombined		= &dispCombined;
	params.labelMap			= &labelMap;
	params.segPixelLists	= &segPixelLists;
	params.canvas			= &canvas;
	params.numDisps			= numDisps;

	cv::imshow("canvas", canvas);
	cv::setMouseCallback("canvas", OnMouseCombineDisparities, &params);
	cv::waitKey(0);

	cv::Mat jetMapCombined = Float2ColorJet(dispCombined, 0, numDisps);
	dispCombined.convertTo(dispCombined, CV_16UC1, 64);
	
	cv::imwrite(filePathDispCombined, dispCombined);
	cv::imwrite(filePathDispCombined + "_jetmap.png", jetMapCombined);
	
	return 0;
}
#endif

#if 0
int main(int argc, char **argv)
{
	if (argc != 6) {
		printf("usage: %s.exe filePathRGB filePathOut numRegionis compactness showImmediately", argv[0]);
		exit(-1);
	}
	std::string filePathRGB = argv[1];
	std::string filePathOut = argv[2];
	int numPreferedRegions = atoi(argv[3]);
	float compactness = atof(argv[4]);
	int showImmediately = atoi(argv[5]);

	printf("numRegions = %d\n", numPreferedRegions);
	printf("compactness = %f\n", compactness);


	int SLICSegmentation(const cv::Mat &img, const int numPreferedRegions, const int compactness, cv::Mat& labelMap, cv::Mat& contourImg);
	cv::Mat DrawSegmentImage(cv::Mat &labelMap);
	std::vector<cv::Vec3b> ComupteSegmentMeanLabColor(cv::Mat &labImg, cv::Mat &labelMap, int numSegs);

	cv::Mat img = cv::imread(filePathRGB);
	cv::Mat labelMap, contourImg;
	int numRegions = SLICSegmentation(img, numPreferedRegions, compactness, labelMap, contourImg);
	std::vector<cv::Vec3b> meanColors = ComupteSegmentMeanLabColor(img, labelMap, numRegions);

	
	cv::Mat segImg = DrawSegmentImage(labelMap);
	cv::Mat smoothedImg = img.clone();
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			int id = labelMap.at<int>(y, x);
			smoothedImg.at<cv::Vec3b>(y, x) = meanColors[id];
		}
	}


	cv::imwrite(filePathOut + "_colorImg.png", img);
	cv::imwrite(filePathOut + "_segImg.png", segImg);
	cv::imwrite(filePathOut + "_smoothedImg.png", smoothedImg);
	cv::imwrite(filePathOut + "_contourImg.png", contourImg);

	if (showImmediately) {
		cv::imshow("tmp", img);
		cv::waitKey(0);
		cv::imshow("tmp", smoothedImg);
		cv::waitKey(0);
		cv::imshow("tmp", contourImg);
		cv::waitKey(0);
	}


	return 0;
}
#endif
#if 0
// calculate the matching cost rescaling factor if i change the census weight from 0.2 to 2.
template<typename T>
void CostVolumeFromYamaguchi(std::string &leftFilePath, std::string &rightFilePath,
	MCImg<T> &dsiL, MCImg<T> &dsiR, int numDisps);


cv::Vec3d CostStatistics(MCImg<unsigned short> &dsi, cv::Mat &disp)
{
	int numRows = dsi.h, numCols = dsi.w, numDisps = dsi.n;
	std::vector<unsigned short> vals;
	vals.reserve(numRows * numCols);

	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			int d = disp.at<float>(y, x);
			if (d < 0 || d >= numDisps) {
				d = rand() % numDisps;
			}
			vals.push_back(dsi.get(y, x)[d]);
		}
	}

	double sum = 0;
	for (int i = 0; i < vals.size(); i++) {
		sum += vals[i];
	}
	double mean = sum / vals.size();
	
	double deviation = 0;
	for (int i = 0; i < vals.size(); i++) {
		deviation += (vals[i] - mean) * (vals[i] - mean);
	}
	deviation /= vals.size();
	double stdev = std::sqrt(deviation);

	std::sort(vals.begin(), vals.end());
	double median = vals[vals.size() / 2];

	return cv::Vec3d(mean, median, stdev);
}

int main(int argc, char **argv)
{
	std::string filePathGT = "D:/data/MiddEval3/trainingH/Piano/disp0GT.pfm";;
	std::string filePathImL = "D:/data/MiddEval3/trainingH/Piano/im0.png";
	std::string filePathImR = "D:/data/MiddEval3/trainingH/Piano/im1.png";;

	cv::Mat imL = cv::imread(filePathImL);
	cv::Mat imR = cv::imread(filePathImR);
	int numRows = imL.rows, numCols = imL.cols, numDisps = 144;

	cv::Mat ReadFilePFM(std::string filePath);
	cv::Mat dispGT = ReadFilePFM(filePathGT);
	
	extern double SGMSTEREO_DEFAULT_CENSUS_WEIGHT_FACTOR;
	MCImg<unsigned short> dsiL(numRows, numCols, numDisps);
	MCImg<unsigned short> dsiR(numRows, numCols, numDisps);

	SGMSTEREO_DEFAULT_CENSUS_WEIGHT_FACTOR = 0.2;
	CostVolumeFromYamaguchi<unsigned short>(filePathImL, filePathImR, dsiL, dsiR, numDisps);
	cv::Vec3d stat1 = CostStatistics(dsiL, dispGT);

	SGMSTEREO_DEFAULT_CENSUS_WEIGHT_FACTOR = 2;
	CostVolumeFromYamaguchi<unsigned short>(filePathImL, filePathImR, dsiL, dsiR, numDisps);
	cv::Vec3d stat2 = CostStatistics(dsiL, dispGT);

	printf("*************** for dispGT *****************\n");
	printf("old0.2: avg = %lf,   median = %lf,  standard_deviation = %lf\n", stat1[0], stat1[1], stat1[2]);
	printf("new2.0: avg = %lf,   median = %lf,  standard_deviation = %lf\n", stat2[0], stat2[1], stat2[2]);




	cv::Mat dispRandom(numRows, numCols, CV_32FC1);
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			dispRandom.at<float>(y, x) = rand() % numDisps;
		}
	}
	SGMSTEREO_DEFAULT_CENSUS_WEIGHT_FACTOR = 0.2;
	CostVolumeFromYamaguchi<unsigned short>(filePathImL, filePathImR, dsiL, dsiR, numDisps);
	stat1 = CostStatistics(dsiL, dispRandom);

	SGMSTEREO_DEFAULT_CENSUS_WEIGHT_FACTOR = 2;
	CostVolumeFromYamaguchi<unsigned short>(filePathImL, filePathImR, dsiL, dsiR, numDisps);
	stat2 = CostStatistics(dsiL, dispRandom);

	printf("*************** for disp Noise *****************\n");
	printf("old0.2: avg = %lf,   median = %lf,  standard_deviation = %lf\n", stat1[0], stat1[1], stat1[2]);
	printf("new2.0: avg = %lf,   median = %lf,  standard_deviation = %lf\n", stat2[0], stat2[1], stat2[2]);

	return 0;

}
#endif