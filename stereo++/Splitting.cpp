#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <set>

#include "StereoAPI.h"
#include "SlantedPlane.h"
#include "Timer.h"
#include "ReleaseAssert.h"
#include "gco-v3.0/GCoptimization.h"


void ConstructNeighboringTriangleGraph(int numRows, int numCols, std::vector<cv::Point2d> &vertexCoords, std::vector<std::vector<int>> &triVertexInds,
	std::vector<cv::Point2d> &baryCenters, std::vector<std::vector<int>> &nbIndices);

void DeterminePixelOwnership(int numRows, int numCols, std::vector<cv::Point2d> &vertexCoords,
	std::vector<std::vector<int>> &triVertexInds, std::vector<std::vector<cv::Point2i>> &triPixelLists);

cv::Mat TriangleLabelToDisparityMap(int numRows, int numCols, std::vector<SlantedPlane> &slantedPlanes,
	std::vector<std::vector<cv::Point2i>> &triPixelLists);

cv::Mat SlantedPlaneMapToDisparityMap(MCImg<SlantedPlane> &slantedPlanes);



static void TriangleLabelVotedFromPixelwiseLabel(MCImg<SlantedPlane> &pixelwiseSlantedPlanes, 
	std::vector<std::vector<cv::Point2i>> &triPixelLists, std::vector<SlantedPlane> &slantedPlanes)
{
	int numTriangles = triPixelLists.size();
	slantedPlanes.resize(numTriangles);

	for (int id = 0; id < numTriangles; id++) {
		std::vector<cv::Point2i> &pixelList = triPixelLists[id];
		int numPixels = pixelList.size();
		std::vector<float> a(numPixels);
		std::vector<float> b(numPixels);
		std::vector<float> c(numPixels);

		for (int i = 0; i < numPixels; i++) {
			cv::Point2i &p = pixelList[i];
			SlantedPlane &plane = pixelwiseSlantedPlanes[p.y][p.x];
			a[i] = plane.a;
			b[i] = plane.b;
			c[i] = plane.c;
		}

		std::nth_element(a.begin(), a.begin() + numPixels / 2, a.end());
		std::nth_element(b.begin(), b.begin() + numPixels / 2, b.end());
		std::nth_element(c.begin(), c.begin() + numPixels / 2, c.end());
	
		float ma = a[numPixels / 2];
		float mb = b[numPixels / 2];
		float mc = c[numPixels / 2];
		slantedPlanes[id] = SlantedPlane::ConstructFromAbc(ma, mb, mc);
	}
}

static cv::Mat DrawSplitMap(int numRows, int numCols, std::vector<cv::Point2d> &vertexCoords, std::vector<int> &labeling)
{
	ASSERT(labeling.size() == vertexCoords.size());
	cv::Mat canvas(numRows, numCols, CV_8UC3);
	for (int i = 0; i < labeling.size(); i++) {
		if (labeling[i]) {
			cv::Point2d &p = vertexCoords[i];
			cv::circle(canvas, p - cv::Point2d(0.5, 0.5), 0, cv::Scalar(0, 0, 255), 2, CV_AA);
		}
	}
	return canvas;
}

static std::vector<std::vector<int>> ConstructNeighboringVertexGraph(int numPublicVertices, std::vector<std::vector<int>> &triVertexInds)
{
	std::vector<std::set<int>> vertexNbInds(numPublicVertices);
	int numTriangles = triVertexInds.size();
	for (int id = 0; id < numTriangles; id++) {
		for (int i = 0; i < 3; i++) {
			for (int j = i + 1; j < 3; j++) {
				int p = triVertexInds[id][i];
				int q = triVertexInds[id][j];
				vertexNbInds[p].insert(q);
				vertexNbInds[q].insert(p);
			}
		}
	}

	std::vector<std::vector<int>> nbGraph(numPublicVertices);
	for (int i = 0; i < numPublicVertices; i++) {
		nbGraph[i] = std::vector<int>(vertexNbInds[i].begin(), vertexNbInds[i].end());
	}

	return nbGraph;
}

void ComputeSplittingMap(std::string rootFolder)
{
	cv::Mat imL = cv::imread("d:/data/stereo/" + rootFolder + "/im2.png");
	cv::Mat imR = cv::imread("d:/data/stereo/" + rootFolder + "/im6.png");
	int numRows = imL.rows, numCols = imL.cols;
	int numDisps, maxDisp, visualizeScale;
	SetupStereoParameters(rootFolder, numDisps, maxDisp, visualizeScale);


	// Step 1 - Obtain a nice pixelwise labeling from PatchMatch on pixels.
	MCImg<SlantedPlane> pixelwiseSlantedPlanesL(numRows, numCols);
	MCImg<SlantedPlane> pixelwiseSlantedPlanesR(numRows, numCols);
	pixelwiseSlantedPlanesL.LoadFromBinaryFile("d:/" + rootFolder + "SlantedPlanesL.bin");
	pixelwiseSlantedPlanesR.LoadFromBinaryFile("d:/" + rootFolder + "SlantedPlanesR.bin");


	// Step 2 - Obtain a labeling of triangles by using the pixelwise label at the bary centers of the triangle.
	std::vector<cv::Point2d> vertexCoordsL;
	std::vector<std::vector<int>> triVertexIndsL;
	Triangulate2DImage(imL, vertexCoordsL, triVertexIndsL);
	cv::Mat triImgL = DrawTriangleImage(numRows, numCols, vertexCoordsL, triVertexIndsL);

	std::vector<cv::Point2d> baryCentersL;
	std::vector<std::vector<int>> nbIndicesL, nbIndicesR;
	ConstructNeighboringTriangleGraph(numRows, numCols, vertexCoordsL, triVertexIndsL, baryCentersL, nbIndicesL);

	std::vector<std::vector<cv::Point2i>> triPixelListsL;
	DeterminePixelOwnership(numRows, numCols, vertexCoordsL, triVertexIndsL, triPixelListsL);

	int numTrianglesL = baryCentersL.size();
	std::vector<SlantedPlane> slantedPlanesL(numTrianglesL);
	for (int id = 0; id < numTrianglesL; id++) {
		int yc = baryCentersL[id].y + 0.5;
		int xc = baryCentersL[id].x + 0.5;
		ASSERT(InBound(yc, xc, numRows, numCols));
		slantedPlanesL[id] = pixelwiseSlantedPlanesL[yc][xc];
	}
	//TriangleLabelVotedFromPixelwiseLabel(pixelwiseSlantedPlanesL, triPixelListsL, slantedPlanesL);

	cv::Mat dispL = TriangleLabelToDisparityMap(numRows, numCols, slantedPlanesL, triPixelListsL);
	//cv::Mat dispL = SlantedPlaneMapToDisparityMap(pixelwiseSlantedPlanesL);
	/*std::vector<std::pair<std::string, void*>> auxParams;
	auxParams.push_back(std::pair<std::string, void*>("triImg", &triImgL));
	EvaluateDisparity(rootFolder, dispL, 0.5f, auxParams);*/



	// Step 3 - Solve the splitting problem by binary graph cut.
	// E(S) = \sum_p E_p(s_p) + \sum_pq E_pq(s_p, s_q).
	int numPublicVerticesL = vertexCoordsL.size();
	std::vector<std::vector<float>> dispSets(numPublicVerticesL);
	for (int id = 0; id < numTrianglesL; id++) {
		for (int j = 0; j < 3; j++) {
			int publicVertexIdx = triVertexIndsL[id][j];
			cv::Point2d &p = vertexCoordsL[publicVertexIdx];
			dispSets[publicVertexIdx].push_back(slantedPlanesL[id].ToDisparity(p.y - 0.5, p.x - 0.5));
		}
	}

	extern float POSTALIGN_TAU1;
	extern float POSTALIGN_TAU2;

	std::vector<float> unaryCosts(2 * numPublicVerticesL);
	for (int id = 0; id < numPublicVerticesL; id++) {
		int groupSize = dispSets[id].size();
		int numPairs = 0;
		float totalCost = 0.f;
		for (int i = 0; i < groupSize; i++) {
			for (int j = i + 1; j < groupSize; j++) {
				float d1 = dispSets[id][i];
				float d2 = dispSets[id][j];
				if (1/*!isnan(d1) && !isnan(d2)*/) {
					totalCost += std::abs(d1 - d2);
					numPairs++;
				}
			}
		}
		/*if (numPairs == 0) {
			printf("GOT YOU !!\n");
		}*/
		numPairs = std::max(1, numPairs);
		unaryCosts[2 * id + 0] = totalCost / numPairs;
		unaryCosts[2 * id + 1] = POSTALIGN_TAU1;
		//printf("%f\n", unaryCosts[2 * id + 0]);
		if (isnan(unaryCosts[2 * id + 0])) {
			printf("BAD!!!!!!!\n");
		}
	}

	float smoothCosts[4] = { 0, POSTALIGN_TAU2, POSTALIGN_TAU2, 0 };

	try {
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(numPublicVerticesL, 2);
		gc->setDataCost(&unaryCosts[0]);
		gc->setSmoothCost(smoothCosts);

		std::vector<std::vector<int>> nbGraph 
			= ConstructNeighboringVertexGraph(numPublicVerticesL, triVertexIndsL);
		for (int id = 0; id < numPublicVerticesL; id++) {
			for (int k = 0; k < nbGraph[id].size(); k++) {
				int nbId = nbGraph[id][k];
				if (id < nbId) {
					gc->setNeighbors(id, nbId);
				}
			}
		}

		printf("Before optimization energy is %f \n", gc->compute_energy());
		bs::Timer::Tic("Alpha Expansion");
		gc->expansion(20000);
		bs::Timer::Toc();
		printf("After  optimization energy is %f \n", gc->compute_energy());

		std::vector<int> labeling(numPublicVerticesL);
		for (int i = 0; i < numPublicVerticesL; i++) {
			labeling[i] = gc->whatLabel(i);
		}
		delete gc;

		int sum = 0;
		for (int i = 0; i < labeling.size(); i++) {
			sum += labeling[i];
		}
		printf("sum = %d\n", sum);
		printf("labeling.size() = %d\n", labeling.size());

		cv::Mat splitImg = DrawSplitMap(numRows, numCols, vertexCoordsL, labeling);
		std::vector<std::pair<std::string, void*>> auxParams;
		auxParams.push_back(std::pair<std::string, void*>("triImg", &splitImg));
		EvaluateDisparity(rootFolder, dispL, 0.5f, auxParams);
		//cv::imshow("splitImg", splitImg);
		//cv::waitKey(0);
	}
	catch (GCException e){
		e.Report();
	}

}

void TestSplittingPostProcess()
{
	ComputeSplittingMap("teddy");
}

