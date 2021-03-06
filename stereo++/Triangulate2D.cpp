#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <stack>
#include <set>
#include <string>
#include <ctime>
#include <iostream>

#include "SLIC/SLIC.h"
#include "poly2tri/poly2tri.h"
#include "StereoAPI.h"
#include "Fade_2D.h"


#ifdef _DEBUG
#pragma comment(lib, "fade2D_vc12_Debug.lib")
#else
#pragma comment(lib, "fade2D_vc12_Release.lib")
#endif


static void traceBoundary(
	cv::Mat& segMap,
	int numSeg,
	cv::Mat& E,						  // OUT: indices of all countour nodes, set to -1 for non-countour node.
	std::vector<cv::Point2f>& X,      // OUT: the set of all contour nodes
	std::vector<std::vector<int>>& I  // OUT: all sequences of contour node indices
	)
{
#define imRef(E, x, y) (E.at<int>(y, x))
	const int HUGE = 99999;

	int width = segMap.cols;
	int height = segMap.rows;
	cv::Mat F = cv::Mat::zeros(height + 1, width + 1, CV_32SC1);
	E = -1 * cv::Mat::ones(height + 1, width + 1, CV_32SC1);

	for (int y = 0; y < height + 1; y++)
	{
		int *rowPxF = F.ptr<int>(y);
		int *rowPxSeg = segMap.ptr<int>(std::min(y, height - 1));
		for (int x = 0; x < width + 1; x++)
		{
			if (y == 0 || y == height || x == 0 || x == width)
			{
				rowPxF[x] = 1;
			}
			else
			{
				int segVal = rowPxSeg[x];
				for (int delta_y = -1; delta_y <= 0; delta_y++)
				{
					for (int delta_x = -1; delta_x <= 0; delta_x++)
					{
						if (imRef(segMap, x + delta_x, y + delta_y) != segVal)
						{
							rowPxF[x] = 1;
							break;
						}
					}
				}
			}
		}
	}


	int c = 0;
	// index all countour nodes
	for (int y = 0; y < height + 1; y++)
	{
		int *rowPxE = E.ptr<int>(y);
		int *rowPxF = F.ptr<int>(y);
		for (int x = 0; x < width + 1; x++)
		{
			if (rowPxF[x] == 1)
			{
				rowPxE[x] = c;
				cv::Point2f pt(x, y);
				X.push_back(pt); c++;
			}
		}
	}

	static int offsetM[4][6] = {
		-1, 0, -1, -1, -1, 0,
		0, 1, -1, 0, 0, 0,
		1, 0, 0, 0, 0, -1,
		0, -1, 0, -1, -1, -1 };
	I.resize(numSeg);
	for (int y = 0; y < height; y++)
	{
		int *rowPxSeg = segMap.ptr<int>(y);
		for (int x = 0; x < width; x++)
		{
			int k = rowPxSeg[x];
			if (I[k].size() == 0)
			{
				//backup F
				//cv::Mat tempF = F.clone();
				std::stack<cv::Point2f> pos;
				//trace boundary_k
				int temp_y = y;
				int temp_x = x;
				I[k].push_back(imRef(E, temp_x, temp_y));

				int curIndex = 2;
				//imRef(tempF, temp_x, temp_y) = 0;
				imRef(F, temp_x, temp_y) = 0;
				pos.push(cv::Point(temp_x, temp_y));

				while (true) {
					temp_y += offsetM[curIndex][0];
					temp_x += offsetM[curIndex][1];
					I[k].push_back(imRef(E, temp_x, temp_y));

					std::vector<int> possibleOffIndex;
					for (int index = 0; index < 4; index++) {
						int val_y = temp_y + offsetM[index][0];
						int val_x = temp_x + offsetM[index][1];
						if (val_y<0 || val_y>height || val_x<0 || val_x>width)
							continue;
						//int fVal = imRef(tempF, val_x, val_y);
						int fVal = imRef(F, val_x, val_y);

						val_y = temp_y + offsetM[index][2];
						val_x = temp_x + offsetM[index][3];

						int kVal1 = -HUGE;  //assign a special value when out of bound
						if (val_y >= 0 && val_y<height && val_x >= 0 && val_x<width)
							kVal1 = imRef(segMap, val_x, val_y);

						val_y = temp_y + offsetM[index][4];
						val_x = temp_x + offsetM[index][5];

						int kVal2 = -HUGE;  //assign a special value when out of bound
						if (val_y >= 0 && val_y<height && val_x >= 0 && val_x<width)
							kVal2 = imRef(segMap, val_x, val_y);

						if (fVal == 1 && k == kVal1 && k != kVal2)
							possibleOffIndex.push_back(index);
					}

					if (possibleOffIndex.size() == 1) {
						//imRef(tempF, temp_x, temp_y) = 0;
						imRef(F, temp_x, temp_y) = 0;
						pos.push(cv::Point(temp_x, temp_y));
						curIndex = possibleOffIndex[0];
					}
					else if (possibleOffIndex.size()>1) {
						int minVal = HUGE;
						int minIndex = -1;
						for (int lIndex = 0; lIndex<(int)possibleOffIndex.size(); lIndex++) {
							int tempVal = (possibleOffIndex.at(lIndex) - curIndex + 4) % 4;
							if (tempVal < minVal) {
								minVal = tempVal;
								minIndex = lIndex;
							}
						}
						curIndex = possibleOffIndex.at(minIndex);
					}
					else break;
				}

				while (!pos.empty()) {
					int y = pos.top().y;
					int x = pos.top().x;
					imRef(F, x, y) = 1;
					pos.pop();
				}
			}

		}
	}
}

static double PointListStraightenCost(std::vector<cv::Point2f>& pointList, int s, int e)
{
	int numNodes = pointList.size();
	cv::Point2f a = pointList[s];
	cv::Point2f b = pointList[e];
	cv::Point2f n = b - a;
	double nL = cv::norm(n);
	if (nL > 1e-6) {
		n *= (1.0 / nL);
	}

	double dist = 0.f;
	for (int i = (s + 1) % numNodes; i != e; i = (i + 1) % numNodes) {
		cv::Point2f& p = pointList[i];
		dist += cv::norm((a - p) - (n.dot(a - p)) * n);
	}
	return dist;
}

static std::vector<int> AddBorderAnchor(std::vector<cv::Point2f>& pointList)
{
	const double maxTol = 2.5;
	std::vector<int> anchorList;

	// check if can be approximated well by a single line
	int numNodes = pointList.size();
	double bestErr = PointListStraightenCost(pointList, 0, numNodes - 1) / numNodes;
	//printf("bestErr = %.2lf\n", bestErr);
	if (bestErr <= maxTol) {
		return anchorList;
	}

	// else approximate by single anchor
	int s = 0, e = numNodes - 1;
	int bestInd = s;
	for (int i = 5; i < numNodes - 5; i++) {  // use 2 to avoid too-short segments
		double err = PointListStraightenCost(pointList, s, i) * 0.5 / (i - s + 1)
			+ PointListStraightenCost(pointList, i, e) * 0.5 / (e - i + 1);
		if (err < bestErr) {
			bestInd = i;
			bestErr = err;
		}
	}
	if (bestInd != s) {
		anchorList.push_back(bestInd);
		return anchorList;
	}

	// else approximate by two anchor
	// do not deal with this case so far.
	return anchorList;
}

static bool IsTwoLineSegmentIntersect(cv::Point2f& A, cv::Point2f& B, cv::Point2f& C, cv::Point2f& D)
{
	// i adopt the cross product approach
	// see http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect

	const double eps = 1e-7;
	cv::Point2f p = A, r = B - A;
	cv::Point2f q = C, s = D - C;
	double rxs = r.cross(s);
	double qmpxr = (q - p).cross(r);
	double u = (q - p).cross(r) / rxs;
	double t = (q - p).cross(s) / rxs;

	// conditioin 1. and 2.
	if (std::abs(rxs) < eps && std::abs(qmpxr) < eps) {
		if ((0 - eps <= (q - p).dot(r) && (q - p).dot(r) <= r.dot(r) + eps)
			|| (0 - eps <= (p - q).dot(s) && (p - q).dot(s) <= s.dot(s) + eps)
			) {
			return true;
		}
		else  {
			return false;
		}
	}
	// condition 3.
	if (std::abs(rxs) < eps && std::abs(qmpxr) >= eps) {
		return false;
	}
	// condition 4.
	if (std::abs(rxs) >= eps
		&& 0 - eps <= t && t <= 1 + eps
		&& 0 - eps <= u && u <= 1 + eps) {
		return true;
	}
	// condition 5.
	return false;
}

static bool IsPolygonValid(std::vector<cv::Point2f>& pointList)
{
	if (pointList.size() <= 2) {
		return false;
	}
	if (pointList.size() == 3) {
		// bad if the three points are colinear
		// by checking whether the area of the triangle is zero
		// S = [Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By)] / 2
		cv::Point2f& A = pointList[0];
		cv::Point2f& B = pointList[1];
		cv::Point2f& C = pointList[2];
		return std::abs(A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) > 1e-6;
	}
	// test if any pair of non-adjacent line segment intersect
	// this is a brute force O(n^2) implementation
	int L = pointList.size();
	for (int i = 0; i < L; i++) {
		for (int j = i + 2; j < L; j++) {
			if (i == 0 && j == L - 1) {
				continue;
			}
			if (IsTwoLineSegmentIntersect(pointList[i], pointList[i + 1], pointList[j], pointList[(j + 1) % L])) {
				return false;
			}
		}
	}
	return true;
}

static cv::Mat TriangulateSLICSegments(cv::Mat& labelMap, int numSeg, int mergeRadius, std::vector<cv::Point2f>& vertexCoords, std::vector<std::vector<int>>& triVertexInds)
{
	/*
	* labelMap:		input vraiable, a label map representing the segmentation
	* numSeg:			input variable, number of segments in the label map
	* vertexCoords:	output variable, the set of all vertices of the output triangles
	* triVertexInds:		output variable, has N element, where N is the number of triangles, each element is a vector
	*					containing exactly three indices pointing to vertexCoords, thus represent a triangle.
	*/

	/////////////////////////////////////////////////////////////////////////
	// step 1 - trace boundary
	/////////////////////////////////////////////////////////////////////////
	int numRows = labelMap.rows;
	int numCols = labelMap.cols;
	cv::Mat XIndMap;
	std::vector<cv::Point2f> X;
	std::vector<std::vector<int>> I;
	int tic = clock();
	traceBoundary(labelMap, numSeg, XIndMap, X, I);
	printf("traceBoundary uses %.2fs\n", (clock() - tic) / 1000.f);
	printf("step 1 finished\n");


	/////////////////////////////////////////////////////////////////////////
	// step 2 - determine anchor nodes
	/////////////////////////////////////////////////////////////////////////
	std::vector<int> ownerCnt(X.size());
	for (int i = 0; i < X.size(); i++) {
		int yc = X[i].y;
		int xc = X[i].x;
		std::set<int> owners;
		for (int x = xc - 1; x <= xc; x++) {
			for (int y = yc - 1; y <= yc; y++) {
				if (0 <= y && y < numRows && 0 <= x && x < numCols) {
					owners.insert(labelMap.at<int>(y, x));
				}
			}
		}
		ownerCnt[i] = owners.size();
	}

	std::vector<bool> isAnchor(X.size());
	for (int i = 0; i < X.size(); i++) {
		int y = X[i].y;
		int x = X[i].x;
		if ((ownerCnt[i] >= 3)
			|| (ownerCnt[i] == 2 && (y == 0 || y == numRows || x == 0 || x == numCols))  // at border
			|| ((y == 0 || y == numRows) && (x == 0 || x == numCols))				     // at the 4 image corners
			) {
			isAnchor[i] = true;
		}
		else {
			isAnchor[i] = false;
		}
	}
	printf("step 2 finished\n");


	/////////////////////////////////////////////////////////////////////////
	// step 2.1 - add possible non-corner node as anchors
	/////////////////////////////////////////////////////////////////////////
#if 1
	int numAnchors = 0;
	for (int i = 0; i < X.size(); i++) {
		numAnchors += isAnchor[i];
	}
	printf("BEFORE numAnchors = %d\n", numAnchors);

	std::vector<bool> isBorderAnchor(X.size());
	for (int i = 0; i < isBorderAnchor.size(); i++) {
		isBorderAnchor[i] = false;
	}
	std::vector<bool> visited(X.size());
	for (int i = 0; i < visited.size(); i++) {
		visited[i] = false;
	}

	for (int id = 0; id < numSeg; id++) {
		if (id == 1624) {
			printf("id = %d\n", id);
		}

		int numNodes = I[id].size();
		const std::vector<int>& cind = I[id];
		std::vector<cv::Point2f> pointList(cind.size());
		for (int i = 0; i < cind.size(); i++) {
			pointList[i] = X[cind[i]];
		}

		for (int s = 0; s < numNodes; s++) {
			if (!isAnchor[cind[s]]) {
				continue;
			}
			int e = (s + 1) % numNodes;
			while (!isAnchor[cind[e]]) {
				e = (e + 1) % numNodes;
			}
			if (s == e) {
				// the polygon contain only a single anchor,
				// we simply skip this case so far.
				continue;
			}

			// check if this border has been processed.
			int i = 0;
			for (i = (s + 1) % numNodes; i != e; i = (i + 1) % numNodes) {
				if (visited[cind[i]])
					break;
			}
			if (i != e) {
				continue;
			}
			int borderLen = (e > s ? e - s + 1 : e + numNodes - s + 1);
			if (borderLen <= 8) {
				continue;
			}

			// border has not been processed, now process.	
			std::vector<cv::Point2f> borderList;
			for (int i = s; i != e; i = (i + 1) % numNodes) {
				borderList.push_back(pointList[i]);
			}

			std::vector<int> anchorInd = AddBorderAnchor(borderList);
			for (int i = 0; i < anchorInd.size(); i++) {
				int ind = (s + anchorInd[i]) % numNodes;
				isAnchor[cind[ind]] = true;
				isBorderAnchor[cind[ind]] = true;
			}

			for (i = s; i != e; i = (i + 1) % numNodes) {
				visited[cind[i]] = true;
			}
			visited[cind[e]] = true;
		}
	}

	numAnchors = 0;
	for (int i = 0; i < X.size(); i++) {
		numAnchors += isAnchor[i];
	}
	printf("AFTER numAnchors = %d\n", numAnchors);
	printf("2-country border anchor localization done.\n");
#endif







	/////////////////////////////////////////////////////////////////////////
	// step 3 - merge very close anchor nodes
	/////////////////////////////////////////////////////////////////////////
	//const int radius = 3;
	const int radius = mergeRadius;
	std::vector<int> redirectId(isAnchor.size());
	for (int i = 0; i < redirectId.size(); i++) {
		redirectId[i] = i;
	}
	for (int i = 0; i < isAnchor.size(); i++) {
		if (isAnchor[i] && redirectId[i] == i) {
			int yc = X[i].y;
			int xc = X[i].x;
			for (int y = yc - radius; y <= yc + radius; y++) {
				for (int x = xc - radius; x <= xc + radius; x++) {
					if ((0 <= y && y <= numRows && 0 <= x && x <= numCols)
						&& (y != yc || x != xc)
						&& (XIndMap.at<int>(y, x) >= 0)
						&& isAnchor[XIndMap.at<int>(y, x)])
					{
						redirectId[XIndMap.at<int>(y, x)] = i;
					}
				}
			}
		}
	}
	printf("step 3 finished\n");


	/////////////////////////////////////////////////////////////////////////
	// step 4 - triangulate each polygon
	/////////////////////////////////////////////////////////////////////////
	std::vector<p2t::Point> P;
	std::vector<p2t::CDT*> cdts;
	std::vector<std::vector<p2t::Point*>> allPolygons;
	P.reserve(8 * X.size());	// 8 times should be enough, but no gaurantee

	for (int id = 0; id < numSeg; id++) {
		//printf("segment id = %d\n", id);
		// obtain the indices of the vertices of the polygon
		std::vector<int> cind = I[id];
		for (int i = 0; i < cind.size(); i++) {
			cind[i] = redirectId[cind[i]];
		}
		// eliminate duplicate vertices caused by merging
		std::vector<int> tmp;
		for (int i = 0; i < cind.size(); i++) {
			if (isAnchor[cind[i]]) {
				if (tmp.empty()) {
					tmp.push_back(cind[i]);
				}
				else {
					int j = 0;
					for (j = 0; j < tmp.size(); j++) {
						if (cind[i] == tmp[j])
							break;
					}
					if (j == tmp.size()) {
						tmp.push_back(cind[i]);
					}
				}
			}
		}
		cind = tmp;
		// retain only simple polygons
		std::vector<cv::Point2f> pointList;
		for (int i = 0; i < cind.size(); i++) {
			pointList.push_back(X[cind[i]]);
		}
		if (!IsPolygonValid(pointList)) {
			continue;
		}

		//if (id == 330) {
		//	for (int i = 0; i < cind.size(); i++) {
		//		printf("%.2lf, %.2lf\n", X[cind[i]].x, X[cind[i]].y);
		//	}
		//}

		// prepare coords of the vertices of the polygon, for triangulation
		std::vector<p2t::Point*> polygon(cind.size());
		for (int i = 0; i < cind.size(); i++) {
			// give each polygon its own vertices in P.
			// if i let different polygons to share vertices, poly2tri will crash.
			// but i do not have time to find out why.
			P.push_back(p2t::Point(X[cind[i]].x, X[cind[i]].y));
			polygon[i] = &P[P.size() - 1];
		}
		p2t::CDT *cdt = new p2t::CDT(polygon);
		cdt->Triangulate();
		cdts.push_back(cdt);
		allPolygons.push_back(polygon);
	}
	printf("step 4 finished\n");



	/////////////////////////////////////////////////////////////////////////
	// step 5 - output triangles
	/////////////////////////////////////////////////////////////////////////
	cv::Mat idMap = -1 * cv::Mat::ones(numRows + 1, numCols + 1, CV_32SC1);
	int curId = -1;
	for (int id = 0; id < cdts.size(); id++) {
		std::vector<p2t::Triangle*> triangles = cdts[id]->GetTriangles();
		for (int i = 0; i < triangles.size(); i++) {
			std::vector<int> tinds;
			for (int j = 0; j < 3; j++) {
				p2t::Point *v = triangles[i]->GetPoint(j);
				int y = v->y;
				int x = v->x;
				if (idMap.at<int>(y, x) < 0) {
					vertexCoords.push_back(cv::Point2f(x, y));
					tinds.push_back(++curId);
					idMap.at<int>(y, x) = curId;
				}
				else {
					tinds.push_back(idMap.at<int>(y, x));
				}
			}
			triVertexInds.push_back(tinds);
		}
	}


	/////////////////////////////////////////////////////////////////////////
	// clean up
	/////////////////////////////////////////////////////////////////////////
	for (int i = 0; i < cdts.size(); i++) {
		delete cdts[i];
	}


	/////////////////////////////////////////////////////////////////////////
	// visualize
	/////////////////////////////////////////////////////////////////////////
	cv::Mat polyImg(numRows, numCols, CV_8UC3);
	cv::Mat triImg(numRows, numCols, CV_8UC3);
	cv::Mat segImg(numRows, numCols, CV_8UC3);
	const cv::Point2f halfOffset(0.5, 0.5);

	// fill polyimg
	for (int i = 0; i < allPolygons.size(); i++) {
		std::vector<p2t::Point*>& polygon = allPolygons[i];
		for (int j = 0; j < polygon.size(); j++) {
			int s = j;
			int t = (j + 1) % polygon.size();
			cv::Point2f ss(polygon[s]->x, polygon[s]->y);
			cv::Point2f tt(polygon[t]->x, polygon[t]->y);
			cv::line(polyImg, ss - halfOffset, tt - halfOffset,
				cv::Scalar(rand() % 255, rand() % 255, rand() % 255, 255), 1, CV_AA);
		}
	}

	// fill triImg
	for (int i = 0; i < triVertexInds.size(); i++) {
		cv::Point2f p0 = vertexCoords[triVertexInds[i][0]];
		cv::Point2f p1 = vertexCoords[triVertexInds[i][1]];
		cv::Point2f p2 = vertexCoords[triVertexInds[i][2]];

		cv::line(triImg, p0 - halfOffset, p1 - halfOffset, cv::Scalar(0, 0, 255, 255), 1, CV_AA);
		cv::line(triImg, p0 - halfOffset, p2 - halfOffset, cv::Scalar(0, 0, 255, 255), 1, CV_AA);
		cv::line(triImg, p1 - halfOffset, p2 - halfOffset, cv::Scalar(0, 0, 255, 255), 1, CV_AA);
	}

	// fill segImg
	std::vector<cv::Vec3b> randColors(numSeg);
	for (int i = 0; i < randColors.size(); i++) {
		randColors[i] = cv::Vec3b(rand() % 255, rand() % 255, rand() % 255);
	}
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			segImg.at<cv::Vec3b>(y, x) = randColors[labelMap.at<int>(y, x)];
		}
	}
	for (int i = 0; i < X.size(); i++) {
		if (isAnchor[i] && !isBorderAnchor[i]) {
			cv::circle(segImg, X[i] - halfOffset, 2, cv::Scalar(0, 0, 255, 0), -1, CV_AA);
		}
		if (isBorderAnchor[i]) {
			cv::circle(segImg, X[i] - halfOffset, 2, cv::Scalar(255, 0, 0, 0), -1, CV_AA);
		}
	}


	cv::Mat canvas;
	cv::hconcat(triImg, polyImg, canvas);
	cv::hconcat(canvas, segImg, canvas);
	/*cv::imshow("canvas", canvas);
	cv::waitKey(0)*/;
	return canvas;
}

int SLICSegmentation(const cv::Mat &img, const int numPreferedRegions, const int compactness, cv::Mat& labelMap, cv::Mat& contourImg)
{
	int numRows = img.rows;
	int numCols = img.cols;


	cv::Mat argb(numRows, numCols, CV_8UC4);
	assert(argb.isContinuous());
	//assert(img.isContinuous());

	int from_to[] = { -1, 0, 0, 3, 1, 2, 2, 1 };
	cv::mixChannels(&img, 1, &argb, 1, from_to, 4);

	int width(numCols), height(numRows), numlabels(0);;
	unsigned int* pbuff = (unsigned int*)argb.data;
	int* klabels = NULL;

	int		k = numPreferedRegions;	// Desired number of superpixels.
	double	m = compactness;		// Compactness factor. use a value ranging from 10 to 40 depending on your needs. Default is 10
	SLIC segment;
	segment.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(pbuff, width, height, klabels, numlabels, k, m);
	segment.DrawContoursAroundSegments(pbuff, klabels, width, height, 0xff0000);

	labelMap.create(numRows, numCols, CV_32SC1);
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			labelMap.at<int>(y, x) = klabels[y * numCols + x];
		}
	}

	contourImg.create(numRows, numCols, CV_8UC3);
	int to_from[] = { 3, 0, 2, 1, 1, 2 };
	cv::mixChannels(&argb, 1, &contourImg, 1, to_from, 3);

	delete[] klabels;
	return numlabels;
}

cv::Mat DrawTriangleImage(int numRows, int numCols, std::vector<cv::Point2f> &vertexCoords, std::vector<std::vector<int>> &triVertexInds)
{
	const cv::Point2f halfOffset(0.5, 0.5);
	cv::Mat triImg = cv::Mat::zeros(numRows, numCols, CV_8UC3);

	triImg.setTo(cv::Vec3b(0, 0, 0));

	for (int i = 0; i < triVertexInds.size(); i++) {
		cv::Point2f p0 = vertexCoords[triVertexInds[i][0]];
		cv::Point2f p1 = vertexCoords[triVertexInds[i][1]];
		cv::Point2f p2 = vertexCoords[triVertexInds[i][2]];

		cv::line(triImg, p0 - halfOffset, p1 - halfOffset, cv::Scalar(0, 0, 255, 255), 1, CV_AA);
		cv::line(triImg, p0 - halfOffset, p2 - halfOffset, cv::Scalar(0, 0, 255, 255), 1, CV_AA);
		cv::line(triImg, p1 - halfOffset, p2 - halfOffset, cv::Scalar(0, 0, 255, 255), 1, CV_AA);
	}

	return triImg;
}

cv::Mat DrawTriangleImage(int numRows, int numCols, std::vector<cv::Point2f> &vertexCoords, std::vector<std::vector<int>> &triVertexInds, cv::Mat &textureImg)
{
	const cv::Point2f halfOffset(0.5, 0.5);
	cv::Mat triImg = textureImg.clone();

	for (int i = 0; i < triVertexInds.size(); i++) {
		cv::Point2f p0 = vertexCoords[triVertexInds[i][0]];
		cv::Point2f p1 = vertexCoords[triVertexInds[i][1]];
		cv::Point2f p2 = vertexCoords[triVertexInds[i][2]];

		cv::line(triImg, p0 - halfOffset, p1 - halfOffset, cv::Scalar(0, 0, 255, 255), 1, CV_AA);
		cv::line(triImg, p0 - halfOffset, p2 - halfOffset, cv::Scalar(0, 0, 255, 255), 1, CV_AA);
		cv::line(triImg, p1 - halfOffset, p2 - halfOffset, cv::Scalar(0, 0, 255, 255), 1, CV_AA);
	}

	return triImg;
}

void Triangulate2DImage(cv::Mat& img, std::vector<cv::Point2f> &vertexCoords, std::vector<std::vector<int>> &triVertexInds)
{
	int numRows = img.rows, numCols = img.cols;
	const int segLen = 10;
	const int numPreferedRegions = (numRows * numCols) / (segLen * segLen);
	const int compactness = 25;
	const int mergeRadius = segLen * 0.375;
	//const int mergeRadius = 3;

	cv::Mat labelMap, contourImg;
	int numSeg = SLICSegmentation(img, numPreferedRegions, compactness, labelMap, contourImg);
	cv::Mat canvas = TriangulateSLICSegments(labelMap, numSeg, mergeRadius, vertexCoords, triVertexInds);
	//cv::imshow("canvas", canvas);
	//cv::waitKey(0);
}

void TestTriangulation2D()
{
	std::string folder = "Baby1";
	std::string folderPath = "d:/data/stereo/" + folder;

	cv::Mat imL = cv::imread(folderPath + "/im2.png");
	cv::Mat imR = cv::imread(folderPath + "/im6.png");

	int numRows = imL.rows, numCols = imL.cols;
	const int segLen = 8;
	const int numPreferedRegions = (numRows * numCols) / (segLen * segLen);
	const int compactness = 30;
	const int mergeRadius = segLen * 0.375;

	cv::Mat labelMap, contourImg;
	std::vector<cv::Point2f> vertexCoords;
	std::vector<std::vector<int>> triVertexInds;

	int numSeg = SLICSegmentation(imL, numPreferedRegions, compactness, labelMap, contourImg);
	cv::Mat canvas = TriangulateSLICSegments(labelMap, numSeg, mergeRadius, vertexCoords, triVertexInds);

	cv::hconcat(canvas, contourImg, canvas);
	cv::imshow("canvas", canvas);
	cv::waitKey(0);
}

std::string type2str(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}


static void UpdateClosestAnchorInfo(cv::Mat &closestAnchorDist, cv::Mat &closestAnchorLocation, 
	int yc, int xc, int scanType = 0)
{
	const std::vector<cv::Point2f> offsetsL = { 
		cv::Point2f(-1, -1), cv::Point2f(0, -1), cv::Point2f(+1, -1), 
		cv::Point2f(-1, 0) };
	const std::vector<cv::Point2f> offsetsR = {
		cv::Point2f(-1, -1), cv::Point2f(0, -1), cv::Point2f(+1, -1),
		cv::Point2f(+1, 0) };

	const std::vector<cv::Point2f> &offsets = (scanType == 0 ? offsetsL : offsetsR);


	int numRows = closestAnchorDist.rows, numCols = closestAnchorDist.cols;
	float bestDist = FLT_MAX;
	cv::Point2f bestAnchor = cv::Point2f(xc, yc);

	for (int i = 0; i < offsets.size(); i++) {
		int y = yc + offsets[i].y;
		int x = xc + offsets[i].x;
		if (InBound(y, x, numRows, numCols)) {
			cv::Point2f candidateAnchor = closestAnchorLocation.at<cv::Vec2f>(y, x);
			float dist = (candidateAnchor.x - xc) * (candidateAnchor.x - xc)
				+ (candidateAnchor.y - yc) * (candidateAnchor.y - yc);
			if (dist < bestDist) {
				bestDist = dist;
				bestAnchor = candidateAnchor;
			}
		}
	}

	closestAnchorDist.at<float>(yc, xc) = bestDist;
	closestAnchorLocation.at<cv::Vec2f>(yc, xc) = bestAnchor;
}

static float DistanceToClosestAnchors(cv::Point2f &p, std::vector<cv::Point2f> &vertexCoordList)
{
	float minDist = FLT_MAX;
	for (int i = 0; i < vertexCoordList.size(); i++) {
		cv::Point2f &q = vertexCoordList[i];
		float dist = (p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y);
		minDist = std::min(dist, minDist);
	}
	return minDist;
}

static cv::Point2f SubpixelGradientShift(cv::Mat &blurredGrayImg, int xc, int yc)
{
	//if (yc == 9 && xc == 232) {
	//	printf("here.\n");
	//	printf("%s\n", type2str(blurredGrayImg.type()).c_str());
	//}
	int bestY = yc, bestX = xc;		// that is, upper-left corner of the pixel (xc, yc)
	float bestDist = -1;
	for (int dy = -1; dy <= +1; dy += 2) {
		for (int dx = -1; dx <= +1; dx += 2) {
			int y = yc + dy;
			int x = xc + dx;
			if (InBound(y, x, blurredGrayImg.rows, blurredGrayImg.cols)) {
				float dist = std::abs(blurredGrayImg.at<unsigned char>(y, x) - blurredGrayImg.at<unsigned char>(yc, xc));
				if (dist > bestDist) {
					bestDist = dist;
					bestX = xc + (dx + 1) / 2;	// -1->0, +1->1
					bestY = yc + (dy + 1) / 2;	// -1->0, +1->1
				}
			}
		}
	}
	return cv::Point2f(bestX, bestY);
}

void ImageDomainTessellation(cv::Mat &img, std::vector<cv::Point2f> &vertexCoordList,
	std::vector<std::vector<int>> &triVertexIndsList)
{
	const float Pmin	= 9;
	const float delta	= 10;
	int numRows = img.rows, numCols = img.cols;
	

	// Step 1 - detect canny edges on three channels seperately and union them
	//        - spread the initial anchors along edges with min distance P_min
	std::vector<cv::Mat> channels;
	cv::split(img, channels);

	cv::Mat edgeMaps[3];
	for (int i = 0; i < 3; i++) {
		cv::blur(channels[i], channels[i], cv::Size(3, 3));
		cv::Canny(channels[i], edgeMaps[i], 100, 200);
		std::cout << type2str(edgeMaps[i].type()) << "\n";
	}

	//for (int i = 0; i < 3; i++) {
	//	cv::imshow("edges", edgeMaps[i]);
	//	cv::waitKey(0);
	//}
	
	cv::bitwise_or(edgeMaps[0], edgeMaps[1], edgeMaps[1]);
	cv::bitwise_or(edgeMaps[1], edgeMaps[2], edgeMaps[2]);
	cv::Mat finalEdgeMap = edgeMaps[2].clone();
	//cv::imshow("final edge", edgeMaps[2]);
	//cv::waitKey(0);
	
	 

	vertexCoordList.clear();
	vertexCoordList.push_back(cv::Point2f(0, 0));
	vertexCoordList.push_back(cv::Point2f(0, numRows));
	vertexCoordList.push_back(cv::Point2f(numCols, 0));
	vertexCoordList.push_back(cv::Point2f(numCols, numRows));


	// add anchors along image borders
	for (int x = 0; x < numCols; x++) {
		float dist = DistanceToClosestAnchors(cv::Point2f(x, 0), vertexCoordList);
		if (dist > Pmin * Pmin) {
			vertexCoordList.push_back(cv::Point2f(x, 0));
		}
		dist = DistanceToClosestAnchors(cv::Point2f(x, numRows), vertexCoordList);
		if (dist > Pmin * Pmin) {
			vertexCoordList.push_back(cv::Point2f(x, numRows));
		}
	}
	for (int y = 0; y < numRows; y++) {
		float dist = DistanceToClosestAnchors(cv::Point2f(0, y), vertexCoordList);
		if (dist > Pmin * Pmin) {
			vertexCoordList.push_back(cv::Point2f(0, y));
		}
		dist = DistanceToClosestAnchors(cv::Point2f(numCols, y), vertexCoordList);
		if (dist > Pmin * Pmin) {
			vertexCoordList.push_back(cv::Point2f(numCols, y));
		}
	}

	
	cv::Mat blurredGrayImg;
	cv::cvtColor(img, blurredGrayImg, CV_BGR2GRAY);
	
	cv::GaussianBlur(blurredGrayImg, blurredGrayImg, cv::Size(0, 0), 1);
	
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			if (finalEdgeMap.at<unsigned char>(y, x) == 255) {
				//printf("22222222222\n");
				float dist = DistanceToClosestAnchors(cv::Point2f(x, y), vertexCoordList);
				if (dist > Pmin * Pmin) {
					// shift the vertex from pixel center to pixel corner according to local gradient.
					//cv::Point2f p(x, y);
					//printf("(y, x) = (%d, %d)\n", y, x);
					cv::Point2f p = SubpixelGradientShift(blurredGrayImg, x, y);
					//printf("22222222222\n");
					vertexCoordList.push_back(p);
				}
			}
		}
	}
	
	

	// Draw anchor image.
	//cv::Mat anchorImg = img.clone();
	//for (int i = 0; i < vertexCoordList.size(); i++) {
	//	cv::Point2f p = vertexCoordList[i];
	//	cv::circle(anchorImg, p, 5, cv::Scalar(0, 0, 255, 0), 1, CV_AA);
	//}
	//cv::imshow("anchorImg", anchorImg);
	//cv::waitKey(0);


	
	// Step 2 - Compute scale map, scan the image to add additional anchors
	double sigmas[6];
	for (int j = 0; j <= 5; j++) {
		sigmas[j] = std::pow(1.9, j);
	}

	cv::Mat blurredImgs[6];
	for (int j = 0; j <= 5; j++) {
		cv::GaussianBlur(img, blurredImgs[j], cv::Size(0, 0), sigmas[j], sigmas[j]);
	}

	cv::Mat scaleMap(numRows, numCols, CV_32FC1);
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			int k = 1;
			for (int j = 1; j <= 5; j++) {
				float L2Dist(const cv::Vec3b &a, const cv::Vec3b &b);
				float diff = L2Dist(blurredImgs[1].at<cv::Vec3b>(y, x), blurredImgs[j].at<cv::Vec3b>(y, x));
				if (diff < delta) {
					k = j;
				}
				/*else {
					break;
				}*/
			}
			scaleMap.at<float>(y, x) = k;
		}
	}
	cv::GaussianBlur(scaleMap, scaleMap, cv::Size(0, 0), 2.0);
	//for (int y = 0; y < numRows; y++) {
	//	for (int x = 0; x < numCols; x++) {
	//		scaleMap.at<float>(y, x) = (int)(scaleMap.at<float>(y, x) + 0.5);
	//	}
	//}

	
	// visualize scale map
	cv::Mat scaleMapImg;
	scaleMap.convertTo(scaleMapImg, CV_8UC1, 25.f);
	//cv::imshow("scaleMap", scaleMapImg);
	//cv::waitKey(0);


	// add additional anchors
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			float dist = DistanceToClosestAnchors(cv::Point2f(x, y), vertexCoordList);
			float thresh = std::pow(1.5, scaleMap.at<float>(y, x) + 6.0);
			if (dist > thresh * thresh) {
				vertexCoordList.push_back(cv::Point2f(x, y));
			}
		}
	}

	cv::Mat anchorImg = img.clone();
	for (int i = 0; i < vertexCoordList.size(); i++) {
		cv::Point2f p = vertexCoordList[i];
		cv::circle(anchorImg, p, 1, cv::Scalar(0, 0, 255, 0), 1, CV_AA);
	}
	//cv::imshow("anchorImg", anchorImg);
	//cv::waitKey(0);

	
	// Step 3 - Delaunay triangulate the anchors and return;
	std::vector<GEOM_FADE2D::Point2> fade2dVertexList(vertexCoordList.size());
	for (int i = 0; i < vertexCoordList.size(); i++) {
		fade2dVertexList[i] = GEOM_FADE2D::Point2(vertexCoordList[i].x, vertexCoordList[i].y);
		fade2dVertexList[i].setCustomIndex(i);
	}

	GEOM_FADE2D::Fade_2D triangulizer;
	std::vector<GEOM_FADE2D::Point2*> vHandles;
	triangulizer.insert(fade2dVertexList, vHandles);

	std::vector<GEOM_FADE2D::Triangle2*> trianglePtrs;
	triangulizer.getTrianglePointers(trianglePtrs);
	printf("trianglePtrs.size() = %d\n", trianglePtrs.size());


	// populate triVertexIndsList
	triVertexIndsList.resize(trianglePtrs.size());
	for (int i = 0; i < trianglePtrs.size(); i++) {
		triVertexIndsList[i].push_back(trianglePtrs[i]->getCorner(0)->getCustomIndex());
		triVertexIndsList[i].push_back(trianglePtrs[i]->getCorner(1)->getCustomIndex());
		triVertexIndsList[i].push_back(trianglePtrs[i]->getCorner(2)->getCustomIndex());
	}


	//cv::Mat triangleImg = img.clone();
	//for (int i = 0; i < trianglePtrs.size(); i++) {
	//	GEOM_FADE2D::Point2 *A = trianglePtrs[i]->getCorner(0);
	//	GEOM_FADE2D::Point2 *B = trianglePtrs[i]->getCorner(1);
	//	GEOM_FADE2D::Point2 *C = trianglePtrs[i]->getCorner(2);
	//	//printf("A = (%.2lf, %.2lf)\n", A->x(), A->y());
	//	//printf("B = (%.2lf, %.2lf)\n", B->x(), B->y());
	//	//printf("C = (%.2lf, %.2lf)\n\n", C->x(), C->y());
	//	cv::Point2f a = cv::Point2f(A->x(), A->y());
	//	cv::Point2f b = cv::Point2f(B->x(), B->y());
	//	cv::Point2f c = cv::Point2f(C->x(), C->y());
	//	cv::line(triangleImg, a, b, cv::Scalar(0, 0, 255), 1, CV_AA);
	//	cv::line(triangleImg, a, c, cv::Scalar(0, 0, 255), 1, CV_AA);
	//	cv::line(triangleImg, b, c, cv::Scalar(0, 0, 255), 1, CV_AA);

	//	printf("A->getCustomIndex() = %d\n", A->getCustomIndex());
	//}
	cv::Mat triangleImg = img.clone();
	for (int i = 0; i < trianglePtrs.size(); i++) {
		
		const cv::Point2f halfOffset(0.5, 0.5);
		cv::Point2f a = halfOffset + vertexCoordList[triVertexIndsList[i][0]];
		cv::Point2f b = halfOffset + vertexCoordList[triVertexIndsList[i][1]];
		cv::Point2f c = halfOffset + vertexCoordList[triVertexIndsList[i][2]];
	
		cv::line(triangleImg, a, b, cv::Scalar(0, 0, 255), 1, CV_AA);
		cv::line(triangleImg, a, c, cv::Scalar(0, 0, 255), 1, CV_AA);
		cv::line(triangleImg, b, c, cv::Scalar(0, 0, 255), 1, CV_AA);
	}




	//cv::imshow("triangleImg", triangleImg);
	//cv::waitKey(0);

	printf("happy!\n");
	

}