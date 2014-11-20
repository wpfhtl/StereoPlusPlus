#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "SlantedPlane.h"
#include "ReleaseAssert.h"


static cv::Vec3f CrossProduct(cv::Vec3f& u, cv::Vec3f& v)
{
	cv::Vec3f w;
	// u /cross v = (u2v3i +u3v1j + u1v2k) - (u3v2i + u1v3j + u2v1k)
	w[0] = u[1] * v[2] - u[2] * v[1];
	w[1] = u[2] * v[0] - u[0] * v[2];
	w[2] = u[0] * v[1] - u[1] * v[0];
	float norm = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
	norm = std::max(0.01f, norm);
	w[0] /= norm;
	w[1] /= norm;
	w[2] /= norm;

	return w;
}

void SaveDisparityToPly(cv::Mat &disp, cv::Mat& img, float maxDisp, 
	std::string workingDir, std::string plyFilePath, cv::Mat &validPixelMap = cv::Mat())
{
	int numRows = img.rows, numCols = img.cols;
	if (validPixelMap.empty()) {
		validPixelMap = cv::Mat(disp.size(), CV_8UC1, cv::Scalar(255));
	}
	int numValidPixels = cv::countNonZero(validPixelMap);

	const float f = 3740;
	const float B = 160;
	const float cutoff = 0;
	float dmin = 270;

	FILE *fid = fopen((workingDir + "/dmin.txt").c_str(), "r");
	if (fid != NULL) {
		fscanf(fid, "%f", &dmin);
		fclose(fid);
	}
	printf("dmin = %f\n", dmin);

	cv::Mat X(disp.size(), CV_32FC1);
	cv::Mat Y(disp.size(), CV_32FC1);
	cv::Mat Z(disp.size(), CV_32FC1);

	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			float d = disp.at<float>(y, x);
			d = std::max(d, cutoff);
			d = std::min(d, (float)maxDisp);
			d += dmin;
			Z.at<float>(y, x) = f * B / d;
			X.at<float>(y, x) = x * Z.at<float>(y, x) / f;
			Y.at<float>(y, x) = y * Z.at<float>(y, x) / f;
		}
	}


	fid = fopen(plyFilePath.c_str(), "w");
	fprintf(fid, "ply\nformat ascii 1.0 \nelement vertex %d \n", numValidPixels);
	fprintf(fid, "property float x \nproperty float y \nproperty float z \nproperty float nx \nproperty float ny \nproperty float nz \n");
	//fprintf(fid, "property uchar diffuse_red \nproperty uchar diffuse_green \nproperty uchar diffuse_blue \n");
	fprintf(fid, "property uchar red \nproperty uchar green \nproperty uchar blue \n");
	/*if (facets.size() > 0) {
		fprintf(fid, "element face %d\nproperty list uchar int vertex_indices\n", facets.size());
	}*/
	fprintf(fid, "end_header \n");

	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {

			if (!validPixelMap.at<unsigned char>(y, x)) {
				continue;
			}

			cv::Vec3b c = img.at<cv::Vec3b>(y, x);
			cv::Vec3f A, B, C;
			A[0] = X.at<float>(y, x);
			A[1] = Y.at<float>(y, x);
			A[2] = Z.at<float>(y, x);
			B = A;
			C = A;
			if (x + 1 < numCols) {
				B[0] = X.at<float>(y, x + 1);
				B[1] = Y.at<float>(y, x + 1);
				B[2] = Z.at<float>(y, x + 1);
			}
			if (y + 1 < numRows) {
				C[0] = X.at<float>(y + 1, x);
				C[1] = Y.at<float>(y + 1, x);
				C[2] = Z.at<float>(y + 1, x);
			}
			cv::Vec3f N = CrossProduct(C - A, B - A);
			fprintf(fid, "%f %f %f %f %f %f %d %d %d\n", A[0], A[1], A[2], N[0], N[1], N[2], c[2], c[1], c[0]);
		}
	}

	//int ind[3];
	//for (int i = 0; i < facets.size(); i++) {
	//	Facet triangle = facets[i];
	//	for (int j = 0; j < 3; j++) {
	//		int y = triangle.vertex[j].y;
	//		int x = triangle.vertex[j].x;
	//		inline float CLAMP(float x, float a, float b);
	//		x = CLAMP(x, 0, numCols - 1);
	//		y = CLAMP(y, 0, numRows - 1);
	//		ind[j] = y * numCols + x;
	//	}
	//	fprintf(fid, "%d %d %d %d\n", 3, ind[2], ind[1], ind[0]);
	//}

	fclose(fid);
}


static cv::Point3d Convert2DCoordDispTo3DCoordDepth(float y, float x, float d,  
	float f, float B, float maxDisp, float dmin,float cutoff)
{
	float X, Y, Z;
	d = std::max(d, cutoff);
	d = std::min(d, (float)maxDisp);
	d += dmin;
	Z = f * B / d;
	X = x * Z / f;
	Y = y * Z / f;
	return cv::Point3d(X, Y, Z);
}

#if 1
void SaveMeshStereoResultToPly(cv::Mat &img, float maxDisp, 
	std::string workingDir, std::string plyFilePath, std::string textureFilePath,
	std::vector<cv::Point2f> &vertexCoords, std::vector<std::vector<int>> &triVertexInds, 
	std::vector<std::vector<SlantedPlane>> &triVertexBestLabels, cv::Mat &splitMap)
{
	//splitMap.setTo(true);

	const float f = 3740;
	const float B = 160;
	const float cutoff = 0;
	float dmin = 270;

	FILE *fid = fopen((workingDir + "/dmin.txt").c_str(), "r");
	if (fid != NULL) {
		fscanf(fid, "%f", &dmin);
		fclose(fid);
	}
	printf("dmin = %f\n", dmin);

	int numRows = img.rows, numCols = img.cols;
	cv::Mat shareDispVal(numRows + 1, numCols + 1, CV_32FC1);
	cv::Mat shareDispCnt(numRows + 1, numCols + 1, CV_32SC1);
	shareDispVal.setTo(0.f);
	shareDispCnt.setTo(0);

	printf("222222222222222\n");
	int numTriangles = triVertexInds.size();
	for (int id = 0; id < numTriangles; id++) {
		for (int j = 0; j < 3; j++) {
			int y = vertexCoords[triVertexInds[id][j]].y;
			int x = vertexCoords[triVertexInds[id][j]].x;
			ASSERT(0 <= y && y <= numRows);
			ASSERT(0 <= x && x <= numCols);
			if (!splitMap.at<bool>(y, x)) {
				shareDispVal.at<float>(y, x) += triVertexBestLabels[id][j].ToDisparity(y - 0.5f, x - 0.5f);
				shareDispCnt.at<int>(y, x) += 1;
			}
		}
	}
	printf("222222222222222\n");
	// Build the 3D vertex list

	cv::Mat elemVertexInds(numRows + 1, numCols + 1, CV_32SC1);
	elemVertexInds.setTo(-1);
	std::vector<cv::Point3d> elemXYZ;
	elemXYZ.reserve(3 * numTriangles);

	printf("222222222222222\n");
	for (int y = 0; y < numRows + 1; y++) {
		for (int x = 0; x < numCols + 1; x++) {
			if (shareDispCnt.at<int>(y, x) > 0) {
				shareDispVal.at<float>(y, x) /= shareDispCnt.at<int>(y, x);
				float d = shareDispVal.at<float>(y, x);
				cv::Point3d p = Convert2DCoordDispTo3DCoordDepth(y - 0.5, x - 0.5, d, f, B, maxDisp, dmin, cutoff);
				elemXYZ.push_back(p);
				elemVertexInds.at<int>(y, x) = elemXYZ.size() - 1;
			}
		}
	}
	printf("222222222222222\n");
	std::vector<std::vector<int>> facetElemVertexInds = triVertexInds;
	for (int id = 0; id < numTriangles; id++) {
		for (int j = 0; j < 3; j++) {
			int y = vertexCoords[triVertexInds[id][j]].y;
			int x = vertexCoords[triVertexInds[id][j]].x;
			
			if (elemVertexInds.at<int>(y, x) > -1) {
				facetElemVertexInds[id][j] = elemVertexInds.at<int>(y, x);
			}
			else {
				float d = triVertexBestLabels[id][j].ToDisparity(y - 0.5, x - 0.5);
				cv::Point3d p = Convert2DCoordDispTo3DCoordDepth(y - 0.5, x - 0.5, d, f, B, maxDisp, dmin, cutoff);
				elemXYZ.push_back(p);
				facetElemVertexInds[id][j] = elemXYZ.size() - 1;
			}
		}
	}

	printf("222222222222222\n");
	int numElementVertices = elemXYZ.size();
	int numElementFacets = triVertexInds.size();

	// Step 1 - print the header
	printf("%s\n", plyFilePath.c_str());
	fid = fopen(plyFilePath.c_str(), "w");
	ASSERT(fid != NULL);
	fprintf(fid,
		"ply\n"
		"format ascii 1.0\n"
		"comment TextureFile %s\n"
		"element vertex %d\n"
		"property float x\n"
		"property float y\n"
		"property float z\n"
		"element face %d\n"
		"property list uchar int vertex_indices\n"
		"property list uchar float texcoord\n"
		"end_header\n",
		textureFilePath.c_str(),
		numElementVertices,
		numElementFacets
		);


	printf("222222222222222\n");
	// Step 2 - print vertices coordinates in 3D
	for (int i = 0; i < numElementVertices; i++) {
		fprintf(fid, "%f %f %f\n", elemXYZ[i].x, elemXYZ[i].y, elemXYZ[i].z);
	}

	printf("222222222222222\n");
	// Step 3 - print each triangle's vertex index and texture coordinates
	cv::Point2f offset = cv::Point2f(0.5, 0.5);
	for (int id = 0; id < numTriangles; id++) {
		cv::Point2f p0 = vertexCoords[triVertexInds[id][0]];
		cv::Point2f p1 = vertexCoords[triVertexInds[id][1]];
		cv::Point2f p2 = vertexCoords[triVertexInds[id][2]];

		p0.x /= numCols;	p0.y = 1.0 - p0.y / numRows;
		p1.x /= numCols;	p1.y = 1.0 - p1.y / numRows;
		p2.x /= numCols;	p2.y = 1.0 - p2.y / numRows;

		int i0 = facetElemVertexInds[id][0];
		int i1 = facetElemVertexInds[id][1];
		int i2 = facetElemVertexInds[id][2];

		fprintf(fid, "3 %d %d %d 6 %f %f %f %f %f %f\n",
			i2, i1, i0, p2.x, p2.y, p1.x, p1.y, p0.x, p0.y);
	}

	fclose(fid);
}
#else 
void SaveMeshStereoResultToPly(cv::Mat &img, float maxDisp,
	std::string workingDir, std::string plyFilePath, std::string textureFilePath,
	std::vector<cv::Point2f> &vertexCoords, std::vector<std::vector<int>> &triVertexInds,
	std::vector<std::vector<SlantedPlane>> &triVertexBestLabels, cv::Mat &splitMap)
{
	splitMap.setTo(1);

	const float f = 3740;
	const float B = 160;
	const float cutoff = 0;
	float dmin = 270;

	FILE *fid = fopen((workingDir + "/dmin.txt").c_str(), "r");
	if (fid != NULL) {
		fscanf(fid, "%f", &dmin);
		fclose(fid);
	}
	printf("dmin = %f\n", dmin);

	
	int numTriangles = triVertexInds.size();
	std::vector<cv::Point3d> elemXYZ;
	elemXYZ.reserve(3 * numTriangles);
	std::vector<std::vector<int>> facetElemVertexInds = triVertexInds;

	for (int id = 0; id < numTriangles; id++) {
		for (int j = 0; j < 3; j++) {
			int y = vertexCoords[triVertexInds[id][j]].y;
			int x = vertexCoords[triVertexInds[id][j]].x;

			float d = triVertexBestLabels[id][0].ToDisparity(y - 0.5, x - 0.5);
			cv::Point3d p = Convert2DCoordDispTo3DCoordDepth(y - 0.5, x - 0.5, d, f, B, maxDisp, dmin, cutoff);
			elemXYZ.push_back(p);
			facetElemVertexInds[id][j] = elemXYZ.size() - 1;
		}
	}

	int numRows = img.rows, numCols = img.cols;
	int numElementVertices = elemXYZ.size();
	int numElementFacets = triVertexInds.size();

	// Step 1 - print the header
	fid = fopen(plyFilePath.c_str(), "w");
	fprintf(fid,
		"ply\n"
		"format ascii 1.0\n"
		"comment TextureFile %s\n"
		"element vertex %d\n"
		"property float x\n"
		"property float y\n"
		"property float z\n"
		"element face %d\n"
		"property list uchar int vertex_indices\n"
		"property list uchar float texcoord\n"
		"end_header\n",
		textureFilePath.c_str(),
		numElementVertices,
		numElementFacets
		);


	// Step 2 - print vertices coordinates in 3D
	for (int i = 0; i < numElementVertices; i++) {
		fprintf(fid, "%f %f %f\n", elemXYZ[i].x, elemXYZ[i].y, elemXYZ[i].z);
	}

	// Step 3 - print each triangle's vertex index and texture coordinates
	cv::Point2f offset = cv::Point2f(0.5, 0.5);
	for (int id = 0; id < numTriangles; id++) {
		cv::Point2f p0 = vertexCoords[triVertexInds[id][0]];
		cv::Point2f p1 = vertexCoords[triVertexInds[id][1]];
		cv::Point2f p2 = vertexCoords[triVertexInds[id][2]];

		p0.x /= numCols;	p0.y = 1.0 - p0.y / numRows;
		p1.x /= numCols;	p1.y = 1.0 - p1.y / numRows;
		p2.x /= numCols;	p2.y = 1.0 - p2.y / numRows;

		int i0 = facetElemVertexInds[id][0];
		int i1 = facetElemVertexInds[id][1];
		int i2 = facetElemVertexInds[id][2];

		/*fprintf(fid, "3 %d %d %d 6 %f %f %f %f %f %f\n",
			i0, i1, i2, p0.x, p0.y, p1.x, p1.y, p2.x, p2.y);*/
		fprintf(fid, "3 %d %d %d 6 %f %f %f %f %f %f\n",
			i2, i1, i0, p2.x, p2.y, p1.x, p1.y, p0.x, p0.y);
	}

	fclose(fid);
}
#endif

void SaveMeshToPly(std::string plyFilePath, int numRows, int numCols, float focalLen, float baselineLen,
	std::vector<cv::Point3f> &meshVertexCoordsXyd, std::vector<std::vector<int>> &facetVetexIndsList, 
	std::string textureFilePath, bool showInstantly = false)
{
	int dmin = 270;
	extern std::string ROOTFOLDER;
	std::string filePathDmin = "D:/data/Midd2/ThirdSize/" + ROOTFOLDER + "/dmin.txt";
	FILE* f = fopen(filePathDmin.c_str(), "r");
	if (f != NULL) {
		fscanf(f, "%d", &dmin);
		fclose(f);
	}
	
	
	printf("meshVertexCoordsXyd.size() = %d\n", meshVertexCoordsXyd.size());

	std::vector<cv::Point3f> meshVertexCoordsXYZ(meshVertexCoordsXyd.size());
	for (int i = 0; i < meshVertexCoordsXyd.size(); i++) {
		float d = meshVertexCoordsXyd[i].z;
		d = std::max(0.f, std::min(79.f, d));
		d += dmin; 
		float Z = focalLen * baselineLen / d;
		float X = meshVertexCoordsXyd[i].x * Z / focalLen;
		float Y = meshVertexCoordsXyd[i].y * Z / focalLen;
		meshVertexCoordsXYZ[i] = cv::Point3f(X, Y, Z);
	}

	int numElementVertices = meshVertexCoordsXYZ.size();
	int numElementFacets = facetVetexIndsList.size();

	// Step 1 - print the header
	printf("saving ply file %s .......\n", plyFilePath.c_str());
	FILE *fid = fopen(plyFilePath.c_str(), "w");
	ASSERT(fid != NULL);
	fprintf(fid,
		"ply\n"
		"format ascii 1.0\n"
		"comment TextureFile %s\n"
		"element vertex %d\n"
		"property float x\n"
		"property float y\n"
		"property float z\n"
		"property uchar red\n"
		"property uchar green\n"
		"property uchar blue\n"
		"element face %d\n"
		"property list uchar int vertex_indices\n"
		"property list uchar float texcoord\n"
		"end_header\n",
		textureFilePath.c_str(),
		numElementVertices,
		numElementFacets
		);

	cv::Mat textureImg = cv::imread(textureFilePath);
	printf("textureFilePath = %s\n", textureFilePath.c_str());
	ASSERT(!textureImg.empty());
	// Step 2 - print vertices coordinates in 3D
	for (int i = 0; i < numElementVertices; i++) {
		int y = meshVertexCoordsXyd[i].y;
		int x = meshVertexCoordsXyd[i].x;
		y = std::max(0, std::min(numRows - 1, y));
		x = std::max(0, std::min(numRows - 1, x));
		cv::Vec3b color = textureImg.at<cv::Vec3b>(y, x);
		fprintf(fid, "%f %f %f %d %d %d\n", 
			meshVertexCoordsXYZ[i].x, meshVertexCoordsXYZ[i].y, meshVertexCoordsXYZ[i].z,
			color[2], color[1], color[0]);
	}

	// Step 3 - print each triangle's vertex index and texture coordinates
	for (int id = 0; id < facetVetexIndsList.size(); id++) {
		cv::Point3f p0 = meshVertexCoordsXyd[facetVetexIndsList[id][0]];
		cv::Point3f p1 = meshVertexCoordsXyd[facetVetexIndsList[id][1]];
		cv::Point3f p2 = meshVertexCoordsXyd[facetVetexIndsList[id][2]];

		p0.x /= numCols;	p0.y = 1.0 - p0.y / numRows;
		p1.x /= numCols;	p1.y = 1.0 - p1.y / numRows;
		p2.x /= numCols;	p2.y = 1.0 - p2.y / numRows;

		int i0 = facetVetexIndsList[id][0];
		int i1 = facetVetexIndsList[id][1];
		int i2 = facetVetexIndsList[id][2];

		fprintf(fid, "3 %d %d %d 6 %f %f %f %f %f %f\n",
			i2, i1, i0, p2.x, p2.y, p1.x, p1.y, p0.x, p0.y);
	}

	fclose(fid);

	if (showInstantly) {
		std::string cmd = "meshlab " + plyFilePath;
		system(cmd.c_str());
	}
}