#if 1
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <GL/glut.h>
#include <stdio.h>
#include <gl\gl.h>
#include <gl\glu.h>
#include <stdlib.h>

#include <vector>
#include <iostream>
#include "ReleaseAssert.h"

//#define ORIGINAL


static bool isLeftFrame = false;
static float alpha = 0.5;
const float focalLen = 3740;
const float baselineLen = 160;

GLuint texture[2];







std::vector<cv::Point2f> gTextureCoords;
std::vector<cv::Point3f> gMeshVertexCoords;
std::vector<std::vector<int>> gFacetVetexIndsList;
cv::Mat gTextureImg;

void myinit(std::string LorR)
{
	cv::Mat textureImg;
	if (LorR == "L") {
		textureImg = cv::imread("d:/data/midd2/thirdSize/Bowling2/view1.png");
	}
	else {
		textureImg = cv::imread("d:/data/midd2/thirdSize/Bowling2/view5.png");
	}
	
	cv::cvtColor(textureImg, textureImg, CV_BGR2RGB);
	ASSERT(textureImg.isContinuous());
	gTextureImg = textureImg;

	std::vector<cv::Point3f> LoadVectorPoint3f(std::string filePath, std::string mode);
	std::vector<std::vector<int>> LoadVectorVectorInt(std::string filePath, std::string mode);
	
	std::vector<cv::Point3f> meshVertexCoordsXyd 
		= LoadVectorPoint3f("d:/meshVertexCoordsXyd" + LorR + ".txt", "r");
	std::vector<std::vector<int>> facetVetexIndsList 
		= LoadVectorVectorInt("d:/facetVetexIndsList" + LorR + ".txt", "r");


	gTextureCoords.resize(meshVertexCoordsXyd.size());
	int numRows = textureImg.rows, numCols = textureImg.cols;
	///////////////////////// flip y /////////////////////////////
	for (int i = 0; i < meshVertexCoordsXyd.size(); i++) {
		meshVertexCoordsXyd[i].y = numRows - meshVertexCoordsXyd[i].y;
	}
	////////////////////////////////////////////////////////////
	for (int i = 0; i < meshVertexCoordsXyd.size(); i++) {
		float x = meshVertexCoordsXyd[i].x;
		float y = meshVertexCoordsXyd[i].y;
		gTextureCoords[i] = cv::Point2f(x / numCols, y / numRows);
	}


	std::vector<cv::Point3f> meshVertexCoords = meshVertexCoordsXyd;
	for (int i = 0; i < meshVertexCoords.size(); i++) {
		cv::Point3f &p = meshVertexCoords[i];
		float Z = focalLen * baselineLen / p.z;
		float X = p.x * Z / focalLen;
		float Y = p.y * Z / focalLen;
		meshVertexCoords[i] = cv::Point3f(X, Y, Z);
	}

	void SaveVectorPoint3f(std::string filePath, std::vector<cv::Point3f> &vertices, std::string mode = "w");
	SaveVectorPoint3f("d:/XYZ.txt", meshVertexCoords, "w");



	gMeshVertexCoords = meshVertexCoords;
	gFacetVetexIndsList = facetVetexIndsList;

	glClearColor(1, 0, 0, 0);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	// Create Texture
	glGenTextures(2, texture);
	glBindTexture(GL_TEXTURE_2D, texture[0]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); //scale linearly when image bigger than texture
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); //scale linearly when image smalled than texture

	glTexImage2D(GL_TEXTURE_2D, 0, 3, textureImg.cols, textureImg.rows, 0,
		GL_RGB, GL_UNSIGNED_BYTE, textureImg.data);

	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	glEnable(GL_TEXTURE_2D);
	glShadeModel(GL_FLAT);
}

void display(void){

	static int frameBufferSaved = 0;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindTexture(GL_TEXTURE_2D, texture[0]);

	glBegin(GL_TRIANGLES);
	for (int id = 0; id < gFacetVetexIndsList.size(); id++) {
		for (int j = 0; j < 3; j++) {
			cv::Point3f pCoord = gMeshVertexCoords[gFacetVetexIndsList[id][j]];
			cv::Point2f tCoord = gTextureCoords[gFacetVetexIndsList[id][j]];
			glTexCoord2f(tCoord.x, 1 - tCoord.y);
			glVertex3f(pCoord.x, pCoord.y, -pCoord.z);
		}
	}
	glEnd();

	if (!frameBufferSaved){
		cv::Mat rendered(gTextureImg.rows, gTextureImg.cols, CV_8UC3);
		ASSERT(rendered.isContinuous());

		void *frameBuffer = rendered.data;
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadPixels(0, 0, gTextureImg.cols, gTextureImg.rows, GL_RGB, GL_UNSIGNED_BYTE, frameBuffer);
		frameBufferSaved = 1;

		cv::cvtColor(rendered, rendered, CV_RGB2BGR);
		if (isLeftFrame) {
			cv::imwrite("d:/renderedL.png", rendered);
		}
		else {
			cv::imwrite("d:/renderedR.png", rendered);
		}
	}

	glutSwapBuffers();
}

void myReshape(int w, int h)
{
	if (!gTextureImg.empty()) {
		w = gTextureImg.cols;
		h = gTextureImg.rows;
	}
	printf("myReshape w = %d,  h = %d\n", w, h);
	const float Znear = 2000;
	const float Zfar = 50000;
	int numRows = gTextureImg.rows, numCols = gTextureImg.cols;



	glViewport(0, 0, w, h);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(0 * Znear / focalLen, numCols * Znear / focalLen,
		0 * Znear / focalLen, numRows * Znear / focalLen, Znear, Zfar);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	cv::Point3d camearaCenter(0, numRows * Znear / focalLen, 0);

	double XShift = (isLeftFrame ? alpha * baselineLen : -(1 - alpha) * baselineLen);
	gluLookAt(XShift, 0, 1, XShift, 0, 0, 0, 1, 0);
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key) {
	case 27: // ¡°esc¡± on keyboard
		exit(0);
		break;

	default: // ¡°a¡± on keyboard
		break;
	}
}

//int main(int argc, char** argv)
int RenderByOpenGL()
{
	alpha = 0.5;

	
	std::string LorR = "L";
	isLeftFrame = true;
	//std::string LorR = "R";
	//isLeftFrame = false;



		int argc = 1;
		char *dummy = "sdsfsdf";
		char **argv = &dummy;
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
		glutInitWindowSize(443, 370);
		glutCreateWindow("Texture Mapping - Programming Techniques");

		myinit(LorR);

		glutReshapeFunc(myReshape);
		glutDisplayFunc(display);
		glutKeyboardFunc(keyboard);
		glutMainLoop();


	
	return 0;
}

#endif





#include "StereoAPI.h"


static unsigned char MedianFilterInWindow(cv::Mat &img, int yc, int xc, int radius)
{
	int numRows = img.rows, numCols = img.cols;
	std::vector<unsigned char> pixels;
	for (int y = yc - radius; y <= yc + radius; y++) {
		for (int x = xc - radius; x <= xc + radius; x++) {
			if (InBound(y, x, numRows, numCols)) {
				pixels.push_back(img.at<unsigned char>(y, x));
			}
		}
	}
	std::sort(pixels.begin(), pixels.end());
	ASSERT(pixels.size() > 0);
	return pixels[pixels.size() / 2];
}

void FillRenderHoles(std::string filePathImageIn, std::string filePathImageOut)
{
	cv::Mat img = cv::imread(filePathImageIn);
	/*cv::Mat imgHsv;
	cv::cvtColor(img, imgHsv, CV_BGR2HSV);*/
	std::vector<cv::Mat> channels;
	cv::split(img, channels);


	int numRows = img.rows, numCols = img.cols;
	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			if (img.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 0, 0)) {
				for (int i = 0; i < 3; i++) {
					channels[i].at<unsigned char>(y, x) = MedianFilterInWindow(channels[i], y, x, 17);
				}
			}
		}
	}


	for (int y = 0; y < numRows; y++) {
		for (int x = 0; x < numCols; x++) {
			cv::Vec3b newColor = cv::Vec3b(
				channels[0].at<unsigned char>(y, x),
				channels[1].at<unsigned char>(y, x),
				channels[2].at<unsigned char>(y, x));
			img.at<cv::Vec3b>(y, x) = newColor;
		}
	}

	cv::imwrite(filePathImageOut, img);
}