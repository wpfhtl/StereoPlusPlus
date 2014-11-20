#if 0
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


const float focalLen = 3740;
const float baselineLen = 160;

GLuint texture[2];







std::vector<cv::Point2f> gTextureCoords;
std::vector<cv::Point3f> gMeshVertexCoords;
std::vector<std::vector<int>> gFacetVetexIndsList;
cv::Mat gTextureImg;

void myinit(void)
{
	


	cv::Mat textureImg = cv::imread("d:/data/midd2/thirdSize/Bowling2/view1.png");
	cv::cvtColor(textureImg, textureImg, CV_BGR2RGB);
	ASSERT(textureImg.isContinuous());
	gTextureImg = textureImg;

	std::vector<cv::Point3f> LoadVectorPoint3f(std::string filePath, std::string mode);
	std::vector<std::vector<int>> LoadVectorVectorInt(std::string filePath, std::string mode);
	/*std::vector<cv::Point3f> meshVertexCoordsXyd = LoadVectorPoint3f("d:/meshVertexCoordsXydL.txt", "r");
	std::vector<std::vector<int>> facetVetexIndsList = LoadVectorVectorInt("d:/facetVetexIndsListL.txt", "r");*/
	std::vector<cv::Point3f> meshVertexCoordsXyd = LoadVectorPoint3f("d:/meshVertexCoordsXydL.txt", "r");
	std::vector<std::vector<int>> facetVetexIndsList = LoadVectorVectorInt("d:/facetVetexIndsListL.txt", "r");


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
	// meshVertexCoords is in (x,y,d) form, project them to (X, Y, Z) form
	//for (int i = 0; i < meshVertexCoords.size(); i++) {
	//	
	//	meshVertexCoords[i].z = std::max(4.f, std::min(79.f, meshVertexCoords[i].z));
	//	meshVertexCoords[i].z += 240;
	//}
	//for (int i = 0; i < meshVertexCoords.size(); i++) {
	//	cv::Point3f &p = meshVertexCoords[i];
	//	float Z = focalLen * baselineLen / p.z;
	//	float X = p.x * Z / focalLen;
	//	float Y = p.y * Z / focalLen;
	//	meshVertexCoords[i] = cv::Point3f(X, Y, Z);
	//}

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
	printf("load success.\n");
	//exit(-1);


	//glClearColor(0.5, 0.5, 0.5, 0.0);
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
	//glBindTexture(GL_TEXTURE_2D, texture[1]);
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	//glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	//glTexImage2D(GL_TEXTURE_2D, 0, 3, checkImageWidth,
	//	checkImageHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, &checkImage[0][0][0]);
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
			glTexCoord2f(tCoord.x, 1-tCoord.y);
			glVertex3f(pCoord.x, pCoord.y, -pCoord.z);
		}
	}

/*	glVertex3f(0, 0.000000, -2233.816406);
	glTexCoord2f(0, 0);
	glVertex3f(0, 400, -2293.338379);
	glTexCoord2f(0, 1);
	glVertex3f(300, 400, -2269.870605);
	glTexCoord2f(1, 1);

		   */

		    

	glEnd();

	if (!frameBufferSaved){
		cv::Mat rendered(gTextureImg.rows, gTextureImg.cols, CV_8UC3);
		ASSERT(rendered.isContinuous());
		
		void *frameBuffer = rendered.data;
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadPixels(0, 0, gTextureImg.cols, gTextureImg.rows, GL_RGB, GL_UNSIGNED_BYTE, frameBuffer);
		//glReadPixels(0, 0, gTextureImg.cols, gTextureImg.rows, GL_RGB, GL_UNSIGNED_BYTE, frameBuffer);
		frameBufferSaved = 1;

		cv::cvtColor(rendered, rendered, CV_RGB2BGR);
		cv::imwrite("d:/rendered.png", rendered);
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
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	//const float Znear = 1200;
	//const float Zfar = 2700;
	const float Znear = 2000;
	const float Zfar = 50000;
	int numRows = gTextureImg.rows, numCols = gTextureImg.cols;
	

	glFrustum(0 * Znear / focalLen, numCols * Znear / focalLen,
		0 * Znear / focalLen, numRows * Znear / focalLen, Znear, Zfar);
	//glFrustum(0 * Znear / focalLen + 10, numCols * Znear / focalLen + 10,
	//	0 * Znear / focalLen, numRows * Znear / focalLen, Znear, Zfar);
	

	//gluPerspective(30, 1.0*(GLfloat)w / (GLfloat)h, 1500, 2700);
	/*gluLookAt(100, 0, 0, 0, 0, -2000, 0, 1, 0);*/
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	cv::Point3d camearaCenter(0, numRows * Znear / focalLen, 0);
	//gluLookAt(camearaCenter.x, camearaCenter.y, 1, camearaCenter.x, camearaCenter.y, camearaCenter.z, 0, 1, 0);

	//gluLookAt(-80, 0, 1, -80, 0, 0, 0, 1, 0);
	gluLookAt(+80, 0, 1, +80, 0, 0, 0, 1, 0);
	//gluLookAt(numCols * Znear / focalLen, numRows * Znear / focalLen, 1,
	//	numCols * Znear / focalLen, numRows * Znear / focalLen, 0,
	//	0, 1, 0);
	//glTranslatef(+100, +100, +0);
	//glTranslatef(0.0, 0.0, -3.6);
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

	int argc = 1;
	char *dummy = "sdsfsdf";
	char **argv = &dummy;
	//glutInit(&argc, argv);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(443, 370);
	//glutInitWindowSize(gTextureImg.cols, gTextureImg.rows);
	glutCreateWindow("Texture Mapping - Programming Techniques");
	
	

	myinit();
	

	glutReshapeFunc(myReshape);
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMainLoop();

	return 0; 

}

#endif