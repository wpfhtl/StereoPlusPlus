#pragma once
#ifndef __SLANTEDPLANE_H__
#define __SLANTEDPLANE_H__

#include <vector>
#include <algorithm>


struct SlantedPlane {
	float a, b, c;
	float nx, ny, nz;
	SlantedPlane() {}
	SlantedPlane(float a_, float b_, float c_, float nx_, float ny_, float nz_)
		:a(a_), b(b_), c(c_), nx(nx_), ny(ny_), nz(nz_) {}
	static float ToDisparity(SlantedPlane &p, float y, float x)
	{
		return p.a * x + p.b * y + p.c;
	}
	static SlantedPlane ConstructFromNormalDepthAndCoord(float nx, float ny, float nz, float z, float y, float x)
	{
		SlantedPlane p;
		const float eps = 1e-4;
		p.nx = nx;  p.ny = ny;  p.nz = nz;
		if (std::abs(p.nz) < eps) {
			if (p.nz > 0)	p.nz = +eps;
			else			p.nz = -eps;
		}
		p.a = -p.nx / p.nz;
		p.b = -p.ny / p.nz;
		p.c = (p.nx * x + p.ny * y + p.nz * z) / p.nz;
		return p;
	}
	static SlantedPlane ConstructFromAbc(float a, float b, float c)
	{
		SlantedPlane p;
		p.a = a;  p.b = b;  p.c = c;
		p.nz = std::sqrt(1.f / (1.f + a*a + b*b));
		p.nx = -a * p.nz;
		p.ny = -b * p.nz;
		return p;
	}
	static SlantedPlane ConstructFromOtherView(SlantedPlane &q, int sign)
	{
		// sign = -1: from LEFT view to RIGHT view
		// sign = +1: from RIGHT view to LEFT view
		float a = q.a / (1 + sign * q.a);
		float b = q.b / (1 + sign * q.a);
		float c = q.c / (1 + sign * q.a);
		return ConstructFromAbc(a, b, c);
	}
	static SlantedPlane ConstructFromRandomInit(float y, float x, float maxDisp)
	{
		const int RAND_HALF = RAND_MAX / 2;

		float z = maxDisp * ((double)rand() / RAND_MAX);
		float nx = ((float)rand() - RAND_HALF) / RAND_HALF;
		float ny = ((float)rand() - RAND_HALF) / RAND_HALF;
		float nz = ((float)rand() - RAND_HALF) / RAND_HALF;

		float norm = std::max(1e-4f, sqrt(nx*nx + ny*ny + nz*nz));
		nx /= norm;
		ny /= norm;
		nz /= norm;

		return ConstructFromNormalDepthAndCoord(nx, ny, nz, z, y, x);
	}
	static SlantedPlane ConstructFromRandomPertube(SlantedPlane &perturbCenter, float y, float x, float nRadius, float zRadius)
	{
		const int RAND_HALF = RAND_MAX / 2;

		float nx = perturbCenter.nx + nRadius * (((float)rand() - RAND_HALF) / RAND_HALF);
		float ny = perturbCenter.ny + nRadius * (((float)rand() - RAND_HALF) / RAND_HALF);
		float nz = perturbCenter.nz + nRadius * (((float)rand() - RAND_HALF) / RAND_HALF);

		float norm = std::max(1e-4f, sqrt(nx*nx + ny*ny + nz*nz));
		nx /= norm;
		ny /= norm;
		nz /= norm;

		float z = perturbCenter.ToDisparity(y, x)
			+ zRadius * (((float)rand() - RAND_HALF) / RAND_HALF);

		return ConstructFromNormalDepthAndCoord(nx, ny, nz, z, y, x);
	}
	float ToDisparity(int y, int x)
	{
		return a * x + b * y + c;
	}
	void SelfConstructFromNormalDepthAndCoord(float nx, float ny, float nz, float z, float y, float x)
	{
		*this = ConstructFromNormalDepthAndCoord(nx, ny, nz, z, y, x);
	}
	void SlefConstructFromAbc(float a, float b, float c)
	{
		*this = ConstructFromAbc(a, b, c);
	}
	void SelfConstructFromOtherView(SlantedPlane &q, int sign)
	{
		*this = ConstructFromOtherView(q, sign);
	}
	void SelfConstructFromRandomInit(float y, float x, float maxDisp)
	{
		*this = ConstructFromRandomInit(y, x, maxDisp);
	}
	void SelfConstructFromRandomPertube(SlantedPlane &perturbCenter, float y, float x, float nRadius, float zRadius)
	{
		*this = ConstructFromRandomPertube(perturbCenter, y, x, nRadius, zRadius);
	}
};

#endif