
#include "OpenCVWrapper.h"
#include "ClipperWrapper.h"
#include "clipper.hpp"

float ClipperWrapper::_cScaling = 0;

/*
================================================================================
================================================================================
*/
ClipperWrapper::ClipperWrapper()
{
}

/*
================================================================================
================================================================================
*/
ClipperWrapper::~ClipperWrapper()
{
}


// ROUND
std::vector<std::vector<A2DVector>>  ClipperWrapper::RoundOffsettingP(std::vector<A2DVector> polygon,
	float offsetVal)
{
	float cScaling = ClipperWrapper::_cScaling;
	ClipperLib::ClipperOffset cOffset;
	cOffset.ArcTolerance = 0.25f * cScaling;

	ClipperLib::Path subj;
	ClipperLib::Paths pSol;
	for (int i = 0; i < polygon.size(); i++)
	{
		subj << ClipperLib::IntPoint(polygon[i].x * cScaling, polygon[i].y * cScaling);
	}

	cOffset.AddPath(subj, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
	cOffset.Execute(pSol, offsetVal * cScaling);

	std::vector<std::vector<A2DVector>>  offPolys;

	for (int a = 0; a < pSol.size(); a++)
	{
		std::vector<A2DVector> offPoly;
		for (int b = 0; b < pSol[a].size(); b++)
		{
			A2DVector iPt(pSol[a][b].X, pSol[a][b].Y);
			iPt /= cScaling;
			offPoly.push_back(iPt);
		}
		offPolys.push_back(offPoly);
	}

	return offPolys;
}

// ROUND
std::vector<std::vector<A2DVector>> ClipperWrapper::RoundOffsettingPP(std::vector<std::vector<A2DVector >> polygons, float offsetVal)
{
	float cScaling = ClipperWrapper::_cScaling;
	ClipperLib::ClipperOffset cOffset;
	cOffset.ArcTolerance = 0.25 * cScaling;

	ClipperLib::Paths subjs(polygons.size());
	ClipperLib::Paths pSol;
	for (int i = 0; i < polygons.size(); i++)
	{
		for (int a = 0; a < polygons[i].size(); a++)
		{
			subjs[i] << ClipperLib::IntPoint(polygons[i][a].x * cScaling, polygons[i][a].y * cScaling);
		}
	}

	cOffset.AddPaths(subjs, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
	cOffset.Execute(pSol, offsetVal * cScaling);

	std::vector<std::vector<A2DVector>>  offPolys;

	std::vector<A2DVector> largestPoly;
	float largestArea = -1000;
	for (int a = 0; a < pSol.size(); a++)
	{
		std::vector<A2DVector> offPoly;
		for (int b = 0; b < pSol[a].size(); b++)
		{
			A2DVector iPt(pSol[a][b].X, pSol[a][b].Y);
			iPt /= cScaling;
			offPoly.push_back(iPt);
		}

		OpenCVWrapper cvWrapper;
		float pArea = cvWrapper.GetArea(offPoly); // check area size
		if (pArea > largestArea)
		{
			largestPoly = offPoly;
			largestArea = pArea;
		}
	}
	offPolys.push_back(largestPoly);

	return offPolys;
}