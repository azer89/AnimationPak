
#include "AnElement.h"
#include "SystemParams.h"
#include "A2DRectangle.h"
#include "UtilityFunctions.h"

#include "PoissonGenerator.h"

#include "OpenCVWrapper.h"
#include "TetWrapper.h"

#include <OgreManualObject.h>
#include <OgreMaterialManager.h>
#include <OgreSceneManager.h>
#include <OgreStringConverter.h>
#include <OgreEntity.h>
#include <OgreMeshManager.h>
#include <OgreHardwareVertexBuffer.h>
#include <OgreHardwareIndexBuffer.h>
#include <OgreSubMesh.h>

#include "NANOFLANNWrapper2D.h"


AnElement::AnElement()
{
	this->_tubeObject          = 0;
	this->_sceneNode           = 0;
	this->_numPointPerLayer    = 0;
	this->_numBoundaryPointPerLayer = 0;
	//this->_uniqueMaterial = false;
}

AnElement::~AnElement()
{
	// still can't create proper destructor ???
	// maybe they're automatically deleted???

	_tubeObject = 0;
	_sceneNode = 0;
	_sceneMgr = 0;
	_material.reset();


	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		_insideFlags.push_back(false);
	}
	/*if (_tubeObject)
	{
	if (_tubeObject->getParentSceneNode())
	_tubeObject->getParentSceneNode()->detachObject(_tubeObject);

	_sceneMgr->destroyManualObject(_tubeObject);
	}

	_material.reset();

	if (_sceneNode)
	{
	_sceneNode->removeAndDestroyAllChildren();
	_sceneNode->getParentSceneNode()->removeAndDestroyChild(_sceneNode->getName());
	_sceneNode = 0;
	}*/

}

void AnElement::ScaleXY(float scVal)
{
	for (int a = 0; a < _massList.size(); a++)
	{
		A3DVector pos = _massList[a]._pos;
		_massList[a]._pos = A3DVector(pos._x * scVal, pos._y * scVal, pos._z);
	}

	ResetSpringRestLengths();
	// TODO
	// need to update edges
	/*for (int a = 0; a < _triEdges.size(); a++)
	{
		if (!_triEdges[a]._isLayer2Layer)
		{
			_triEdges[a]._oriDist *= scVal;
			_triEdges[a]._dist *= scVal;
		}
	}*/
}

void AnElement::TranslateXY(float x, float y)
{

	for (int a = 0; a < _massList.size(); a++)
	{
		A3DVector pos = _massList[a]._pos;
		_massList[a]._pos = A3DVector(pos._x + x, pos._y + y, pos._z);
	}

	ResetSpringRestLengths();
}

void AnElement::AdjustEndPosition(A2DVector endPt2D, bool lockEnds)
{
	float zGap = SystemParams::_upscaleFactor / (float)(SystemParams::_num_layer - 1);
	A2DVector startPt = _massList[0]._pos.GetA2DVector();
	A2DVector dirVector = startPt.DirectionTo(endPt2D);
	float xyGap = dirVector.Length() / (float)(SystemParams::_num_layer - 1);
	dirVector = dirVector.Norm();

	for (int a = 0; a < _massList.size(); a++)
	{
		float which_layer = _massList[a]._layer_idx;
		A2DVector moveVector2D = dirVector * (xyGap * which_layer);
		_massList[a]._pos._x += moveVector2D.x;
		_massList[a]._pos._y += moveVector2D.y;
		_massList[a]._pos._z = -(zGap * which_layer);

		if (which_layer == 0 || which_layer == (SystemParams::_num_layer - 1))
		{
			_massList[a]._isDocked = lockEnds;
			_massList[a]._dockPoint = _massList[a]._pos;
		}
	}

	ResetSpringRestLengths();
}

void AnElement::AdjustEnds(A2DVector startPt2D, A2DVector endPt2D, bool lockEnds)
{

	A3DVector startPt(startPt2D.x, startPt2D.y, 0);
	A3DVector endPt(endPt2D.x, endPt2D.y, 0);
	A3DVector dirVector = startPt.DirectionTo(endPt).Norm();
	float ln = startPt.Distance(endPt);
	float gapCounter = ln / (float)(SystemParams::_num_layer - 1);
	
	for (int a = 0; a < _massList.size(); a++)
	{
		int which_layer = _massList[a]._layer_idx;
		A3DVector moveVector = dirVector * (gapCounter * which_layer);
		_massList[a]._pos += moveVector;

		if (which_layer == 0 || which_layer == (SystemParams::_num_layer - 1))
		{
			_massList[a]._isDocked = lockEnds;
			_massList[a]._dockPoint = _massList[a]._pos;
		}
	}

	ResetSpringRestLengths();
}

void AnElement::DrawEdges()
{

}

void AnElement::DrawRandomPoints(std::vector<A2DVector> randomPoints)
{
	float imgScale = 2.0f;
	CVImg img;
	img.CreateColorImage(SystemParams::_upscaleFactor * imgScale);
	img.SetColorImageToWhite();

	for(int a = 0; a < randomPoints.size(); a++)
	{
		OpenCVWrapper cvWrapper;
		cvWrapper.DrawCircle(img._img, randomPoints[a] * imgScale, MyColor(255, 0, 0), imgScale);
	}

	std::stringstream ss;
	ss << SystemParams::_save_folder << "PNG\\random_points_" << std::to_string(_elem_idx) << ".png";
	img.SaveImage(ss.str());


}

float AnElement::GetMaxDistRandomPoints(const std::vector<A2DVector>& randomPoints)
{
	NANOFLANNWrapper2D nn;
	nn.SetPointData(randomPoints);
	nn.CreatePointKDTree();

	float maxDist = 0;
	for (int a = 0; a < randomPoints.size(); a++)
	{
		A2DVector pt1 = randomPoints[a];
		std::vector<A2DVector> pts = nn.GetClosestPoints(pt1, 2);
		float d1 = pt1.Distance(pts[0]);
		float d2 = pt1.Distance(pts[1]);

		float d = d2;
		if (d1 > d2) { d = d1; }

		if (d > maxDist) { maxDist = d; }
	}

	std::cout << "max dist = " << maxDist << "\n";

	return maxDist;
}

void AnElement::Triangularization(int self_idx)
{
	this->_elem_idx = self_idx;

	// define a star
	std::vector<A2DVector> star_points_2d;	
	star_points_2d.push_back(A2DVector(0, 193));
	star_points_2d.push_back(A2DVector(172, 168));
	star_points_2d.push_back(A2DVector(250, 12));
	star_points_2d.push_back(A2DVector(327, 168));
	star_points_2d.push_back(A2DVector(500, 193));
	star_points_2d.push_back(A2DVector(375, 315));
	star_points_2d.push_back(A2DVector(404, 487));
	star_points_2d.push_back(A2DVector(250, 406));
	star_points_2d.push_back(A2DVector(95, 487));
	star_points_2d.push_back(A2DVector(125, 315));
	//star_points.push_back(A3DVector(250, 250, 0)); // should be in end

	// why do we need bounding box?
	A2DRectangle bb = UtilityFunctions::GetBoundingBox(star_points_2d);
	float img_length = bb.witdh;
	if (bb.height > bb.witdh) { img_length = bb.height; }
	A2DVector centerPt = bb.GetCenter();

	// moving to new center
	img_length += 5.0f; // triangulation error without this ?
	A2DVector newCenter = A2DVector((img_length / 2.0f), (img_length / 2.0f));
	star_points_2d = UtilityFunctions::MovePoly(star_points_2d, centerPt, newCenter);

	// random points
	//int boundaryPointNum = 0; // num of boundary points per layer
	std::vector<A2DVector> randomPoints;
	CreateRandomPoints(star_points_2d, img_length, randomPoints, this->_numBoundaryPointPerLayer);
	this->_numPointPerLayer = randomPoints.size(); // ASSIGN
	//this->_numBoundaryPerLayer = boundaryPointNum;
	// randomPoints.size() == num of all points per layer

	// debug delete me
	//DrawRandomPoints(randomPoints);

	// ---------- triangulation ----------
	OpenCVWrapper cvWrapper;
	cvWrapper.Triangulate(_triangles,
						randomPoints,
						star_points_2d,
						img_length);	
	// ---------- triangulation ----------


	float zOffset = -((float)SystemParams::_upscaleFactor) / ((float)SystemParams::_num_layer - 1);

	// --- generate mass ---
	int massCounter = 0;
	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		float zPos = zOffset * a;
		for (int b = 0; b < randomPoints.size(); b++)
		{
			_massList.push_back(AMass(randomPoints[b].x, // x
				randomPoints[b].y,                       // y
				zPos,                                       // z
				massCounter++,                           // self_idx
				_elem_idx,                               // parent_idx
				a));                                     // debug_which_layer
		}
	}
	// ---------- generate mass  ----------

	// ---------- triangle edge springs ----------
	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		float massIdxOffset = a * randomPoints.size();
		for (unsigned int b = 0; b < _triangles.size(); b++)
		{
			int idx0 = _triangles[b].idx0 + massIdxOffset;
			int idx1 = _triangles[b].idx1 + massIdxOffset;
			int idx2 = _triangles[b].idx2 + massIdxOffset;

			TryToAddTriangleEdge(AnIndexedLine(idx0, idx1)); // 0 - 1		
			TryToAddTriangleEdge(AnIndexedLine(idx1, idx2)); // 1 - 2		
			TryToAddTriangleEdge(AnIndexedLine(idx2, idx0)); // 2 - 0
		}
	}
	for (int a = 0; a < SystemParams::_num_layer - 1; a++)
	{
		int massIdxOffset1 = a * randomPoints.size();
		int massIdxOffset2 = massIdxOffset1 + randomPoints.size();
		for (int b = 0; b < randomPoints.size(); b++)
		{
			TryToAddTriangleEdge(AnIndexedLine(b + massIdxOffset1, b + massIdxOffset2, true));
		}
	}
	// ---------- triangle edge springs ----------

	ResetSpringRestLengths();

	// some precomputation
	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		_per_layer_boundary.push_back(std::vector<A2DVector>());
	}for (int a = 0; a < _massList.size(); a++)
	{
		int perLayerIdx = a % _numPointPerLayer;
		if (perLayerIdx < _numBoundaryPointPerLayer)
		{
			int layerIdx = _massList[a]._layer_idx;
			_per_layer_boundary[layerIdx].push_back ( _massList[a]._pos.GetA2DVector() );
		}
	}

	// generate points
	/*for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		float zPos = zOffset * a;
		for (int b = 0; b < randomPoints.size(); b++)
		{
			bool isBoundary = false;
			if (b < boundaryPointNum) { isBoundary = true; }

			_massList.push_back(AMass(randomPoints[b].x, randomPoints[b].y, zPos, _massList.size(), _elem_idx, a));

		}
	}*/

	//float maxDistRandPt = GetMaxDistRandomPoints(randomPoints);

	// CGAL
	//TetWrapper tWrapper;
	//tWrapper.GenerateTet(_massList, maxDistRandPt, _tetEdges);

	std::cout << "done\n";
}

void AnElement::CreateRandomPoints(std::vector<A2DVector> ornamentBoundary,
							float img_length,
							std::vector<A2DVector>& randomPoints,
							int& boundaryPointNum)
{
	// how many points (really weird code...)
	float fVal = img_length / SystemParams::_upscaleFactor;
	fVal *= fVal;
	int numPoints = SystemParams::_sampling_density * fVal;
	float resamplingGap = std::sqrt(float(numPoints)) / float(numPoints) * img_length;

	std::vector<A2DVector> resampledBoundary;

	ornamentBoundary.push_back(ornamentBoundary[0]); // closed sampling
	float rGap = (float)(resamplingGap * SystemParams::_boundary_sampling_factor);
	UtilityFunctions::UniformResample(ornamentBoundary, resampledBoundary, rGap);
	// bug !!! nasty code
	if (resampledBoundary[resampledBoundary.size() - 1].Distance(resampledBoundary[0]) < rGap * 0.5) // r gap
	{
		resampledBoundary.pop_back();
	}
	
	PoissonGenerator::DefaultPRNG PRNG;
	if (SystemParams::_seed > 0)
	{
		PRNG = PoissonGenerator::DefaultPRNG(SystemParams::_seed);
	}
	const auto points = PoissonGenerator::GeneratePoissonPoints(numPoints, PRNG);

	randomPoints.insert(randomPoints.begin(), resampledBoundary.begin(), resampledBoundary.end());
	boundaryPointNum = resampledBoundary.size();

	float sc = img_length * std::sqrt(2.0f);
	float ofVal = 0.5f * (sc - img_length);
	//float ofVal = 0;
	// ---------- iterate points ----------
	for (auto i = points.begin(); i != points.end(); i++)
	{
		float x = (i->x * sc) - ofVal;
		float y = (i->y * sc) - ofVal;
		A2DVector pt(x, y);

		if (UtilityFunctions::InsidePolygon(ornamentBoundary, pt.x, pt.y))
		{
			float d = UtilityFunctions::DistanceToClosedCurve(resampledBoundary, pt);
			if (d > resamplingGap)
			{
				randomPoints.push_back(pt);
			}
			//AVector cPt = knn->GetClosestPoints(pt, 1)[0];
			//if (cPt.Distance(pt) > resamplingGap)
			//	{ randomPoints.push_back(pt); }
		}
	}
}

// visualization
void AnElement::InitMeshOgre3D(Ogre::SceneManager* sceneMgr,
	Ogre::SceneNode* sceneNode,
	const Ogre::String& name,
	const Ogre::String& materialName)
{
	this->_sceneMgr = sceneMgr;
	this->_sceneNode = sceneNode;

	if (_tubeObject) return;

	float rVal = (float)(rand() % 255) / 255.0f;
	float gVal = (float)(rand() % 255) / 255.0f;
	float bVal = (float)(rand() % 255) / 255.0f;

	/*_material = Ogre::MaterialManager::getSingleton().getByName(materialName)->clone(materialName + std::to_string(_self_idx));
	_material->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(rVal, gVal, bVal, 1));

	// clone material
	//_material = Ogre::MaterialManager::getSingleton().getByName(materialName)->clone(materialName + std::to_string(_self_idx));
	_tubeObject = _sceneMgr->createManualObject(name);
	_tubeObject->setDynamic(true);

	UpdateMeshOgre3D();

	if (_sceneNode)
		_sceneNode->attachObject(_tubeObject);
	else
		std::cout << "_sceneNode is null\n";*/

	// material
	Ogre::MaterialPtr line_material = Ogre::MaterialManager::getSingleton().getByName("Examples/RedMat")->clone("ElementLines" + std::to_string(_elem_idx));	
	line_material->getTechnique(0)->getPass(0)->setDiffuse(Ogre::ColourValue(rVal, gVal, bVal, 1));
	

	// Line drawing
	//In the initialization somewhere, create the initial lines object :
	_debug_lines = new DynamicLines(line_material, Ogre::RenderOperation::OT_LINE_LIST);
	_debug_lines->update();
	_debugNode = _sceneMgr->getRootSceneNode()->createChildSceneNode("debug_lines" + std::to_string(_elem_idx));
	_debugNode->attachObject(_debug_lines);

	_spring_lines = new DynamicLines(line_material, Ogre::RenderOperation::OT_LINE_LIST);

	// springs
	//{
	//for (int i = 0; i < _sWorker->_element_list.size(); i++)
	//{
	//AnElement elem = _sWorker->_element_list[i];
	for (int a = 0; a < _triEdges.size(); a++)
	{
		AnIndexedLine ln = _triEdges[a];

		//if (!ln._isLayer2Layer) { continue; }

		A3DVector pt1 = _massList[ln._index0]._pos;
		A3DVector pt2 = _massList[ln._index1]._pos;
		_spring_lines->addPoint(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
		_spring_lines->addPoint(Ogre::Vector3(pt2._x, pt2._y, pt2._z));
	}
	//}
	//}

	//In the initialization somewhere, create the initial lines object :

	//for (int i = 0; i < _spring_points.size(); i++) {
	//	_spring_lines->addPoint(_spring_points[i]);
	//}

	_spring_lines->update();
	_springNode = _sceneMgr->getRootSceneNode()->createChildSceneNode("SpringNode" + std::to_string(_elem_idx));
	_springNode->attachObject(_spring_lines);
}

void AnElement::UpdateSpringDisplayOgre3D()
{
	if (!SystemParams::_show_time_springs) { return; }

	int idx = 0;
	//for (int i = 0; i < _sWorker->_element_list.size(); i++)
	//{
		//AnElement elem = _sWorker->_element_list[i];
		for (int a = 0; a < _triEdges.size(); a++)
		{
			AnIndexedLine ln = _triEdges[a];

			//if (!ln._isLayer2Layer) { continue; }

			A3DVector pt1 = _massList[ln._index0]._pos;
			A3DVector pt2 = _massList[ln._index1]._pos;
			_spring_lines->setPoint(idx++, Ogre::Vector3(pt1._x, pt1._y, pt1._z));
			_spring_lines->setPoint(idx++, Ogre::Vector3(pt2._x, pt2._y, pt2._z));
		}
	//}

	_spring_lines->update();
}


// visualization
void AnElement::UpdateMeshOgre3D()
{
	if (_tubeObject->getDynamic() == true && _tubeObject->getNumSections() > 0)
		_tubeObject->beginUpdate(0);
	else
		_tubeObject->begin(_material->getName());

	for (int a = 0; a < _massList.size(); a++)
	{
		if (a % 11 == 0) continue;

		A3DVector pos = _massList[a]._pos;
		_tubeObject->position(pos._x, pos._y, pos._z);

		// normal doesn't work???
		//A3DVector normVec = A3DVector(250, 250, pos._z).DirectionTo(pos).Norm();
		//_tubeObject->normal(normVec._x, normVec._y, normVec._z);

		// uv
		int curLayer = a / 1;
		int idx = a % 10;
		float u = (float)idx / 10.0;
		float v = (float)curLayer / (float)(5);
		_tubeObject->textureCoord(u, v);
	}

	//std::cout << "_massList size = " << _massList.size() << "\n";
	//std::cout << "render vertex = " << _tubeObject->getCurrentVertexCount() << "\n";

	int indexOffset = 10;
	int A, B, C, D;
	int maxIdx = SystemParams::_num_layer - 1;
	for (int i = 0; i < maxIdx; i++)
	{
		int startIdx = i * indexOffset;
		// 0
		{
			A = startIdx;
			B = A + 1;
			C = A + indexOffset;
			D = B + indexOffset;
			_tubeObject->quad(C, D, B, A);
		}

		// 1
		{
			A = startIdx + 1;
			B = A + 1;
			C = A + indexOffset;
			D = B + indexOffset;
			_tubeObject->quad(C, D, B, A);
		}

		// 2
		{
			A = startIdx + 2;
			B = A + 1;
			C = A + indexOffset;
			D = B + indexOffset;
			_tubeObject->quad(C, D, B, A);
		}

		// 3
		{
			A = startIdx + 3;
			B = A + 1;
			C = A + indexOffset;
			D = B + indexOffset;
			_tubeObject->quad(C, D, B, A);
		}

		// 4
		{
			A = startIdx + 4;
			B = A + 1;
			C = A + indexOffset;
			D = B + indexOffset;
			_tubeObject->quad(C, D, B, A);
		}

		// 5
		{
			A = startIdx + 5;
			B = A + 1;
			C = A + indexOffset;
			D = B + indexOffset;
			_tubeObject->quad(C, D, B, A);
		}

		// 6
		{
			A = startIdx + 6;
			B = A + 1;
			C = A + indexOffset;
			D = B + indexOffset;
			_tubeObject->quad(C, D, B, A);
		}

		// 7
		{
			A = startIdx + 7;
			B = A + 1;
			C = A + indexOffset;
			D = B + indexOffset;
			_tubeObject->quad(C, D, B, A);
		}

		// 8
		{
			A = startIdx + 8;
			B = A + 1;
			C = A + indexOffset;
			D = B + indexOffset;
			_tubeObject->quad(C, D, B, A);
		}

		// 9
		{
			A = startIdx + 9;
			B = startIdx;
			C = A + indexOffset;
			D = B + indexOffset;
			_tubeObject->quad(C, D, B, A);
		}
	}

	_tubeObject->end();

}

/*void AnElement::UpdateSpringLengths()
{
	for (int a = 0; a < _triEdges.size(); a++)
	{
		A3DVector pt1 = _massList[_triEdges[a]._index0]._pos;
		A3DVector pt2 = _massList[_triEdges[a]._index1]._pos;
		float d = pt1.Distance(pt2);
		_triEdges[a]._dist = d;
	}
}*/

void AnElement::RandomizeLayerSize()
{
	float scale1 = (float)((rand() % 100) + 50) / 100.0f;
	float scale2 = (float)((rand() % 100) + 50) / 100.0f;
	float scale3 = (float)((rand() % 100) + 50) / 100.0f;
	float scale4 = (float)((rand() % 100) + 50) / 100.0f;

	int idx2 = SystemParams::_num_layer / 3;
	int idx3 = 2 * SystemParams::_num_layer / 3;

	float athird = SystemParams::_num_layer / 3;


	std::vector<float> randomScale;

	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		float scVal = 1.0f;
		if (a >= 0 && a < idx2)
		{
			float ratioVal = (float)a / athird;
			scVal = (1.0 - ratioVal) * scale1 + ratioVal * scale2;
		}
		else if (a >= idx2 && a < idx3)
		{
			float ratioVal = (float)(a - idx2) / athird;
			scVal = (1.0 - ratioVal) * scale2 + ratioVal * scale3;
		}
		else
		{
			float ratioVal = (float)(a - idx3) / athird;
			scVal = (1.0 - ratioVal) * scale3 + ratioVal * scale4;
		}

		//int randVal = (rand() % 100) + 50;
		//float scaleVal = (float)randVal / 100.0;
		randomScale.push_back(scVal);
	}

	A2DVector ctr(250, 250);
	for (int a = 0; a < _massList.size(); a++)
	{
		A2DVector pos2D = _massList[a]._pos.GetA2DVector();
		pos2D -= ctr;
		
		int lyr = _massList[a]._layer_idx;
		pos2D *= randomScale[lyr];

		pos2D += ctr;

		_massList[a]._pos.SetXY(pos2D.x, pos2D.y);

	}
}

void  AnElement::CreateHelix()
{
	for (int a = 0; a < _massList.size(); a++)
	{
		if (a % 11 == 0) { continue; }
		A2DVector pos(_massList[a]._pos._x, _massList[a]._pos._y);
		int curLayer = _massList[a]._layer_idx;
		float radAngle = (6.28318530718 / (float)SystemParams::_num_layer) * (float)curLayer;
		A2DVector rotPos = UtilityFunctions::Rotate(pos, A2DVector(250, 250), radAngle);
		_massList[a]._pos._x = rotPos.x;
		_massList[a]._pos._y = rotPos.y;
	}
}

void AnElement::UpdateBackend()
{
	//
	//UpdateSpringLengths();
	

	// for closest point
	/*for (int a = 0; a < _massList.size(); a++)
	{
		A2DVector pt(_massList[a]._pos._x, _massList[a]._pos._y);
		int layer_idx = _massList[a]._layer_idx;
		int mass_idx = a % 11;
		_per_layer_points[layer_idx][mass_idx] = pt;
	}*/

	// per layer boundary
	for (int a = 0; a < _massList.size(); a++)
	{
		int perLayerIdx = a % _numPointPerLayer;
		if (perLayerIdx < _numBoundaryPointPerLayer)
		{
			int layerIdx = _massList[a]._layer_idx;
			_per_layer_boundary[layerIdx][perLayerIdx] = _massList[a]._pos.GetA2DVector();
		}
	}
	/*for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		for (int b = 1; b < 11; b++)
		{
			_per_layer_boundary[a][b-1] = _per_layer_points[a][b];
		}
	}*/
}

// back end
void AnElement::CreateStarTube(int self_idx)
{
	// for identification
	_elem_idx = self_idx;

	/*
	center = 250, 250, 0
	0, 193, 0
	172, 168, 0
	250, 12, 0
	327, 168, 0
	500, 193, 0
	375, 315, 0
	404, 487, 0
	250, 406, 0
	95, 487, 0
	125, 315, 0
	*/

	//_massList.push_back(AMass());
	float zPos = 0;
	float zOffset = -(SystemParams::_upscaleFactor / (SystemParams::_num_layer - 1));
	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		int idxGap = a * 11;

		// x y z mass_idx element_idx layer_idx
		_massList.push_back(AMass(250, 250, zPos, 0 + idxGap, _elem_idx, a)); // 0 center
		_massList.push_back(AMass(0,   193, zPos, 1 + idxGap, _elem_idx, a, true)); // 1
		_massList.push_back(AMass(172, 168, zPos, 2 + idxGap, _elem_idx, a, true)); // 2
		_massList.push_back(AMass(250, 12,  zPos, 3 + idxGap, _elem_idx, a, true)); // 3
		_massList.push_back(AMass(327, 168, zPos, 4 + idxGap, _elem_idx, a, true)); // 4
		_massList.push_back(AMass(500, 193, zPos, 5 + idxGap, _elem_idx, a, true)); // 5
		_massList.push_back(AMass(375, 315, zPos, 6 + idxGap, _elem_idx, a, true)); // 6
		_massList.push_back(AMass(404, 487, zPos, 7 + idxGap, _elem_idx, a, true)); // 7
		_massList.push_back(AMass(250, 406, zPos, 8 + idxGap, _elem_idx, a, true)); // 8
		_massList.push_back(AMass(95,  487, zPos, 9 + idxGap, _elem_idx, a, true)); // 9
		_massList.push_back(AMass(125, 315, zPos, 10 + idxGap, _elem_idx, a, true)); // 10

		zPos += zOffset;
	}

	// ???
	//RandomizeLayerSize();
	//CreateHelix();

	//if (createAcrossTube)
	//{
	//	BuildAcrossTube();
	//}
	int idxOffset = 0;
	int offsetGap = 11;
	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		// center to side
		/*_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 1));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 2));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 3));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 4));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 5));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 6));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 7));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 8));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 9));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 10));

																	  // pentagon
		_triEdges.push_back(AnIndexedLine(idxOffset + 10, idxOffset + 2));
		_triEdges.push_back(AnIndexedLine(idxOffset + 2, idxOffset + 4));
		_triEdges.push_back(AnIndexedLine(idxOffset + 4, idxOffset + 6));
		_triEdges.push_back(AnIndexedLine(idxOffset + 6, idxOffset + 8));
		_triEdges.push_back(AnIndexedLine(idxOffset + 8, idxOffset + 10));

																		  // side to side
		_triEdges.push_back(AnIndexedLine(idxOffset + 1, idxOffset + 2));
		_triEdges.push_back(AnIndexedLine(idxOffset + 2, idxOffset + 3));
		_triEdges.push_back(AnIndexedLine(idxOffset + 3, idxOffset + 4));
		_triEdges.push_back(AnIndexedLine(idxOffset + 4, idxOffset + 5));
		_triEdges.push_back(AnIndexedLine(idxOffset + 5, idxOffset + 6));
		_triEdges.push_back(AnIndexedLine(idxOffset + 6, idxOffset + 7));
		_triEdges.push_back(AnIndexedLine(idxOffset + 7, idxOffset + 8));
		_triEdges.push_back(AnIndexedLine(idxOffset + 8, idxOffset + 9));
		_triEdges.push_back(AnIndexedLine(idxOffset + 9, idxOffset + 10));
		_triEdges.push_back(AnIndexedLine(idxOffset + 10, idxOffset + 1));*/

		if (idxOffset > 0)
		{
			int prevOffset = idxOffset - offsetGap;

			// layer to layer
			/*_triEdges.push_back(AnIndexedLine(prevOffset, idxOffset, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 1, idxOffset + 1, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 2, idxOffset + 2, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 3, idxOffset + 3, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 4, idxOffset + 4, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 5, idxOffset + 5, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 6, idxOffset + 6, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 7, idxOffset + 7, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 8, idxOffset + 8, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 9, idxOffset + 9, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 10, idxOffset + 10, true));

			_triEdges.push_back(AnIndexedLine(prevOffset + 1, idxOffset + 2, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 2, idxOffset + 3, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 3, idxOffset + 4, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 4, idxOffset + 5, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 5, idxOffset + 6, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 6, idxOffset + 7, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 7, idxOffset + 8, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 8, idxOffset + 9, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 9, idxOffset + 10, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 10, idxOffset + 1, true));

			_triEdges.push_back(AnIndexedLine(prevOffset + 2, idxOffset + 1, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 3, idxOffset + 2, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 4, idxOffset + 3, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 5, idxOffset + 4, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 6, idxOffset + 5, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 7, idxOffset + 6, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 8, idxOffset + 7, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 9, idxOffset + 8, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 10, idxOffset + 9, true));
			_triEdges.push_back(AnIndexedLine(prevOffset + 1, idxOffset + 10, true));*/
		}

		idxOffset += offsetGap;
	}

	// 
	ResetSpringRestLengths();

	// _per_layer_points
	/*for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		_per_layer_points.push_back(std::vector<A2DVector>());
		_per_layer_boundary.push_back(std::vector<A2DVector>());
	}
	for (int a = 0; a < _massList.size(); a++)
	{
		A2DVector pt(_massList[a]._pos._x, _massList[a]._pos._y);
		int layer_idx = _massList[a]._layer_idx;
		_per_layer_points[layer_idx].push_back(pt);
	}
	// per layer boundary
	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		for (int b = 1; b < 11; b++)
		{
			_per_layer_boundary[a].push_back(_per_layer_points[a][b]);
		}
	}*/
}

void AnElement::ResetSpringRestLengths()
{
	// PLEASE UNCOMMENT ME
	// PLEASE FIX ME

	//for (int a = 0; a < _triEdges.size(); a++)
	//{
	//	/*if(_triEdges[a]._isLayer2Layer)
	//	{
	//		A2DVector p1 = _massList[_triEdges[a]._index0]._pos;
	//		A2DVector p2 = _massList[_triEdges[a]._index1]._pos;
	//		_triEdges[a].SetActualOriDistance(p1.Distance(p2));
	//	}
	//	else*/
	//	{
	//		A2DVector p1 = _massList[_triEdges[a]._index0]._pos.GetA2DVector();
	//		A2DVector p2 = _massList[_triEdges[a]._index1]._pos.GetA2DVector();
	//		_triEdges[a].SetActualOriDistance(p1.Distance(p2));
	//	}
	//}
}

A2DVector AnElement::ClosestPtOnALayer(A2DVector pt, int layer_idx)
{
	float dist = 10000000000000;
	A2DVector closestPt;
	closestPt = UtilityFunctions::GetClosestPtOnClosedCurve(_per_layer_boundary[layer_idx], pt);

	/*for (int a = 0; a < _per_layer_points[layer_idx].size(); a++)
	{
		float d = _per_layer_points[layer_idx][a].Distance(pt);
		if (d < dist)
		{
			dist = d;
			closestPt = _per_layer_points[layer_idx][a];
		}
	}*/

	//closestPt.Print();

	return closestPt;
}

void AnElement::Grow(float growth_scale_iter, float dt)
{
	//for (unsigned int a = 0; a < _triEdges.size(); a++)
	//{
	//	//if (!_triEdges[a]._isLayer2Layer)
	//	{
	//		_triEdges[a].MakeLonger(/* _shrinking_state * */ growth_scale_iter, dt);
	//	}
	//}
}

void AnElement::SolveForSprings()
{
	//float k_edge = SystemParams::_k_edge;

	A2DVector pt0;
	A2DVector pt1;
	A2DVector dir;
	A2DVector eForce;
	float dist = 0;
	float diff = 0;
	float k = 0;

	// PLEASE UNCOMMENT ME
	// PLEASE FIX ME

	for (unsigned int a = 0; a < _triEdges.size(); a++)
	{
		int idx0 = _triEdges[a]._index0;
		int idx1 = _triEdges[a]._index1;

		pt0 = _massList[idx0]._pos.GetA2DVector();
		pt1 = _massList[idx1]._pos.GetA2DVector();
				
		//if (_triEdges[a]._isLayer2Layer)
		/*{
			k = SystemParams::_k_time_edge;
			dist = pt0.Distance(pt1);
			dir = pt0.DirectionTo(pt1).Norm();
			diff = dist - _triEdges[a]._dist;
		}*/
		//else

		if (_triEdges[a]._isLayer2Layer)
		{
			k = SystemParams::_k_time_edge;
		}
		else
		{
			k = SystemParams::_k_edge;
		}

		//{
			
		dir = pt0.DirectionTo(pt1);
		//dir._z = 0;
		dir = dir.Norm();
		dist = pt0.Distance(pt1);
		diff = dist - _triEdges[a]._dist;
		//}
		
		/*float signVal = 1;
		if (diff < 0) { signVal = -1; }
		eForce = (dir * k *  signVal * diff * diff);
		*/

		eForce = dir * k *  diff;

		// 2D
		/*if(_triEdges[a]._isLayer2Layer)
		{
			//eForce._z = 0;
			//eForce = eForce.Norm();
		}*/
		if (!eForce.IsBad())
		{
			_massList[idx0]._edgeForce += A3DVector(eForce.x, eForce.y, 0);	// _massList[idx0]._distToBoundary;
			_massList[idx1]._edgeForce -= A3DVector(eForce.x, eForce.y, 0);	// _massList[idx1]._distToBoundary;
		}
	}
}

void AnElement::UpdatePerLayerBoundaryOgre3D()
{
	//_debug_points.clear();
	_debug_lines->clear();

	float zOffset = -(SystemParams::_upscaleFactor / (SystemParams::_num_layer - 1));
	//int elem_sz = _sWorker->_element_list.size();
	//for (int a = 0; a < elem_sz; a++) // iterate element
	//{
	for (int b = 0; b < SystemParams::_num_layer; b++) // iterate layer
	{
		float zPos = b * zOffset;

		int boundary_sz = _per_layer_boundary[b].size();
		for (int c = 0; c < boundary_sz; c++) // iterate point
		{
			if (c == 0)
			{
				A2DVector pt1 = _per_layer_boundary[b][boundary_sz - 1];
				A2DVector pt2 = _per_layer_boundary[b][c];
				_debug_lines->addPoint(Ogre::Vector3(pt1.x, pt1.y, zPos));
				_debug_lines->addPoint(Ogre::Vector3(pt2.x, pt2.y, zPos));

				continue;
			}

			A2DVector pt1 = _per_layer_boundary[b][c - 1];
			A2DVector pt2 = _per_layer_boundary[b][c];
			_debug_lines->addPoint(Ogre::Vector3(pt1.x, pt1.y, zPos));
			_debug_lines->addPoint(Ogre::Vector3(pt2.x, pt2.y, zPos));
		}

	}
	//}
	_debug_lines->update();
}

void AnElement::UpdateClosestPtsDisplayOgre3D()
{
	//_debug_points.clear();
	_debug_lines->clear();

	for (int b = 0; b < _massList.size(); b++)
	{
		for (int c = 0; c < _massList[b]._closestPt_fill_sz; c++)
		{
			A3DVector pt1 = _massList[b]._pos;
			A2DVector pt22D = _massList[b]._closestPoints[c];
			A3DVector pt2(pt22D.x, pt22D.y, pt1._z);

			_debug_lines->addPoint(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
			_debug_lines->addPoint(Ogre::Vector3(pt2._x, pt2._y, pt2._z));
		}
	}


	//for (int i = 0; i < _debug_points.size(); i++) 
	//{
	//	_debug_lines->addPoint(_debug_points[i]);
	//}

	_debug_lines->update();
}

bool AnElement::TryToAddTriangleEdge(AnIndexedLine anEdge/*, int triIndex*/)
{
	int edgeIndex = FindTriangleEdge(anEdge);
	if (edgeIndex < 0)
	{
		A3DVector pt1 = _massList[anEdge._index0]._pos;
		A3DVector pt2 = _massList[anEdge._index1]._pos;
		float d = pt1.Distance(pt2);
		anEdge.SetDist(d);

		// push to edge list
		_triEdges.push_back(anEdge);

		// UNCOMMENT PERHAPS?
		//// push to edge-to-triangle list
		//std::vector<int> indices;
		//indices.push_back(triIndex);
		//_edgeToTri.push_back(indices);

		return true;
	}

	// UNCOMMENT PERHAPS?
	//// push to edge-to-triangle list
	//_edgeToTri[edgeIndex].push_back(triIndex);

	return false;
}


// triangle edges
int AnElement::FindTriangleEdge(AnIndexedLine anEdge)
{
	for (unsigned int a = 0; a < _triEdges.size(); a++)
	{
		if (_triEdges[a]._index0 == anEdge._index0 &&
			_triEdges[a]._index1 == anEdge._index1)
		{
			return a;
		}

		if (_triEdges[a]._index1 == anEdge._index0 &&
			_triEdges[a]._index0 == anEdge._index1)
		{
			return a;
		}

	}

	return -1;
}

