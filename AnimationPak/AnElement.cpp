
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

	this->_layer_center = A2DVector(250, 250);

	this->_scale = 1.0f;
	this->_maxScale = SystemParams::_element_max_scale;
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

void  AnElement::RotateXY(float radAngle)
{
	for (int a = 0; a < _massList.size(); a++)
	{
		A2DVector pos = _massList[a]._pos.GetA2DVector();
		//_massList[a]._pos = A3DVector(pos._x * scVal, pos._y * scVal, pos._z);
		A2DVector rotPos = UtilityFunctions::Rotate(pos, _layer_center, radAngle);
		_massList[a]._pos._x = rotPos.x;
		_massList[a]._pos._y = rotPos.y;
	}

	ResetSpringRestLengths();
}

void AnElement::ScaleXY(float scVal)
{
	for (int a = 0; a < _massList.size(); a++)
	{
		A3DVector pos = _massList[a]._pos;
		_massList[a]._pos = A3DVector(pos._x * scVal, pos._y * scVal, pos._z);
	}

	ResetSpringRestLengths();
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

void AnElement::CreateDockPoint(A2DVector queryPos, A2DVector lockPos, int layer_idx)
{
	int massListIdx = -1;
	float dist = 100000;
	int l1 = layer_idx * _numPointPerLayer;
	int l2 = l1 + _numBoundaryPointPerLayer;
	for (int a = l1; a < l2; a++)
	{
		//if (_massList[a]._layer_idx == layer_idx)
		//{
		float d = _massList[a]._pos.GetA2DVector().Distance(queryPos);
		if (d < dist)
		{
			dist = d;
			massListIdx = a;
		}
		//}
	}

	if (massListIdx == -1)
	{
		std::cout << "error massListIdx == -1\n";
	}

	_massList[massListIdx]._isDocked = true;
	// you probably want the dockpoint be 2D?
	_massList[massListIdx]._dockPoint = A3DVector(lockPos.x, lockPos.y, _massList[massListIdx]._pos._z);
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

		/*if (which_layer == 0 || which_layer == (SystemParams::_num_layer - 1))
		{
			_massList[a]._isDocked = lockEnds;
			_massList[a]._dockPoint = _massList[a]._pos;
		}*/
		if (which_layer == 0)
		{
			CreateDockPoint(A2DVector(-40, -40), A2DVector(5, 5), 0);
		}
		else if (which_layer == (SystemParams::_num_layer - 1))
		{
			CreateDockPoint(endPt2D + A2DVector(40, 40), endPt2D, (SystemParams::_num_layer - 1));
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
	// ----- element index -----
	this->_elem_idx = self_idx;

	// ----- define a star ----- 
	std::vector<A2DVector> star_points_2d;	

	star_points_2d.push_back(A2DVector(125, 315)); // 10
	star_points_2d.push_back(A2DVector(95, 487)); // 9
	star_points_2d.push_back(A2DVector(250, 406)); // 8
	star_points_2d.push_back(A2DVector(404, 487)); // 7
	star_points_2d.push_back(A2DVector(375, 315)); // 6
	star_points_2d.push_back(A2DVector(500, 193)); // 5
	star_points_2d.push_back(A2DVector(327, 168)); // 4
	star_points_2d.push_back(A2DVector(250, 12)); // 3
	star_points_2d.push_back(A2DVector(172, 168)); // 2
	star_points_2d.push_back(A2DVector(0, 193));   // 1
	
	// -----  why do we need bounding box? ----- 
	A2DRectangle bb = UtilityFunctions::GetBoundingBox(star_points_2d);
	float img_length = bb.witdh;
	if (bb.height > bb.witdh) { img_length = bb.height; }
	A2DVector centerPt = bb.GetCenter();

	// -----  moving to new center ----- 
	img_length += 5.0f; // triangulation error without this ?
	A2DVector newCenter = A2DVector((img_length / 2.0f), (img_length / 2.0f));
	star_points_2d = UtilityFunctions::MovePoly(star_points_2d, centerPt, newCenter);

	// -----  random points ----- 
	std::vector<A2DVector> randomPoints;
	CreateRandomPoints(star_points_2d, img_length, randomPoints, this->_numBoundaryPointPerLayer);
	this->_numPointPerLayer = randomPoints.size(); // ASSIGN

	// debug delete me
	//DrawRandomPoints(randomPoints);

	// -----  triangulation ----- 
	OpenCVWrapper cvWrapper;
	std::vector<AnIdxTriangle> tempTriangles;
	cvWrapper.Triangulate(tempTriangles,
						  randomPoints,
						  star_points_2d,
						  img_length);	
	// duplicate triangles
	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		float massIdxOffset = a * randomPoints.size();
		for (unsigned int b = 0; b < tempTriangles.size(); b++)
		{
			int idx0 = tempTriangles[b].idx0 + massIdxOffset;
			int idx1 = tempTriangles[b].idx1 + massIdxOffset;
			int idx2 = tempTriangles[b].idx2 + massIdxOffset;
			AnIdxTriangle tri(idx0, idx1, idx2);
			_triangles.push_back(tri);
		}
	}
	// -----  triangulation ----- 


	// z axis offset
	float zOffset = -((float)SystemParams::_upscaleFactor) / ((float)SystemParams::_num_layer - 1);

	// -----  generate mass ----- 
	int massCounter = 0;
	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		float zPos = zOffset * a;
		for (int b = 0; b < randomPoints.size(); b++)
		{
			AMass m(randomPoints[b].x, // x
				randomPoints[b].y,                       // y
				zPos,                                       // z
				massCounter++,                           // self_idx
				_elem_idx,                               // parent_idx
				a); // debug_which_layer
			if (b < _numBoundaryPointPerLayer) { m._is_boundary = true; }
			_massList.push_back(m);                                     
		}
	}
	// ----- generate mass ----- 

	// -----  triangle edge springs ----- 
	// intra layer
	/*for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		float massIdxOffset = a * randomPoints.size();
		for (unsigned int b = 0; b < _triangles.size(); b++)
		{
			int idx0 = _triangles[b].idx0 + massIdxOffset;
			int idx1 = _triangles[b].idx1 + massIdxOffset;
			int idx2 = _triangles[b].idx2 + massIdxOffset;

			// TO DO the b index is wrong?
			TryToAddTriangleEdge(AnIndexedLine(idx0, idx1), b); // 0 - 1		
			TryToAddTriangleEdge(AnIndexedLine(idx1, idx2), b); // 1 - 2		
			TryToAddTriangleEdge(AnIndexedLine(idx2, idx0), b); // 2 - 0

			// ----- add triangles to mass -----
			AnIdxTriangle tri(idx0, idx1, idx2);
			_massList[idx0]._triangles.push_back(tri);
			_massList[idx1]._triangles.push_back(tri);
			_massList[idx2]._triangles.push_back(tri);
		}
	}*/
	for (unsigned int a = 0; a < _triangles.size(); a++)
	{
		int idx0 = _triangles[a].idx0;
		int idx1 = _triangles[a].idx1;
		int idx2 = _triangles[a].idx2;

		TryToAddTriangleEdge(AnIndexedLine(idx0, idx1), a); // 0 - 1		
		TryToAddTriangleEdge(AnIndexedLine(idx1, idx2), a); // 1 - 2		
		TryToAddTriangleEdge(AnIndexedLine(idx2, idx0), a); // 2 - 0

		// ----- add triangles to mass -----
		AnIdxTriangle tri(idx0, idx1, idx2);
		_massList[idx0]._triangles.push_back(tri);
		_massList[idx1]._triangles.push_back(tri);
		_massList[idx2]._triangles.push_back(tri);
	}

	// ----- bending springs ----- 
	_auxiliaryEdges = CreateBendingSprings();
	/*std::vector<AnIndexedLine> aux_edge_temp = CreateBendingSprings();
	_auxiliaryEdges.insert(_auxiliaryEdges.end(), aux_edge_temp.begin(), aux_edge_temp.end());
	for (int a = 1; a < SystemParams::_num_layer; a++)
	{
		int layerOffset = a * _numPointPerLayer;
		for (int b = 0; b < aux_edge_temp.size(); b++)
		{
			AnIndexedLine ln = aux_edge_temp[b];
			ln._index0 += layerOffset;
			ln._index1 += layerOffset;
			_auxiliaryEdges.push_back(ln);
		}
	}*/
	// ----- bending springs ----- 

	// 1-1 pattern
	/*for (int a = 0; a < SystemParams::_num_layer - 1; a++)
	{
		int massIdxOffset1 = a * randomPoints.size();
		int massIdxOffset2 = massIdxOffset1 + randomPoints.size();
		for (int b = 0; b < randomPoints.size(); b++)
		{
			TryToAddTriangleEdge(AnIndexedLine(b + massIdxOffset1, b + massIdxOffset2, true));
		}
	}*/
	// cross pattern
	for (int a = 0; a < SystemParams::_num_layer - 1; a++)
	{
		int massIdxOffset1 = a * randomPoints.size();
		int massIdxOffset2 = massIdxOffset1 + randomPoints.size();
		for (int b = 0; b < _numBoundaryPointPerLayer; b++)
		{
			int idxA = b - 1; // prev
			int idxB = b + 1; // next
			if (b == 0)
			{
				idxA = _numBoundaryPointPerLayer - 1;
			}
			else if (b == _numBoundaryPointPerLayer - 1)
			{
				idxB = 0;
			}

			TryToAddTriangleEdge(AnIndexedLine(b + massIdxOffset1, idxA + massIdxOffset2, true), -1);
			TryToAddTriangleEdge(AnIndexedLine(b + massIdxOffset1, idxB + massIdxOffset2, true), -1);
		}
	}
	// -----  triangle edge springs ----- 

	// rotate
	CreateHelix();

	// reset !!!
	ResetSpringRestLengths();

	// ----- some precomputation ----- 
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

}

void AnElement::CalculateRestStructure()
{
	UpdateBackend(); // update per_layer_boundary

	//std::vector<A2DVector> _layer_center_array; // OpenCVWrapper::GetCenter
	OpenCVWrapper cvWrapper;
	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		A2DVector centerPt = cvWrapper.GetCenter(_per_layer_boundary[a]);
		_layer_center_array.push_back(centerPt);
	}

	//std::vector<A3DVector> _rest_mass_pos_array;
	_rest_mass_pos_array.clear();
	for (int a = 0; a < _massList.size(); a++)
	{
		_ori_rest_mass_pos_array.push_back(_massList[a]._pos);
		_rest_mass_pos_array.push_back(_massList[a]._pos);
	}
	
}

void AnElement::Grow(float growth_scale_iter, float dt)
{
	if (_scale > _maxScale)
	{
		return;
	}

	_scale += growth_scale_iter * dt;

	// iterate rest_mass_pos
	for (int a = 0; a < _rest_mass_pos_array.size(); a++)
	{
		int layer_idx = _massList[a]._layer_idx;
		A2DVector pos = _ori_rest_mass_pos_array[a].GetA2DVector();
		pos -= _layer_center_array[layer_idx];		
		pos *= _scale;
		pos += _layer_center_array[layer_idx];
		_rest_mass_pos_array[a]._x = pos.x;
		_rest_mass_pos_array[a]._y = pos.y;
	}

	// iterate edges
	for (unsigned int a = 0; a < _triEdges.size(); a++)
	{
		A2DVector p1 = _rest_mass_pos_array[_triEdges[a]._index0].GetA2DVector();
		A2DVector p2 = _rest_mass_pos_array[_triEdges[a]._index1].GetA2DVector();
		_triEdges[a].SetActualOriDistance(p1.Distance(p2));
	}

	for (unsigned int a = 0; a < _auxiliaryEdges.size(); a++)
	{
		A2DVector p1 = _rest_mass_pos_array[_auxiliaryEdges[a]._index0].GetA2DVector();
		A2DVector p2 = _rest_mass_pos_array[_auxiliaryEdges[a]._index1].GetA2DVector();
		_auxiliaryEdges[a].SetActualOriDistance(p1.Distance(p2));
	}
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

	_color = MyColor(rVal * 255, gVal * 255, bVal * 255);

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
	
	

	// ---------- springs ----------
	/*_spring_lines = new DynamicLines(line_material, Ogre::RenderOperation::OT_LINE_LIST);
	for (int a = 0; a < _triEdges.size(); a++)
	{
		AnIndexedLine ln = _triEdges[a];

		if (ln._isLayer2Layer) { continue; }

		A3DVector pt1 = _massList[ln._index0]._pos;
		A3DVector pt2 = _massList[ln._index1]._pos;
		_spring_lines->addPoint(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
		_spring_lines->addPoint(Ogre::Vector3(pt2._x, pt2._y, pt2._z));
	}/
	for (int a = 0; a < _auxiliaryEdges.size(); a++)
	{
		AnIndexedLine ln = _auxiliaryEdges[a];
		
		A3DVector pt1 = _massList[ln._index0]._pos;
		A3DVector pt2 = _massList[ln._index1]._pos;
		_spring_lines->addPoint(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
		_spring_lines->addPoint(Ogre::Vector3(pt2._x, pt2._y, pt2._z));
	}
	_spring_lines->update();
	_springNode = _sceneMgr->getRootSceneNode()->createChildSceneNode("SpringNode" + std::to_string(_elem_idx));
	_springNode->attachObject(_spring_lines);*/
	// ---------- springs ----------

	// ---------- boundary ----------
	//In the initialization somewhere, create the initial lines object :
	_debug_lines = new DynamicLines(line_material, Ogre::RenderOperation::OT_LINE_LIST);
	for (int l = 0; l < SystemParams::_num_layer; l++)
	{
		int layerOffset = l * _numPointPerLayer;
		for (int b = 0; b < _numBoundaryPointPerLayer; b++)
		{
			int massIdx1 = b + layerOffset;
			int massIdx2 = b + layerOffset + 1;
			if (b == _numBoundaryPointPerLayer - 1)
			{
				massIdx2 = layerOffset;
			}
			A3DVector pt1 = _massList[massIdx1]._pos;
			A3DVector pt2 = _massList[massIdx2]._pos;
			_debug_lines->addPoint(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
			_debug_lines->addPoint(Ogre::Vector3(pt2._x, pt2._y, pt2._z));
		}
	}	
	_debug_lines->update();
	_debugNode = _sceneMgr->getRootSceneNode()->createChildSceneNode("debug_lines" + std::to_string(_elem_idx));
	_debugNode->attachObject(_debug_lines);
	// ---------- boundary ----------
		
}

void AnElement::UpdateBoundaryDisplayOgre3D()
{
	int idx = 0;

	for (int l = 0; l < SystemParams::_num_layer; l++)
	{
		int layerOffset = l * _numPointPerLayer;
		for (int b = 0; b < _numBoundaryPointPerLayer; b++)
		{
			int massIdx1 = b + layerOffset;
			int massIdx2 = b + layerOffset + 1;
			if (b == _numBoundaryPointPerLayer - 1)
			{
				massIdx2 = layerOffset;
			}
			A3DVector pt1 = _massList[massIdx1]._pos;
			A3DVector pt2 = _massList[massIdx2]._pos;
			_debug_lines->setPoint(idx++, Ogre::Vector3(pt1._x, pt1._y, pt1._z));
			_debug_lines->setPoint(idx++, Ogre::Vector3(pt2._x, pt2._y, pt2._z));
		}
	}

	_debug_lines->update();
}

void AnElement::UpdateSpringDisplayOgre3D()
{
	if (!SystemParams::_show_time_springs) { return; }

	int idx = 0;

	/*for (int a = 0; a < _triEdges.size(); a++)
	{
		AnIndexedLine ln = _triEdges[a];

		if (ln._isLayer2Layer) { continue; }

		A3DVector pt1 = _massList[ln._index0]._pos;
		A3DVector pt2 = _massList[ln._index1]._pos;
		_spring_lines->setPoint(idx++, Ogre::Vector3(pt1._x, pt1._y, pt1._z));
		_spring_lines->setPoint(idx++, Ogre::Vector3(pt2._x, pt2._y, pt2._z));
	}*/
	for (int a = 0; a < _auxiliaryEdges.size(); a++)
	{
		AnIndexedLine ln = _auxiliaryEdges[a];


		A3DVector pt1 = _massList[ln._index0]._pos;
		A3DVector pt2 = _massList[ln._index1]._pos;
		_spring_lines->setPoint(idx++, Ogre::Vector3(pt1._x, pt1._y, pt1._z));
		_spring_lines->setPoint(idx++, Ogre::Vector3(pt2._x, pt2._y, pt2._z));
	}

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

	A2DVector ctr = _layer_center;
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
	float ggg = 6.28318530718 * 5;

	int randVal = rand() % 2;
	if (randVal == 0)
	{
		ggg = -ggg;
	}

	for (int a = 0; a < _massList.size(); a++)
	{
		//if (a % 11 == 0) { continue; }
		A2DVector pos(_massList[a]._pos._x, _massList[a]._pos._y);
		int curLayer = _massList[a]._layer_idx;
		float radAngle = (ggg / (float)SystemParams::_num_layer) * (float)curLayer;
		A2DVector rotPos = UtilityFunctions::Rotate(pos, _layer_center, radAngle);
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

	for (int a = 0; a < _triEdges.size(); a++)
	{		
		//{
		A2DVector p1 = _massList[_triEdges[a]._index0]._pos.GetA2DVector();
		A2DVector p2 = _massList[_triEdges[a]._index1]._pos.GetA2DVector();
		_triEdges[a].SetActualOriDistance(p1.Distance(p2));
		//}
		/*A3DVector p1 = _massList[_triEdges[a]._index0]._pos;
		A3DVector p2 = _massList[_triEdges[a]._index1]._pos;
		_triEdges[a].SetActualOriDistance(p1.Distance(p2));*/
	}

	for (int a = 0; a < _auxiliaryEdges.size(); a++)
	{
		//{
		A2DVector p1 = _massList[_auxiliaryEdges[a]._index0]._pos.GetA2DVector();
		A2DVector p2 = _massList[_auxiliaryEdges[a]._index1]._pos.GetA2DVector();
		_auxiliaryEdges[a].SetActualOriDistance(p1.Distance(p2));
	}
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



void AnElement::SolveForSprings2D()
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

			if (_scale > 3.0f)
			{
				k *= 0.25;
			}
		}
		else
		{
			k = SystemParams::_k_edge;

			if (_scale < 3.0f)
			{
				k *= 5;
			}
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

	for (unsigned int a = 0; a < _auxiliaryEdges.size(); a++)
	{
		int idx0 = _auxiliaryEdges[a]._index0;
		int idx1 = _auxiliaryEdges[a]._index1;

		pt0 = _massList[idx0]._pos.GetA2DVector();
		pt1 = _massList[idx1]._pos.GetA2DVector();

		
		k = SystemParams::_k_edge;

		if (_scale < 3.0f)
		{
			k *= 5;
		}

		dir = pt0.DirectionTo(pt1);
		dir = dir.Norm();
		dist = pt0.Distance(pt1);
		diff = dist - _auxiliaryEdges[a]._dist;

		eForce = dir * k *  diff;

		if (!eForce.IsBad())
		{
			_massList[idx0]._edgeForce += A3DVector(eForce.x, eForce.y, 0);	// _massList[idx0]._distToBoundary;
			_massList[idx1]._edgeForce -= A3DVector(eForce.x, eForce.y, 0);	// _massList[idx1]._distToBoundary;
		}
	}
}

void AnElement::SolveForSprings3D()
{
	//float k_edge = SystemParams::_k_edge;

	A3DVector pt0;
	A3DVector pt1;
	A3DVector dir;
	A3DVector eForce;
	float dist = 0;
	float diff = 0;
	float k = 0;

	// PLEASE UNCOMMENT ME
	// PLEASE FIX ME

	for (unsigned int a = 0; a < _triEdges.size(); a++)
	{
		int idx0 = _triEdges[a]._index0;
		int idx1 = _triEdges[a]._index1;

		pt0 = _massList[idx0]._pos;
		pt1 = _massList[idx1]._pos;

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
			_massList[idx0]._edgeForce += eForce;	// _massList[idx0]._distToBoundary;
			_massList[idx1]._edgeForce -= eForce;	// _massList[idx1]._distToBoundary;
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

int AnElement::GetUnsharedVertexIndex(AnIdxTriangle tri, AnIndexedLine edge)
{
	if (tri.idx0 != edge._index0 && tri.idx0 != edge._index1) { return tri.idx0; }

	if (tri.idx1 != edge._index0 && tri.idx1 != edge._index1) { return tri.idx1; }

	if (tri.idx2 != edge._index0 && tri.idx2 != edge._index1) { return tri.idx2; }

	return -1;
}

std::vector<AnIndexedLine> AnElement::CreateBendingSprings()
{
	std::vector<AnIndexedLine> auxiliaryEdges;
	for (unsigned a = 0; a < _edgeToTri.size(); a++)
	{
		if (_edgeToTri[a].size() != 2) { continue; }

		int idx1 = GetUnsharedVertexIndex(_triangles[_edgeToTri[a][0]], _triEdges[a]);
		if (idx1 < 0) { continue; }

		int idx2 = GetUnsharedVertexIndex(_triangles[_edgeToTri[a][1]], _triEdges[a]);
		if (idx2 < 0) { continue; }

		AnIndexedLine anEdge(idx1, idx2);
		A2DVector pt1 = _massList[idx1]._pos.GetA2DVector();
		A2DVector pt2 = _massList[idx2]._pos.GetA2DVector();
		//anEdge._dist = pt1.Distance(pt2);
		float d = pt1.Distance(pt2);
		anEdge.SetDist(d);

		// push to edge list
		auxiliaryEdges.push_back(anEdge);
	}
	return auxiliaryEdges;
}

bool AnElement::TryToAddTriangleEdge(AnIndexedLine anEdge, int triIndex)
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

		// push to edge-to-triangle list
		std::vector<int> indices;
		indices.push_back(triIndex);
		_edgeToTri.push_back(indices);

		return true;
	}

	// UNCOMMENT PERHAPS?
	//// push to edge-to-triangle list
	_edgeToTri[edgeIndex].push_back(triIndex);

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

