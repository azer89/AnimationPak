
#include "AnElement.h"
#include "SystemParams.h"

#include <OgreManualObject.h>
#include <OgreMaterialManager.h>
#include <OgreSceneManager.h>
#include <OgreStringConverter.h>
#include <OgreEntity.h>
#include <OgreMeshManager.h>
#include <OgreHardwareVertexBuffer.h>
#include <OgreHardwareIndexBuffer.h>
#include <OgreSubMesh.h>

#include "UtilityFunctions.h"


AnElement::AnElement()
{
	this->_tubeObject = 0;
	this->_sceneNode = 0;
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

void AnElement::AdjustEnds(A2DVector startPt2D, A2DVector endPt2D, bool lockEnds)
{

	A3DVector startPt(startPt2D.x, startPt2D.y, 0);
	A3DVector endPt(endPt2D.x, endPt2D.y, 0);
	A3DVector dirVector = startPt.DirectionTo(endPt).Norm();
	float ln = startPt.Distance(endPt);
	float gapCounter = ln / (float)(SystemParams::_num_layer - 1);
	
	for (int a = 0; a < _massList.size(); a++)
	{
		int which_layer = _massList[a]._debug_which_layer;
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

// visualization
void AnElement::InitMesh(Ogre::SceneManager* sceneMgr,
	Ogre::SceneNode* sceneNode,
	const Ogre::String& name,
	const Ogre::String& materialName)
{
	this->_sceneMgr = sceneMgr;
	this->_sceneNode = sceneNode;

	if (_tubeObject) return;

	_material = Ogre::MaterialManager::getSingleton().getByName(materialName);

	_tubeObject = _sceneMgr->createManualObject(name);
	_tubeObject->setDynamic(true);

	UpdateMesh2();

	if (_sceneNode)
		_sceneNode->attachObject(_tubeObject);
	else
		std::cout << "_sceneNode is null\n";
}

// visualization
void AnElement::UpdateMesh2()
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
		
		int lyr = _massList[a]._debug_which_layer;
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
		int curLayer = _massList[a]._debug_which_layer;
		float radAngle = (3.14159265359 / (float)SystemParams::_num_layer) * (float)curLayer;
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
	for (int a = 0; a < _massList.size(); a++)
	{
		A2DVector pt(_massList[a]._pos._x, _massList[a]._pos._y);
		int layer_idx = _massList[a]._debug_which_layer;
		int mass_idx = _massList[a]._self_idx;
		_per_layer_points[layer_idx][mass_idx] = pt;
	}
}

// back end
void AnElement::CreateStarTube(int self_idx)
{
	// for identification
	_self_idx = self_idx;

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
		// x y z mass_idx element_idx layer_idx
		_massList.push_back(AMass(250, 250, zPos, 0, _self_idx, a)); // 0 center
		_massList.push_back(AMass(0, 193, zPos, 1, _self_idx, a)); // 1
		_massList.push_back(AMass(172, 168, zPos, 2, _self_idx, a)); // 2
		_massList.push_back(AMass(250, 12, zPos, 3, _self_idx, a)); // 3
		_massList.push_back(AMass(327, 168, zPos, 4, _self_idx, a)); // 4
		_massList.push_back(AMass(500, 193, zPos, 5, _self_idx, a)); // 5
		_massList.push_back(AMass(375, 315, zPos, 6, _self_idx, a)); // 6
		_massList.push_back(AMass(404, 487, zPos, 7, _self_idx, a)); // 7
		_massList.push_back(AMass(250, 406, zPos, 8, _self_idx, a)); // 8
		_massList.push_back(AMass(95, 487, zPos, 9, _self_idx, a)); // 9
		_massList.push_back(AMass(125, 315, zPos, 10, _self_idx, a)); // 10

		zPos += zOffset;
	}

	// ???
	//RandomizeLayerSize();
	CreateHelix();

	//if (createAcrossTube)
	//{
	//	BuildAcrossTube();
	//}

	// dist
	//float c2s_dist1 = A2DVector(250, 250).Distance(A2DVector(0, 193)); // center to side
	//float c2s_dist2 = A2DVector(250, 250).Distance(A2DVector(172, 168)); // center to side
	//float c2s_dist3 = A2DVector(172, 168).Distance(A2DVector(327, 168)); // center to side
	//float s2s_dist = A2DVector(172, 168).Distance(A2DVector(0, 193)); // side to side
	//float l2l_dist = SystemParams::_upscaleFactor / (float)(SystemParams::_num_layer - 1);

	int idxOffset = 0;
	int offsetGap = 11;
	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		// center to side
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 1));//, c2s_dist1, c2s_dist1));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 2));//, c2s_dist2, c2s_dist2));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 3));//, c2s_dist1, c2s_dist1));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 4));//, c2s_dist2, c2s_dist2));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 5));//, c2s_dist1, c2s_dist1));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 6));//, c2s_dist2, c2s_dist2));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 7));//, c2s_dist1, c2s_dist1));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 8));//, c2s_dist2, c2s_dist2));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 9));//, c2s_dist1, c2s_dist1));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 10));//, c2s_dist2, c2s_dist2));

																	  // pentagon
		_triEdges.push_back(AnIndexedLine(idxOffset + 10, idxOffset + 2));//, c2s_dist3, c2s_dist3));
		_triEdges.push_back(AnIndexedLine(idxOffset + 2, idxOffset + 4));//, c2s_dist3, c2s_dist3));
		_triEdges.push_back(AnIndexedLine(idxOffset + 4, idxOffset + 6));//, c2s_dist3, c2s_dist3));
		_triEdges.push_back(AnIndexedLine(idxOffset + 6, idxOffset + 8));//, c2s_dist3, c2s_dist3));
		_triEdges.push_back(AnIndexedLine(idxOffset + 8, idxOffset + 10));//, c2s_dist3, c2s_dist3));

																		  // side to side
		_triEdges.push_back(AnIndexedLine(idxOffset + 1, idxOffset + 2));//, s2s_dist, s2s_dist));
		_triEdges.push_back(AnIndexedLine(idxOffset + 2, idxOffset + 3));//, s2s_dist, s2s_dist));
		_triEdges.push_back(AnIndexedLine(idxOffset + 3, idxOffset + 4));//, s2s_dist, s2s_dist));
		_triEdges.push_back(AnIndexedLine(idxOffset + 4, idxOffset + 5));//, s2s_dist, s2s_dist));
		_triEdges.push_back(AnIndexedLine(idxOffset + 5, idxOffset + 6));//, s2s_dist, s2s_dist));
		_triEdges.push_back(AnIndexedLine(idxOffset + 6, idxOffset + 7));//, s2s_dist, s2s_dist));
		_triEdges.push_back(AnIndexedLine(idxOffset + 7, idxOffset + 8));//, s2s_dist, s2s_dist));
		_triEdges.push_back(AnIndexedLine(idxOffset + 8, idxOffset + 9));//, s2s_dist, s2s_dist));
		_triEdges.push_back(AnIndexedLine(idxOffset + 9, idxOffset + 10));//, s2s_dist, s2s_dist));
		_triEdges.push_back(AnIndexedLine(idxOffset + 10, idxOffset + 1));//, s2s_dist, s2s_dist));

		if (idxOffset > 0)
		{
			int prevOffset = idxOffset - offsetGap;

			// layer to layer
			_triEdges.push_back(AnIndexedLine(prevOffset, idxOffset, true));//, l2l_dist, l2l_dist, true)); // 0
			_triEdges.push_back(AnIndexedLine(prevOffset + 1, idxOffset + 1, true));//, l2l_dist, l2l_dist, true)); // 1
			_triEdges.push_back(AnIndexedLine(prevOffset + 2, idxOffset + 2, true));//, l2l_dist, l2l_dist, true)); // 2
			_triEdges.push_back(AnIndexedLine(prevOffset + 3, idxOffset + 3, true));//, l2l_dist, l2l_dist, true)); // 3
			_triEdges.push_back(AnIndexedLine(prevOffset + 4, idxOffset + 4, true));//, l2l_dist, l2l_dist, true)); // 4
			_triEdges.push_back(AnIndexedLine(prevOffset + 5, idxOffset + 5, true));//, l2l_dist, l2l_dist, true)); // 5
			_triEdges.push_back(AnIndexedLine(prevOffset + 6, idxOffset + 6, true));//, l2l_dist, l2l_dist, true)); // 6
			_triEdges.push_back(AnIndexedLine(prevOffset + 7, idxOffset + 7, true));//, l2l_dist, l2l_dist, true)); // 7
			_triEdges.push_back(AnIndexedLine(prevOffset + 8, idxOffset + 8, true));//, l2l_dist, l2l_dist, true)); // 8
			_triEdges.push_back(AnIndexedLine(prevOffset + 9, idxOffset + 9, true));//, l2l_dist, l2l_dist, true)); // 9
			_triEdges.push_back(AnIndexedLine(prevOffset + 10, idxOffset + 10, true));//, l2l_dist, l2l_dist, true)); // 10

			_triEdges.push_back(AnIndexedLine(prevOffset + 1, idxOffset + 2, true));//, l2l_dist, l2l_dist, true)); // 1
			_triEdges.push_back(AnIndexedLine(prevOffset + 2, idxOffset + 3, true));//, l2l_dist, l2l_dist, true)); // 2
			_triEdges.push_back(AnIndexedLine(prevOffset + 3, idxOffset + 4, true));//, l2l_dist, l2l_dist, true)); // 3
			_triEdges.push_back(AnIndexedLine(prevOffset + 4, idxOffset + 5, true));//, l2l_dist, l2l_dist, true)); // 4
			_triEdges.push_back(AnIndexedLine(prevOffset + 5, idxOffset + 6, true));//, l2l_dist, l2l_dist, true)); // 5
			_triEdges.push_back(AnIndexedLine(prevOffset + 6, idxOffset + 7, true));//, l2l_dist, l2l_dist, true)); // 6
			_triEdges.push_back(AnIndexedLine(prevOffset + 7, idxOffset + 8, true));//, l2l_dist, l2l_dist, true)); // 7
			_triEdges.push_back(AnIndexedLine(prevOffset + 8, idxOffset + 9, true));//, l2l_dist, l2l_dist, true)); // 8
			_triEdges.push_back(AnIndexedLine(prevOffset + 9, idxOffset + 10, true));//, l2l_dist, l2l_dist, true)); // 9
			_triEdges.push_back(AnIndexedLine(prevOffset + 10, idxOffset + 1, true));//, l2l_dist, l2l_dist, true)); // 10

			_triEdges.push_back(AnIndexedLine(prevOffset + 2, idxOffset + 1, true));//, l2l_dist, l2l_dist, true)); // 1
			_triEdges.push_back(AnIndexedLine(prevOffset + 3, idxOffset + 2, true));//, l2l_dist, l2l_dist, true)); // 2
			_triEdges.push_back(AnIndexedLine(prevOffset + 4, idxOffset + 3, true));//, l2l_dist, l2l_dist, true)); // 3
			_triEdges.push_back(AnIndexedLine(prevOffset + 5, idxOffset + 4, true));//, l2l_dist, l2l_dist, true)); // 4
			_triEdges.push_back(AnIndexedLine(prevOffset + 6, idxOffset + 5, true));//, l2l_dist, l2l_dist, true)); // 5
			_triEdges.push_back(AnIndexedLine(prevOffset + 7, idxOffset + 6, true));//, l2l_dist, l2l_dist, true)); // 6
			_triEdges.push_back(AnIndexedLine(prevOffset + 8, idxOffset + 7, true));//, l2l_dist, l2l_dist, true)); // 7
			_triEdges.push_back(AnIndexedLine(prevOffset + 9, idxOffset + 8, true));//, l2l_dist, l2l_dist, true)); // 8
			_triEdges.push_back(AnIndexedLine(prevOffset + 10, idxOffset + 9, true));//, l2l_dist, l2l_dist, true)); // 9
			_triEdges.push_back(AnIndexedLine(prevOffset + 1, idxOffset + 10, true));//, l2l_dist, l2l_dist, true)); // 10*/
		}

		idxOffset += offsetGap;
	}

	// 
	ResetSpringRestLengths();

	// _per_layer_points
	for (int a = 0; a < SystemParams::_num_layer; a++)
	{
		_per_layer_points.push_back(std::vector<A2DVector>());
	}
	for (int a = 0; a < _massList.size(); a++)
	{
		A2DVector pt(_massList[a]._pos._x, _massList[a]._pos._y);
		int layer_idx = _massList[a]._debug_which_layer;
		_per_layer_points[layer_idx].push_back(pt);
	}
}

void AnElement::ResetSpringRestLengths()
{
	for (int a = 0; a < _triEdges.size(); a++)
	{
		A3DVector p1 = _massList[_triEdges[a]._index0]._pos;
		A3DVector p2 = _massList[_triEdges[a]._index1]._pos;
		_triEdges[a].SetActualOriDistance(p1.Distance(p2));
	}
}

A2DVector AnElement::ClosestPtOnALayer(A2DVector pt, int layer_idx)
{
	float dist = 10000000000000;
	A2DVector closetPt;

	for (int a = 0; a < _per_layer_points[layer_idx].size(); a++)
	{
		float d = _per_layer_points[layer_idx][a].Distance(pt);
		if (d < dist)
		{
			dist = d;
			closetPt = _per_layer_points[layer_idx][a];
		}
	}

	//closetPt.Print();

	return closetPt;
}

void AnElement::Grow(float growth_scale_iter, float dt)
{
	for (unsigned int a = 0; a < _triEdges.size(); a++)
	{
		if (!_triEdges[a]._isLayer2Layer)
		{
			_triEdges[a].MakeLonger(/* _shrinking_state * */ growth_scale_iter, dt);
		}
	}
}

void AnElement::SolveForSprings()
{
	//float k_edge = SystemParams::_k_edge;

	A3DVector pt0;
	A3DVector pt1;
	A3DVector dir;
	A3DVector eForce;
	float dist = 0;
	float diff = 0;
	float k = 0;

	for (unsigned int a = 0; a < _triEdges.size(); a++)
	{
		int idx0 = _triEdges[a]._index0;
		int idx1 = _triEdges[a]._index1;

		pt0 = _massList[idx0]._pos;
		pt1 = _massList[idx1]._pos;

		

		if (_triEdges[a]._isLayer2Layer)
		{
			k = SystemParams::_k_time_edge;
			dist = pt0.Distance(pt1);
			dir = pt0.DirectionTo(pt1).Norm();
			diff = dist - _triEdges[a]._dist;
		}
		else
		{
			k = SystemParams::_k_edge;
			dir = pt0.DirectionTo(pt1);
			dir._z = 0;
			dir = dir.Norm();
			dist = pt0.GetA2DVector().Distance(pt1.GetA2DVector());
			diff = dist - _triEdges[a]._dist;
		}
		
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