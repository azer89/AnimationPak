
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
	_sceneNode  = 0;
	_sceneMgr   = 0;
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

	// TODO
	// need to update edges
}

void AnElement::TranslateXY(float x, float y)
{

	for (int a = 0; a < _massList.size(); a++)
	{
		A3DVector pos = _massList[a]._pos;
		_massList[a]._pos = A3DVector(pos._x + x, pos._y + y, pos._z);
	}
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

	std::cout << "_massList size = " << _massList.size() << "\n";
	std::cout << "render vertex = " << _tubeObject->getCurrentVertexCount() << "\n";

	int indexOffset = 10;
	int A, B, C, D;
	for (int i = 0; i < 5; i++)
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

void AnElement::UpdateSpringLengths()
{

}

void AnElement::UpdateBackend()
{
	//
	UpdateSpringLengths();

	//
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
	float zOffset = -(SystemParams::_upscaleFactor / (SystemParams::_num_layer - 1) );
	for(int a = 0; a < SystemParams::_num_layer; a++)
	{
		// x y z mass_idx element_idx layer_idx
		_massList.push_back(AMass(250, 250, zPos, 0, _self_idx, a)); // 0 center
		_massList.push_back(AMass(0,   193, zPos, 1, _self_idx, a)); // 1
		_massList.push_back(AMass(172, 168, zPos, 2, _self_idx, a)); // 2
		_massList.push_back(AMass(250, 12,  zPos, 3, _self_idx, a)); // 3
		_massList.push_back(AMass(327, 168, zPos, 4, _self_idx, a)); // 4
		_massList.push_back(AMass(500, 193, zPos, 5, _self_idx, a)); // 5
		_massList.push_back(AMass(375, 315, zPos, 6, _self_idx, a)); // 6
		_massList.push_back(AMass(404, 487, zPos, 7, _self_idx, a)); // 7
		_massList.push_back(AMass(250, 406, zPos, 8, _self_idx, a)); // 8
		_massList.push_back(AMass(95,  487, zPos, 9, _self_idx, a)); // 9
		_massList.push_back(AMass(125, 315, zPos, 10, _self_idx, a)); // 10

		zPos += zOffset;
	}

	int idxOffset = 0;
	int offsetGap = 11;
	for (int a = 0; a <= 5; a++)
	{
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 1));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 2));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 3));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 4));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 5));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 6));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 7));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 8));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 9));
		_triEdges.push_back(AnIndexedLine(idxOffset, idxOffset + 10));

		_triEdges.push_back(AnIndexedLine(idxOffset + 1, idxOffset + 2));
		_triEdges.push_back(AnIndexedLine(idxOffset + 2, idxOffset + 3));
		_triEdges.push_back(AnIndexedLine(idxOffset + 3, idxOffset + 4));
		_triEdges.push_back(AnIndexedLine(idxOffset + 4, idxOffset + 5));
		_triEdges.push_back(AnIndexedLine(idxOffset + 5, idxOffset + 6));
		_triEdges.push_back(AnIndexedLine(idxOffset + 6, idxOffset + 7));
		_triEdges.push_back(AnIndexedLine(idxOffset + 7, idxOffset + 8));
		_triEdges.push_back(AnIndexedLine(idxOffset + 8, idxOffset + 9));
		_triEdges.push_back(AnIndexedLine(idxOffset + 9, idxOffset + 10));
		_triEdges.push_back(AnIndexedLine(idxOffset + 10, idxOffset + 1));

		if (idxOffset > 0)
		{
			int prevOffset = idxOffset - offsetGap;
			_triEdges.push_back(AnIndexedLine(prevOffset,      idxOffset)); // 0
			_triEdges.push_back(AnIndexedLine(prevOffset + 1,  idxOffset + 1)); // 1
			_triEdges.push_back(AnIndexedLine(prevOffset + 2,  idxOffset + 2)); // 2
			_triEdges.push_back(AnIndexedLine(prevOffset + 3,  idxOffset + 3)); // 3
			_triEdges.push_back(AnIndexedLine(prevOffset + 4,  idxOffset + 4)); // 4
			_triEdges.push_back(AnIndexedLine(prevOffset + 5,  idxOffset + 5)); // 5
			_triEdges.push_back(AnIndexedLine(prevOffset + 6,  idxOffset + 6)); // 6
			_triEdges.push_back(AnIndexedLine(prevOffset + 7,  idxOffset + 7)); // 7
			_triEdges.push_back(AnIndexedLine(prevOffset + 8,  idxOffset + 8)); // 8
			_triEdges.push_back(AnIndexedLine(prevOffset + 9,  idxOffset + 9)); // 9
			_triEdges.push_back(AnIndexedLine(prevOffset + 10, idxOffset + 10)); // 10
		}

		idxOffset += offsetGap;
	}

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

A2DVector AnElement::ClosestPtOnALayer(A2DVector pt, int layer_idx)
{
	float dist = 10000000000000;
	A2DVector closetPt;

	for(int a = 0; a < _per_layer_points[layer_idx].size(); a++)
	{
		float d = _per_layer_points[layer_idx][a].Distance(pt);
		if (d < dist)
		{
			dist = d;
			closetPt = _per_layer_points[layer_idx][a];
		}
	}

	return closetPt;
}