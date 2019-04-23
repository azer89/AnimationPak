

#ifndef _AN_ELEMENT_H_
#define _AN_ELEMENT_H_

#include "A3DVector.h"
#include "AMass.h"
#include "AnIndexedLine.h"
#include "AnIdxTriangle.h"

// ogre
#include <Ogre.h>
#include <OgreApplicationContext.h>
#include <OgrePrerequisites.h>
#include <OgreMaterial.h>
#include <OgreMesh.h>
#include <OgreSceneManager.h>

#include "DynamicLines.h"

class AnElement
{
public:
	AnElement();
	~AnElement();

	void CreateStarTube(int self_idx);
	void ResetSpringRestLengths();
	void CreateHelix();
	void RandomizeLayerSize();

	void InitMeshOgre3D(Ogre::SceneManager* sceneMgr,
		Ogre::SceneNode* sceneNode,
		const Ogre::String& name,
		const Ogre::String& materialName);

	void UpdateBackend();
	//void UpdateSpringLengths();


	void SolveForSprings();

	void UpdateMeshOgre3D();
	//void UpdateMesh();

	void DrawEdges();
	void DrawRandomPoints(std::vector<A2DVector> randomPoints); // debug

	float GetMaxDistRandomPoints(const std::vector<A2DVector>& randomPoints);
	void Triangularization(int self_idx);
	void CreateRandomPoints(std::vector<A2DVector> ornamentBoundary, // called from Tetrahedralization()
						float img_length,
						std::vector<A2DVector>& randomPoints,
						int& boundaryPointNum);

	void ScaleXY(float scVal);
	void TranslateXY(float x, float y);
	void AdjustEnds(A2DVector startPt2D, A2DVector endPt2D, bool lockEnds = true);

	void AdjustEndPosition(A2DVector endPt2D, bool lockEnds = true);

	A2DVector ClosestPtOnALayer(A2DVector pt, int layer_idx);

	void Grow(float growth_scale_iter, float dt);

	void UpdateClosestPtsDisplayOgre3D();
	void UpdatePerLayerBoundaryOgre3D();
	void UpdateSpringDisplayOgre3D();

	// edges
	bool TryToAddTriangleEdge(AnIndexedLine anEdge/*, int triIndex*/);
	int FindTriangleEdge(AnIndexedLine anEdge);



public:
	int _elem_idx; // for identification

	//std::vector<std::vector<A2DVector>> _per_layer_points; // need to delete this, check function ClosestPtOnALayer()
	std::vector<std::vector<A2DVector>> _per_layer_boundary; // only need this

public:
	std::vector<bool> _insideFlags;

	int _numPointPerLayer;
	int _numBoundaryPointPerLayer;

	std::vector<AMass>         _massList;       // list of the masses
	std::vector<AnIndexedLine> _triEdges;  // for edge forces
	std::vector<AnIdxTriangle> _triangles;
	//std::vector<AnIndexedLine> _tetEdges;

	Ogre::SceneManager* _sceneMgr;
	Ogre::SceneNode*    _sceneNode;
	Ogre::MaterialPtr   _material;
	//bool _uniqueMaterial;
	Ogre::ManualObject* _tubeObject;

	DynamicLines*    _spring_lines;
	Ogre::SceneNode* _springNode;
	DynamicLines*    _debug_lines;
	Ogre::SceneNode* _debugNode;

	//std::vector<A3DVector>       _skin;       // boundary points. temporary data, always updated every step

};

#endif
