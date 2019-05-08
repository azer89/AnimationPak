

#ifndef _AN_ELEMENT_H_
#define _AN_ELEMENT_H_

#include "A3DVector.h"
#include "AMass.h"
#include "AnIndexedLine.h"
#include "AnIdxTriangle.h"

#include "OpenCVWrapper.h"

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
	
	void UpdateLayerBoundaries();
	//void UpdateSpringLengths();

	void SolveForSprings3D();
	void SolveForSprings2D();

	
		
	void DrawRandomPoints(std::vector<A2DVector> randomPoints); // debug

	float GetMaxDistRandomPoints(const std::vector<A2DVector>& randomPoints);
	void Triangularization(int self_idx);
	void CreateRandomPoints(std::vector<A2DVector> ornamentBoundary, // called from Tetrahedralization()
							float img_length,
							std::vector<A2DVector>& randomPoints,
							int& boundaryPointNum);

	void RotateXY(float radAngle);
	void ScaleXY(float scVal);
	void TranslateXY(float x, float y);
	void AdjustEnds(A2DVector startPt2D, A2DVector endPt2D, bool lockEnds = true);

	void CreateDockPoint(A2DVector queryPos, A2DVector lockPos, int layer_idx);
	void AdjustEndPosition(A2DVector endPt2D, bool lockEnds = true);

	A2DVector ClosestPtOnALayer(A2DVector pt, int layer_idx);

	void Grow(float growth_scale_iter, float dt);

	void CalculateRestStructure();

	// edges
	bool TryToAddTriangleEdge(AnIndexedLine anEdge, int triIndex, std::vector<AnIndexedLine>& tEdges, std::vector<std::vector<int>>& e2t);
	int FindTriangleEdge(AnIndexedLine anEdge, std::vector<AnIndexedLine>& tEdges);
	std::vector<AnIndexedLine> CreateBendingSprings(std::vector<AMass>& mList, 
													const std::vector<AnIdxTriangle>& tris,
		                                            const std::vector<AnIndexedLine>& tEdges,
													const std::vector<std::vector<int>>& e2t);
	int GetUnsharedVertexIndex(AnIdxTriangle tri, AnIndexedLine edge);
	void DrawEdges();

	// ---------- Ogre 3D ----------
	void InitMeshOgre3D(Ogre::SceneManager* sceneMgr,
						Ogre::SceneNode* sceneNode,
						const Ogre::String& name,
						const Ogre::String& materialName);	
	void UpdateMeshOgre3D();
	void UpdateClosestPtsDisplayOgre3D();
	void UpdatePerLayerBoundaryOgre3D();
	void UpdateSpringDisplayOgre3D();
	void UpdateBoundaryDisplayOgre3D();
	// ---------- Ogre 3D ----------

	


public:
	int _elem_idx; // for identification

	bool _predefined_time_path; // time path isn't straight

	std::vector<std::vector<A2DVector>> _per_layer_boundary; // for closest point

	MyColor _color; // drawing
	A2DVector _layer_center; // star only, please delete me!

private:
	//std::vector<std::vector<A2DVector>> _temp_per_layer_boundary; // for interpolation mode	
	std::vector<bool> _insideFlags; // for interpolation mode	

public:
	//void EnableInterpolationMode();  // for interpolation mode	
	//void DisableInterpolationMode(); // for interpolation mode	

public:
	int _numPointPerLayer;
	int _numBoundaryPointPerLayer;

	// for growing
	float _scale; // initially 1.0f, check SystemParamas
	float _maxScale; // check SystemParamas
	std::vector<A2DVector> _layer_center_array; // 
	std::vector<A3DVector> _ori_rest_mass_pos_array; // before scaling
	std::vector<A3DVector> _rest_mass_pos_array; // after scaling

	// ---------- important stuff ----------
	std::vector<AMass>            _massList;       // list of the masses
	std::vector<AnIndexedLine>    _auxiliaryEdges; // for edge forces // UNCOMMENT
	std::vector<AnIndexedLine>    _triEdges;  // for edge forces
	std::vector<std::vector<int>> _edgeToTri; // for aux edges, if -1 means time springs!!!
	std::vector<AnIdxTriangle>    _triangles;

	// ---------- interpolation stuff ----------
	std::vector<AMass>            _interp_massList;       // list of the masses
	std::vector<AnIndexedLine>    _interp_auxiliaryEdges; // for edge forces
	std::vector<AnIndexedLine>    _interp_triEdges;  // for edge forces
	std::vector<std::vector<int>> _interp_edgeToTri;
	std::vector<AnIdxTriangle>    _interp_triangles; // 

	// ---------- Ogre 3D ----------
	Ogre::SceneManager* _sceneMgr;
	Ogre::SceneNode*    _sceneNode;
	Ogre::MaterialPtr   _material;
	Ogre::ManualObject* _tubeObject;

	// ---------- Ogre 3D ----------
	DynamicLines*    _spring_lines;
	Ogre::SceneNode* _springNode;

	DynamicLines*    _debug_lines;
	Ogre::SceneNode* _debugNode;

	DynamicLines*    _debug_lines_2;
	Ogre::SceneNode* _debugNode_2;
};

#endif
