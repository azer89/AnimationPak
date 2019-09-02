

#ifndef _AN_ELEMENT_H_
#define _AN_ELEMENT_H_

#include "A3DVector.h"
#include "AMass.h"
#include "AnIndexedLine.h"
#include "AnIdxTriangle.h"

#include "OpenCVWrapper.h"

// Ogre
#include <Ogre.h>
#include <OgreApplicationContext.h>
#include <OgrePrerequisites.h>
#include <OgreMaterial.h>
#include <OgreMesh.h>
#include <OgreSceneManager.h>

#include "DynamicLines.h"

#include "ABary.h"

class AnElement
{
public:
	AnElement();
	~AnElement();

	void CreateStarTube(int self_idx);
	void ResetSpringRestLengths();
	void CreateHelix();
	void RandomizeLayerSize();
	
	void BiliniearInterpolation(std::vector<A3DVector>& boundaryA, 
		                        std::vector<A3DVector>& boundaryB, 
		                        std::vector<A3DVector>& boundaryInterp, 
		                        float interVal);
	void BiliniearInterpolationTriangle(std::vector<std::vector<A3DVector>>& triangleA,
		                                std::vector<std::vector<A3DVector>>& triangleB,
		                                std::vector<std::vector<A2DVector>>& triangleInterp,
		                                float interVal);
	std::vector<std::vector<A2DVector>> GetBilinearInterpolatedArt(std::vector<std::vector<A2DVector>> triangles);
	void CalculateLayerBoundaries_Drawing();
	void CalculateLayerTriangles_Drawing();
	void UpdateLayerBoundaries();
	//void UpdateSpringLengths();
	void UpdateInterpMasses();
	void UpdateZConstraint();


	void SolveForSprings3D();

	bool IsInsideApprox(int layer_idx, A3DVector pos);
	bool IsInside(int layer_idx, A3DVector pos, std::vector<A3DVector>& boundary_slice);
	
		
	void DrawRandomPoints(std::vector<A2DVector> randomPoints); // debug

	float GetMaxDistRandomPoints(const std::vector<A2DVector>& randomPoints);
	void ComputeBary();
	void Triangularization(std::vector<std::vector<A2DVector>> art_path, int self_idx);
	void CreateRandomPoints(std::vector<A2DVector> ornamentBoundary, // called from Tetrahedralization()
							float img_length,
							std::vector<A2DVector>& randomPoints,
							int& boundaryPointNum);

	void RotateXY(float radAngle);
	void ScaleXY(float scVal);
	void TranslateXY(float x, float y);
	//void AdjustEnds(A2DVector startPt2D, A2DVector endPt2D, bool lockEnds = true);

	void CreateDockPoint(A2DVector queryPos, A2DVector lockPos, int layer_idx);
	void DockEnds(A2DVector startPt2D, A2DVector endPt2D, bool lockEnds = true);

	//A2DVector ClosestPtOnALayer(A2DVector pt, int layer_idx);

	A3DVector ClosestPtOnTriSurfaces(std::vector<int>& triIndices, A3DVector pos);
	A3DVector ClosestPtOnATriSurface(int triIdx, A3DVector pos);

	void Grow(float growth_scale_iter, float dt);

	void CalculateRestStructure();

	// edges
	bool TryToAddTriangleEdge(AnIndexedLine anEdge, int triIndex, std::vector<AnIndexedLine>& tEdges, std::vector<std::vector<int>>& e2t);
	void ForceAddTriangleEdge(AnIndexedLine anEdge, int triIndex, std::vector<AnIndexedLine>& tEdges, std::vector<std::vector<int>>& e2t);
	int FindTriangleEdge(AnIndexedLine anEdge, std::vector<AnIndexedLine>& tEdges);
	std::vector<AnIndexedLine> CreateBendingSprings(std::vector<AMass>& mList, 
													const std::vector<AnIdxTriangle>& tris,
		                                            const std::vector<AnIndexedLine>& tEdges,
													const std::vector<std::vector<int>>& e2t);
	int GetUnsharedVertexIndex(AnIdxTriangle tri, AnIndexedLine edge);
	void DrawEdges();

	void UpdatePerLayerInsideFlags();
	

	// ---------- Ogre 3D ----------
	void InitMeshOgre3D(Ogre::SceneManager* sceneMgr,
						Ogre::SceneNode* sceneNode,
						const Ogre::String& name,
						const Ogre::String& materialName);	
	void UpdateMeshOgre3D();
	void UpdateClosestPtsDisplayOgre3D();
	//void UpdatePerLayerBoundaryOgre3D();
	void UpdateSpringDisplayOgre3D();
	void UpdateBoundaryDisplayOgre3D();
	void UpdateDockLinesOgre3D();
	void UpdateClosestSliceOgre3D();
	void UpdateClosestTriOgre3D();
	void UpdateSurfaceTriangleOgre3D();
	void UpdateOverlapOgre3D();
	void UpdateNegSpaceEdgeOgre3D();
	void UpdateMassListOgre3D();
	void UpdateForceOgre3D();
	void UpdateTimeEdgesOgre3D();
	// ---------- Ogre 3D ----------

	// ----- interpolation ----- 
	/*A2DVector Interp_ClosestPtOnALayer(A2DVector pt, int layer_idx);
	void Interp_UpdateLayerBoundaries();
	bool Interp_HasOverlap();
	void Interp_SolveForSprings2D();
	void Interp_ResetSpringRestLengths();*/
	// ----- interpolation ----- 

public:
	// ---------- interpolation stuff ----------
	/*std::vector<std::vector<A2DVector>> _interp_per_layer_boundary;

	std::vector<AMass>            _interp_massList;       // list of the masses
	std::vector<AnIndexedLine>    _interp_auxiliaryEdges; // for edge forces
	std::vector<AnIndexedLine>    _interp_triEdges;  // for edge forces
	std::vector<std::vector<int>> _interp_edgeToTri;
	std::vector<AnIdxTriangle>    _interp_triangles; //

	std::vector<AnIndexedLine>    _timeEdgesA; // first index is from original, second index is from interpolation
	std::vector<AnIndexedLine>    _timeEdgesB; // first index is from interpolation, second index is from original
	std::vector<std::vector<int>> _interp_edgeToTriA; // not used
	std::vector<std::vector<int>> _interp_edgeToTriB; // not used
	// ---------- interpolation stuff ----------*/

public:
	int _elem_idx; // for identification

	bool _predefined_time_path; // time path isn't straight

	std::vector<std::vector<A3DVector>> _per_layer_boundary; // for closest point and Ogre3D
	std::vector<A2DVector> _a_layer_boundary;// for closest point
	std::vector<A3DVector> _a_layer_boundary_3d;// for closest point
	std::vector<std::vector<A3DVector>> _per_layer_boundary_drawing;
	std::vector<std::vector<std::vector<A2DVector>>> _per_layer_triangle_drawing;
	//std::vector<float> _per_layer_boundary_z_pos;

	MyColor _color; // drawing
	A2DVector _layer_center; // for some transformation

	std::vector<float> _z_pos_array; // for UpdateZConstaint();

private:
	//std::vector<std::vector<A2DVector>> _temp_per_layer_boundary; // for interpolation mode	
	std::vector<bool> _insideFlags; // for interpolation mode	

public:
	//void EnableInterpolationMode();  // for interpolation mode	
	//void DisableInterpolationMode(); // for interpolation mode	

	void ShowTimeSprings(bool yesno);

public:
	// DO NOT CONFUSE BETWEEN THESE TWO!
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
	std::vector<AnIndexedLine>    _negSpaceEdges;

	std::vector<AnIdxTriangle>    _surfaceTriangles; // for 3D collision grid
	std::vector<A3DVector>		  _tempTri3;

	// ---------- Ogre 3D ----------
	Ogre::SceneManager* _sceneMgr;
	Ogre::SceneNode*    _sceneNode;
	Ogre::MaterialPtr   _material;
	Ogre::ManualObject* _tubeObject;

	// ---------- Ogre 3D ----------
	DynamicLines*    _spring_lines;
	Ogre::SceneNode* _springNode;

	DynamicLines*    _boundary_lines;
	Ogre::SceneNode* _boundary_node;

	DynamicLines*    _dock_lines;
	Ogre::SceneNode* _dock_node;
	std::vector<int> _dock_mass_idx;


	DynamicLines*    _neg_space_edge_lines;
	Ogre::SceneNode* _neg_space_edge_node;

	// testing surface triangle mesh
	DynamicLines*    _surface_tri_lines;
	Ogre::SceneNode* _surface_tri_node;
	//DynamicLines*    _debug_lines_3;
	//Ogre::SceneNode* _debugNode_3;

	//
	DynamicLines*    _time_edge_lines;
	Ogre::SceneNode* _time_edge_node;

	// testing closest slice
	DynamicLines*    _closest_slice_lines; // for debugging repulsion forces
	Ogre::SceneNode* _closest_slice_node;  // for debugging repulsion forces

	//
	DynamicLines*    _closest_tri_lines; // for debugging repulsion forces
	Ogre::SceneNode* _closest_tri_node;  // for debugging repulsion forces

	// testing closest points
	DynamicLines*    _closet_pt_lines; // for debugging repulsion forces
	Ogre::SceneNode* _closet_pt_node;  // for debugging repulsion forces

	DynamicLines*    _closet_pt_approx_lines; // for debugging repulsion forces
	Ogre::SceneNode* _closet_pt_approx_node;  // for debugging repulsion forces

	// testing closest points
	//DynamicLines*    _closet_pt_lines_back;
	//Ogre::SceneNode* _closet_pt_node_back;

	//DynamicLines*    _closet_pt_approx_lines_back;
	//Ogre::SceneNode* _closet_pt_approx_node_back;


	// testing overlap
	DynamicLines*    _overlap_lines;
	Ogre::SceneNode* _overlap_node;

	// testing massList
	DynamicLines*    _massList_lines;
	Ogre::SceneNode* _massList_node;

	// testing force
	DynamicLines*    _force_lines;
	Ogre::SceneNode* _force_node;

	//
	int                                 _numTrianglePerLayer;  // number of triangle in just one layer
	std::vector<std::vector<A2DVector>> _arts;                 // vector graphics
	std::vector<std::vector<int>>       _arts2Triangles;       // mapping vector graphics to triangles
	std::vector<std::vector<ABary>>     _baryCoords;           // barycentric coordinates
};

#endif
