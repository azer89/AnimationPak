

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

#include "TubeConnector.h"

class AnElement
{
public:
	AnElement();
	~AnElement();


	// call it exactly once before simulation
	// or the collision grid gets angry
	// see StuffWorker::Update()
	// see StuffWorker::InitElements2(Ogre::SceneManager* scnMgr)
	void InitSurfaceTriangleMidPts();

	//void CreateStarTube(int self_idx);
	void ResetSpringRestLengths();
	void CreateHelix(float val = 2.0f);
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
	//void UpdateInterpMasses();
	void UpdateZConstraint();
	//void UpdateAvgLayerSpringLength();


	void SolveForSprings3D();

	bool IsInsideApprox(int layer_idx, A3DVector pos);
	bool IsInside(int layer_idx, A3DVector pos, std::vector<A3DVector>& boundary_slice);
	bool IsInside_Const(int layer_idx, A3DVector pos, std::vector<A3DVector>& boundary_slice) const;
	
		
	void DrawRandomPoints(std::vector<A2DVector> randomPoints); // debug

	float GetMaxDistRandomPoints(const std::vector<A2DVector>& randomPoints);
	void ComputeBary();

	void TriangularizationThatIsnt(int self_idx); // replacement for Triangularization()
	void Triangularization(std::vector<std::vector<A2DVector>> art_path, int self_idx);
	void CreateRandomPoints(std::vector<A2DVector> ornamentBoundary, // called from Tetrahedralization()
							float img_length,
							std::vector<A2DVector>& randomPoints,
							int& boundaryPointNum);
	void SetIndex(int idx);
	void RotateXY(float radAngle);
	void ScaleXY(float scVal);
	void TranslateXY(float x, float y);
	void TranslateXY(float x, float y, int start_mass_idx, int end_mass_idx);
	void TranslateCenterXY(float x, float y);
	//void AdjustEnds(A2DVector startPt2D, A2DVector endPt2D, bool lockEnds = true);

	void CreateDockPoint(A2DVector queryPos, A2DVector lockPos, int layer_idx);
	void DockEnds(A2DVector startPt2D, A2DVector endPt2D, bool lockEnds = true);
	void Docking(std::vector<A3DVector> aPath, std::vector<int> layer_indices);

	//A2DVector ClosestPtOnALayer(A2DVector pt, int layer_idx);

	A3DVector ClosestPtOnTriSurfaces(std::vector<int>& triIndices, A3DVector pos);
	A3DVector ClosestPtOnATriSurface(int triIdx, A3DVector pos);
	A3DVector ClosestPtOnATriSurface_Const(int triIdx, A3DVector pos) const;

	void Grow(float growth_scale_iter, float dt);

	//void PrintKEdgeArray();

	void CalculateRestStructure();

	// edges
	void AddSpring(AnIndexedLine anEdge, std::vector<AnIndexedLine>& tSpring);
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
	void UpdateLayerSpringsOgre3D();
	void UpdateAuxSpringsOgre3D();
	void UpdateMassListOgre3D();
	void UpdateVelocityMagnitudeOgre3D();
	void UpdateTimeEdgesOgre3D();
	void UpdateGrowingOgre3D();
	// ---------- Ogre 3D ----------

	// ----- interpolation ----- 
	//A2DVector Interp_ClosestPtOnALayer(A2DVector pt, int layer_idx);
	//void Interp_UpdateLayerBoundaries();
	//bool Interp_HasOverlap();
	//void Interp_SolveForSprings2D();
	//void Interp_ResetSpringRestLengths();
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
	std::vector<std::vector<int>> _interp_edgeToTriB; // not used*/
	// ---------- interpolation stuff ----------

public:
	int _elem_idx; // for identification

	//bool _predefined_time_path; // time path isn't straight

	std::vector<std::vector<A3DVector>> _per_layer_boundary; // for closest point and Ogre3D
	std::vector<A2DVector> _a_layer_boundary;// for closest point
	std::vector<A3DVector> _a_layer_boundary_3d;// for closest point
	std::vector<std::vector<A3DVector>> _per_layer_boundary_drawing;
	std::vector<std::vector<std::vector<A2DVector>>> _per_layer_triangle_drawing;
	//std::vector<float> _per_layer_boundary_z_pos;

	MyColor _color; // drawing
	A2DVector _layer_center; // for some transformation

	std::vector<float> _z_pos_array; // for UpdateZConstaint();

public:
	
	void AddConnector(int other_elem_idx, int ur_layer_idx, int their_layer_idx);

	std::vector<TubeConnector> _t_connectors;

private:
	//std::vector<std::vector<A2DVector>> _temp_per_layer_boundary; // for interpolation mode	
	std::vector<bool> _insideFlags; 	
	std::vector<float> _layer_scale_array;
	std::vector<float> _layer_k_edge_array;

	//std::vector<bool> _growFlags; // see Grow()
	//bool _is_growing;              // see Grow()
	

//public:
	//void EnableInterpolationMode();  // for interpolation mode	
	//void DisableInterpolationMode(); // for interpolation mode	
	//void ShowTimeSprings(bool yesno);

public:
	// DO NOT CONFUSE BETWEEN THESE TWO!
	int _numPointPerLayer;   
	int _numBoundaryPointPerLayer;

	

	//float _avg_layer_springs_length;

	// for growing
	float _scale; // initially 1.0f, check SystemParamas
	//float _maxScale; // check SystemParamas
	std::vector<A2DVector> _layer_center_array; // 
	std::vector<A3DVector> _ori_rest_mass_pos_array; // before scaling
	std::vector<A3DVector> _rest_mass_pos_array; // after scaling

	// ---------- important stuff ----------
	std::vector<AMass>            _massList;       // list of the masses


	//std::vector<AnIndexedLine>    _auxiliaryEdges; // for edge forces // UNCOMMENT
	//std::vector<AnIndexedLine>    _triEdges;  // for edge forces
	//std::vector<AnIndexedLine>    _negSpaceEdges;
	float _k_edge; // see Grow()

	std::vector<AnIndexedLine>    _layer_springs;     // 0
	std::vector<AnIndexedLine>    _time_springs;      // 1
	std::vector<AnIndexedLine>    _auxiliary_springs; // 2 
	std::vector<AnIndexedLine>    _neg_space_springs; // 3


	std::vector<std::vector<int>> _edgeToTri; // for aux edges, if -1 means time springs!!!
	std::vector<AnIdxTriangle>    _triangles;
	

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

	/* 0 */
	DynamicLines*    _layer_springs_lines;
	Ogre::SceneNode* _layer_springs_node;

	/* 1 */
	DynamicLines*    _time_springs_lines;
	Ogre::SceneNode* _time_springs_node;

	/* 2 */
	DynamicLines*    _aux_springs_lines;
	Ogre::SceneNode* _aux_springs_node;

	/* 3 */
	DynamicLines*    _neg_space_springs_lines;
	Ogre::SceneNode* _neg_space_springs_node;

	// testing surface triangle mesh
	DynamicLines*    _surface_tri_lines;
	Ogre::SceneNode* _surface_tri_node;	

	// testing closest slice
	DynamicLines*    _closest_slice_lines; // for debugging repulsion forces
	Ogre::SceneNode* _closest_slice_node;  // for debugging repulsion forces

	//
	DynamicLines*    _closest_tri_lines; // for debugging repulsion forces
	Ogre::SceneNode* _closest_tri_node;  // for debugging repulsion forces

	// repulsion forces
	DynamicLines*    _exact_r_force_lines; // for debugging repulsion forces
	Ogre::SceneNode* _exact_r_force_node;  // for debugging repulsion forces

	// repulsion forces
	DynamicLines*    _approx_r_force_lines; // for debugging repulsion forces
	Ogre::SceneNode* _approx_r_force_node;  // for debugging repulsion forces

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

	// velocity magnitude
	DynamicLines*    _v_magnitude_lines;
	Ogre::SceneNode* _v_magnitude_node;

	DynamicLines*    _growing_elements_lines;
	Ogre::SceneNode* _growing_elements_node;

	DynamicLines*    _not_growing_elements_lines;
	Ogre::SceneNode* _not_growing_elements_node;

	//
	int                                 _numTrianglePerLayer;  // number of triangle in just one layer
	std::vector<std::vector<A2DVector>> _arts;                 // vector graphics
	std::vector<MyColor> _art_f_colors;
	std::vector<MyColor> _art_b_colors;
	std::vector<std::vector<int>>       _arts2Triangles;       // mapping vector graphics to triangles
	std::vector<std::vector<ABary>>     _baryCoords;           // barycentric coordinates
};

#endif
