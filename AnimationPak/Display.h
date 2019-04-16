
/*
================================================================================
Reza Adhitya Saputra
radhitya@uwaterloo.ca
================================================================================
*/

#ifndef DISPLAY_H
#define DISPLAY_H

#include <string>
#include <iostream>
#include <memory>
#include <vector>

#include <Ogre.h>
#include <OgreApplicationContext.h>

#include "ImguiManager.h" // ogre
#include "OgreCameraMan.h" // ogre
#include "DynamicLines.h"  // ogre

#include "StuffWorker.h"

#include "AnElement.h"

class Display : public OgreBites::ApplicationContext, public OgreBites::InputListener
{
public:

	Display();
	~Display();
	bool frameStarted(const Ogre::FrameEvent& evt);


	void setup();

	bool keyPressed(const OgreBites::KeyboardEvent& evt);
	bool 	keyReleased(const OgreBites::KeyboardEvent &evt);
	bool 	mouseMoved(const OgreBites::MouseMotionEvent &evt);
	bool 	mousePressed(const OgreBites::MouseButtonEvent &evt);
	bool 	mouseReleased(const OgreBites::MouseButtonEvent &evt);
	bool 	mouseWheelRolled(const OgreBites::MouseWheelEvent &evt);

	//void DoStuff();
	void shutdown();

	void CreateCubeFromLines();
	void UpdateClosestPtsDisplay();
	void UpdatePerLayerBoundary();

	void CreateSpringLines();
	void UpdateSpringDisplay();

	//void Draw();
	//void Update(int nScreenWidth = 0, int nScreenHeight = 0);
	//bool KeyboardEvent(unsigned char nChar, int x, int y);
	//bool MouseEvent(int button, int state, int x, int y);

public:
	bool _cameraActivated;            // activate or deactivate a camera
	OgreBites::CameraMan* _cameraMan; // 1st person shooter camera
	Ogre::SceneNode* _cameraNode;

	StuffWorker* _sWorker;

	Ogre::Root* _root;
	Ogre::SceneManager* _scnMgr;

	//cube
	int _maxDebugLines; // TODO
	std::deque<Ogre::Vector3> _debug_points;
	DynamicLines* _debug_lines;
	Ogre::SceneNode* _debugNode;

	// element springs
	std::deque<Ogre::Vector3> _spring_points;
	DynamicLines* _spring_lines;
	Ogre::SceneNode* _springNode;

	//AnElement* _debug_elem;
	//static std::shared_ptr<Display> GetInstance();

	//static void ShowGL(int argc, char **argv);
	//static void ResizeCallback(int w, int h);
	//static void ShowCallback();



	//static void SpecialKeyboardCallback(int key, int x, int y);
	//static void KeyboardCallback(unsigned char nChar, int x, int y);
	//static void MouseCallback(int button, int state, int x, int y);
	//static void MouseWheel(int button, int dir, int x, int y);
	//static void MouseDragCallback(int x, int y);
	//static void MouseMoveCallback(int x, int y);

	//AVector MapScreenToFieldSpace(float nScreenX, float nScreenY);

	//void ThreadTask(std::string msg);

	/*
	_svg_snapshot_capture_time; // 1
	_png_snapshot_capture_time; // 2
	_sdf_capture_time;          // 3
	_rms_capture_time;          // 4
	*/

	/*
	void DrawSVGSnapshot(float time_delta);
	void DrawPNGSnapshot(float time_delta);
	void CalculateSDF(float time_delta);
	void CalculateFillRMS(float time_delta);

	void DeleteFolders();
	void DeleteFiles();
	*/
	
	//static std::shared_ptr<Display> _static_instance;


	/*
	float _zoomFactor;
	float _xDragOffset;
	float _yDragOffset;
	float _oriXDragOffset;
	float _oriYDragOffset;

	static float _initScreenWidth;
	static float _initScreenHeight;

	float _screenWidth;
	float _screenHeight;
	*/
	/*
	std::string _window_title;

	// limiting drawing frame-rate
	int _time_sum;

	int _noise_time_counter;
	int _noise_time;

	int _previous_time;
	int _simulation_time;
	

	int  _frameCounter;
	
	StuffWorker _sWorker;
	AVector _clickPoint;

	float _svg_time_counter;
	int   _svg_int_counter;
	float _png_time_counter;
	int   _png_int_counter;
	float _sdf_time_counter;
	int   _sdf_int_counter;
	float _rms_time_counter;
	int   _rms_int_counter;

	int _prev_snapshot_time;
	int _prev_opengl_draw;
	
	OpenCVWrapper* _cvWrapper;
	*/

public:

#ifndef OGRE_BUILD_COMPONENT_RTSHADERSYSTEM
		void locateResources()
		{
			OgreBites::ApplicationContext::locateResources();
			// we have to manually specify the shaders
			Ogre::ResourceGroupManager::getSingleton().addResourceLocation(
				"../resources", "FileSystem", Ogre::ResourceGroupManager::INTERNAL_RESOURCE_GROUP_NAME);
		}
#endif
};

#endif
