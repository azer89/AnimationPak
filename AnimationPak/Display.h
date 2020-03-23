
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
	bool keyReleased(const OgreBites::KeyboardEvent &evt);
	bool mouseMoved(const OgreBites::MouseMotionEvent &evt);
	bool mousePressed(const OgreBites::MouseButtonEvent &evt);
	bool mouseReleased(const OgreBites::MouseButtonEvent &evt);
	bool mouseWheelRolled(const OgreBites::MouseWheelEvent &evt);

	//void DoStuff();
	void shutdown();

	//void CreateCubeFromLines();
	

	//void CreateSpringLines();
	//void UpdateSpringDisplay();

	//void Draw();
	//void Update(int nScreenWidth = 0, int nScreenHeight = 0);
	//bool KeyboardEvent(unsigned char nChar, int x, int y);
	//bool MouseEvent(int button, int state, int x, int y);

public:
	bool                  _cameraActivated;            // activate or deactivate a camera
	OgreBites::CameraMan* _cameraMan; // 1st person shooter camera
	Ogre::SceneNode*      _cameraNode;

	Ogre::Root*         _root;
	Ogre::SceneManager* _scnMgr;

	StuffWorker* _sWorker; // GOD class
	
	float _obj_time_ctr;
	float _obj_time_gap;


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
