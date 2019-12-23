
#include "Display.h"

#include "StuffWorker.h"
#include "ContainerWorker.h"

#include "DynamicLines.h" // Ogre tutorial
#include "Tubes.h"		  // Ogre tutorial

#include "AnElement.h" // delete this!

#include <iostream>

// rename title for opening setup
Display::Display()  : OgreBites::ApplicationContext("AnimationPak"), 
	_cameraMan(0), 
	_cameraNode(0), 
	_cameraActivated(false), 
	_sWorker(0),
	_root(0),
    _scnMgr(0)
{
}

Display::~Display()
{
	// TODO
	//if (_debug_elem) { delete _debug_elem; } // still can't create proper destructor ???
	if (_cameraMan) { delete _cameraMan; }
	if (_root)      { _root = 0; }
	if (_scnMgr)    { _scnMgr = 0; }
}

// called first before destructor
void Display::shutdown()
{
	std::cout << "quit 2\n";
	//std::cout << "shutdownshutdownshutdownshutdownshutdownshutdownshutdownshutdownshutdownshutdownshutdownshutdown";
	if (_sWorker) { delete _sWorker; }
	std::cout << "stuff worker destroyed\n";
	OgreBites::ApplicationContext::shutdown();
	std::cout << "ApplicationContext::shutdown <-- did this\n";
}

// loop
bool Display::frameStarted(const Ogre::FrameEvent& evt)
{
	OgreBites::ApplicationContext::frameStarted(evt);
	
	Ogre::ImguiManager::getSingleton().newFrame(
									evt.timeSinceLastFrame,
									Ogre::Rect(0, 0, getRenderWindow()->getWidth(), getRenderWindow()->getHeight()));
	
	if(_cameraActivated)
	{
		_cameraMan->frameRendered(evt);
	}

	// SIMULATION here
	_sWorker->Update();				// Update collision grid
	_sWorker->AlmostAllUrShit();	// All other stuff
	_sWorker->UpdateOgre3D();		// drawing	

	// Stopping
	bool shouldQuit = false;
	int num_layer_growing = _sWorker->StillGrowing();
	if (num_layer_growing < SystemParams::_num_layer_growing_threshold)
	{
		shouldQuit = true;
	}
	
	// IMGUI
	ImGui::SetNextWindowPos(ImVec2(5, 5), ImGuiCond_Always);
	ImGui::SetNextWindowSize(ImVec2(340, 850), ImGuiCond_Always);
	bool* p_open = NULL;
	ImGuiWindowFlags window_flags = 0;
	ImGui::Begin("AnimationPak", p_open, window_flags);

	

	if (ImGui::Button("Reload parameters")) { SystemParams::LoadParameters(); }
	if (ImGui::Button("Save Triangles to PNGs")) { _sWorker->SaveFrames3(); }
	if (ImGui::Button("Save Elements to PNGs")) 
	{ 
		// delete folder
		std::stringstream ss;
		ss << "del /Q " << SystemParams::_save_folder << "*.*";
		std::system(ss.str().c_str());

		// stuff, see code far below when you quit
		_sWorker->SaveFrames4();
		_sWorker->SaveStatistics();
		_sWorker->SaveScene();
	}
	if (ImGui::Button("Pause/Resume Simulation")) { _sWorker->_is_paused = !_sWorker->_is_paused; }
		
	ImGui::Text("Visualization");
	ImGui::Checkbox("Container",                  &SystemParams::_show_container);
	ImGui::Checkbox("Mass list",                  &SystemParams::_show_mass_list);
	ImGui::Checkbox("Element boundaries",         &SystemParams::_show_element_boundaries);
	ImGui::Checkbox("0 - Layer springs",          &SystemParams::_show_layer_springs);
	ImGui::Checkbox("1 - Time springs",           &SystemParams::_show_time_springs);
	ImGui::Checkbox("2 - Auxiliary springs",      &SystemParams::_show_aux_springs);
	ImGui::Checkbox("3 - Negative space springs", &SystemParams::_show_negative_space_springs);

	ImGui::Checkbox("Exact repulsion forces",     &SystemParams::_show_exact_repulsion_forces);
	ImGui::Checkbox("Approx repulsion forces",    &SystemParams::_show_approx_repulsion_forces);
	ImGui::Checkbox("Collision grid",             &SystemParams::_show_collision_grid);
	ImGui::Checkbox("Collision grid objects",     &SystemParams::_show_collision_grid_object);

	ImGui::Checkbox("Surface triangles",          &SystemParams::_show_surface_tri);
	ImGui::Checkbox("Growth",                     &SystemParams::_show_growing_elements);

	ImGui::Checkbox("Velocity",                   &SystemParams::_show_force);
	ImGui::Checkbox("Overlap",                    &SystemParams::_show_overlap);
	ImGui::Checkbox("Docking",                    &SystemParams::_show_dock_points);
	
	
	ImGui::Checkbox("Multithread test",           &SystemParams::_multithread_test);
	ImGui::Checkbox("Arts", &SystemParams::_show_arts);
	ImGui::Checkbox("Centers", &SystemParams::_show_centers);
	ImGui::SliderInt("Layer select",              &SystemParams::_layer_slider_int, -1, SystemParams::_num_layer - 1);

	ImGui::Text(("Num elements = " + std::to_string(_sWorker->_element_list.size())).c_str());

	if (evt.timeSinceLastFrame > 0) { ImGui::Text(("FPS: " + std::to_string(1.0f / evt.timeSinceLastFrame)).c_str()); }
	else { ImGui::Text("FPS : -"); }

	ImGui::Text(("Scale = " + std::to_string(_sWorker->_element_list[0]._scale)).c_str());
	ImGui::Text(("Num vertex = " + std::to_string(_sWorker->_num_vertex)).c_str());
	ImGui::Text(("_max_c_pts = " + std::to_string(_sWorker->_max_c_pts)).c_str());
	ImGui::Text(("_max_c_pts_approx = " + std::to_string(_sWorker->_max_c_pts_approx)).c_str());

	ImGui::Text(("_k_edge = " + std::to_string(_sWorker->_element_list[0]._k_edge)).c_str());

	ImGui::Text(("num_layer_growing = " + std::to_string(num_layer_growing)).c_str());

	if (SystemParams::_multithread_test)
	{
		ImGui::Text(("C grid (N vs 1)          = " + std::to_string((int)_sWorker->_cg_multi_t.Avg()) + " vs " + std::to_string((int)_sWorker->_cg_single_t.Avg())).c_str());
		ImGui::Text(("Everything else (N vs 1) = " + std::to_string((int)_sWorker->_almostall_multi_t.Avg()) + " vs " + std::to_string((int)_sWorker->_almostall_single_t.Avg())).c_str());
	}
	else
	{
		ImGui::Text(("C grid          = " + std::to_string((int)_sWorker->_cg_multi_t.Avg())).c_str());
		ImGui::Text(("Everything else = " + std::to_string((int)_sWorker->_almostall_multi_t.Avg())).c_str());
	}

	ImGui::Text("Press C to activate or deactivate camera");
	ImGui::Text("Press X to pause/resume simulation");
	
	ImGui::End();

	if (shouldQuit)
	{
		// delete folder
		std::stringstream ss;
		ss << "del /Q " << SystemParams::_save_folder << "*.*";
		std::system(ss.str().c_str());

		// stuff
		_sWorker->SaveFrames4();
		_sWorker->SaveStatistics();
		_sWorker->SaveScene();

		std::cout << "quit 1\n";

		return false;
	}

	return true;
}

// setup OGRE
void Display::setup()
{
	OgreBites::ApplicationContext::setup();
	this->getRenderWindow()->resize(1400, 900); // window size
	this->getRenderWindow()->reposition(5, 5);
	addInputListener(this);

	Ogre::ImguiManager::createSingleton();
	addInputListener(Ogre::ImguiManager::getSingletonPtr());

	// get a pointer to the already created root
	_root = getRoot();
	_scnMgr = _root->createSceneManager();
	Ogre::ImguiManager::getSingleton().init(_scnMgr);

	// register our scene with the RTSS
	Ogre::RTShader::ShaderGenerator* shadergen = Ogre::RTShader::ShaderGenerator::getSingletonPtr();
	shadergen->addSceneManager(_scnMgr);

	{
		Ogre::Light* light = _scnMgr->createLight("Light1");
		Ogre::SceneNode* lightNode = _scnMgr->getRootSceneNode()->createChildSceneNode();
		lightNode->setPosition(250, 250, -600);
		lightNode->attachObject(light);
	}

	{
		Ogre::Light* light = _scnMgr->createLight("Light2");
		Ogre::SceneNode* lightNode = _scnMgr->getRootSceneNode()->createChildSceneNode();
		lightNode->setPosition(250, 250, 100);
		lightNode->attachObject(light);
	}

	_cameraNode = _scnMgr->getRootSceneNode()->createChildSceneNode();
	_cameraNode->setPosition(400, 400, 600);
	_cameraNode->lookAt(Ogre::Vector3(250, 250, -250), Ogre::Node::TS_PARENT);

	Ogre::Camera* cam = _scnMgr->createCamera("myCam");
	cam->setNearClipDistance(0.1); // specific to this sample
	cam->setAutoAspectRatio(true);
	_cameraNode->attachObject(cam);
	Ogre::Viewport* vp = getRenderWindow()->addViewport(cam);
	_cameraMan = new OgreBites::CameraMan(_cameraNode);
	_cameraMan->setStyle(OgreBites::CameraStyle::CS_MANUAL);
	
	// background color
	vp->setBackgroundColour(Ogre::ColourValue(1, 1, 1));

	_sWorker = new StuffWorker;
	_sWorker->InitElementsAndCGrid(_scnMgr); // NEVER REPLACE THIS FUNCTION
	_sWorker->_containerWorker->CreateOgreContainer(_scnMgr);
}



bool Display::keyPressed(const OgreBites::KeyboardEvent& evt)
{
	if (evt.keysym.sym == 27)
	{
		getRoot()->queueEndRendering();
	}
	if (evt.keysym.sym == 'c' || evt.keysym.sym == 'C')
	{
		// Activate or deactivate camera
		_cameraActivated = !_cameraActivated;
		if (!_cameraActivated)
		{
			std::cout << "stop camera\n";
			_cameraMan->manualStop();
			_cameraMan->setStyle(OgreBites::CameraStyle::CS_MANUAL);
		}
		else
		{
			_cameraMan->setStyle(OgreBites::CameraStyle::CS_FREELOOK);
		}
	}
	if (evt.keysym.sym == 'x' || evt.keysym.sym == 'X')
	{
		_sWorker->_is_paused = !_sWorker->_is_paused;
	}

	_cameraMan->keyPressed(evt);
	return true;
}

bool Display::keyReleased(const OgreBites::KeyboardEvent &evt)
{
	_cameraMan->keyReleased(evt);
	return true;
}

bool Display::mouseMoved(const OgreBites::MouseMotionEvent &evt)
{
	_cameraMan->mouseMoved(evt);
	return true;
}

bool Display::mousePressed(const OgreBites::MouseButtonEvent &evt)
{	
	
	
	_cameraMan->mousePressed(evt);
	return true;
}

bool Display::mouseReleased(const OgreBites::MouseButtonEvent &evt)
{
	_cameraMan->mouseReleased(evt);
	return true;
}

bool Display::mouseWheelRolled(const OgreBites::MouseWheelEvent &evt)
{
	_cameraMan->mouseWheelRolled(evt);
	return true;
}

// static
/*std::shared_ptr<Display> Display::GetInstance()
{


	if (_static_instance == nullptr)
	{
		//_initScreenWidth = SystemParams::_screen_width;
		//_initScreenHeight = SystemParams::_screen_height;

		_static_instance = std::shared_ptr<Display>(new Display());
		//_static_instance->_screenWidth = _initScreenWidth;
		//_static_instance->_screenHeight = _initScreenHeight;
	}
	return _static_instance;
}
*/