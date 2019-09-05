
#include "Display.h"

#include "StuffWorker.h"
#include "ContainerWorker.h"

#include "DynamicLines.h" // Ogre tutorial
#include "Tubes.h"		  // Ogre tutorial

#include "AnElement.h" // delete this!

#include <iostream>

//#define IM_ARRAYSIZE(_ARR)  ((int)(sizeof(_ARR)/sizeof(*_ARR)))

//std::shared_ptr<Display> Display::_static_instance = nullptr;


// rename title for opening setup
Display::Display()  : OgreBites::ApplicationContext("AnimationPak"), 
	_cameraMan(0), 
	_cameraNode(0), 
	_cameraActivated(false), 
	_sWorker(0),
	_root(0),
    _scnMgr(0)
	//_maxDebugLines(500),  // TODO
	//_debug_lines(0),
	//_debugNode(0),
	//_spring_lines(0),
	//_springNode(0)

{
	//_debug_points.clear();
	//_spring_points.clear();
}

Display::~Display()
{
	// TODO
	//if (_debug_elem) { delete _debug_elem; } // still can't create proper destructor ???
	if (_cameraMan) { delete _cameraMan; }
	if (_root)      { _root = 0; }
	if (_scnMgr)    { _scnMgr = 0; }

	// TODO
	//if (_debug_lines) {}
	//if (_debugNode) {}


	// TODO
	//if (_spring_lines) {}
	//if (_springNode) {}
	
}

// called first before destructor
void Display::shutdown()
{
	//std::cout << "shutdownshutdownshutdownshutdownshutdownshutdownshutdownshutdownshutdownshutdownshutdownshutdown";
	if (_sWorker) { delete _sWorker; }
	OgreBites::ApplicationContext::shutdown();
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

	///// UPDATE
	/*if (StuffWorker::_interp_mode)
	{
		_sWorker->Interp_Update();
		_sWorker->Interp_Reset();
		_sWorker->Interp_Solve();
		_sWorker->Interp_Simulate();
	}
	else*/
	{
		
		_sWorker->Reset();
		_sWorker->Solve();
		_sWorker->Simulate();	
		_sWorker->ImposeConstraints();
		_sWorker->Update();
		_sWorker->UpdateOgre3D(); // drawing
	}
	
	//UpdateSpringDisplay(); // init CreateSpringLines()
	//UpdatePerLayerBoundary();
	//UpdateClosestPtsDisplay();
	
	//ImGui::ShowDemoWindow();
	//ImGui::ShowDemoWindow();
	// Draw imgui
	//ImGui_ImplGLUT_NewFrame(getRenderWindow()->getWidth(), getRenderWindow()->getHeight(), 1.0f / 30.0f);
	//ImGui::SetNextWindowPos(ImVec2(5, 15), ImGuiSetCond_FirstUseEver);  // set position
	//bool show_another_window = false;
	//ImGui::Begin("AnimationPak", &show_another_window, ImVec2(240, 540));
	ImGui::SetNextWindowPos(ImVec2(5, 5), ImGuiCond_Always);
	ImGui::SetNextWindowSize(ImVec2(300, 700), ImGuiCond_Always);
	bool* p_open = NULL;
	ImGuiWindowFlags window_flags = 0;
	ImGui::Begin("AnimationPak", p_open, window_flags);

	ImGui::Text(("Num elements = " + std::to_string(_sWorker->_element_list.size())).c_str());

	// interpolation
	/*
	std::string interp_str = " ";
	if (StuffWorker::_interp_mode)
	{
		interp_str = "Interpolation mode is active";
	}

	ImGui::Text(interp_str.c_str());
	
	if (ImGui::Button("Interpolation Mode")) 
	{ 
		StuffWorker::_interp_mode = !StuffWorker::_interp_mode;
		if (StuffWorker::_interp_mode)
		{
			_sWorker->EnableInterpolationMode();
		}
	}
	*/
	if (evt.timeSinceLastFrame > 0) { ImGui::Text(("FPS: " + std::to_string(1.0f / evt.timeSinceLastFrame)).c_str()); }
	else { ImGui::Text("FPS : -"); }

	ImGui::Text(("Scale = " + std::to_string(_sWorker->_element_list[0]._scale)).c_str());
	ImGui::Text(("Num vertices = " + std::to_string(_sWorker->_num_vertex)).c_str());
	ImGui::Text(("Num springs = " + std::to_string(_sWorker->_num_spring)).c_str());

	if (ImGui::Button("Reload parameters")) { SystemParams::LoadParameters(); }
	if (ImGui::Button("Save Triangles to PNGs")) { _sWorker->SaveFrames3(); }
	if (ImGui::Button("Save Elements to PNGs")) { _sWorker->SaveFrames4(); }
	if (ImGui::Button("Pause/Resume Simulation")) { _sWorker->_is_paused = !_sWorker->_is_paused; }
	//if (ImGui::Button("Button B")) {}
	//if (ImGui::Button("Button C")) {}
	
	ImGui::Text("Visualization:");
	ImGui::Checkbox("Container", &SystemParams::_show_container);
	ImGui::Checkbox("Mass list", &SystemParams::_show_mass_list);
	ImGui::Checkbox("Element boundaries", &SystemParams::_show_element_boundaries);
	ImGui::Checkbox("Exact repulsion forces", &SystemParams::_show_exact_repulsion_forces);
	ImGui::Checkbox("Approx repulsion forces", &SystemParams::_show_approx_repulsion_forces);
	ImGui::Checkbox("Collision grid", &SystemParams::_show_collision_grid);
	ImGui::Checkbox("Collision grid objects", &SystemParams::_show_collision_grid_object);
	ImGui::Checkbox("Exact closest points (debug)", &SystemParams::_show_c_pt_cg);
	ImGui::Checkbox("Approx closest points (debug)", &SystemParams::_show_c_pt_approx_cg);
	ImGui::Checkbox("Surface triangles", &SystemParams::_show_surface_tri);
	ImGui::Checkbox("Time springs", &SystemParams::_show_time_edges);
	ImGui::Checkbox("Negative space springs", &SystemParams::_show_negative_space_springs);
	//bool SystemParams::_show_force = false;
	//bool SystemParams::_show_overlap = false;
	ImGui::Checkbox("Velocity", &SystemParams::_show_force);
	ImGui::Checkbox("Overlap", &SystemParams::_show_overlap);
	ImGui::Checkbox("Closest Triangle", &SystemParams::_show_closest_tri);
	ImGui::SliderInt("Layer select", &SystemParams::_layer_slider_int, -1, SystemParams::_num_layer - 1);
	/*if (ImGui::Checkbox("Show time springs", &SystemParams::_show_time_springs))
	{
		for (int a = 0; a < _sWorker->_element_list.size(); a++)
		{
			_sWorker->_element_list[a].ShowTimeSprings(SystemParams::_show_time_springs);
		}
	}*/
	ImGui::Text("Keys:");
	ImGui::Text("C to activate or deactivate camera");
	ImGui::Text("X to pause/resume simulation");
	

	/*Ogre::Vector3 camPos = _cameraNode->getPosition();
	ImGui::Text("Camera position = ");
	ImGui::Text( (std::to_string(camPos.x)).c_str());
	ImGui::Text((std::to_string(camPos.y)).c_str());
	ImGui::Text((std::to_string(camPos.z)).c_str());*/
	//ImGui::Text( ("Camera position = " + std::to_string(camPos.x) + ", " + std::to_string(camPos.y) + ", " + std::to_string(camPos.z)).c_str()  );
	// //Ogre::Vector3 camPos = _cameraNode->getPosition();
	
	ImGui::End();
	//ImGui::Render();

	return true;
}

//void Display::CreateCubeFromLines()
//{	
//}




// setup OGRE
void Display::setup()
{
	OgreBites::ApplicationContext::setup();
	this->getRenderWindow()->resize(1400, 840); // window size
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
		lightNode->setPosition(0, 0, -500);
		lightNode->attachObject(light);
	}

	{
		Ogre::Light* light = _scnMgr->createLight("Light2");
		Ogre::SceneNode* lightNode = _scnMgr->getRootSceneNode()->createChildSceneNode();
		lightNode->setPosition(500, 500, 900);
		lightNode->attachObject(light);
	}

	_cameraNode = _scnMgr->getRootSceneNode()->createChildSceneNode();
	_cameraNode->setPosition(250, 250, 700);
	//_cameraNode->setPosition(1200, 250, -250);
	_cameraNode->lookAt(Ogre::Vector3(250, 250, 0), Ogre::Node::TS_PARENT);
	//_cameraNode->lookAt(Ogre::Vector3(250, 250, -250), Ogre::Node::TS_PARENT);

	Ogre::Camera* cam = _scnMgr->createCamera("myCam");
	cam->setNearClipDistance(0.1); // specific to this sample
	cam->setAutoAspectRatio(true);
	_cameraNode->attachObject(cam);
	Ogre::Viewport* vp = getRenderWindow()->addViewport(cam);
	_cameraMan = new OgreBites::CameraMan(_cameraNode);
	_cameraMan->setStyle(OgreBites::CameraStyle::CS_MANUAL);
	
	// background color
	vp->setBackgroundColour(Ogre::ColourValue(1, 1, 1));
	//vp->setBackgroundColour(Ogre::ColourValue(0, 0, 0));
	//vp->setBackgroundColour(Ogre::ColourValue(0.5, 0.5, 0.5));

	/*Ogre::Entity* ent = scnMgr->createEntity("Sinbad.mesh");
	ent->setMaterialName("Examples/TransparentTest2");
	Ogre::SceneNode* node = scnMgr->getRootSceneNode()->createChildSceneNode();
	node->attachObject(ent);
	node->setScale(10, 10, 10);*/

	// what is this???
	//Ogre::RenderSystemList::const_iterator renderers = mRoot->getAvailableRenderers().begin();


	
	//mCameraMan->manualStop();
	/*
	std::cout << "\n\n renderers \n\n";
	while (renderers != mRoot->getAvailableRenderers().end())
	{
		Ogre::String rName = (*renderers)->getName();
		std::cout << rName << "\n";
		renderers++;
	}
	*/

	_sWorker = new StuffWorker;
	_sWorker->InitElements(_scnMgr);
	_sWorker->_containerWorker->CreateOgreContainer(_scnMgr);

	//CreateCubeFromLines();
	//CreateSpringLines(); // if you want to display springs!!!
	
	
	// add points
	//somePoints.push_back(Ogre::Vector3(0.0f, 0.0f, 0.0f));
	//somePoints.push_back(Ogre::Vector3(452.0f, 2345.0f, 453.0f));

	




	/*for (int i = 0; i < _sWorker->_element_list.size(); i++)
	{
		AnElement elem = _sWorker->_element_list[i];
		for (int a = 0; a < elem._triEdges.size(); a++)
		{
			AnIndexedLine ln = elem._triEdges[a];
			A3DVector pt1 = elem._massList[ln._index0]._pos;
			A3DVector pt2 = elem._massList[ln._index1]._pos;
			somePoints.push_back(Ogre::Vector3(pt1._x, pt1._y, pt1._z));
			somePoints.push_back(Ogre::Vector3(pt2._x, pt2._y, pt2._z));
		}
	}*/

	
	// tubes
	/*Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode();

	SeriesOfTubes* mTubes = new SeriesOfTubes(scnMgr, 16, 10.0, 12, 12, 12.0);

	mTubes->addPoint(Ogre::Vector3(400, 10, 0));
	mTubes->addPoint(Ogre::Vector3(300, 20, -100));
	mTubes->addPoint(Ogre::Vector3(100, 50, -200));
	mTubes->addPoint(Ogre::Vector3(100, 100, -300));
	mTubes->addPoint(Ogre::Vector3(30, 250, -400));
	mTubes->addPoint(Ogre::Vector3(0, 350, -500));

	mTubes->setSceneNode(pNode);
	mTubes->createTubes("MyTubes", "Examples/TransparentTest2");

	delete mTubes;
	*/

	
	
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