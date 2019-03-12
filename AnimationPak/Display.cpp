
#include "Display.h"

#include "DynamicLines.h"
#include "Tubes.h"

#include <iostream>

//#define IM_ARRAYSIZE(_ARR)  ((int)(sizeof(_ARR)/sizeof(*_ARR)))

//std::shared_ptr<Display> Display::_static_instance = nullptr;


// rename title for opening setup
Display::Display()  : OgreBites::ApplicationContext("AnimationPak"), mCameraMan(0)
{
}



Display::~Display()
{
	if (mCameraMan) { delete mCameraMan; }
}

bool Display::frameStarted(const Ogre::FrameEvent& evt)
{
	OgreBites::ApplicationContext::frameStarted(evt);

	Ogre::ImguiManager::getSingleton().newFrame(
		evt.timeSinceLastFrame,
		Ogre::Rect(0, 0, getRenderWindow()->getWidth(), getRenderWindow()->getHeight()));


	mCameraMan->frameRendered(evt);

	//ImGui::ShowDemoWindow();
	//ImGui::ShowDemoWindow();
	// Draw imgui
	//ImGui_ImplGLUT_NewFrame(getRenderWindow()->getWidth(), getRenderWindow()->getHeight(), 1.0f / 30.0f);
	//ImGui::SetNextWindowPos(ImVec2(5, 15), ImGuiSetCond_FirstUseEver);  // set position
	//bool show_another_window = false;
	//ImGui::Begin("AnimationPak", &show_another_window, ImVec2(240, 540));
	ImGui::SetNextWindowPos(ImVec2(5, 5), ImGuiCond_Always);
	ImGui::SetNextWindowSize(ImVec2(300, 500), ImGuiCond_Always);
	bool* p_open = NULL;
	ImGuiWindowFlags window_flags = 0;
	ImGui::Begin("AnimationPak", p_open, window_flags);

	if (ImGui::Button("Button")) {  }

	if (ImGui::Button("Button")) {}

	if (ImGui::Button("Button")) {}

	ImGui::End();
	//ImGui::Render();

	return true;
}

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
	Ogre::Root* root = getRoot();
	Ogre::SceneManager* scnMgr = root->createSceneManager();
	Ogre::ImguiManager::getSingleton().init(scnMgr);

	// register our scene with the RTSS
	Ogre::RTShader::ShaderGenerator* shadergen = Ogre::RTShader::ShaderGenerator::getSingletonPtr();
	shadergen->addSceneManager(scnMgr);


	Ogre::Light* light = scnMgr->createLight("MainLight");
	Ogre::SceneNode* lightNode = scnMgr->getRootSceneNode()->createChildSceneNode();
	lightNode->setPosition(250, 250, 100);
	lightNode->attachObject(light);


	Ogre::SceneNode* camNode = scnMgr->getRootSceneNode()->createChildSceneNode();
	camNode->setPosition(250, 250, 1000);
	camNode->lookAt(Ogre::Vector3(250, 250, 0), Ogre::Node::TS_PARENT);

	Ogre::Camera* cam = scnMgr->createCamera("myCam");
	cam->setNearClipDistance(5); // specific to this sample
	cam->setAutoAspectRatio(true);
	camNode->attachObject(cam);
	Ogre::Viewport* vp = getRenderWindow()->addViewport(cam);
	
	// background color
	vp->setBackgroundColour(Ogre::ColourValue(1, 1, 1));
	//vp->setBackgroundColour(Ogre::ColourValue(0, 0, 0));

	Ogre::Entity* ent = scnMgr->createEntity("Sinbad.mesh");
	ent->setMaterialName("Examples/TransparentTest2");
	Ogre::SceneNode* node = scnMgr->getRootSceneNode()->createChildSceneNode();
	node->attachObject(ent);
	node->setScale(10, 10, 10);

	Ogre::RenderSystemList::const_iterator renderers = mRoot->getAvailableRenderers().begin();


	mCameraMan = new OgreBites::CameraMan(camNode);
	mCameraMan->setStyle(OgreBites::CameraStyle::CS_FREELOOK);
	/*
	std::cout << "\n\n renderers \n\n";
	while (renderers != mRoot->getAvailableRenderers().end())
	{
		Ogre::String rName = (*renderers)->getName();
		std::cout << rName << "\n";
		renderers++;
	}
	*/

	std::deque<Ogre::Vector3> somePoints;
	// add points
	//somePoints.push_back(Ogre::Vector3(0.0f, 0.0f, 0.0f));
	//somePoints.push_back(Ogre::Vector3(452.0f, 2345.0f, 453.0f));
	// front
	somePoints.push_back(Ogre::Vector3(0.0f, 0.0f, 0.0f));
	somePoints.push_back(Ogre::Vector3(500.0f, 0.0f, 0.0f));

	somePoints.push_back(Ogre::Vector3(500.0f, 0.0f, 0.0f));
	somePoints.push_back(Ogre::Vector3(500.0f, 500.0f, 0.0f));

	somePoints.push_back(Ogre::Vector3(500.0f, 500.0f, 0.0f));
	somePoints.push_back(Ogre::Vector3(0.0f, 500.0f, 0.0f));

	somePoints.push_back(Ogre::Vector3(0.0f, 500.0f, 0.0f));
	somePoints.push_back(Ogre::Vector3(0.0f, 0.0f, 0.0f));

	// back
	somePoints.push_back(Ogre::Vector3(0.0f, 0.0f, -500.0f));
	somePoints.push_back(Ogre::Vector3(500.0f, 0.0f, -500.0f));

	somePoints.push_back(Ogre::Vector3(500.0f, 0.0f, -500.0f));
	somePoints.push_back(Ogre::Vector3(500.0f, 500.0f, -500.0f));

	somePoints.push_back(Ogre::Vector3(500.0f, 500.0f, -500.0f));
	somePoints.push_back(Ogre::Vector3(0.0f, 500.0f, -500.0f));

	somePoints.push_back(Ogre::Vector3(0.0f, 500.0f, -500.0f));
	somePoints.push_back(Ogre::Vector3(0.0f, 0.0f, -500.0f));

	// left
	somePoints.push_back(Ogre::Vector3(0.0f, 500.0f, 0.0f));
	somePoints.push_back(Ogre::Vector3(0.0f, 500.0f, -500.0f));

	somePoints.push_back(Ogre::Vector3(0.0f, 0.0f, 0.0f));
	somePoints.push_back(Ogre::Vector3(0.0f, 0.0f, -500.0f));

	// right
	somePoints.push_back(Ogre::Vector3(500.0f, 500.0f, 0.0f));
	somePoints.push_back(Ogre::Vector3(500.0f, 500.0f, -500.0f));

	somePoints.push_back(Ogre::Vector3(500.0f, 0.0f, 0.0f));
	somePoints.push_back(Ogre::Vector3(500.0f, 0.0f, -500.0f));

	// tubes
	Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode();

	SeriesOfTubes* mTubes = new SeriesOfTubes(scnMgr, 16, 10.0, 12, 12, 12.0);

	/*mTubes->addPoint(Ogre::Vector3(0, 0, 0));
	mTubes->addPoint(Ogre::Vector3(100, 0, 0));
	mTubes->addPoint(Ogre::Vector3(0, 200, 200));
	mTubes->addPoint(Ogre::Vector3(50, 340, 100));
	mTubes->addPoint(Ogre::Vector3(500, 340, 000));
	mTubes->addPoint(Ogre::Vector3(400, 100, -100));
	mTubes->addPoint(Ogre::Vector3(50, -20, -190));
	mTubes->addPoint(Ogre::Vector3(0, -100, -500));
	*/
	mTubes->addPoint(Ogre::Vector3(400, 10, 0));
	mTubes->addPoint(Ogre::Vector3(300, 20, -100));
	mTubes->addPoint(Ogre::Vector3(100, 50, -200));
	mTubes->addPoint(Ogre::Vector3(100, 100, -300));
	mTubes->addPoint(Ogre::Vector3(30, 250, -400));
	mTubes->addPoint(Ogre::Vector3(0, 350, -500));

	mTubes->setSceneNode(pNode);
	mTubes->createTubes("MyTubes", "Examples/TransparentTest2");


	//In the initialization somewhere, create the initial lines object :
	DynamicLines * lines = new DynamicLines(Ogre::RenderOperation::OT_LINE_LIST);
	for (int i = 0; i<somePoints.size(); i++) {
		lines->addPoint(somePoints[i]);
	}
	lines->update();
	Ogre::SceneNode *linesNode = scnMgr->getRootSceneNode()->createChildSceneNode("lines");
	linesNode->attachObject(lines);
}

bool Display::keyPressed(const OgreBites::KeyboardEvent& evt)
{
	if (evt.keysym.sym == 27)
	{
		getRoot()->queueEndRendering();
	}
	mCameraMan->keyPressed(evt);
	return true;
}

bool Display::keyReleased(const OgreBites::KeyboardEvent &evt)
{
	mCameraMan->keyReleased(evt);
	return true;
}

bool Display::mouseMoved(const OgreBites::MouseMotionEvent &evt)
{
	mCameraMan->mouseMoved(evt);
	return true;
}

bool Display::mousePressed(const OgreBites::MouseButtonEvent &evt)
{
	mCameraMan->mousePressed(evt);
	return true;
}

bool Display::mouseReleased(const OgreBites::MouseButtonEvent &evt)
{
	mCameraMan->mouseReleased(evt);
	return true;
}

bool Display::mouseWheelRolled(const OgreBites::MouseWheelEvent &evt)
{
	mCameraMan->mouseWheelRolled(evt);
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