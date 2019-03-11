
#include "Display.h"

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
	lightNode->setPosition(0, 10, 15);
	lightNode->attachObject(light);


	Ogre::SceneNode* camNode = scnMgr->getRootSceneNode()->createChildSceneNode();
	camNode->setPosition(0, 0, 15);
	camNode->lookAt(Ogre::Vector3(0, 0, -1), Ogre::Node::TS_PARENT);

	Ogre::Camera* cam = scnMgr->createCamera("myCam");
	cam->setNearClipDistance(5); // specific to this sample
	cam->setAutoAspectRatio(true);
	camNode->attachObject(cam);
	Ogre::Viewport* vp = getRenderWindow()->addViewport(cam);
	
	// background color
	vp->setBackgroundColour(Ogre::ColourValue(1, 1, 1));

	Ogre::Entity* ent = scnMgr->createEntity("Sinbad.mesh");
	ent->setMaterialName("Examples/TransparentTest2");
	Ogre::SceneNode* node = scnMgr->getRootSceneNode()->createChildSceneNode();
	node->attachObject(ent);

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