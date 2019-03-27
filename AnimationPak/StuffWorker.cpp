
#include "StuffWorker.h"
#include "OpenCVWrapper.h"

StuffWorker::StuffWorker(): _elem(0)
{

}

StuffWorker::~StuffWorker()
{


	//_element_list.clear();
	if (_elem)
	{
		delete _elem;
	}

	//Ogre::MaterialManager::getSingleton().remove("Examples/TransparentTest2");
}

void StuffWorker::InitElements(Ogre::SceneManager* scnMgr)
{
	
	_elem = new AnElement;
	_elem->CreateStarTube();
	_elem->ScaleXY(0.2);
	Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode("Wow");
	_elem->InitMesh(scnMgr, pNode, "StarTube1", "Examples/TransparentTest2");
	
	
	/*{
		AnElement elem;
		elem.CreateStarTube();
		elem.ScaleXY(0.2);
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode();
		elem.InitMesh(scnMgr, pNode, "StarTube1", "Examples/TransparentTest2");
		_element_list.push_back(elem);
	}*/
	/*
	{
		AnElement elem;
		elem.CreateStarTube();
		elem.ScaleXY(0.2);
		elem.TranslateXY(250, 200);
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode();
		elem.InitMesh(scnMgr, pNode, "StarTube2", "Examples/TransparentTest2");
		//_element_list.push_back(elem);
	}

	{
		AnElement elem;
		elem.CreateStarTube();
		elem.ScaleXY(0.2);
		elem.TranslateXY(250, 0);
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode();
		elem.InitMesh(scnMgr, pNode, "StarTube3", "Examples/TransparentTest2");
		//_element_list.push_back(elem);
	}

	{
		AnElement elem;
		elem.CreateStarTube();
		elem.ScaleXY(0.2);
		elem.TranslateXY(0, 350);
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode();
		elem.InitMesh(scnMgr, pNode, "StarTube4", "Examples/TransparentTest2");
		//_element_list.push_back(elem);
	}

	{
		AnElement elem;
		elem.CreateStarTube();
		elem.ScaleXY(0.2);
		elem.TranslateXY(400, 400);
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode();
		elem.InitMesh(scnMgr, pNode, "StarTube5", "Examples/TransparentTest2");
		//_element_list.push_back(elem);
	}

	{
		AnElement elem;
		elem.CreateStarTube();
		elem.ScaleXY(0.2);
		elem.TranslateXY(400, 0);
		Ogre::SceneNode* pNode = scnMgr->getRootSceneNode()->createChildSceneNode();
		elem.InitMesh(scnMgr, pNode, "StarTube6", "Examples/TransparentTest2");
		//_element_list.push_back(elem);
	}*/
}

