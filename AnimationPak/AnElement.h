

#ifndef _AN_ELEMENT_H_
#define _AN_ELEMENT_H_

#include "A3DVector.h"
#include "AMass.h"
#include "AnIndexedLine.h"

// ogre
#include <Ogre.h>
#include <OgreApplicationContext.h>
#include <OgrePrerequisites.h>
#include <OgreMaterial.h>
#include <OgreMesh.h>
#include <OgreSceneManager.h>

class AnElement
{
public:
	AnElement();
	~AnElement();

	void CreateStarTube(int self_idx);

	void InitMesh(Ogre::SceneManager* sceneMgr, 
		Ogre::SceneNode* sceneNode,
		const Ogre::String& name,
		const Ogre::String& materialName);

	void UpdateMesh2();
	//void UpdateMesh();

	void ScaleXY(float scVal);
	void TranslateXY(float x, float y);

public:
	int _self_idx; // for identification

public:
	std::vector<AMass>   _massList;       // list of the masses
	std::vector<AnIndexedLine> _triEdges;  // for edge forces

	Ogre::SceneManager* _sceneMgr;
	Ogre::SceneNode* _sceneNode;
	Ogre::MaterialPtr   _material;
	//bool _uniqueMaterial;
	Ogre::ManualObject* _tubeObject;

	//std::vector<A3DVector>       _skin;       // boundary points. temporary data, always updated every step

};

#endif
