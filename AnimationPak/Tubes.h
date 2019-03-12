
#ifndef _TUBES_H_
#define _TUBES_H_

#include <Ogre.h>
#include <OgreApplicationContext.h>

#include <OgrePrerequisites.h>
#include <OgreMaterial.h>
#include <OgreMesh.h>
#include <OgreSceneManager.h>

//namespace Ogre
//{
class SeriesOfTubes
{
public:
	SeriesOfTubes(
		Ogre::SceneManager* sceneMgr,
		const Ogre::uint numberOfSides = 0,
		const Ogre::Real radius = 0.0,
		const Ogre::uint sphereRings = 0,
		const Ogre::uint sphereSegments = 0,
		const Ogre::Real sphereRadius = 0.0,
		const Ogre::Real sphereMaxVisibilityDistance = 0.0);
	~SeriesOfTubes();

	void addPoint(const Ogre::Vector3& pos);
	void removePoint(const Ogre::uint pointNumber);

	void setRadius(const Ogre::Real radius) { mRadius = radius; }
	void setSides(const Ogre::uint numberOfSides) { mSideCount = numberOfSides; }

	const Ogre::Real getRadius() { return mRadius; }
	const Ogre::uint getSides() { return mSideCount; }

	void setSceneNode(Ogre::SceneNode* sceneNode) { mSceneNode = sceneNode; }
	Ogre::SceneNode* getSceneNode() { return mSceneNode; }

	Ogre::MaterialPtr getMaterial() { return mMaterial; }

	Ogre::ManualObject* createTubes(
		const Ogre::String& name,
		const Ogre::String& materialName,
		bool uniqueMaterial = false,
		bool isDynamic = false,
		bool disableUVs = false,
		bool disableNormals = false);

	void _update(bool disableUVs = false, bool disableNormals = false);
	void _destroy();

	void _createSphere(const Ogre::String& strName);

	Ogre::ManualObject* createDebug(const Ogre::String& name);


public:
	Ogre::SceneManager* mSceneMgr;

	typedef std::vector<Ogre::Vector3> VVec;
	VVec mLineVertices;

	Ogre::uint mSideCount;
	Ogre::Real mRadius;
	bool mUniqueMaterial;

	Ogre::uint mSphereRings;
	Ogre::uint mSphereSegments;
	Ogre::Real mSphereRadius;
	Ogre::Real mSphereMaxVisDistance;

	Ogre::MaterialPtr mMaterial;
	Ogre::ManualObject* mTubeObject;

	//typedef std::vector<Ogre::Entity*> SphereStorage;
	//SphereStorage mSpheresJoints;
	//Ogre::MeshPtr mSphereMesh;

	Ogre::SceneNode* mSceneNode;


};
//}

#endif