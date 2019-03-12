#include "Tubes.h"

#include <OgreManualObject.h>
#include <OgreMaterialManager.h>
#include <OgreSceneManager.h>
#include <OgreStringConverter.h>
#include <OgreEntity.h>
#include <OgreMeshManager.h>
#include <OgreHardwareVertexBuffer.h>
#include <OgreHardwareIndexBuffer.h>
#include <OgreSubMesh.h>

using namespace Ogre;

//namespace Ogre
//{

	SeriesOfTubes::SeriesOfTubes(
		Ogre::SceneManager* sceneMgr,
		const Ogre::uint numberOfSides /*= 0*/,
		const Ogre::Real radius /*= 0.0*/,
		const Ogre::uint sphereRings /*= 0*/,
		const Ogre::uint sphereSegments /*= 0*/,
		const Ogre::Real sphereRadius /*= 0.0*/,
		const Ogre::Real sphereMaxVisibilityDistance /*= 0.0*/)
		: mSceneMgr(sceneMgr),
		mSideCount(numberOfSides),
		mRadius(radius),
		mTubeObject(0),
		mUniqueMaterial(false),
		mSphereRings(sphereRings),
		mSphereSegments(sphereSegments),
		mSphereRadius(sphereRadius),
		mSphereMaxVisDistance(sphereMaxVisibilityDistance),
		mSceneNode(0)
	{

	}

	SeriesOfTubes::~SeriesOfTubes()
	{
		_destroy();
	}

	void SeriesOfTubes::addPoint(const Ogre::Vector3& pos)
	{
		mLineVertices.push_back(pos);
	}

	void SeriesOfTubes::removePoint(const Ogre::uint pointNumber)
	{
		if (pointNumber < mLineVertices.size())
		{
			VVec::iterator it = mLineVertices.begin() + pointNumber;
			mLineVertices.erase(it);
		}
	}

	Ogre::ManualObject* SeriesOfTubes::createTubes(
		const Ogre::String& name,
		const Ogre::String& materialName,
		bool uniqueMaterial /* = false*/,
		bool isDynamic /*= false*/,
		bool disableUVs /*= false*/,
		bool disableNormals /*= false*/)
	{
		if (mTubeObject)
			return mTubeObject;

		mMaterial = MaterialManager::getSingleton().getByName(materialName);

		mUniqueMaterial = uniqueMaterial;

		if (mUniqueMaterial)
			mMaterial = mMaterial->clone(materialName + "_" + name);


		mTubeObject = mSceneMgr->createManualObject(name);
		mTubeObject->setDynamic(isDynamic);

		_update(disableUVs, disableNormals);

		if (mSceneNode)
			mSceneNode->attachObject(mTubeObject);

		return mTubeObject;


	}

	void SeriesOfTubes::_update(bool disableUVs /*= false*/, bool disableNormals /*= false*/)
	{
		if (mTubeObject == 0 || mLineVertices.size() < 2)
			return;

		if (mTubeObject->getDynamic() == true && mTubeObject->getNumSections() > 0)
			mTubeObject->beginUpdate(0);
		else
			mTubeObject->begin(mMaterial->getName());

		Quaternion qRotation(Degree(360.0 / (Real)mSideCount), Vector3::UNIT_Z);

		const uint iVertCount = mSideCount + 1;

		Vector3* vCoreVerts = new Vector3[iVertCount];
		Vector3 vPos = Vector3::UNIT_Y * mRadius;


		for (int i = 0; i<iVertCount; i++)
		{
			vCoreVerts[i] = vPos;
			vPos = qRotation * vPos;
		}

		Vector3 vLineVertA, vLineVertB;
		Vector3 vLine;
		Real dDistance;
		int A, B, C, D;
		int iOffset;

		Vector3* vCylinderVerts = new Vector3[iVertCount * 2];

		for (int i = 1; i<mLineVertices.size(); i++)
		{
			vLineVertA = mLineVertices[i - 1];
			vLineVertB = mLineVertices[i];

			vLine = vLineVertB - vLineVertA;
			dDistance = vLine.normalise();

			qRotation = Vector3::UNIT_Z.getRotationTo(vLine);

			for (int j = 0; j<iVertCount; j++)
			{
				vCylinderVerts[j] = (qRotation * vCoreVerts[j]);
				vCylinderVerts[j + iVertCount] = (qRotation * (vCoreVerts[j] + (Vector3::UNIT_Z * dDistance)));
			}

			Real u, v, delta;
			delta = 1.0 / (Real)(iVertCount - 1);
			u = 0.0;
			v = 1.0;

			for (int j = 0; j<(iVertCount * 2); j++)
			{
				mTubeObject->position(vCylinderVerts[j] + vLineVertA);
				if (disableNormals == false)
				{
					mTubeObject->normal(vCylinderVerts[j].normalisedCopy());
				}
				if (disableUVs == false)
				{
					if (j == iVertCount) {
						u = 0.0;
						v = 0.0;
					}
					mTubeObject->textureCoord(u, v);
					u += delta;
				}
			}

			iOffset = (i - 1) * (iVertCount * 2);
			for (int j = 0; j<iVertCount; j++)
			{
				// End A: 0-(Sides-1)
				// End B: Sides-(Sides*2-1)

				// Verts:
				/*

				A = (j+1)%Sides        C = A + Sides
				B = j                D = B + Sides

				*/



				A = ((j + 1) % iVertCount);
				B = j;
				C = A + iVertCount;
				D = B + iVertCount;

				A += iOffset;
				B += iOffset;
				C += iOffset;
				D += iOffset;

				// Tri #1
				// C,B,A

				mTubeObject->triangle(C, B, A);

				// Tri #2
				// C,D,B

				mTubeObject->triangle(C, D, B);

			}

		}

		delete[] vCoreVerts;
		delete[] vCylinderVerts;
		vCoreVerts = 0;
		vCylinderVerts = 0;

		//if (mSphereMesh == nullptr)
		//	_createSphere(mTubeObject->getName() + "_SphereMesh");

		if (mSceneNode)
			mSceneNode->removeAndDestroyAllChildren();


		/*Entity* pEnt = 0;
		SceneNode* pChildNode = 0;
		VVec::iterator it = mLineVertices.begin() + 1;
		for (int i = 1; it != (mLineVertices.end() - 1); ++it, i++)
		{
			if (mSpheresJoints.size() < i)
			{
				pEnt = mSceneMgr->createEntity(mTubeObject->getName() + "_SphereEnt" + StringConverter::toString(i), mSphereMesh->getName());
				pEnt->setMaterialName(mMaterial->getName());
				mSpheresJoints.push_back(pEnt);
			}
			else
			{
				pEnt = mSpheresJoints[i - 1];
			}
			pEnt->setRenderingDistance(mSphereMaxVisDistance);

			if (mSceneNode)
			{
				pChildNode = mSceneNode->createChildSceneNode();
				pChildNode->setPosition((*it));
				pChildNode->attachObject(pEnt);
			}
		}*/

		mTubeObject->end();

	}

	void SeriesOfTubes::_destroy()
	{
		if (mTubeObject)
		{
			if (mTubeObject->getParentSceneNode())
				mTubeObject->getParentSceneNode()->detachObject(mTubeObject);

			mSceneMgr->destroyManualObject(mTubeObject);
		}



		if (mUniqueMaterial)
		{
			MaterialManager::getSingleton().remove(mMaterial->getName());
		}
		mMaterial.reset();

		/*if (mSpheresJoints.size() > 0)
		{
			Entity* pEnt = 0;
			SphereStorage::iterator it = mSpheresJoints.begin();
			for (; it != mSpheresJoints.end(); ++it)
			{
				pEnt = (*it);
				pEnt->getParentSceneNode()->detachObject(pEnt);
				mSceneMgr->destroyEntity(pEnt);
			}
		}

		if (mSphereMesh == nullptr)
		{
			MeshManager::getSingleton().remove(mSphereMesh->getName());
			mSphereMesh.reset();
		}*/

		if (mSceneNode)
		{
			mSceneNode->removeAndDestroyAllChildren();
			mSceneNode->getParentSceneNode()->removeAndDestroyChild(mSceneNode->getName());
			mSceneNode = 0;
		}

	}

	Ogre::ManualObject* SeriesOfTubes::createDebug(const Ogre::String& name)
	{
		ManualObject* pObj = mSceneMgr->createManualObject(name);
		pObj->begin("BaseWhiteNoLighting", RenderOperation::OT_LINE_STRIP);

		VVec::iterator it = mLineVertices.begin();
		for (; it != mLineVertices.end(); ++it)
		{
			pObj->position((*it));
			pObj->colour(Math::UnitRandom(), Math::UnitRandom(), Math::UnitRandom());
		}
		pObj->end();

		return pObj;

	}

	///////////////////////////////////////////////////////////////////////////////////
	// Courtesy of the Wiki: http://www.ogre3d.org/wiki/index.php/ManualSphereMeshes //
	///////////////////////////////////////////////////////////////////////////////////

	void SeriesOfTubes::_createSphere(const Ogre::String& strName)
	{
		/*mSphereMesh = MeshManager::getSingleton().createManual(strName, ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
		SubMesh *pSphereVertex = mSphereMesh->createSubMesh();

		mSphereMesh->sharedVertexData = new VertexData();
		VertexData* vertexData = mSphereMesh->sharedVertexData;

		// define the vertex format
		VertexDeclaration* vertexDecl = vertexData->vertexDeclaration;
		size_t currOffset = 0;
		// positions
		vertexDecl->addElement(0, currOffset, VET_FLOAT3, VES_POSITION);
		currOffset += VertexElement::getTypeSize(VET_FLOAT3);
		// normals
		vertexDecl->addElement(0, currOffset, VET_FLOAT3, VES_NORMAL);
		currOffset += VertexElement::getTypeSize(VET_FLOAT3);
		// two dimensional texture coordinates
		vertexDecl->addElement(0, currOffset, VET_FLOAT2, VES_TEXTURE_COORDINATES, 0);
		currOffset += VertexElement::getTypeSize(VET_FLOAT2);

		// allocate the vertex buffer
		vertexData->vertexCount = (mSphereRings + 1) * (mSphereSegments + 1);
		HardwareVertexBufferSharedPtr vBuf = HardwareBufferManager::getSingleton().createVertexBuffer(vertexDecl->getVertexSize(0), vertexData->vertexCount, HardwareBuffer::HBU_STATIC_WRITE_ONLY, false);
		VertexBufferBinding* binding = vertexData->vertexBufferBinding;
		binding->setBinding(0, vBuf);
		float* pVertex = static_cast<float*>(vBuf->lock(HardwareBuffer::HBL_DISCARD));

		// allocate index buffer
		pSphereVertex->indexData->indexCount = 6 * mSphereRings * (mSphereSegments + 1);
		pSphereVertex->indexData->indexBuffer = HardwareBufferManager::getSingleton().createIndexBuffer(HardwareIndexBuffer::IT_16BIT, pSphereVertex->indexData->indexCount, HardwareBuffer::HBU_STATIC_WRITE_ONLY, false);
		HardwareIndexBufferSharedPtr iBuf = pSphereVertex->indexData->indexBuffer;
		unsigned short* pIndices = static_cast<unsigned short*>(iBuf->lock(HardwareBuffer::HBL_DISCARD));

		float fDeltaRingAngle = (Math::PI / mSphereRings);
		float fDeltaSegAngle = (2 * Math::PI / mSphereSegments);
		unsigned short wVerticeIndex = 0;

		// Generate the group of rings for the sphere
		for (int ring = 0; ring <= mSphereRings; ring++) {
			float r0 = mSphereRadius * sinf(ring * fDeltaRingAngle);
			float y0 = mSphereRadius * cosf(ring * fDeltaRingAngle);

			// Generate the group of segments for the current ring
			for (int seg = 0; seg <= mSphereSegments; seg++) {
				float x0 = r0 * sinf(seg * fDeltaSegAngle);
				float z0 = r0 * cosf(seg * fDeltaSegAngle);

				// Add one vertex to the strip which makes up the sphere
				*pVertex++ = x0;
				*pVertex++ = y0;
				*pVertex++ = z0;

				Vector3 vNormal = Vector3(x0, y0, z0).normalisedCopy();
				*pVertex++ = vNormal.x;
				*pVertex++ = vNormal.y;
				*pVertex++ = vNormal.z;

				*pVertex++ = (float)seg / (float)mSphereSegments;
				*pVertex++ = (float)ring / (float)mSphereRings;

				if (ring != mSphereRings) {
					// each vertex (except the last) has six indices pointing to it
					*pIndices++ = wVerticeIndex + mSphereSegments + 1;
					*pIndices++ = wVerticeIndex;
					*pIndices++ = wVerticeIndex + mSphereSegments;
					*pIndices++ = wVerticeIndex + mSphereSegments + 1;
					*pIndices++ = wVerticeIndex + 1;
					*pIndices++ = wVerticeIndex;
					wVerticeIndex++;
				}
			}; // end for seg
		} // end for ring

		  // Unlock
		vBuf->unlock();
		iBuf->unlock();
		// Generate face list
		pSphereVertex->useSharedVertices = true;

		// the original code was missing this line:
		mSphereMesh->_setBounds(
			AxisAlignedBox(
				Vector3(-mSphereRadius, -mSphereRadius, -mSphereRadius),
				Vector3(mSphereRadius, mSphereRadius, mSphereRadius)
			), false);
		mSphereMesh->_setBoundingSphereRadius(mSphereRadius);
		// this line makes clear the mesh is loaded (avoids memory leaks)
		mSphereMesh->load();
		*/

	}
