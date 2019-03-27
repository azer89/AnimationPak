#ifndef _STUFF_WORKER_H_
#define _STUFF_WORKER_H_

#include "AnElement.h"

#include <vector>

class StuffWorker
{
public:
	StuffWorker();

	~StuffWorker();

	void InitElements(Ogre::SceneManager* scnMgr);
public:

	AnElement* _elem;
	std::vector<AnElement> _element_list;
};

#endif