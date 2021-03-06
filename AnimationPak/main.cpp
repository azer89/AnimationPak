

#include "Display.h"

#include "SystemParams.h"

#include <time.h> // time seed
#include <stdlib.h>     /* srand, rand */
#include <time.h> 

#include <windows.h>
//#include <winuser.h>

#include "ClipperWrapper.h"

int main(int argc, char *argv[])
{
	ClipperWrapper::_cScaling = 1e10;

	std::cout << "====================================\n\n";
	std::cout << "\nSystemParams::LoadParameters\n";
	SystemParams::LoadParameters();
	std::cout << "SystemParams::LoadParameters DONE\n\n";


	if (SystemParams::_seed <= 0)
	{
		SystemParams::_seed = time(NULL) % 1000000;
		SystemParams::_seed %= 1000000;
		SystemParams::_seed %= 1000000;
		SystemParams::_seed %= 1000000;
		SystemParams::_seed %= 1000000;
	}

	std::cout << "seed is " << SystemParams::_seed << "\n";
	std::cout << "====================================\n\n";
	srand(SystemParams::_seed);


	Display app;
	app.initApp();
	
	//Ogre::RenderWindow* window = app.getRoot()->getAutoCreatedWindow();
	//HWND hwin;
	//window->getCustomAttribute("HWND", &hwin);
	//SetWindowTextA(hwin, "test");

	app.getRoot()->startRendering();
	app.closeApp();

	return 0;
}
