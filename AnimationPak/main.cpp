

#include "Display.h"

#include "SystemParams.h"


int main(int argc, char *argv[])
{
	SystemParams::LoadParameters();

	Display app;
	app.initApp();
	app.getRoot()->startRendering();
	app.closeApp();

	return 0;
}
