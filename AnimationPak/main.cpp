

#include "Display.h"


int main(int argc, char *argv[])
{
	Display app;
	app.initApp();
	app.getRoot()->startRendering();
	app.closeApp();

	return 0;
}
