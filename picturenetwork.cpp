/**********************************************************
picturenetwork.cpp - project network on z = 0 plane.  TWS Dec. 07
Labels nodes with nodevar and segments with segvar (must be float).
Generates a postscript file.
Version 2.0, May 1, 2010.  Added abbrevations of setlinewidth and stroke, May 09.
Added visualization of tissue points, April 2010.
Version 3.0, May 17, 2011.  Plots in z-order, for better results with 3D networks.
***********************************************************/
#define _CRT_SECURE_NO_DEPRECATE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nrutil.h"

void picturenetwork(float* segvar, const char fname[])
{
	extern int nseg, nnod;
	extern int* segtyp, * ista, * iend, * nodname;
	extern float* diam, ** cnode;
	extern float alx, aly, alz, lb, maxl;
	int k, iseg, inod, ilevel, nlevel = 100;
	float xmin, xmax, ymin, ymax, xs, ys, picfac, red, green, blue, xz, xzmin, xzmax;
	float diamfac = 1., zcoord, zbottom, ztop, zmin, zmax;

	FILE* ofp;

	xmin = 0.;
	xmax = alx;
	ymin = 0.;
	ymax = aly;

	//Determine range of z values
	zmin = 1.e6;
	zmax = -1.e6;
	for (iseg = 1; iseg <= nseg; iseg++) if (segtyp[iseg] == 4 || segtyp[iseg] == 5) {
		zcoord = (cnode[3][ista[iseg]] + cnode[3][iend[iseg]]) / 2.;
		zmin = FMIN(zmin, zcoord - 1.);
		zmax = FMAX(zmax, zcoord + 1.);
	}

	picfac = FMIN(500. / xmax, 700. / ymax);//updated April 2010
	ofp = fopen(fname, "w");
	fprintf(ofp, "%%!PS-Adobe-2.0\n");
	fprintf(ofp, "%%%%Pages: 1\n");
	fprintf(ofp, "%%%%EndComments\n");
	fprintf(ofp, "%%%%Page: 1 1\n");
	fprintf(ofp, "/mx {%g mul 50 add} def\n", picfac);
	fprintf(ofp, "/my {%g mul 50 add} def\n", picfac);//updated April 2010
	fprintf(ofp, "/cf {closepath fill} def\n");
	fprintf(ofp, "/cs {closepath stroke} def\n");
	fprintf(ofp, "/m {moveto} def\n");
	fprintf(ofp, "/n {newpath} def\n");
	fprintf(ofp, "/l {lineto} def\n");
	fprintf(ofp, "/sl {setlinewidth} def\n");
	fprintf(ofp, "/sc {setrgbcolor} def\n");
	fprintf(ofp, "/s {stroke} def\n");
	fprintf(ofp, "/Times-Roman findfont\n");
	fprintf(ofp, "8 scalefont\n");
	fprintf(ofp, "setfont\n");

	fprintf(ofp, "newpath\n");
	fprintf(ofp, "%g mx %g my m\n", xmin, ymin);
	fprintf(ofp, "%g mx %g my l\n", xmax, ymin);
	fprintf(ofp, "%g mx %g my l\n", xmax, ymax);
	fprintf(ofp, "%g mx %g my l\n", xmin, ymax);
	fprintf(ofp, "closepath\n");
	fprintf(ofp, "stroke\n");

	fprintf(ofp, "/Times-Roman findfont\n");
	fprintf(ofp, "4 scalefont\n");
	fprintf(ofp, "setfont\n");
	//xzmin = 1.e6;
	//xzmax = -1.e6;
	//for (iseg = 1; iseg <= nseg; iseg++) if (segvar[iseg] > 0) {
	//	xzmin = FMIN(xzmin, segvar[iseg]);
	//	xzmax = FMAX(xzmax, segvar[iseg]);
	//}
	xzmin = 0.;
	xzmax = 7.;

	//plot vessels according to segvar in order from bottom to top according to z-coordinate
	for (ilevel = 1; ilevel <= nlevel; ilevel++) {
		zbottom = zmin + (ilevel - 1) * (zmax - zmin) / nlevel;
		ztop = zmin + ilevel * (zmax - zmin) / nlevel;
		for (iseg = 1; iseg <= nseg; iseg++) if (segvar[iseg] > 0) {
			zcoord = (cnode[3][ista[iseg]] + cnode[3][iend[iseg]]) / 2.;
			if (zcoord >= zbottom && zcoord < ztop) {
				if (xzmin != xzmax) xz = (segvar[iseg] - xzmin) / (xzmax - xzmin);
				else xz = 0.75;
				blue = FMIN(FMAX(1.5 - 4. * fabs(xz - 0.25), 0.), 1.);//Set up colors using Matlab 'jet' scheme
				green = FMIN(FMAX(1.5 - 4. * fabs(xz - 0.5), 0.), 1.);
				red = FMIN(FMAX(1.5 - 4. * fabs(xz - 0.75), 0.), 1.);
				fprintf(ofp, "%f %f %f sc\n", red, green, blue);
				fprintf(ofp, "%g sl\n", picfac * diam[iseg] * diamfac);//line widths scaled up by diamfac
				xs = cnode[1][ista[iseg]];
				ys = cnode[2][ista[iseg]];
				fprintf(ofp, "%g mx %g my m ", xs, ys);
				xs = cnode[1][iend[iseg]];
				ys = cnode[2][iend[iseg]];
				fprintf(ofp, "%g mx %g my l s \n", xs, ys);
			}
		}
	}
	fprintf(ofp, "0 0 0 setrgbcolor\n");	//label nodes in black
	for (iseg = 1; iseg <= nseg; iseg++) if (segvar[iseg] > 0) {
		inod = ista[iseg];
		xs = cnode[1][inod];
		ys = cnode[2][inod];
		fprintf(ofp, "%g mx %g my m ", xs + 0.5 / picfac, ys);
		fprintf(ofp, "(%i) show\n", nodname[inod]);
		inod = iend[iseg];
		xs = cnode[1][inod];
		ys = cnode[2][inod];
		fprintf(ofp, "%g mx %g my m ", xs + 0.5 / picfac, ys);
		fprintf(ofp, "(%i) show\n", nodname[inod]);
	}
	fprintf(ofp, "0 0 1 setrgbcolor\n");	//label segments in blue
	for (iseg = 1; iseg <= nseg; iseg++) if (segvar[iseg] > 0) {
		xs = (cnode[1][ista[iseg]] + cnode[1][iend[iseg]]) / 2.;
		ys = (cnode[2][ista[iseg]] + cnode[2][iend[iseg]]) / 2.;
		fprintf(ofp, "%g mx %g my m ", xs + 0.5 / picfac, ys);
		fprintf(ofp, "(%g) show\n", segvar[iseg]);
	}
	//create a color bar
	float c;
	float cbbox = 15.; //size of boxes
	float cbx = 560; //origin of color bar
	float cby = 100;//origin of color bar
	fprintf(ofp, "0.5 setlinewidth\n");
	fprintf(ofp, "/Times-Roman findfont\n");
	fprintf(ofp, "8 scalefont\n");
	fprintf(ofp, "setfont\n");
	for (k = 0; k <= 10; k++) {
		xz = k * 0.1;
		c = xzmin + (xzmax - xzmin) * xz;
		blue = FMIN(FMAX(1.5 - 4. * fabs(xz - 0.25), 0.), 1.);//Set up colors using Matlab 'jet' scheme
		green = FMIN(FMAX(1.5 - 4. * fabs(xz - 0.5), 0.), 1.);
		red = FMIN(FMAX(1.5 - 4. * fabs(xz - 0.75), 0.), 1.);
		fprintf(ofp, "%f %f %f setrgbcolor\n", red, green, blue);
		fprintf(ofp, "n %g %g m %g %g l %g %g l %g %g l cf\n",
			cbx, cby + k * cbbox, cbx + cbbox, cby + k * cbbox, cbx + cbbox, cby + (k + 1) * cbbox, cbx, cby + (k + 1) * cbbox);
		if (k > 0) fprintf(ofp, "%g %f m 0 0 0 setrgbcolor (%g) show\n", cbx + cbbox * 1.1, cby + cbbox * (k - 0.1), c);
	}
	fprintf(ofp, "n %g %g m %g %g l %g %g l %g %g l cs\n",
		cbx, cby, cbx + cbbox, cby, cbx + cbbox, cby + cbbox * 11, cbx, cby + cbbox * 11);
	fprintf(ofp, "showpage\n");
	fclose(ofp);
}