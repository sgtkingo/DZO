// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <cmath>
#include <opencv2/opencv.hpp>

#define B 0
#define G 1
#define R 2
// TODO: reference additional headers your program requires here
void SetConvolution(cv::Mat piUc1, cv::Mat convultionMatrix, int x, int y, int devider);
uchar SetGamaToPixel(uchar pixel, double gama, uchar brightness);
void ImgGamaEdit(cv::Mat picUc3, double gama, uchar bright);

double calc_g(double I, double o);
double AnisotropicFormula(double I, double cn, double ce, double cw, double cs, double in, double ie, double iw, double is, double lambda);
void SetAnisotropic(cv::Mat picUf1, double o, double lambda);