#pragma once
#include <stdio.h>
#include <cmath>
#include <opencv2/opencv.hpp>

#define B 0
#define G 1
#define R 2
#define A 3

#define REAL 0
#define COMPLEX 1

cv::Mat CreateConvolutionMatrix(int size_r, int size_c, bool BoxOrGaussian = true);
void SetConvolution(cv::Mat piUc1, cv::Mat convultionMatrix, int x, int y, int devider);
void DoConvolution(cv::Mat pi8uc1, cv::Mat convultionMatrix);

uchar SetGamaToPixel(uchar pixel, double gama, uchar brightness);
void ImgGamaEdit(cv::Mat picUc3, double gama, uchar bright);

double calc_g(double I, double o);
double AnisotropicFormula(double I, double cn, double ce, double cw, double cs, double in, double ie, double iw, double is, double lambda);
void SetAnisotropic(cv::Mat pic64f1, double o, double lambda);
void DoAnisoptropicIterations(cv::Mat pic64f1, int iteration_ratio, double o, double lambda);

void switch_quadrants(cv::Mat & src);
double CalcSpectrumAmplitude(double Real, double Imagine, bool Power = false);
cv::Mat ConvertToSpectrumAmplitude(cv::Mat ComplexMatrix);
cv::Mat GetPowerSpectrum(cv::Mat ComplexMatrix);
cv::Mat GetPhasseImage(cv::Mat ComplexMatrix);
cv::Mat DiscreteFourierTransform(cv::Mat pic64f1);