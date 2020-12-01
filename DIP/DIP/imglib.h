#pragma once
#include <stdio.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>

#define B 0
#define G 1
#define R 2
#define A 3

#define L 256

#define REAL 0
#define IMAGINE 1

#define LOW_PASS 0
#define HIGH_PASS 1

//Share function
void ImageShow(cv::Mat img, std::string name = "INPUT IMG");
cv::Mat OpenImage(std::string file_path, int open_mode = cv::IMREAD_COLOR);

//BoxOrGaussian => Box = true, Gausian = false
cv::Mat CreateConvolutionMatrix(uint mSize = 3, bool BoxOrGaussian = true);
void SetConvolution(cv::Mat pic8uc1, cv::Mat convultionMatrix, int x, int y, int devider);
void DoConvolution(cv::Mat pic8uc1, cv::Mat convultionMatrix);

float GetConvolution_F(const cv::Mat pic32uc1, cv::Mat convultionMatrix, int x, int y, int devider);
cv::Mat DoConvolution_F(const cv::Mat pic32uc1, cv::Mat convultionMatrix);

uchar SetGamaToPixel(uchar pixel, double gama, uchar brightness);
void ImgGamaEdit(cv::Mat pic8Uc3, double gama, uchar bright);

double calc_g(double I, double o);
double AnisotropicFormula(double I, double cn, double ce, double cw, double cs, double in, double ie, double iw, double is, double lambda);
void SetAnisotropic(cv::Mat pic64f1, double o, double lambda);
void DoAnisoptropicIterations(cv::Mat pic64f1, int iteration_ratio, double o, double lambda);

double CalcComplexNumberABS(cv::Vec2d cn);
cv::Vec2d CalcComplexNumberMultiple(cv::Vec2d cn_a, cv::Vec2d cn_b);
double CalcComplexNumberScalarMultiple(cv::Vec2d cn_a, cv::Vec2d cn_b);

//DFT
void switch_quadrants(cv::Mat & src);
double CalcSpectrumAmplitude(double Real, double Imagine);
double CalcSpectrumPower(double Real, double Imagine);

cv::Mat GetSpectrumAmplitude(cv::Mat ComplexMatrix);
cv::Mat GetPowerSpectrum(cv::Mat ComplexMatrix);
cv::Mat GetPhasseImage(cv::Mat ComplexMatrix);
cv::Mat DiscreteFourierTransform(cv::Mat pic64f1);
cv::Mat InverseDiscreteFourierTransform(cv::Mat matrixFreqSpectrum);

//Filters
cv::Mat CircleFilterMask(int rows, int cols, double diametr_ration, int mode);
cv::Mat FilterMask(cv::Mat powerSpectrum, double ratio);

cv::Mat LowPassFilter(cv::Mat matrixFreqSpectrum, double ratio);
cv::Mat HighPassFilter(cv::Mat matrixFreqSpectrum, double ratio);

cv::Mat Filter(cv::Mat matrixFreqSpectrum, cv::Mat filterMask);

//Distorion
void apply_rld(int id, void * user_data);
void geom_dist(cv::Mat & src_8uc3_img, cv::Mat & dst_8uc3_img, bool bili, double K1 = 1.0, double K2 = 1.0);

inline cv::Vec3b pixel_bilinear_interpolation(const cv::Mat& img, double x, double y);
cv::Mat bilinear_interpolation(const cv::Mat& img, double K1, double K2);

inline cv::Vec3b pixel_average_pixel(const cv::Mat& img, double x, double y);
cv::Mat average_pixel(const cv::Mat& img, double K1, double K2);

int ex_rld();

//Histogramn
uint cumulative_distribution_function(const std::vector<uint> histogram, uchar brightnessValue);
std::vector<uint> create_histogram(const cv::Mat & src, int L_value);	
std::vector<uchar> equalized_brightness(const std::vector<uint> histogram, const uint width, const uint height, int L_value);
cv::Mat Histogram(const cv::Mat src);

//Perspective
cv::Mat perspective_transformation(const cv::Mat & input, const cv::Mat & overlay, const std::vector<std::pair<cv::Point2i, cv::Point2i>>& points);

//Edge Detection
void apply_sobel_operator(cv::Mat & input_bw_32F);
cv::Mat EdgeDetection_Sobel(const cv::Mat & input);

//Edge Thining
void apply_non_maxima_operator(cv::Mat & input_bw_32F);
cv::Mat EdgeThining(const cv::Mat & input);
void EdgeThining_DoubleTreashold(cv::Mat &input);