#include "stdafx.h"

void SetConvolution(cv::Mat piUc1, cv::Mat convultionMatrix, int x, int y, int devider) {
	int resultValue = 0;

	uchar pixelValue = 0;
	uchar convValue = 0;
	for (int i = 0; i < (convultionMatrix.rows); i++)
	{
		for (int j = 0; j < (convultionMatrix.cols); j++)
		{
			if ((x - i) >= 0 && (y - j) >= 0) {
				pixelValue = piUc1.at<uchar>(x - i, y - j);
				convValue = convultionMatrix.at<uchar>(i, j);
				resultValue += (pixelValue * convValue);
			}
		}
	}
	resultValue /= devider;
	piUc1.at<uchar>(x, y) = resultValue;
}

uchar SetGamaToPixel(uchar pixel, double gama, uchar brightness) {
	//return (uchar)(pow(pixel, gama) / 255) + brightness;
	uchar result = ((((double)pixel / 255)*gama) * 255) + brightness;
	return result;
}

void ImgGamaEdit(cv::Mat picUc3, double gama, uchar bright) {
	uchar r, g, b;
	for (int y = 0; y < picUc3.rows; y++) {
		for (int x = 0; x < picUc3.cols; x++) {
			b = SetGamaToPixel(picUc3.at<cv::Vec3b>(y, x)[B], gama, bright);
			g = SetGamaToPixel(picUc3.at<cv::Vec3b>(y, x)[G], gama, bright);
			r = SetGamaToPixel(picUc3.at<cv::Vec3b>(y, x)[R], gama, bright);

			picUc3.at<cv::Vec3b>(y, x) = { b, g, r };
		}
	}
}

double calc_g(double I, double o) {
	double e_exp = (std::pow(abs(I), 2) / std::pow(o, 2));
	return std::exp(-e_exp);
}

double AnisotropicFormula(double I, double cn, double ce, double cw, double cs, double in, double ie, double iw, double is, double lambda) {
	return ((I*(1.0 - (lambda*(cn + cs + ce + cw)))) + (lambda*((cn*in) + (cs*is) + (ce*ie) + (cw*iw))));
}

void SetAnisotropic(cv::Mat picUf1, double o, double lambda) {
	//Create copy of origin
	cv::Mat picUf1_copy;
	picUf1.copyTo(picUf1_copy);

	double I = 0;
	double newI = 0;
	double cn, ce, cw, cs, in, ie, iw, is;
	//Apply algorythm
	const int border_px = 1;
	for (int i = border_px; i < picUf1.rows - border_px; i++)
	{
		for (int j = border_px; j < picUf1.cols - border_px; j++)
		{
			I = picUf1_copy.at<double>(i, j);
			in = picUf1_copy.at<double>(i, j - 1);
			is = picUf1_copy.at<double>(i, j + 1);
			ie = picUf1_copy.at<double>(i + 1, j);
			iw = picUf1_copy.at<double>(i - 1, j);

			cn = calc_g(in - I, o);
			ce = calc_g(ie - I, o);
			cw = calc_g(iw - I, o);
			cs = calc_g(is - I, o);

			newI = AnisotropicFormula(I, cn, ce, cw, cs, in, ie, iw, is, lambda);
			picUf1.at<double>(i, j) = newI;
		}
	}
}
