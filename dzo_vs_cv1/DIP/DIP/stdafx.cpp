// stdafx.cpp : source file that includes just the standard includes
// DIP.pch will be the pre-compiled header
// stdafx.obj will contain the pre-compiled type information

#include "stdafx.h"

// TODO: reference any additional headers you need in STDAFX.H
// and not in this file

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

double calc_g(double I, double o){
	const double e = 2.71828182845904523536;

	double I_abs = pow(abs(I), 2);
	double e_exp = (I_abs/pow(o,2))*-1;
	return pow(e,e_exp);
}

double AnisotropicFormula(double I, double cn, double ce, double cw, double cs, double in, double ie, double iw, double is, double lambda){
	return (I*(1-(lambda*(cn+ce+cs+cw))+(lambda*((cn*in)+(cw*iw)+(ce*ie)+(cw*iw)))));
}

void SetAnisotropic(cv::Mat picUf1, double o, double lambda){
	double I = 0;
	double newI = 0;
	double cn, ce, cw, cs, in, ie, iw, is;
	for (int i = 0; i < picUf1.rows; i++)
	{
		for (int j = 0; j < picUf1.cols; j++)
		{
			if ((j - 1 < 0) || (j + 1 >= picUf1.cols) || (i - 1 < 0) || (i + 1 >= picUf1.rows)){
				continue;
			}
			I = picUf1.at<float>(i, j);
			in = picUf1.at<float>(i, j - 1);
			is = picUf1.at<float>(i, j + 1);
			ie = picUf1.at<float>(i + 1, j);
			iw = picUf1.at<float>(i - 1, j);

			cn = calc_g(abs(in - I), o);
			ce = calc_g(abs(ie - I), o);
			cw = calc_g(abs(iw - I), o);
			cs = calc_g(abs(is - I), o);

			newI = AnisotropicFormula(I,cn, ce, cw, cs, in, ie, iw, is, lambda);
			picUf1.at<float>(i, j) = newI;
		}
	}
}
