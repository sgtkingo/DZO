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
	return (uchar)(pow(pixel, gama) / 255) + brightness;
}

void ImgGamaEdit(cv::Mat picUc3, double gama, uchar bright) {
	uchar r, g, b;
	for (int y = 0; y < picUc3.rows; y++) {
		for (int x = 0; x < picUc3.cols; x++) {
			b = picUc3.at<cv::Vec3b>(y, x)[B];
			g = picUc3.at<cv::Vec3b>(y, x)[G];
			r = picUc3.at<cv::Vec3b>(y, x)[R];
			picUc3.at<cv::Vec3b>(y, x) = { SetGamaToPixel(b,gama,bright),SetGamaToPixel(g,gama,bright), SetGamaToPixel(r,gama,bright) };
		}
	}
}
