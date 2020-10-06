#include "stdafx.h"


cv::Mat CreateConvolutionMatrix(int size_r, int size_c, bool BoxOrGaussian) {
	printf("Generating matrix..\n");
	cv::Mat convultionMatrix3X3(3, 3, CV_8UC1);
	uchar val = 0;
	for (int y = 0; y < convultionMatrix3X3.rows; y++) {
		for (int x = 0; x < convultionMatrix3X3.cols; x++) {
			//Use box or gaussian blur
			if (BoxOrGaussian) val = 1;
			else val = (x + y + 1);
			convultionMatrix3X3.at<uchar>(y, x) = val;
			printf("[%d,%d] %d \n", y, x, val);
		}
	}
	printf("Matrix completed..\n");
	return convultionMatrix3X3;
}

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

void DoConvolution(cv::Mat pi8uc1, cv::Mat convultionMatrix){
	printf("Prepare for Convultion...\n");
	for (int y = 0; y < pi8uc1.rows; y++) {
		for (int x = 0; x < pi8uc1.cols; x++) {
			SetConvolution(pi8uc1, convultionMatrix, x, y, 9);
		}
	}
	printf("Convultion complete..\n");
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

void SetAnisotropic(cv::Mat picFc1, double o, double lambda) {
	//Create copy of origin
	cv::Mat picDc1_copy;
	picFc1.copyTo(picDc1_copy);

	double I = 0;
	double newI = 0;
	double cn, ce, cw, cs, in, ie, iw, is;
	//Apply algorythm
	const int border_px = 1;
	for (int i = border_px; i < picFc1.rows - border_px; i++)
	{
		for (int j = border_px; j < picFc1.cols - border_px; j++)
		{
			I = picDc1_copy.at<double>(i, j);
			in = picDc1_copy.at<double>(i, j - 1);
			is = picDc1_copy.at<double>(i, j + 1);
			ie = picDc1_copy.at<double>(i + 1, j);
			iw = picDc1_copy.at<double>(i - 1, j);

			cn = calc_g(in - I, o);
			ce = calc_g(ie - I, o);
			cw = calc_g(iw - I, o);
			cs = calc_g(is - I, o);

			newI = AnisotropicFormula(I, cn, ce, cw, cs, in, ie, iw, is, lambda);
			picFc1.at<double>(i, j) = newI;
		}
	}
}

void DoAnisoptropicIterations(cv::Mat pic64f1, int iteration_ratio, double o, double lambda){
	printf("Try for anisotropic..[%d times]\n", iteration_ratio);
	for (int i = 0; i < iteration_ratio; i++)
	{
		SetAnisotropic(pic64f1, o, lambda);
		//cv::imshow("VALVE ANTISOPTROPIC", pic64f1);
		//cv::waitKey(1);
		if (!(i % ((iteration_ratio / 100) + 1)))printf("*");
	}
	printf("\n");
	printf("Anisotropic complete..\n");
}

double CalcSpectrumAmplitude(double Real, double Imagine, bool Power) {
	double result = ((Real * Real) + (Imagine * Imagine));
	if (!Power){
		result = std::sqrt(result);
	}
	return result;
}

cv::Mat ConvertToSpectrumAmplitude(cv::Mat ComplexMatrix){
	printf("Converting to Fourier Spectrum Amplitude..\n");
	const size_t rows = ComplexMatrix.rows;
	const size_t cols = ComplexMatrix.cols;
	cv::Mat img = cv::Mat(rows,cols,CV_64FC1);
	double realPart, complexPart;

	for (size_t k = 0; k < rows; k++)
	{
		for (size_t l = 0; l < cols; l++)
		{
			realPart = ComplexMatrix.at<cv::Vec2d>(k, l)[REAL];
			complexPart = ComplexMatrix.at<cv::Vec2d>(k, l)[COMPLEX];
			img.at<double>(k, l) = CalcSpectrumAmplitude(realPart, complexPart);
		}
	}
	printf("Convertion success!\n");
	return img;
}

cv::Mat GetPhasseImage(cv::Mat ComplexMatrix) {
	printf("Getting Fourier Phasse image..\n");
	const size_t rows = ComplexMatrix.rows;
	const size_t cols = ComplexMatrix.cols;
	cv::Mat img = cv::Mat(rows, cols, CV_64FC1);
	double realPart, complexPart;

	for (size_t k = 0; k < rows; k++)
	{
		for (size_t l = 0; l < cols; l++)
		{
			realPart = ComplexMatrix.at<cv::Vec2d>(k, l)[REAL];
			complexPart = ComplexMatrix.at<cv::Vec2d>(k, l)[COMPLEX];
			img.at<double>(k, l) = std::atan(realPart / complexPart);
		}
	}
	printf("Convertion success!\n");
	return img;
}

cv::Mat GetPowerSpectrum(cv::Mat ComplexMatrix) {
	printf("Getting Fourier Power Spectrum..\n");
	const size_t rows = ComplexMatrix.rows;
	const size_t cols = ComplexMatrix.cols;
	cv::Mat img = cv::Mat(rows, cols, CV_64FC1);
	double realPart, complexPart;

	for (size_t k = 0; k < rows; k++)
	{
		for (size_t l = 0; l < cols; l++)
		{
			realPart = ComplexMatrix.at<cv::Vec2d>(k, l)[REAL];
			complexPart = ComplexMatrix.at<cv::Vec2d>(k, l)[COMPLEX];
			img.at<double>(k, l) = std::log(CalcSpectrumAmplitude(realPart, complexPart, true));
		}
	}
	//Normalize result and switch Q
	cv::normalize(img, img, 0.0, 1.0, cv::NORM_MINMAX);
	switch_quadrants(img);
	printf("Convertion success!\n");
	return img;
}

cv::Mat DiscreteFourierTransform(cv::Mat pic64f1) {
	printf("Setting Discrete Fourier Transform..\n");
	const int M = pic64f1.rows;
	const int N = pic64f1.cols;
	const double normalization = 1 / std::sqrt(M*N);
	//Normalize input img
	pic64f1 *= normalization;
	//Complex matrix
	cv::Mat BaseMatrix = cv::Mat(M,N,CV_64FC2);

	double imgVal, realPart, complexPart, x;
	double realSum = 0;
	double complexSum = 0;
	cv::Vec2d base;

	printf("Calculating DFT..\n");
	for (size_t k = 0; k < M; k++)
	{
		printf("DFT progress: %2.0f %%\n", (double)k / M * 100.0);
		for (size_t l = 0; l < N; l++)
		{
			for (size_t m = 0; m < M; m++)
			{
				for (size_t n = 0; n < N; n++)
				{
					imgVal = pic64f1.at<double>(m, n);
					x = (2 * CV_PI)*(((double)(k*m) / M) + ((double)(l*n) / N));
					realPart = (imgVal * std::cos(x));
					complexPart = (-imgVal * std::sin(x));

					realSum += realPart;
					complexSum += complexPart;
				}
			}
			base = cv::Vec2d(realSum, complexSum);
			BaseMatrix.at<cv::Vec2d>(k, l) = base;
		}
	}
	printf("DFT done!\n");
	//switch_quadrants(BaseMatrix);
	return BaseMatrix;
}
//Switch Q 1<->3 and Q 2<->4
void switch_quadrants(cv::Mat & src)
{
	uint widthHalf = src.cols / 2;
	uint heightHalf = src.rows / 2;

	cv::Mat q0(src, cv::Rect(0, 0, widthHalf, heightHalf));						// Top-Left - Create a ROI per quadrant
	cv::Mat q1(src, cv::Rect(widthHalf, 0, widthHalf, heightHalf));				// Top-Right
	cv::Mat q2(src, cv::Rect(0, heightHalf, widthHalf, heightHalf));			// Bottom-Left
	cv::Mat q3(src, cv::Rect(widthHalf, heightHalf, widthHalf, heightHalf));	// Bottom-Right

	cv::Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}
