#include "stdafx.h"

void ImageShow(cv::Mat img) 
{
	cv::imshow("INPUT IMG", img);
	cv::waitKey();
}

cv::Mat CreateConvolutionMatrix(uint mSize, bool BoxOrGaussian) {
	printf("Generating matrix..\n");
	cv::Mat convultionMatrix(mSize, mSize, CV_8UC1);
	int val = 0;
	for (int y = 0; y < mSize; y++) {
		for (int x = 0; x < mSize; x++) {
			//Use box or gaussian blur
			if (BoxOrGaussian) val = 1;
			else {	
				if (y <= mSize / 2) {
					val = x + y;
					if (x <= mSize / 2) {
						val += 1;
					}
					else {
						val -= 1;
					}
				}
				else {
					val = y - x;
					if (x < mSize / 2) {
						val -= 1;
					}
					else {
						val += 1;
					}
				}
			} 
			convultionMatrix.at<uchar>(y, x) = (uchar)val;
			printf("[%d,%d] %d \n", y, x, val);
			val = 0;
		}
	}
	printf("Matrix completed..\n");
	return convultionMatrix;
}

void SetConvolution(cv::Mat pic8uc1, cv::Mat convultionMatrix, int x, int y, int devider) {
	int resultValue = 0;

	uchar pixelValue = 0;
	uchar convValue = 0;
	for (int i = 0; i < (convultionMatrix.rows); i++)
	{
		for (int j = 0; j < (convultionMatrix.cols); j++)
		{
			if ((x - i) >= 0 && (y - j) >= 0) {
				pixelValue = pic8uc1.at<uchar>(x - i, y - j);
				convValue = convultionMatrix.at<uchar>(i, j);
				resultValue += (pixelValue * convValue);
			}
		}
	}
	resultValue /= devider;
	pic8uc1.at<uchar>(x, y) = resultValue;
}

void DoConvolution(cv::Mat pic8uc1, cv::Mat convultionMatrix){
	const int matrixSize = convultionMatrix.rows * convultionMatrix.cols;
	printf("Prepare for Convultion...\n");
	for (int y = 0; y < pic8uc1.rows; y++) {
		for (int x = 0; x < pic8uc1.cols; x++) {
			SetConvolution(pic8uc1, convultionMatrix, x, y, matrixSize);
		}
	}
	printf("Convultion complete..\n");
}


uchar SetGamaToPixel(uchar pixel, double gama, uchar brightness) {
	//return (uchar)(pow(pixel, gama) / 255) + brightness;
	uchar result = ((((double)pixel / 255)*gama) * 255) + brightness;
	return result;
}

void ImgGamaEdit(cv::Mat pic8Uc3, double gama, uchar bright) {
	uchar r, g, b;
	for (int y = 0; y < pic8Uc3.rows; y++) {
		for (int x = 0; x < pic8Uc3.cols; x++) {
			b = SetGamaToPixel(pic8Uc3.at<cv::Vec3b>(y, x)[B], gama, bright);
			g = SetGamaToPixel(pic8Uc3.at<cv::Vec3b>(y, x)[G], gama, bright);
			r = SetGamaToPixel(pic8Uc3.at<cv::Vec3b>(y, x)[R], gama, bright);

			pic8Uc3.at<cv::Vec3b>(y, x) = { b, g, r };
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

double CalcComplexNumberABS(cv::Vec2d cn) {
	return (std::sqrt((cn[REAL]*cn[REAL])+(cn[IMAGINE]*cn[IMAGINE])));
}
//Base math: z1⋅z2=(x1+y1i)⋅(x2+y2i)=(x1x2−y1y2)+(x1y2+x2y1)i
cv::Vec2d CalcComplexNumberMultiple(cv::Vec2d cn_a, cv::Vec2d cn_b) {
	double realPart = (cn_a[REAL] * cn_b[REAL]) - (cn_a[IMAGINE] * cn_b[IMAGINE]);
	double complexPart = (cn_a[REAL] * cn_b[IMAGINE]) + (cn_b[REAL] * cn_a[IMAGINE]);
	return cv::Vec2d(realPart,complexPart);
}

double CalcComplexNumberScalarMultiple(cv::Vec2d cn_a, cv::Vec2d cn_b) {
	return (cn_a[REAL] * cn_b[REAL]) + (-1.0)*(cn_a[IMAGINE] * cn_b[IMAGINE]);
}

double CalcSpectrumAmplitude(double Real, double Imagine) {
	return CalcComplexNumberABS(cv::Vec2d(Real,Imagine));
}

double CalcSpectrumPower(double Real, double Imagine) {
	return (std::log((Real * Real) + (Imagine * Imagine)));
}

cv::Mat GetSpectrumAmplitude(cv::Mat ComplexMatrix){
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
			complexPart = ComplexMatrix.at<cv::Vec2d>(k, l)[IMAGINE];
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
			complexPart = ComplexMatrix.at<cv::Vec2d>(k, l)[IMAGINE];
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
			complexPart = ComplexMatrix.at<cv::Vec2d>(k, l)[IMAGINE];
			img.at<double>(k, l) = CalcSpectrumPower(realPart,complexPart);
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
	cv::Mat FreqSpectrumMatrix = cv::Mat(M,N,CV_64FC2);


	//Used variables:
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
					//Calculate exp(x) arg:
					x = (2 * CV_PI)*(((double)(k*m) / M) + ((double)(l*n) / N));
					realPart = (imgVal * std::cos(x));
					complexPart = (-imgVal * std::sin(x));

					realSum += realPart;
					complexSum += complexPart;
				}
			}
			base = cv::Vec2d(realSum, complexSum);
			FreqSpectrumMatrix.at<cv::Vec2d>(k, l) = base;
			realSum = complexSum = 0.0;
		}
	}
	printf("DFT done!\n");
	//switch_quadrants(BaseMatrix);
	return FreqSpectrumMatrix;
}

cv::Mat InverseDiscreteFourierTransform(cv::Mat matrixFreqSpectrum) {
	printf("Setting Inverse Discrete Fourier Transform..\n");
	const int M = matrixFreqSpectrum.rows;
	const int N = matrixFreqSpectrum.cols;

	//Output image
	cv::Mat img = cv::Mat(M, N, CV_64FC1);

	//Used variables
	double imgVal = 0, x = 0;
	cv::Vec2d Base, F, MultiplicationResult;

	printf("Calculating DFT..\n");
	for (size_t m = 0; m < M; m++)
	{
		printf("Inverse DFT progress: %2.0f %%\n", (double)m / M * 100.0);
		for (size_t n = 0; n < N; n++)
		{
			for (size_t k = 0; k < M; k++)
			{
				for (size_t l = 0; l < N; l++)
				{
					//Freq matrix value:
					F = matrixFreqSpectrum.at<cv::Vec2d>(k, l);
					//Calculate exp(x) arg:
					x = (2 * CV_PI)*(((double)(k*m) / M) + ((double)(l*n) / N));
					Base = cv::Vec2d(std::cos(x), std::sin(x));

					imgVal += CalcComplexNumberScalarMultiple(F, Base);
				}
			}
			img.at<double>(m, n) = imgVal;
			imgVal = 0;
		}
	}
	printf("Inverse DFT done!\n");
	cv::normalize(img, img, 0.0, 1.0, cv::NORM_MINMAX);
	return img;
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

cv::Mat CircleFilterMask(int rows, int cols, double diametr_ration, int mode){
	int radius = ((rows + cols) / 3)*diametr_ration;

	cv::Mat imgMask = cv::Mat(rows, cols, CV_64FC1);
	imgMask.setTo(cv::Scalar(1.0 - mode));

	cv::circle(imgMask, cv::Point(rows / 2, cols / 2), radius, cv::Scalar(mode), CV_FILLED);
	//switch_quadrants(imgMask);
	return imgMask;
}

cv::Mat LowPassFilter(cv::Mat matrixFreqSpectrum, double ratio){
	printf("Using LowPass Filter...\n");
	const int M = matrixFreqSpectrum.rows;
	const int N = matrixFreqSpectrum.cols;

	cv::Mat outMat = cv::Mat(M, N, CV_64FC2);
	cv::Mat filterMask = CircleFilterMask(M,N,ratio,LOW_PASS);
	//matrixFreqSpectrum *= filterMask;
	cv::Vec2d newPointValue = 0;

	for (int r = 0; r < M; r++)
	{
		for (int c = 0; c < N; c++)
		{
			newPointValue = matrixFreqSpectrum.at<cv::Vec2d>(r, c) * filterMask.at<double>(r, c);
			outMat.at<cv::Vec2d>(r, c) = newPointValue;
		}
	}
	//switch_quadrants(outMat);
	return outMat;
}
