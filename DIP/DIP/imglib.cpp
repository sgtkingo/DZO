#include "stdafx.h"

void ImageShow(cv::Mat img, std::string name)
{
	cv::imshow(name, img);
	cv::waitKey();
}

cv::Mat OpenImage(std::string file_path, int open_mode) {
	cv::Mat img = cv::imread(file_path, open_mode);
	if (img.empty())
	{
		printf("Unable to load image!\n");
		exit(-1);
	}
	std::string img_name = file_path.substr(file_path.find_last_of('/')+1);
	ImageShow(img, img_name);

	return img;
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
	for (int x = 0; x < pic8uc1.rows; x++) {
		for (int y = 0; y < pic8uc1.cols; y++) {
			SetConvolution(pic8uc1, convultionMatrix, x, y, matrixSize);
		}
	}
	printf("Convultion complete..\n");
}

float GetConvolution_F(const cv::Mat pic32uc1, cv::Mat convultionMatrix, int x, int y, int devider) {
	float resultValue = 0;

	float pixelValue = 0;
	float convValue = 0;
	for (int i = 0; i < (convultionMatrix.rows); i++)
	{
		for (int j = 0; j < (convultionMatrix.cols); j++)
		{
			if ((x - i) >= 0 && (y - j) >= 0) {
				pixelValue = pic32uc1.at<float>(x - i, y - j);
				convValue = convultionMatrix.at<float>(i, j);
				resultValue += (pixelValue * convValue);
			}
		}
	}
	resultValue /= devider;
	return resultValue;
}

cv::Mat DoConvolution_F(const cv::Mat pic32uc1, cv::Mat convultionMatrix) {
	cv::Mat result;
	result = cv::Mat(pic32uc1.rows, pic32uc1.cols, pic32uc1.type());

	const int matrixSize = convultionMatrix.rows * convultionMatrix.cols;
	float convValue = 0;
	printf("Prepare for Convultion...\n");
	for (int x = 0; x < result.rows; x++) {
		for (int y = 0; y < result.cols; y++) {
			convValue = GetConvolution_F(pic32uc1, convultionMatrix, x, y, matrixSize);
			result.at<float>(x, y) = convValue;
		}
	}
	printf("Convultion complete..\n");
	return result;
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

cv::Mat FilterMask(cv::Mat powerSpectrum, double ratio){
	const int M = powerSpectrum.rows;
	const int N = powerSpectrum.cols;
	cv::Mat imgMask = cv::Mat(M, N, CV_64FC1);

	for (int r = 0; r < M; r++)
	{
		for (int c = 0; c < N; c++)
		{
			if (powerSpectrum.at<double>(r, c) <= ratio ) {
				imgMask.at<double>(r, c) = 1.0;
			}else imgMask.at<double>(r, c) = 0.0;
		}
	}
	return imgMask;
}

cv::Mat LowPassFilter(cv::Mat matrixFreqSpectrum, double ratio){
	printf("Using LowPass Filter...\n");
	const int M = matrixFreqSpectrum.rows;
	const int N = matrixFreqSpectrum.cols;

	cv::Mat filterMask = CircleFilterMask(M,N,ratio,LOW_PASS);

	return Filter(matrixFreqSpectrum, filterMask);
}

cv::Mat HighPassFilter(cv::Mat matrixFreqSpectrum, double ratio) {
	printf("Using HighPass Filter...\n");
	const int M = matrixFreqSpectrum.rows;
	const int N = matrixFreqSpectrum.cols;

	cv::Mat filterMask = CircleFilterMask(M, N, ratio, HIGH_PASS);

	return Filter(matrixFreqSpectrum, filterMask);
}


cv::Mat Filter(cv::Mat matrixFreqSpectrum, cv::Mat filterMask) {
	printf("Using Filter...\n");
	ImageShow(filterMask);

	const int M = matrixFreqSpectrum.rows;
	const int N = matrixFreqSpectrum.cols;

	cv::Mat outMat = cv::Mat(M, N, CV_64FC2);
	cv::Vec2d newPointValue = 0;

	for (int r = 0; r < M; r++)
	{
		for (int c = 0; c < N; c++)
		{
			newPointValue = matrixFreqSpectrum.at<cv::Vec2d>(r, c) * filterMask.at<double>(r, c);
			outMat.at<cv::Vec2d>(r, c) = newPointValue;
		}
	}
	return outMat;
}

//Geo
struct RLDUserData {
	cv::Mat & src_8uc3_img;
	cv::Mat & undistorted_8uc3_img;
	int K1;
	int K2;

	RLDUserData(const int K1, const int K2, cv::Mat & src_8uc3_img, cv::Mat & undistorted_8uc3_img) : K1(K1), K2(K2), src_8uc3_img(src_8uc3_img), undistorted_8uc3_img(undistorted_8uc3_img) {}
};

inline cv::Vec3b pixel_bilinear_interpolation(const cv::Mat& img, double x, double y)
{
	int lowX = (int)ceil(x);
	int highX = (int)floor(x);

	int lowY = (int)ceil(y);
	int highY = (int)floor(y);

	cv::Vec3b bottomLeft = img.at<cv::Vec3b>(cv::Point(highX, highY));
	cv::Vec3b topLeft = img.at<cv::Vec3b>(cv::Point(highX, lowY));
	cv::Vec3b bottomRight = img.at<cv::Vec3b>(cv::Point(lowX, highY));
	cv::Vec3b topRight = img.at<cv::Vec3b>(cv::Point(lowX, lowY));

	cv::Vec3b R1 = (((lowX - x) / (lowX - highX)) * bottomLeft) + (((x - highX) / (lowX - highX)) * bottomRight);
	cv::Vec3b R2 = (((lowX - x) / (lowX - highX)) * topLeft) + (((x - highX) / (lowX - highX)) * topRight);
	cv::Vec3b result = ((((lowY - y) / (lowY - highY)) * R1) + (((y - highY) / (lowY - highY))*R2));
	return result;
}

cv::Mat bilinear_interpolation(const cv::Mat& img, double K1, double K2) 
{
	double _R2, fi, xD, yD, actualX, actualY;

	const int M = img.rows;
	const int N = img.cols;

	float Cv = (double)M / 2.0;
	float Cu = (double)N / 2.0;

	double _R = sqrt((pow(Cu, 2) + pow(Cv, 2)));

	cv::Mat outMat = cv::Mat(img.size(), img.type());

	for (double r = 0; r < M; r++)
	{
		actualY = (r - Cv) / _R;

		for (int c = 0; c < N; c++)
		{
			actualX = (c - Cu) / _R;

			_R2 = (pow(actualX, 2) + pow(actualY, 2));

			fi = (1 + (K1 * _R2) + (K2 * pow(_R2, 2)));
			fi = 1.0 / fi;

			xD = ((c - Cu) * fi) + Cu;
			yD = ((r - Cv) * fi) + Cv;

			outMat.at<cv::Vec3b>(r, c) = pixel_bilinear_interpolation(img, xD, yD);
		}
	}
	return outMat;
}

inline cv::Vec3b pixel_average_pixel(const cv::Mat& img, double x, double y)
{
	int lowX = (int)ceil(x);
	int highX = (int)floor(x);

	int lowY = (int)ceil(y);
	int highY = (int)floor(y);

	cv::Vec3f bottomLeft = img.at<cv::Vec3b>(cv::Point(highX, highY));
	cv::Vec3f topLeft = img.at<cv::Vec3b>(cv::Point(highX, lowY));
	cv::Vec3f bottomRight = img.at<cv::Vec3b>(cv::Point(lowX, highY));
	cv::Vec3f topRight = img.at<cv::Vec3b>(cv::Point(lowX, lowY));


	cv::Vec3b result = (bottomLeft + topLeft + bottomRight + topRight) / 4;
	return result;
}

cv::Mat average_pixel(const cv::Mat& img, double K1, double K2)
{
	double _R2, fi, xD, yD, actualX, actualY;

	const int M = img.rows;
	const int N = img.cols;

	float Cv = (double)M / 2.0;
	float Cu = (double)N / 2.0;
	float _R = sqrt((pow(Cu, 2) + pow(Cv, 2)));

	cv::Mat outMat = cv::Mat(img.size(), img.type());

	for (double r = 0; r < M; r++)
	{
		actualY = (r - Cv) / _R;

		for (int c = 0; c < N; c++)
		{
			actualX = (c - Cu) / _R;

			_R2 = (pow(actualX, 2) + pow(actualY, 2));

			fi = (1 + (K1 * _R2) + (K2 * pow(_R2, 2)));
			fi = 1.0 / fi;

			xD = ((c - Cu) * fi) + Cu;
			yD = ((r - Cv) * fi) + Cv;

			outMat.at<cv::Vec3b>(r, c) = pixel_average_pixel(img, xD, yD);
		}
	}
	return outMat;
}

void geom_dist(cv::Mat & src_8uc3_img, cv::Mat & dst_8uc3_img, bool bili, double K1, double K2)
{
	//cv::resize(src_8uc3_img, dst_8uc3_img,cv::Size(),K1,K2);
	if (bili)
		dst_8uc3_img = bilinear_interpolation(src_8uc3_img, K1, K2);
	else
		dst_8uc3_img = average_pixel(src_8uc3_img, K1, K2);
}

void apply_rld(int id, void * user_data)
{
	RLDUserData *rld_user_data = (RLDUserData*)user_data;

	geom_dist(rld_user_data->src_8uc3_img, rld_user_data->undistorted_8uc3_img, !false, rld_user_data->K1 / 100.0, rld_user_data->K2 / 100.0);
	cv::imshow("Geom Dist", rld_user_data->undistorted_8uc3_img);
}

int ex_rld()
{
	cv::Mat src_8uc3_img, geom_8uc3_img;
	RLDUserData rld_user_data(50.0, 50.0, src_8uc3_img, geom_8uc3_img);

	src_8uc3_img = cv::imread("images/distorted_window.jpg", cv::IMREAD_COLOR);
	if (src_8uc3_img.empty())
	{
		printf("Unable to load image!\n");
		exit(-1);
	}

	cv::namedWindow("Original Image");
	cv::imshow("Original Image", src_8uc3_img);

	geom_8uc3_img = src_8uc3_img.clone();
	apply_rld(0, (void*)&rld_user_data);

	cv::namedWindow("Geom Dist");
	cv::imshow("Geom Dist", geom_8uc3_img);

	cv::createTrackbar("K1", "Geom Dist", &rld_user_data.K1, 100, apply_rld, &rld_user_data);
	cv::createTrackbar("K2", "Geom Dist", &rld_user_data.K2, 100, apply_rld, &rld_user_data);

	cv::waitKey(0);

	return 0;
}

//Histogram
uint cumulative_distribution_function(const std::vector<uint> histogram, uchar brightnessValue)
{
	uint result = 0;
	for (int j = 0; j <= brightnessValue; j++)
	{
		result += histogram[j];
	}
	return result;
}

std::vector<uint> create_histogram(const cv::Mat & src, int L_value)
{
	printf("Creating Histogram vector...\n");
	std::vector<uint> histogram ;
	// Init histogram
	histogram.resize(L_value);
	for (size_t i = 0; i < L_value; i++)
	{
		histogram[i] = 0;
	}

	uchar pixelValue = 0;
	for (size_t r = 0; r < src.rows; r++)
	{
		for (size_t c = 0; c < src.cols; c++)
		{
			pixelValue = src.at<uchar>(r, c);
			histogram[pixelValue] += 1;
		}
	}
	return histogram;
}

std::vector<uchar> equalized_brightness(const std::vector<uint> histogram, const uint width, const uint height, int L_value)
{
	printf("Creating Equalized Brightness vector...\n");
	std::vector<uchar> output;
	// Init histogram
	output.resize(L_value);
	for (size_t i = 0; i < L_value; i++)
	{
		output[i] = 0;
	}

	double minCdf = *min_element(histogram.begin(), histogram.end());
	double cdf = 0;
	double correctValue = 0;
	for (size_t i = 0; i < L_value; i++)
	{
		cdf = cumulative_distribution_function(histogram, i);
		correctValue = round(((cdf - minCdf) / ((width*height) - minCdf)) * (L_value - 1));

		if (correctValue < 0) correctValue = 0;
		output[i] = (uchar)(correctValue);
	}
	return output;
}

cv::Mat Histogram(const cv::Mat src) 
{
	printf("Preparing Histogram...\n");
	cv::Mat outImg = cv::Mat(src.size(), src.type());

	std::vector<uint> histogram = create_histogram(src, L);
	std::vector<uchar> lut = equalized_brightness(histogram, src.cols, src.rows, L);

	uchar pixelValue = 0;
	for (unsigned int row = 0; row < src.rows; row++)
	{
		for (unsigned int col = 0; col < src.cols; col++)
		{
			pixelValue = src.at<uchar>(row, col);
			outImg.at<uchar>(row, col) = lut[pixelValue];
		}
	}
	printf("Histogram Eq Done!\n");
	return outImg;
}
/*
void FillTransformationMatrix(cv::Mat & A_mat, cv::Mat & B_mat, const std::vector<cv::Point2i> inPoints, const std::vector<cv::Point2i> outPoints) {
	printf("Filling transform matrix...\n");
	cv::Point2i xI, xO;

	const int rows = (A_mat.rows / 2);
	//getting values to the matrix A and B 
	for (size_t row = 0; row < rows; row++)
	{
		xI= inPoints[row];
		xO = outPoints[row];

		A_mat.at<double>(row * 2, 0) = (double)xO.y;
		A_mat.at<double>(row * 2, 1) = 1.0f;
		A_mat.at<double>(row * 2, 2) = 0.0f;
		A_mat.at<double>(row * 2, 3) = 0.0f;
		A_mat.at<double>(row * 2, 4) = 0.0f;
		A_mat.at<double>(row * 2, 5) = (double)(-xI.x * xO.x);
		A_mat.at<double>(row * 2, 6) = (double)(-xI.x * xO.y);
		A_mat.at<double>(row * 2, 7) = (double)(-xI.x);

		A_mat.at<double>((row * 2) + 1, 0) = 0.0f;
		A_mat.at<double>((row * 2) + 1, 1) = 0.0f;
		A_mat.at<double>((row * 2) + 1, 2) = (double)xO.x;
		A_mat.at<double>((row * 2) + 1, 3) = (double)xO.y;
		A_mat.at<double>((row * 2) + 1, 4) = 1.0f;
		A_mat.at<double>((row * 2) + 1, 5) = (double)(-xI.y * xO.x);
		A_mat.at<double>((row * 2) + 1, 6) = (double)(-xI.y * xO.y);
		A_mat.at<double>((row * 2) + 1, 7) = (double)(-xI.y);

		B_mat.at<double>(row * 2, 0) = (double)-xO.x;
		B_mat.at<double>((row * 2) + 1, 0) = 0.0f;
	}
}

cv::Mat PrepareTransformationMatrix(const std::vector<cv::Point2i> bgPoints, const std::vector<cv::Point2i> overlayPoints) {
	printf("Prepare transform matrix...\n");
	//Prepare matrixs
	cv::Mat A_mat = cv::Mat(8, 8, CV_64FC1);
	cv::Mat B_mat = cv::Mat(8, 1, CV_64FC1);
	FillTransformationMatrix(A_mat, B_mat, overlayPoints, bgPoints);

	//Do math
	cv::Mat matResult = cv::Mat(8, 1, CV_64FC1);
	cv::solve(A_mat, B_mat, matResult);
	for (size_t r = 0; r < matResult.rows; r++)
	{
		for (size_t c = 0; c < matResult.cols; c++)
		{
			printf("[%d][%d] %f \n", r, c, matResult.at<double>(r,c));
		}
	}

	return matResult;
}

cv::Mat CalculateTransformationMatrix(const cv::Mat & transMatrix) {
	printf("Calculate transform matrix...\n");
	//Tranform prepare
	cv::Mat transformationMatrix = cv::Mat(3, 3, CV_64FC1);
	transformationMatrix.at<double>(0, 0) = 1.0f;

	unsigned int index = 0;
	for (unsigned int r = 0; r < transformationMatrix.rows; r++)
	{
		for (unsigned int c = 0; c < transformationMatrix.cols; c++)
		{
			if (!(r == 0 && r == 0))
				transformationMatrix.at<double>(r, c) = transMatrix.at<double>(index++, 0);
			printf("[%d][%d] %f \n", r, c, transformationMatrix.at<double>(r, c));
		}
	}
	return transformationMatrix;
}

cv::Mat PerspectiveTransformation(const cv::Mat & input, const cv::Mat & overlay, const std::vector<cv::Point2i> bgPoints, const std::vector<cv::Point2i> overlayPoints)
{
	printf("Perspective transform starting...\n");
	if (bgPoints.size() != 4 || overlayPoints.size() != 4)
		throw new std::exception("There must be 4 pairs of points.");
	//Prepare result img
	cv::Mat result = cv::Mat(input.rows, input.cols, input.type());
	input.copyTo(result);

	cv::Mat matResult = PrepareTransformationMatrix(bgPoints, overlayPoints);
	cv::Mat transformationMatrix = CalculateTransformationMatrix(matResult);

	cv::Mat coord = cv::Mat(3, 1, CV_64FC1);
	cv::Mat homog = cv::Mat(3, 1, CV_64FC1);
	coord.at<double>(2, 0) = 1.0f;

	//Do transformation
	printf("Transformation...\n");
	uint transformedX, transformedY;
	double homogX, homogY, homogW;

	for (unsigned int r = 0; r < overlay.rows; r++)
	{
		coord.at<double>(1, 0) = (double)r;
		for (unsigned int c = 0; c < overlay.cols; c++)
		{
			coord.at<double>(0, 0) = (double)c;

			homog = transformationMatrix * coord;
			homogX = homog.at<double>(0, 0);
			homogY = homog.at<double>(1, 0);
			homogW = homog.at<double>(2, 0);

			transformedX = (uint)(homogX / homogW);
			transformedY = (uint)(homogY / homogW);

			cv::Vec3b overlayPixel = overlay.at<cv::Vec3b>(r, c);

			if( transformedX < result.rows && transformedY < result.cols )
				result.at<cv::Vec3b>(transformedX, transformedY) = overlayPixel;
		}
		cv::imshow("Result", result);
		cv::waitKey(50);
	}

	return result;
}
*/

cv::Mat perspective_transformation(const cv::Mat & input, const cv::Mat & overlay, const std::vector<std::pair<cv::Point2i, cv::Point2i>>& points)
{
	printf("Perspective transform starting...\n");
	if (points.size() != 4)
		throw new std::exception("There must be 4 pairs of points.");
	//Prepare result
	cv::Mat result = cv::Mat(input.rows, input.cols, input.type());
	input.copyTo(result);

	cv::Mat A_mat = cv::Mat(8, 8, CV_64FC1);
	cv::Mat B_mat = cv::Mat(8, 1, CV_64FC1);

	cv::Point2i x, xF;

	for (size_t row = 0; row < (A_mat.rows / 2); row++)
	{
		xF = points[row].second;
		x = points[row].first;

		A_mat.at<double>(row * 2, 0) = (double)x.y;
		A_mat.at<double>(row * 2, 1) = 1.0f;
		A_mat.at<double>(row * 2, 2) = 0.0f;
		A_mat.at<double>(row * 2, 3) = 0.0f;
		A_mat.at<double>(row * 2, 4) = 0.0f;
		A_mat.at<double>(row * 2, 5) = (double)(-xF.x * x.x);
		A_mat.at<double>(row * 2, 6) = (double)(-xF.x * x.y);
		A_mat.at<double>(row * 2, 7) = (double)(-xF.x);

		A_mat.at<double>((row * 2) + 1, 0) = 0.0f;
		A_mat.at<double>((row * 2) + 1, 1) = 0.0f;
		A_mat.at<double>((row * 2) + 1, 2) = (double)x.x;
		A_mat.at<double>((row * 2) + 1, 3) = (double)x.y;
		A_mat.at<double>((row * 2) + 1, 4) = 1.0f;
		A_mat.at<double>((row * 2) + 1, 5) = (double)(-xF.y * x.x);
		A_mat.at<double>((row * 2) + 1, 6) = (double)(-xF.y * x.y);
		A_mat.at<double>((row * 2) + 1, 7) = (double)(-xF.y);

		B_mat.at<double>(row * 2, 0) = (double)-x.x;
		B_mat.at<double>((row * 2) + 1, 0) = 0.0f;
	}

	cv::Mat matResult = cv::Mat(8, 1, CV_64FC1);

	cv::solve(A_mat, B_mat, matResult);

	cv::Mat transformationMatrix = cv::Mat(3, 3, CV_64FC1);
	transformationMatrix.at<double>(0, 0) = 1.0f;

	unsigned int index = 0;
	for (unsigned int row = 0; row < transformationMatrix.rows; row++)
	{
		for (unsigned int col = 0; col < transformationMatrix.cols; col++)
		{
			if (!(row == 0 && col == 0))
				transformationMatrix.at<double>(row, col) = matResult.at<double>(index++, 0);
		}
	}

	cv::Mat coord = cv::Mat(3, 1, CV_64FC1);
	cv::Mat homog = cv::Mat(3, 1, CV_64FC1);
	coord.at<double>(2, 0) = 1.0f;

	uint transformedX, transformedY;
	double homogX, homogY, homogW;
	for (unsigned int row = 0; row < overlay.rows; row++)
	{
		coord.at<double>(1, 0) = (double)row;
		for (unsigned int col = 0; col < overlay.cols; col++)
		{
			coord.at<double>(0, 0) = (double)col;

			homog = transformationMatrix * coord;
			homogX = homog.at<double>(0, 0);
			homogY = homog.at<double>(1, 0);
			homogW = homog.at<double>(2, 0);

			transformedX = (uint)(homogX / homogW);
			transformedY = (uint)(homogY / homogW);

			cv::Vec3b overlayPixel = overlay.at<cv::Vec3b>(row, col);
			if (transformedX < result.rows && transformedY < result.cols)
				result.at<cv::Vec3b>(transformedY, transformedX) = overlayPixel;
		}
	}

	return result;
}

void apply_sobel_operator(cv::Mat & input_bw_32F)
{
	printf("Appling Sobel operator...\n");
	float xMaskData[9] = { 1.0f,0.0f,-1.0f,2.0f,0.0f,-2.0f,1.0f,0.0f,-1.0f };
	float yMaskData[9] = { 1.0f,2.0f,1.0f,0.0f,0.0f,0.0f,-1.0f,-2.0f,-1.0f };
	cv::Mat xMask = cv::Mat(3, 3, CV_32FC1, xMaskData);
	cv::Mat yMask = cv::Mat(3, 3, CV_32FC1, yMaskData);

	cv::Mat Gx = DoConvolution_F(input_bw_32F, xMask);
	cv::Mat Gy = DoConvolution_F(input_bw_32F, yMask);;

	const int N = input_bw_32F.rows;
	const int M = input_bw_32F.cols;

	float G_value, Gx_value, Gy_value;
	for (size_t r = 0; r < N; r++)
	{
		for (size_t c = 0; c < M; c++)
		{
			Gx_value = Gx.at<float>(r, c);
			Gy_value = Gy.at<float>(r, c);
			G_value = sqrt((Gx_value*Gx_value)+ (Gy_value*Gy_value));
			input_bw_32F.at<float>(r, c) = G_value;
		}
	}
	printf("Appling Sobel operator DONE!\n");
}

cv::Mat EdgeDetection_Sobel(const cv::Mat & input) {
	printf("Edge Detection starting...\n");
	cv::Mat img_bw_32F; input.copyTo(img_bw_32F);

	if (img_bw_32F.type() != CV_32FC1) {
		printf("Bad input format! Converting...\n");
		img_bw_32F.convertTo(img_bw_32F, CV_32FC1, 1.0 / 255);
	}
	apply_sobel_operator(img_bw_32F);

	return img_bw_32F;
}

void apply_non_maxima_operator(cv::Mat & input_bw_32F) {
	float fx = 0;
	float fy = 0;

	float alpha = 0, alpha_deg = 0, a = 0, EpO = 0, EnO = 0, E = 0;

	for (size_t r = 1; r < input_bw_32F.rows -1; r++)
	{
		for (size_t c = 1; c < input_bw_32F.cols -1; c++)
		{
			fx = (input_bw_32F.at<float>(r - 1,c) - input_bw_32F.at<float>(r + 1, c)) / 2;
			fy = (input_bw_32F.at<float>(r, c - 1) - input_bw_32F.at<float>(r, c + 1)) / 2;
			E = input_bw_32F.at<float>(r, c);

			alpha = atan2(fy, fx);
			alpha_deg = (CV_PI/180)*alpha;
			a = tan(alpha) * input_bw_32F.at<float>(r + 1, c);

			EpO = a*(input_bw_32F.at<float>(r + 1, c + 1) + ((1 - a)*input_bw_32F.at<float>(r + 1, c)));
			EnO = a*(input_bw_32F.at<float>(r - 1, c - 1) + ((1 - a)*input_bw_32F.at<float>(r - 1, c)));

			if (alpha_deg >= 0.0f && alpha_deg <= 90.0f)
			{
				EpO = (a * input_bw_32F.at<float>(r + 1, c + 1)) + ((1 - a) * input_bw_32F.at<float>(r, c + 1));
				EnO = (a * input_bw_32F.at<float>(r - 1, c - 1)) + ((1 - a) * input_bw_32F.at<float>(r, c - 1));
			}
			else if (alpha_deg >= 90.0f && alpha_deg <= 180.0f)
			{
				EpO = (a * input_bw_32F.at<float>(r + 1, c - 1)) + ((1 - a) * input_bw_32F.at<float>(r, c - 1));
				EnO = (a * input_bw_32F.at<float>(r + 1, c + 1)) + ((1 - a) * input_bw_32F.at<float>(r, c + 1));
			}
			else if (alpha_deg >= 180.0f && alpha_deg <= 270.0f)
			{
				EpO = (a * input_bw_32F.at<float>(r - 1, c - 1)) + ((1 - a) * input_bw_32F.at<float>(r, c - 1));
				EnO = (a * input_bw_32F.at<float>(r + 1, c + 1)) + ((1 - a) * input_bw_32F.at<float>(r, c + 1));
			}
			else if (alpha_deg >= 270.0f && alpha_deg <= 360.0f)
			{
				EpO = (a * input_bw_32F.at<float>(r + 1, c + 1)) + ((1 - a) * input_bw_32F.at<float>(r, c + 1));
				EnO = (a * input_bw_32F.at<float>(r + 1, c - 1)) + ((1 - a) * input_bw_32F.at<float>(r, c - 1));
			}
			else
			{
				continue;
			}
			if (!(E > EpO && E > EnO))
				input_bw_32F.at<float>(r, c) = 0.0;
		}
	}

}

cv::Mat EdgeThining(const cv::Mat & input){
	printf("Edge Thining starting...\n");
	cv::Mat img_bw_32F; input.copyTo(img_bw_32F);

	if (img_bw_32F.type() != CV_32FC1) {
		printf("Bad input format! Converting...\n");
		img_bw_32F.convertTo(img_bw_32F, CV_32FC1, 1.0 / 255);
	}
	apply_non_maxima_operator(img_bw_32F);
	printf("Edge Thining done!\n");
	return img_bw_32F;
}

bool is_edge(const cv::Mat &src, const uint row, const uint col, const double t1, const double t2)
{
	float pixel_val = src.at<float>(row, col);
	if (pixel_val >= t2)
	{
		return true;
	}
	else if (pixel_val >= t1 && pixel_val <= t2)
	{
		return (src.at<float>(row - 1, col) > t2) || (src.at<float>(row + 1, col) > t2) || (src.at<float>(row, col - 1) > t2) || (src.at<float>(row, col + 1) > t2);
		//return is_edge(src, row - 1, col, t1, t2) || is_edge(src, row + 1, col, t1, t2) || is_edge(src, row, col - 1, t1, t2) || is_edge(src, row, col + 1, t1, t2);
	}
	else
	{
		return false;
	}
}

cv::Mat double_threshold(const cv::Mat &src, const int t1, const int t2, const int t_max)
{
	cv::Mat dst = cv::Mat(src.rows, src.cols, src.type());
	uint border = 1;
	for (unsigned int row = border; row < src.rows - border; row++)
	{
		for (unsigned int col = border; col < src.cols - border; col++)
		{
			dst.at<float>(row, col) = is_edge(src, row, col, (double)t1/t_max, (double)t2/t_max) ? 1.0 : 0.0;
		}
	}
	return dst;
}

struct DT_UserData {
	cv::Mat & src_32uc1_img;
	cv::Mat & result_32uc1_img;
	int TR1;
	int TR2;
	int TR_MAX;
	DT_UserData(const int TR1, const int TR2, const int TR_MAX, cv::Mat & src_32uc1_img, cv::Mat & result_32uc1_img) : TR1(TR1), TR2(TR2), TR_MAX(TR_MAX), src_32uc1_img(src_32uc1_img), result_32uc1_img(result_32uc1_img) {}
};

void apply_double_treashold(int id, void * user_data)
{
	DT_UserData *dt_user_data = (DT_UserData*)user_data;

	dt_user_data->result_32uc1_img = double_threshold(dt_user_data->src_32uc1_img, dt_user_data->TR1, dt_user_data->TR2, dt_user_data->TR_MAX);
	cv::imshow("Double_Treashold", dt_user_data->result_32uc1_img);
}

void EdgeThining_DoubleTreashold(cv::Mat &input)
{
	printf("Prepairing Double Treashold Playground...!\n");
	cv::Mat src_img_sobel = EdgeDetection_Sobel(input);
	//ImageShow(src_img_sobel, "Valve SOBEL");
	cv::Mat src_img_thin_edge = EdgeThining(src_img_sobel);
	//ImageShow(src_img_thin_edge, "Valve EDGE THINING");

	cv::Mat result;
	const int tr_max = 25;
	DT_UserData dt_user_data(10, 12, tr_max, src_img_thin_edge, result);

	cv::namedWindow("Double_Treashold");

	cv::createTrackbar("TR1", "Double_Treashold", &dt_user_data.TR1, dt_user_data.TR_MAX, apply_double_treashold, &dt_user_data);
	cv::createTrackbar("TR2", "Double_Treashold", &dt_user_data.TR2, dt_user_data.TR_MAX, apply_double_treashold, &dt_user_data);
	printf("Double Treashold Playground READY!\n");
	cv::waitKey(0);
}