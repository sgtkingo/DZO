// DIP.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

//original image reference
cv::Mat src_8uc3_img;

void MouseEventHandler(int event, int x, int y, int flag, void* Data) {
	cv::Mat* image = (cv::Mat*)(Data);
	
	double gama = 0;
	uchar brightness = 0;
	if (event == CV_EVENT_LBUTTONDOWN) {
		//recopy original to clone
		src_8uc3_img.copyTo(*image);

		//calcula gama and brightness values
		gama = ((double)x/image->rows)*1.0;
		brightness = ((double)y/image->cols)*25;
		printf("X: %d (Gama: %f), Y:%d (Brightness: %d) \n", x, gama, y, brightness);
		//apply gama edit function
		ImgGamaEdit(*image, gama, brightness);

		cv::imshow("LENA with GAMA", *image);
	}
}

int main()
{
	
	src_8uc3_img = cv::imread("images/lena.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	if (src_8uc3_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
		return 1;
	}

	/*
	cv::Mat anisotropic_8uc1_img = cv::imread("images/valve.png", CV_LOAD_IMAGE_GRAYSCALE); // load grayscale image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	if (anisotropic_8uc1_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
		return 1;
	}
	//Image for Discrete Fourier Transform
	cv::Mat earth_64f1_img = cv::imread("images/earth.png", CV_LOAD_IMAGE_GRAYSCALE); // load grayscale image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	if (earth_64f1_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
		return 1;
	}
	cv::Mat lena64noise_64f1_img = cv::imread("images/lena64_noise.png", CV_LOAD_IMAGE_GRAYSCALE); // load grayscale image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	if (lena64noise_64f1_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
		return 1;
	}
	cv::Mat lena64bars_64f1_img = cv::imread("images/lena64_bars.png", CV_LOAD_IMAGE_GRAYSCALE); // load grayscale image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	if (lena64bars_64f1_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
		return 1;
	}
	*/
	/*
	//Prepare images
	printf("Image preparing..\n");
	earth_64f1_img.convertTo(earth_64f1_img, CV_64FC1, 1.0 / 255.0);
	lena64noise_64f1_img.convertTo(lena64noise_64f1_img, CV_64FC1, 1.0 / 255.0);
	lena64bars_64f1_img.convertTo(lena64bars_64f1_img, CV_64FC1, 1.0 / 255.0);
	
	//ImageShow(lena64bars_64f1_img);
	
	//Convert 8bit uchar to 64bit double format
	cv::Mat anisotropic_64f1_img;
	anisotropic_8uc1_img.convertTo(anisotropic_64f1_img, CV_64FC1, 1.0 / 255.0);
	//greayscale convertion
	cv::Mat gray_8uc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 8 bits (uchar)
	cv::cvtColor(src_8uc3_img, gray_8uc1_img, CV_BGR2GRAY); // convert input color image to grayscale one, CV_BGR2GRAY specifies direction of conversion
	//Blur img
	cv::Mat blur_8uc1_img;
	gray_8uc1_img.copyTo(blur_8uc1_img);
	*/

	//Presetting change gama
	//Gama img
	cv::Mat src_8uc3_img_gama;
	src_8uc3_img.copyTo(src_8uc3_img_gama);
	printf("Setting gama event..\n");
	//Set mouse callback event
	cv::namedWindow("LENA with GAMA");
	cv::setMouseCallback("LENA with GAMA", MouseEventHandler,(void*)&src_8uc3_img_gama);
	/*
	//Try using convultion
	cv::Mat convulutionMatrix = CreateConvolutionMatrix(3);
	DoConvolution(blur_8uc1_img, convulutionMatrix);

	//Try use anisotropic
	printf("Setting anisotropic..\n");
	double o = 0.015;
	double lambda = 0.1;
	int iteration_ratio = 1;
	DoAnisoptropicIterations(anisotropic_64f1_img, iteration_ratio, o, lambda);

	//Try Founier Transform
	cv::Mat small_img_64f1;

	cv::Mat FourierFreqSpectrum;

	cv::Mat FourierAmplitudeSpectre;
	cv::Mat FourierPhasseImage;
	cv::Mat FourierPowerSpectre;

	cv::Mat FiltredFreqSpectrum;
	double filter_ratio = 0.8;

	cv::Mat ReconstructedImg;

	//LOW-PASS
	printf("Try DFT...\n");
	small_img_64f1 = lena64noise_64f1_img.clone();
	cv::imshow("INPUT", small_img_64f1);
	FourierFreqSpectrum = DiscreteFourierTransform(small_img_64f1);

	FourierAmplitudeSpectre = GetSpectrumAmplitude(FourierFreqSpectrum);
	FourierPhasseImage = GetPhasseImage(FourierFreqSpectrum);
	FourierPowerSpectre = GetPowerSpectrum(FourierFreqSpectrum);

	//show images
	cv::imshow("INPUT - BASE AMPLITUDE SPECTRE", FourierAmplitudeSpectre);
	cv::imshow("INPUT - BASE PHASSE IMAGE", FourierPhasseImage);
	cv::imshow("INPUT - BASE POWER SPECTRE", FourierPowerSpectre);
	//Wait
	cv::waitKey();
	//Filter
	FiltredFreqSpectrum = LowPassFilter(FourierFreqSpectrum, filter_ratio);
	//Try Inverse Founier Transform
	printf("Try Inverse DFT...\n");
	ReconstructedImg = InverseDiscreteFourierTransform(FiltredFreqSpectrum);
	cv::imshow("RECONSTRUCTED IMAGE", ReconstructedImg);

	//HIGH-PASS
	small_img_64f1 = lena64noise_64f1_img.clone();
	cv::imshow("INPUT", small_img_64f1);
	FourierFreqSpectrum = DiscreteFourierTransform(small_img_64f1);

	FourierAmplitudeSpectre = GetSpectrumAmplitude(FourierFreqSpectrum);
	FourierPhasseImage = GetPhasseImage(FourierFreqSpectrum);
	FourierPowerSpectre = GetPowerSpectrum(FourierFreqSpectrum);

	//show images
	cv::imshow("INPUT - BASE AMPLITUDE SPECTRE", FourierAmplitudeSpectre);
	cv::imshow("INPUT - BASE PHASSE IMAGE", FourierPhasseImage);
	cv::imshow("INPUT - BASE POWER SPECTRE", FourierPowerSpectre);
	//Wait
	cv::waitKey();
	//Filter
	FiltredFreqSpectrum = HighPassFilter(FourierFreqSpectrum, filter_ratio);
	//Try Inverse Founier Transform
	printf("Try Inverse DFT...\n");
	ReconstructedImg = InverseDiscreteFourierTransform(FiltredFreqSpectrum);
	cv::imshow("RECONSTRUCTED IMAGE", ReconstructedImg);

	//BARS
	printf("Try DFT...\n");
	small_img_64f1 = lena64bars_64f1_img.clone();
	cv::imshow("INPUT", small_img_64f1);
	FourierFreqSpectrum = DiscreteFourierTransform(small_img_64f1);

	FourierAmplitudeSpectre = GetSpectrumAmplitude(FourierFreqSpectrum);
	FourierPhasseImage = GetPhasseImage(FourierFreqSpectrum);
	FourierPowerSpectre = GetPowerSpectrum(FourierFreqSpectrum);

	//show images
	cv::imshow("INPUT - BASE AMPLITUDE SPECTRE", FourierAmplitudeSpectre);
	cv::imshow("INPUT - BASE PHASSE IMAGE", FourierPhasseImage);
	cv::imshow("INPUT - BASE POWER SPECTRE", FourierPowerSpectre);
	//Wait
	cv::waitKey();
	//Filter
	FiltredFreqSpectrum = Filter(FourierFreqSpectrum, FilterMask(FourierAmplitudeSpectre, 0.9));

	//Try Inverse Founier Transform
	printf("Try Inverse DFT...\n");
	ReconstructedImg = InverseDiscreteFourierTransform(FiltredFreqSpectrum);
	cv::imshow("RECONSTRUCTED IMAGE", ReconstructedImg);
	
	/*
	cv::imshow("VALVE", anisotropic_8uc1_img);
	cv::imshow("VALVE ANTISOPTROPIC", anisotropic_64f1_img);
	cv::imshow("LENA BLUR", blur_8uc1_img);
	cv::imshow("LENA with GAMA", src_8uc3_img_gama);
	cv::imshow("LENA", src_8uc3_img);
	*/
	/*
	printf("Try Distortion...\n");
	if (!ex_rld()) {
		printf("Distortion playground exit...\n");
	}
	*/
	/*
	printf("Try Histogram...\n");
	cv::Mat src_img_hist  = OpenImage("images/uneq.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat histogram_img = Histogram(src_img_hist);
	ImageShow(histogram_img, "Histogram Equalized Image");
	*/
	/*
	printf("Try Perspective...\n");
	cv::Mat src_img_pers= OpenImage("images/vsb.jpg", CV_LOAD_IMAGE_COLOR);
	cv::Mat src_img_overlay = OpenImage("images/flag.png", CV_LOAD_IMAGE_COLOR);
	printf("Points prepairing...\n");
	
	std::vector<std::pair<cv::Point2i, cv::Point2i>> points;

	std::pair<cv::Point2i, cv::Point2i> pair_1;
	pair_1.second = cv::Point2i(69, 107);
	pair_1.first = cv::Point2i(0, 0);

	std::pair<cv::Point2i, cv::Point2i> pair_2;
	pair_2.second = cv::Point2i(227, 76);
	pair_2.first = cv::Point2i(323, 0);

	std::pair<cv::Point2i, cv::Point2i> pair_3;
	pair_3.second = cv::Point2i(228, 122);
	pair_3.first = cv::Point2i(323, 215);

	std::pair<cv::Point2i, cv::Point2i> pair_4;
	pair_4.second = cv::Point2i(66, 134);
	pair_4.first = cv::Point2i(0, 215);

	points.push_back(pair_1);
	points.push_back(pair_2);
	points.push_back(pair_3);
	points.push_back(pair_4);
	
	cv::Mat perspective_img = perspective_transformation(src_img_pers, src_img_overlay, points);
	ImageShow(perspective_img, "Perspective transform Image");
	*/

	printf("Try Sobel Edge Detection...\n");
	cv::Mat src_img_valve = OpenImage("images/valve.png", CV_LOAD_IMAGE_GRAYSCALE);
	/*
	cv::Mat src_img_sobel = EdgeDetection_Sobel(src_img_valve);
	ImageShow(src_img_sobel, "Valve SOBEL");

	cv::Mat src_img_thin_edge = EdgeThining(src_img_sobel);
	ImageShow(src_img_thin_edge, "Valve EDGE THINING");
	*/
	printf("Try Double Treashold...\n");
	EdgeThining_DoubleTreashold(src_img_valve);
	
	ImageShow(src_8uc3_img_gama, "LENA with GAMA");
	//Press key to exit
	return 0;
}
