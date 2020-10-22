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
	//cv::imshow( "LENA", src_8uc3_img );

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
	cv::Mat lena64_64f1_img = cv::imread("images/lena64.png", CV_LOAD_IMAGE_GRAYSCALE); // load grayscale image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	if (lena64_64f1_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
		return 1;
	}
	//Prepare images
	printf("Image preparing..\n");
	earth_64f1_img.convertTo(earth_64f1_img, CV_64FC1, 1.0 / 255.0);
	lena64_64f1_img.convertTo(lena64_64f1_img, CV_64FC1, 1.0 / 255.0);
	//Convert 8bit uchar to 64bit double format
	cv::Mat anisotropic_64f1_img;
	anisotropic_8uc1_img.convertTo(anisotropic_64f1_img, CV_64FC1, 1.0 / 255.0);
	//greayscale convertion
	cv::Mat gray_8uc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 8 bits (uchar)
	cv::cvtColor(src_8uc3_img, gray_8uc1_img, CV_BGR2GRAY); // convert input color image to grayscale one, CV_BGR2GRAY specifies direction of conversion
	//Gama img
	cv::Mat src_8uc3_img_gama;
	src_8uc3_img.copyTo(src_8uc3_img_gama);
	//Blir img
	cv::Mat blur_8uc1_img;
	gray_8uc1_img.copyTo(blur_8uc1_img);
	
	//Presetting change gama
	printf("Setting gama event..\n");
	//Set mouse callback event
	cv::namedWindow("LENA with GAMA");
	cv::setMouseCallback("LENA with GAMA", MouseEventHandler,(void*)&src_8uc3_img_gama);

	//Try using convultion
	cv::Mat convulutionMatrix = CreateConvolutionMatrix(3);
	DoConvolution(blur_8uc1_img, convulutionMatrix);

	//Try use anisotropic
	printf("Setting anisotropic..\n");
	double o = 0.015;
	double lambda = 0.1;
	int iteration_ratio = 1;
	DoAnisoptropicIterations(anisotropic_64f1_img, iteration_ratio, o, lambda);

	/**/
	cv::Mat filterMask = FilterMask(100, 100, 0.7, LOW_PASS);
	cv::imshow("LOWPASS FILTERED MASK", filterMask);
	cv::waitKey();
	*/
	//Try Founier Transform
	printf("Try DFT...\n");
	cv::Mat earth_64f1_clone = earth_64f1_img.clone();
	cv::Mat FourierFreqSpectrum = DiscreteFourierTransform(earth_64f1_clone);
	//LowPassFilter
	cv::Mat FiltredFreqSpectrum = LowPassFilter(FourierFreqSpectrum, 0.7);

	cv::Mat FourierAmplitudeSpectre = GetSpectrumAmplitude(FiltredFreqSpectrum);
	cv::Mat FourierPhasseImage = GetPhasseImage(FiltredFreqSpectrum);
	cv::Mat FourierPowerSpectre = GetPowerSpectrum(FiltredFreqSpectrum);
	//show images
	cv::imshow("EARTH", earth_64f1_img);
	cv::imshow("EARTH - BASE AMPLITUDE SPECTRE", FourierAmplitudeSpectre);
	cv::imshow("EARTH - BASE PHASSE IMAGE", FourierPhasseImage);
	cv::imshow("EARTH - BASE POWER SPECTRE", FourierPowerSpectre);
	//Wait
	cv::waitKey();
	//Try Inverse Founier Transform
	/*
	printf("Try Inverse DFT...\n");
	cv::Mat ReconstructedImg = InverseDiscreteFourierTransform(FourierFreqSpectrum);
	cv::imshow("RECONSTRUCTED IMAGE", ReconstructedImg);
	*/
	/*
	cv::imshow("VALVE", anisotropic_8uc1_img);
	cv::imshow("VALVE ANTISOPTROPIC", anisotropic_64f1_img);
	cv::imshow("LENA BLUR", blur_8uc1_img);
	cv::imshow("LENA with GAMA", src_8uc3_img_gama);
	cv::imshow("LENA", src_8uc3_img);
	*/
	//Press key to exit
	cv::waitKey();
	return 0;
}
