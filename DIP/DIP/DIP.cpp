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
	printf("Try Distortion...\n");
	if (!ex_rld()) {
		printf("Distortion playground exit...\n");
	}
	//Img show
	cv::imshow("LENA with GAMA", src_8uc3_img_gama);
	//Press key to exit
	cv::waitKey();
	return 0;
}
