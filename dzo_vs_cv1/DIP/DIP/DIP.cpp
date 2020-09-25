// DIP.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


void MouseEventHandler(int event, int x, int y, int flag, void* Img) {
	if (event == CV_EVENT_LBUTTONDOWN) {
		printf("X: %d, Y:%d", x, y);
	}
}


int main()
{
	cv::Mat src_8uc3_img = cv::imread("images/lena.png", CV_LOAD_IMAGE_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)
	if (src_8uc3_img.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
		return 1;
	}


	//cv::imshow( "LENA", src_8uc3_img );

	cv::Mat gray_8uc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 8 bits (uchar)
	cv::Mat gray_32fc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 32 bits (float)

	cv::cvtColor(src_8uc3_img, gray_8uc1_img, CV_BGR2GRAY); // convert input color image to grayscale one, CV_BGR2GRAY specifies direction of conversion
	gray_8uc1_img.convertTo(gray_32fc1_img, CV_32FC1, 1.0 / 255.0); // convert grayscale image from 8 bits to 32 bits, resulting values will be in the interval 0.0 - 1.0

	/*int x = 10, y = 15; // pixel coordinates

	uchar p1 = gray_8uc1_img.at<uchar>(y, x); // read grayscale value of a pixel, image represented using 8 bits
	float p2 = gray_32fc1_img.at<float>(y, x); // read grayscale value of a pixel, image represented using 32 bits
	cv::Vec3b p3 = src_8uc3_img.at<cv::Vec3b>(y, x); // read color value of a pixel, image represented using 8 bits per color channel

	// print values of pixels
	printf("p1 = %d\n", p1);
	printf("p2 = %f\n", p2);
	printf("p3[ 0 ] = %d, p3[ 1 ] = %d, p3[ 2 ] = %d\n", p3[0], p3[1], p3[2]);

	gray_8uc1_img.at<uchar>(y, x) = 0; // set pixel value to 0 (black)
	// draw a rectangle
	cv::rectangle(gray_8uc1_img, cv::Point(65, 84), cv::Point(75, 94),
		cv::Scalar(50), CV_FILLED);

	// declare variable to hold gradient image with dimensions: width= 256 pixels, height= 50 pixels.
	// Gray levels wil be represented using 8 bits (uchar)
	cv::Mat gradient_8uc1_img(50, 256, CV_8UC1);

	// For every pixel in image, assign a brightness value according to the x coordinate.
	// This wil create a horizontal gradient.
	for (int y = 0; y < gradient_8uc1_img.rows; y++) {
		for (int x = 0; x < gradient_8uc1_img.cols; x++) {
			gradient_8uc1_img.at<uchar>(y, x) = x;
		}
	}
	*/
	//Try change gama
	cv::Mat src_8uc3_img_gama;
	cv::Mat blur_8uc1_img;

	src_8uc3_img.copyTo(src_8uc3_img_gama);
	gray_8uc1_img.copyTo(blur_8uc1_img);

	//cv::imshow("Lena rgb", src_8uc3_img_gama);
	//cv::imshow("Lena gray blur", blur_8uc1_img);

	double gama = 1.9;
	uchar brightness = 15;
	printf("Setting gama..\n");

	ImgGamaEdit(src_8uc3_img_gama, gama, brightness);
	//Set mouse callback event
	cv::setMouseCallback("LENA with GAMA", MouseEventHandler);

	//Try using convultion
	printf("Generating matrix..\n");
	cv::Mat convultionMatrix3X3(3,3,CV_8UC1);
	uchar val = 0;
	for (int y = 0; y < convultionMatrix3X3.rows; y++) {
		for (int x = 0; x < convultionMatrix3X3.cols; x++) {
			//val = (x + y + 1);
			val = 1;
			convultionMatrix3X3.at<uchar>(y, x) = val;
			printf("[%d,%d] %d \n", y, x, val);
		}
	}
	printf("Matrix completed..\n");
	for (int y = 0; y < blur_8uc1_img.rows; y++) {
		for (int x = 0; x < blur_8uc1_img.cols; x++) {
			SetConvolution(blur_8uc1_img, convultionMatrix3X3, x, y, 9);
		}
	}
	printf("Convultion complete..\n");
	// diplay images
	/*
	cv::imshow("Gradient", gradient_8uc1_img);
	cv::imshow("Lena gray", gray_8uc1_img);
	cv::imshow("Lena gray 32f", gray_32fc1_img);
	*/
	cv::imshow("LENA BLUR", blur_8uc1_img);
	cv::imshow("LENA", src_8uc3_img);
	cv::imshow("LENA with GAMA", src_8uc3_img_gama);

	cv::waitKey(0); // wait until keypressed

	return 0;
}
