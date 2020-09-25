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

	cv::Mat gray_8uc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 8 bits (uchar)
	cv::cvtColor(src_8uc3_img, gray_8uc1_img, CV_BGR2GRAY); // convert input color image to grayscale one, CV_BGR2GRAY specifies direction of conversion

	//Try change gama
	cv::Mat src_8uc3_img_gama;
	cv::Mat blur_8uc1_img;

	src_8uc3_img.copyTo(src_8uc3_img_gama);
	gray_8uc1_img.copyTo(blur_8uc1_img);

	//cv::imshow("Lena rgb", src_8uc3_img_gama);
	//cv::imshow("Lena gray blur", blur_8uc1_img);

	//Try change gama
	printf("Setting gama..\n");
	//Set mouse callback event
	cv::namedWindow("LENA with GAMA");
	cv::setMouseCallback("LENA with GAMA", MouseEventHandler,(void*)&src_8uc3_img_gama);

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
	//show images
	cv::imshow("LENA BLUR", blur_8uc1_img);
	cv::imshow("LENA", src_8uc3_img);

	// wait until keypressed
	while (cv::waitKey(50) != 0) {
		// refresh images with callback
		cv::imshow("LENA with GAMA", src_8uc3_img_gama);
	} 

	return 0;
}
