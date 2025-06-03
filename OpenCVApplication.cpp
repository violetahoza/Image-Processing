// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <math.h>
#include "LicensePlateRecognition.h"

using namespace std;
using namespace cv;

wchar_t* projectPath;

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

//  Implement a function which changes the gray levels of an image by an additive factor.
void additive_factor(int factor) {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++) {
				int result = src.at<uchar>(i, j) + factor;
				if (result > 255) {
					dst.at<uchar>(i, j) = 255;
				}
				else if (result < 0) {
					dst.at<uchar>(i, j) = 0;
				}
				else {
					dst.at<uchar>(i, j) = result;
				}
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("new image", dst);
		waitKey();
	}
}

// Implement a function which changes the gray levels of an image by a multiplicative factor.
void multiplicative_factor(float factor) {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++) {
				int result = src.at<uchar>(i, j) * factor;
				if (result > 255) {
					dst.at<uchar>(i, j) = 255;
				}
				else if (result < 0) {
					dst.at<uchar>(i, j) = 0;
				}
				else {
					dst.at<uchar>(i, j) = result;
				}
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("new image", dst);
		waitKey();
	}
}

// Create a color image of dimension 256 x 256. Divide it into 4 squares and color the squares  from 
// top to bottom, left to right as : white, red, green, yellow.
void make_squares() {
	Mat img(256, 256, CV_8UC3);

	for (int i = 0; i < img.rows / 2; i++) {
		for (int j = 0; j < img.cols / 2; j++) {
			img.at<Vec3b>(i, j)[0] = 255;
			img.at<Vec3b>(i, j)[1] = 255;
			img.at<Vec3b>(i, j)[2] = 255;
		}
		for (int j = img.cols / 2; j < img.cols; j++) {
			img.at<Vec3b>(i, j)[0] = 0;
			img.at<Vec3b>(i, j)[1] = 0;
			img.at<Vec3b>(i, j)[2] = 255;
		}
	}
	for (int i = img.rows / 2; i < img.rows; i++) {
		for (int j = 0; j < img.cols / 2; j++) {
			img.at<Vec3b>(i, j)[0] = 0;
			img.at<Vec3b>(i, j)[1] = 255;
			img.at<Vec3b>(i, j)[2] = 0;
		}
		for (int j = img.cols / 2; j < img.cols; j++) {
			img.at<Vec3b>(i, j)[0] = 0;
			img.at<Vec3b>(i, j)[1] = 255;
			img.at<Vec3b>(i, j)[2] = 255;
		}
	}

	imshow("Colored squares image", img);
	waitKey();
}

// Create a 3x3 float matrix, determine its inverse and print it.
void inverse_matrix() {
	float values[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 10 };
	Mat matrix(3, 3, CV_32F, values);
	std::cout << "Matrix: " << std::endl << matrix << std::endl;

	double det = determinant(matrix);
	if (det == 0) {
		std::cout << "Matrix is singular (det = 0) and cannot be inverted." << std::endl;
		return;
	}

	Mat inverse = matrix.inv(DECOMP_LU);
	std::cout << "Inverse matrix: " << std::endl << inverse << std::endl;

	Mat identity = matrix * inverse;
	std::cout << "matrix * inverse (should be identity): " << std::endl << identity << std::endl;
	waitKey(5000);
}

/*  Create a function that will copy the R, G and B channels of a color, RGB24
image (CV_8UC3 type) into three matrices of type CV_8UC1 (grayscale images).
Display these matrices in three distinct windows.
*/
void getRGB() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_COLOR);

		int height = src.rows;
		int width = src.cols;

		Mat red(height, width, CV_8UC1);
		Mat	green(height, width, CV_8UC1);
		Mat	blue(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++) {
				red.at<uchar>(i, j) = src.at<Vec3b>(i, j)[2];
				green.at<uchar>(i, j) = src.at<Vec3b>(i, j)[1];
				blue.at<uchar>(i, j) = src.at<Vec3b>(i, j)[0];
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("red image", red);
		imshow("green image", green);
		imshow("blue image", blue);

		waitKey();
	}
}

// Create a function for converting from grayscale to black and white (binary).
Mat gray_to_binary(Mat src, int threshold) {
	//char fname[MAX_PATH];

	//while (openFileDlg(fname))
	//{
	//	double t = (double)getTickCount(); // Get the current time [s]

	//	Mat src = imread(fname, IMREAD_GRAYSCALE);

		int height = src.rows;
		int width = src.cols;

		Mat dst(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++) {
				int r = src.at<uchar>(i, j);
				if (r < threshold) {
					dst.at<uchar>(i, j) = 0;
				}
				else {
					dst.at<uchar>(i, j) = 255;
				}
			}
		}

		return dst;

		// Get the current time again and compute the time difference [s]
		//t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		//printf("Time = %.3f [ms]\n", t * 1000);

		/*imshow("input image", src);
		imshow("binary image", dst);

		waitKey();*/
	//}
}

/* Create a function that will convert a color RGB24 image (CV_8UC3 type) to
a grayscale image (CV_8UC1) and display the result image in a destination window.
*/
Mat RGBtoGray(Mat src) {
	int height = src.rows;
	int width = src.cols;

	Mat dst(height, width, CV_8UC1);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++) {
			int result = (src.at<Vec3b>(i, j)[2] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[0]) / 3;
			if (result > 255) {
				dst.at<uchar>(i, j) = 255;
			}
			else if (result < 0) {
				dst.at<uchar>(i, j) = 0;
			}
			else {
				dst.at<uchar>(i, j) = result;
			}
		}
	}

	return dst;
}

/*  Create a function that will compute the H, S and V values from the R, G, B channels of an image.
*  Store each value (H, S, V) in a CV_8UC1 matrix.  Display these matrices in distinct windows.
*/
void RGBtoHSV() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_COLOR);

		int height = src.rows;
		int width = src.cols;

		Mat dst(height, width, CV_8UC3);
		Mat dstH(height, width, CV_8UC1);
		Mat dstS(height, width, CV_8UC1);
		Mat dstV(height, width, CV_8UC1);

		float H, S, V;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++) {
				float blue = (float)src.at<Vec3b>(i, j)[0] / 255;
				float green = (float)src.at<Vec3b>(i, j)[1] / 255;
				float red = (float)src.at<Vec3b>(i, j)[2] / 255;

				float M = max(max(red, green), blue);
				float m = min(min(red, green), blue);

				float C = M - m;
				V = M;

				if (V != 0)
					S = C / V;
				else S = 0; // black

				if (C != 0) {
					if (M == red)
						H = 60 * (green - blue) / C;
					else if (M == green)
						H = 60 * (blue - red) / C + 120;
					else if (M == blue)
						H = 60 * (red - green) / C + 240;
				}
				else H = 0; // grayscale

				if (H < 0)
					H += 360;

				float H_norm = H * 255 / 360;
				float S_norm = S * 255;
				float V_norm = V * 255;

				dst.at<Vec3b>(i, j)[0] = H_norm;
				dst.at<Vec3b>(i, j)[1] = S_norm;
				dst.at<Vec3b>(i, j)[2] = V_norm;

				dstH.at<uchar>(i, j) = H_norm;
				dstS.at<uchar>(i, j) = S_norm;
				dstV.at<uchar>(i, j) = V_norm;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("HSV image", dst);
		imshow("H", dstH);
		imshow("S", dstS);
		imshow("V", dstV);

		waitKey();
	}
}


void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}


void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

int get_object_area(Mat* img, uchar R, uchar G, uchar B) {
	int area = 0;
	for (int i = 0; i < (*img).rows; i++)
		for (int j = 0; j < (*img).cols; j++)
			if ((*img).at<Vec3b>(i, j)[2] == R && (*img).at<Vec3b>(i, j)[1] == G && (*img).at<Vec3b>(i, j)[0] == B)
				area++;
	return area;
}

int* get_object_center_of_mass(Mat* img, uchar R, uchar G, uchar B, Mat dst) {
	int area = get_object_area(img, R, G, B);
	static int center[2] = { 0, 0 };

	if (area == 0) {
		return center;
	}

	for (int i = 0; i < (*img).rows; i++)
		for (int j = 0; j < (*img).cols; j++)
			if ((*img).at<Vec3b>(i, j)[2] == R && (*img).at<Vec3b>(i, j)[1] == G && (*img).at<Vec3b>(i, j)[0] == B) {
				center[0] += i;
				center[1] += j;
			}

	center[0] /= area;
	center[1] /= area;

	return center;
}

double get_axis_of_elongation(Mat* img, uchar R, uchar G, uchar B, Mat dst) {
	int* center_of_mass = get_object_center_of_mass(img, R, G, B, dst);
	int center_row = center_of_mass[0];
	int center_col = center_of_mass[1];
	double numerator = 0, denominator = 0;
	double phi = 0, deg = 0;

	for (int i = 0; i < (*img).rows; i++)
		for (int j = 0; j < (*img).cols; j++)
			if ((*img).at<Vec3b>(i, j)[2] == R && (*img).at<Vec3b>(i, j)[1] == G && (*img).at<Vec3b>(i, j)[0] == B) {
				numerator += (i - center_row) * (j - center_col);
				denominator += pow(j - center_col, 2) - pow(i - center_row, 2);
			}
	numerator *= 2;
	phi = atan2(numerator, denominator) / 2;
	deg = phi * 180.0 / PI;
	if (deg < 0) deg += 180.0;


	int lineLength = 30;

	Point center(center_col, center_row);
	circle(dst, center, 5, Scalar(255, 255, 255), -1);

	Point p1(center_col + (int)(lineLength * cos(phi)),
		center_row + (int)(lineLength * sin(phi)));
	Point p2(center_col - (int)(lineLength * cos(phi)),
		center_row - (int)(lineLength * sin(phi)));

	line(dst, p1, p2, Vec3b(0, 0, 0), 2);

	return deg;
}

bool is_background(Mat* img, int i, int j) {
	// First check if coordinates are valid
	if (i < 0 || i >= (*img).rows || j < 0 || j >= (*img).cols)
		return true; // Treat out-of-bounds as background

	Vec3b pixel = (*img).at<Vec3b>(i, j);
	return (pixel[2] == 255 && pixel[1] == 255 && pixel[0] == 255);
}



int get_perimeter(Mat* img, uchar R, uchar G, uchar  B, Mat dst) {
	Mat newimg = (*img).clone();
	int perimeter = 0;
	for (int i = 0; i < (*img).rows; i++)
		for (int j = 0; j < (*img).cols; j++)
			if ((*img).at<Vec3b>(i, j)[2] == R && (*img).at<Vec3b>(i, j)[1] == G && (*img).at<Vec3b>(i, j)[0] == B) {
				if (is_background(img, i - 1, j - 1) ||
					is_background(img, i - 1, j) ||
					is_background(img, i - 1, j + 1) ||
					is_background(img, i, j - 1) ||
					is_background(img, i, j + 1) ||
					is_background(img, i + 1, j - 1) ||
					is_background(img, i + 1, j) ||
					is_background(img, i + 1, j + 1)) {
					perimeter++;
					line(dst, Point(j, i), Point(j, i), Vec3b(0, 0, 0), 2);
				}
			}
	return perimeter * PI / 4;
}

double get_thinness_ratio(Mat* img, uchar R, uchar G, uchar B, Mat dst) {
	int area = get_object_area(img, R, G, B);
	int perimeter = get_perimeter(img, R, G, B, dst);
	if (perimeter == 0)
		return 0;
	double thinness_ratio = 4 * PI * area / pow(perimeter, 2);
	return thinness_ratio;
}

double get_aspect_ratio(Mat* img, uchar R, uchar G, uchar B) {
	int rmax = 0, cmax = 0, rmin = (*img).rows, cmin = (*img).cols;
	bool object_found = false;

	for (int i = 0; i < (*img).rows; i++)
		for (int j = 0; j < (*img).cols; j++) {
			uchar Red = (*img).at<Vec3b>(i, j)[2];
			uchar Green = (*img).at<Vec3b>(i, j)[1];
			uchar Blue = (*img).at<Vec3b>(i, j)[0];

			if (Red == R && Green == G && Blue == B) {
				object_found = true;
				if (i < rmin)
					rmin = i;
				if (i > rmax)
					rmax = i;
				if (j < cmin)
					cmin = j;
				if (j > cmax)
					cmax = j;
			}
		}

	if (!object_found || rmax == rmin)
		return 0;

	return (double)(cmax - cmin + 1) / (rmax - rmin + 1);
}

Mat compute_projection(Mat src, Vec3b obj) {
	int height = src.rows;
	int width = src.cols;

	Mat projection(height, width, CV_8UC3, Scalar(255, 255, 255));

	// Horizontal Projection
	for (int i = 0; i < height; i++) {
		int sum = 0;
		for (int j = 0; j < width; j++) {
			if (src.at<Vec3b>(i, j) == obj) {
				sum++;
			}
		}
		line(projection, Point(0, i), Point(sum, i), Vec3b(255, 0, 255), 1);
	}

	// Vertical Projection
	for (int j = 0; j < width; j++) {
		int sum = 0;
		for (int i = 0; i < height; i++) {
			if (src.at<Vec3b>(i, j) == obj) {
				sum++;
			}
		}
		line(projection, Point(j, height), Point(j, height - sum), Vec3b(255, 0, 255), 1);
	}

	return projection;
}

struct CompareVec3b {
	bool operator()(const Vec3b& a, const Vec3b& b) const {
		if (a[0] != b[0]) return a[0] < b[0];
		if (a[1] != b[1]) return a[1] < b[1];
		return a[2] < b[2];
	}
};

Mat filterObjectsByAreaAndOrientation(Mat* labeledImage, double areaThreshold, double phiLow, double phiHigh) {
	int height = (*labeledImage).rows;
	int width = (*labeledImage).cols;

	// Create an output image with the same size as input
	Mat outputImage(height, width, CV_8UC3, Scalar(255, 255, 255));

	// Find all unique colors (objects) in the image
	set<Vec3b, CompareVec3b> uniqueColors;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			Vec3b color = (*labeledImage).at<Vec3b>(i, j);
			// Skip black background (0,0,0)
			if (color != Vec3b(0, 0, 0)) {
				uniqueColors.insert(color);
			}
		}
	}

	// Process each unique object
	for (const Vec3b& objectColor : uniqueColors) {
		uchar R = objectColor[2];
		uchar G = objectColor[1];
		uchar B = objectColor[0];
		// Calculate area of the object
		int area = get_object_area(labeledImage, R, G, B);

		// Skip objects with area >= areaThreshold
		if (area >= areaThreshold) {
			continue;
		}

		// Create a temporary image for elongation calculation
		Mat tempImage = (*labeledImage).clone();

		// Calculate orientation angle
		double phi = get_axis_of_elongation(labeledImage, R, G, B, tempImage);

		// Check if orientation is within the specified range
		if (phi >= phiLow && phi <= phiHigh) {
			// If object meets both criteria, add it to the output image
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					if ((*labeledImage).at<Vec3b>(i, j) == objectColor) {
						outputImage.at<Vec3b>(i, j) = objectColor;
					}
				}
			}
		}
	}

	return outputImage;
}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN) {
		Mat newimg = (*src).clone();

		uchar R = (*src).at<Vec3b>(y, x)[2];
		uchar G = (*src).at<Vec3b>(y, x)[1];
		uchar B = (*src).at<Vec3b>(y, x)[0];

		Vec3b objectColor(B, G, R);

		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n", x, y,
			(int)R, (int)G, (int)B);

		printf("Object area is: %d\n", get_object_area(src, R, G, B));

		int* center = get_object_center_of_mass(src, R, G, B, newimg);
		printf("Center of mass is: row %d, column %d\n", center[0], center[1]);

		printf("Angle of elongation is: %f degrees\n", get_axis_of_elongation(src, R, G, B, newimg));
		printf("Object perimeter is: %d\n", get_perimeter(src, R, G, B, newimg));
		printf("Thinness ratio is: %f\n", get_thinness_ratio(src, R, G, B, newimg));
		printf("Aspect ratio is: %f\n", get_aspect_ratio(src, R, G, B));

		//draw_contour(src, R, G, B, &newimg);

		Mat projection = compute_projection(*src, objectColor);
		imshow("Projection", projection);

		imshow("New image", newimg);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

// Compute the histogram for a given grayscale image (in an array of integers having dimension 256).
int* computeHistogram(Mat& image) {
	int* histogram = (int*)calloc(256, sizeof(int));
	for (int i = 0; i < 256; i++)
		histogram[i] = 0;

	for (int i = 0; i < image.rows; i++)
		for (int j = 0; j < image.cols; j++)
			histogram[image.at<uchar>(i, j)]++;

	return histogram;
}

// Compute the histogram for a given number of bins m ≤ 256.
int* histogramReduced(Mat image, int m) {
	int* histogram = computeHistogram(image);
	int* histogramBin = (int*)calloc(m, sizeof(int));
	for (int i = 0; i < m; i++)
		histogramBin[i] = 0;

	for (int i = 0; i < 256; i++)
	{
		int newBin = (int)(i * m / 256);
		histogramBin[newBin] += histogram[i];
	}

	return histogramBin;
}

// Compute the PDF (in an array of floats of dimension 256).
float* computePDF(Mat image) {
	int* histogram = computeHistogram(image);
	float* pdf = (float*)calloc(256, sizeof(float));
	for (int i = 0; i < 256; i++)
		pdf[i] = 0.0;

	int height = image.rows;
	int width = image.cols;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int index = image.at<uchar>(i, j);
			pdf[index] = (double)histogram[index] / (height * width);
		}
	}

	return pdf;
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

float computeMean(Mat src) {
	float mean = 0;
	float* pdf = computePDF(src);
	for (int i = 0; i < 256; i++)
		mean += i * pdf[i];
	return mean;
}

float computeStandardDeviation(Mat src) {
	float mean = computeMean(src);
	float* pdf = computePDF(src);
	float stdDev = 0;
	for (int i = 0; i < 256; i++)
		stdDev += (i - mean) * (i - mean) * pdf[i];
	return sqrt(stdDev);
}

int* computeCumulativeHistogram(Mat src) {
	int* histogram = computeHistogram(src);
	int* cumulativeHistogram = (int*)calloc(256, sizeof(int));

	int cumulative = 0;
	for (int i = 0; i < 256; i++) {
		cumulative += histogram[i];
		cumulativeHistogram[i] = cumulative;
	}

	free(histogram);
	return cumulativeHistogram;
}

Mat histogramEqualization(Mat src) {
	Mat dst = Mat(src.rows, src.cols, CV_8UC1);
	int* cumulativeHist = computeCumulativeHistogram(src);
	
	float* cpdf = (float*)calloc(256, sizeof(float));
	int totalPixels = src.rows * src.cols;
	for (int i = 0; i < 256; i++) {
		cpdf[i] = (float)cumulativeHist[i] / totalPixels;
	}

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			uchar pixelValue = src.at<uchar>(i, j);
			dst.at<uchar>(i, j) = cvRound(255 * cpdf[pixelValue]);
		}
	}

	free(cumulativeHist);
	free(cpdf);

	return dst;
}

Mat modifyContrast(Mat src, int iOutMin, int iOutMax)
{
	int* histogram = computeHistogram(src);
	int iMin, iMax;

	for (int i = 0; i < 256; i++)
		if (histogram[i] != 0)
		{
			iMin = i;
			break;
		}

	for (int i = 255; i >= 0; i--)
		if (histogram[i] != 0)
		{
			iMax = i;
			break;
		}

	int height = src.rows;
	int width = src.cols;

	Mat dst = src.clone();

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int pixel = src.at<uchar>(i, j);
			pixel = (iOutMax - iOutMin) * (pixel - iMin) / (iMax - iMin) + iOutMin;
			if (pixel < 0)
			{
				pixel = 0;
			}
			if (pixel > 255)
			{
				pixel = 255;
			}
			dst.at<uchar>(i, j) = pixel;
		}
	}

	return dst;
}

Mat gammaCorrection(Mat src, float gamma) {
	Mat dst = src.clone();
	
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int pixel = src.at<uchar>(i, j);
			pixel = 255 * pow((float)pixel / 255, gamma);
			dst.at<uchar>(i, j) = pixel;
		}
	}
	return dst;
}

Mat modifyBrightness(Mat src, int delta) {
	Mat dst = src.clone();

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int pixel = src.at<uchar>(i, j);
			pixel += delta;
			if (pixel < 0)
				pixel = 0;
			if (pixel > 255)
				pixel = 255;
			dst.at<uchar>(i, j) = pixel;
		}
	}
	return dst;
}

float computeThreshold(Mat src, float error) {
	int* histogram = computeHistogram(src);
	int imin, imax;

	for (int i = 0; i < 256; i++) {
		if (histogram[i] != 0) {
			imin = i;
			break;
		}
	}
	for (int i = 255; i >= 0; i--) {
		if (histogram[i] != 0) {
			imax = i;
			break;
		}
	}

	float prevThreshold = (imin + imax) / 2.0;
	float threshold = prevThreshold;

	do {
		float mean1 = 0, mean2 = 0;
		int count1 = 0, count2 = 0;

		for (int i = imin; i <= threshold; i++) {
			mean1 += i * histogram[i];
			count1 += histogram[i];
		}
		mean1 /= count1;

		for (int i = threshold + 1; i <= imax; i++) {
			mean2 += i * histogram[i];
			count2 += histogram[i];
		}
		mean2 /= count2;

		prevThreshold = threshold;
		threshold = (mean1 + mean2) / 2.0;
	} while (abs(threshold - prevThreshold) > error);

	return threshold;
}

void MultilevelThresholding() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);

		float* fdp = computePDF(src);

		int wh = 5;
		float th = 0.0003;
		int nrSteps = 2;

		std::vector<int> steps; // the local maximas vector
		steps.push_back(0);

		for (int i = wh; i < 255 - wh; i++) {
			float v = 0;
			int ok = 1;
			for (int k = i - wh; k <= i + wh; k++) {
				if (fdp[i] < fdp[k]) {
					ok = 0;
				}
				v = v + fdp[k];
			}
			v = v / (2 * wh + 1);
			if (ok == 1 && fdp[i] > v + th) {
				steps.push_back(i);
				nrSteps++;
			}
		}

		steps.push_back(255);

		for (int i = 0; i < nrSteps; i++) {
			std::cout << steps[i] << " ";

		}

		int min = 0;
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				min = abs(src.at<uchar>(i, j) - steps[0]);
				int step = 0;
				for (int k = 1; k < nrSteps; k++) {
					if (min > abs(src.at<uchar>(i, j) - steps[k])) {
						min = abs(src.at<uchar>(i, j) - steps[k]);
						step = steps[k];
					}
				}
				dst.at<uchar>(i, j) = step;

			}
		}

		imshow("input image", src);
		imshow("after multi-level thresholding", dst);
		showHistogram("Histogram", computeHistogram(dst), 256, 200);

		waitKey();
	}
}

void FloydSteinberg() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		Mat dstTemp = Mat(src.rows, src.cols, CV_32FC1);

		float* pdf = computePDF(src);

		int w = 5;
		float th = 0.0003;
		int nrSteps = 2;
		std::vector<int> steps;
		steps.push_back(0);

		for (int i = w; i < 255 - w; i++) {
			float v = 0;
			int ok = 1;
			for (int k = i - w; k <= i + w; k++) {
				if (pdf[i] < pdf[k]) {
					ok = 0;
				}
				v = v + pdf[k];
			}
			v = v / (2 * w + 1);
			if (ok == 1 && pdf[i] > v + th) {
				steps.push_back(i);
				nrSteps++;
			}
		}

		steps.push_back(255);

		for (int i = 0; i < nrSteps; i++) {
			std::cout << steps[i] << " ";

		}

		int min = 0;
		for (int i = 1; i < src.rows - 1; i++) {
			for (int j = 1; j < src.cols - 1; j++) {
				min = abs(dstTemp.at<float>(i, j) - steps[0]);
				int step = 0;
				for (int k = 1; k < nrSteps; k++) {
					if (min > abs(dstTemp.at<float>(i, j) - steps[k])) {
						min = abs(dstTemp.at<float>(i, j) - steps[k]);
						step = steps[k];
					}
				}

				dst.at<uchar>(i, j) = step;
				float error = dstTemp.at<float>(i, j) - step;
				dstTemp.at<float>(i, j) = step;

				dstTemp.at<float>(i, j + 1) = dstTemp.at<float>(i, j + 1) + 7 * error / 16;
				dstTemp.at<float>(i + 1, j - 1) = dstTemp.at<float>(i + 1, j - 1) + 3 * error / 16;
				dstTemp.at<float>(i + 1, j) = dstTemp.at<float>(i + 1, j) + 5 * error / 16;
				dstTemp.at<float>(i + 1, j + 1) = dstTemp.at<float>(i + 1, j + 1) + error / 16;
			}
		}

		imshow("input image", src);
		imshow("after Floyd Steinberg dithering", dst);
		//showHistogram("Histogram", computeHistogram(dst), 256, 200);

		waitKey();
	}
}

std::vector<Point2i> get_neighbors(int img_height, int img_width, int i, int j) {
	std::vector<Point2i> neighbors;
	neighbors.push_back(Point2i(i - 1, j - 1));
	neighbors.push_back(Point2i(i - 1, j));
	neighbors.push_back(Point2i(i - 1, j + 1));
	neighbors.push_back(Point2i(i, j - 1));
	neighbors.push_back(Point2i(i, j + 1));
	neighbors.push_back(Point2i(i + 1, j - 1));
	neighbors.push_back(Point2i(i + 1, j));
	neighbors.push_back(Point2i(i + 1, j + 1));
	int index = 0;
	for (Point2i p : neighbors) {
		if (p.x < 0 || p.y < 0 || p.x >= img_height || p.y >= img_width) {
			neighbors.erase(neighbors.begin() + index);
		}
		else {
			i++;
		}
	}
	return neighbors;
}

void generate_colors(Mat& labels, Mat& newimg) {
	Vec3b colors[50] = { Vec3b(0,0,0) };
	for (int i = 0; i < labels.rows; i++) {
		for (int j = 0; j < labels.cols; j++) {
			int label = labels.at<uchar>(i, j);
			if (label == 0) {
				newimg.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
			else {
				if (colors[label] == Vec3b(0, 0, 0)) {
					colors[label][0] = rand() % 256;
					colors[label][1] = rand() % 256;
					colors[label][2] = rand() % 256;
					//printf("label %d %d %d %d\n", label, colors[label][0], colors[label][1], colors[label][2]);
				}
				newimg.at<Vec3b>(i, j) = colors[label];
			}
		}
	}
}

void bfs() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		imshow("Original image", src);
		Mat newimg(src.rows, src.cols, CV_8UC3);
		uchar label = 0;
		uchar R = 0;
		uchar G = 0;
		uchar B = 0;

		Mat labels(src.rows, src.cols, CV_8UC1, Scalar(0));

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (src.at<uchar>(i, j) == 0 && labels.at<uchar>(i, j) == 0) {
					label++;
					std::queue<Point2i> q;
					labels.at<uchar>(i, j) = label;
					newimg.at<Vec3b>(i, j)[0] = B;
					newimg.at<Vec3b>(i, j)[1] = G;
					newimg.at<Vec3b>(i, j)[2] = R;

					q.push(Point2i(i, j));
					while (!q.empty()) {
						Point2i p = q.front();
						q.pop();
						std::vector<Point2i> neighbors = get_neighbors(src.rows, src.cols, p.x, p.y);
						for (Point2i p : neighbors) {
							int x = p.x;
							int y = p.y;
							if (src.at<uchar>(x, y) == 0 && labels.at<uchar>(x, y) == 0) {
								labels.at<uchar>(x, y) = label;
								q.push(p);
							}
						}
					}
				}
			}
		}

		generate_colors(labels, newimg);
		imshow("Color image", newimg);
		waitKey(0);
	}
}

std::vector<Point2i> get_prev_neighb(int img_height, int img_width, int i, int j) {
	std::vector<Point2i> neighbors;
	neighbors.push_back(Point2i(i - 1, j - 1));
	neighbors.push_back(Point2i(i - 1, j));
	neighbors.push_back(Point2i(i - 1, j + 1));
	neighbors.push_back(Point2i(i, j - 1));
	int index = 0;

	std::vector<Point2i>::iterator it = neighbors.begin();
	while (it != neighbors.end()) {
		Point2i p = *it;
		if (p.x < 0 || p.y < 0 || p.x >= img_height || p.y >= img_width) {
			it = neighbors.erase(it);
		}
		else ++it;
	}
	return neighbors;
}

void two_pass_labeling() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		imshow("Original image", src);

		Mat newimg(src.rows, src.cols, CV_8UC3);
		uchar label = 0;
		Mat labels(src.rows, src.cols, CV_8UC1, Scalar(0));

		int nrAccess = 0;

		std::vector<std::vector<int>> edges(1000);
		// first pass
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (src.at<uchar>(i, j) == 0 && labels.at<uchar>(i, j) == 0) {
					std::vector<uchar> L; // list of neighboring labels
					std::vector<Point2i> prev_n = get_prev_neighb(src.rows, src.cols, i, j);
					for (Point2i n : prev_n) {
						if (labels.at<uchar>(n.x, n.y) > 0) {
							L.push_back(labels.at<uchar>(n.x, n.y));
						}
					}
					if (L.size() == 0) {
						// assign new label
						label++;
						labels.at<uchar>(i, j) = label;
					}
					else {
						// assign the smallestlabel from the neighbors
						uchar x = min_element(L.begin(), L.end())[0];
						labels.at<uchar>(i, j) = x;
						// record equivalence relationships
						for (uchar y : L) {
							if (x != y) {
								//printf("nr access: %d\n", nrAccess++);
								//printf("x: %d\n", x);
								//printf("y: %d\n", y);
								edges[x].push_back(y);
								edges[y].push_back(x);
							}
						}
					}

				}
			}
		}

		// display intermediate results after first pass using generate_colors
		Mat firstPassImg(src.rows, src.cols, CV_8UC3);
		generate_colors(labels, firstPassImg);
		imshow("First Pass Results", firstPassImg);

		uchar newlabel = 0;
		uchar* newlabels = (uchar*)malloc(src.rows * sizeof(uchar));
		for (int i = 0; i < src.rows; i++) {
			newlabels[i] = 0;

		}

		for (int i = 1; i <= label; i++) {
			if (newlabels[i] == 0) {
				newlabel++;
				std::queue<uchar> Q;
				newlabels[i] = newlabel;
				Q.push(i);
				while (!Q.empty()) {
					int x = Q.front();
					Q.pop();
					for (int y : edges[x]) {
						if (newlabels[y] == 0) {
							newlabels[y] = newlabel;
							Q.push(y);
						}
					}
				}
			}
		}

		// assign final label to the label matrix
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				// only relabel if it was labeled in first pass
				labels.at<uchar>(i, j) = newlabels[labels.at<uchar>(i, j)];
			}
		}

		generate_colors(labels, newimg);
		imshow("New image", newimg);

		waitKey(0);
		free(newlabels);
	}
}

Point2i getNextPixel(int dir, int x, int y) {
	switch (dir) {
	case 0: return Point2i(x, y + 1); 
	case 1: return Point2i(x - 1, y + 1); 
	case 2: return Point2i(x - 1, y); 
	case 3: return Point2i(x - 1, y - 1); 
	case 4: return Point2i(x, y - 1);
	case 5: return Point2i(x + 1, y - 1);
	case 6: return Point2i(x + 1, y);
	case 7: return Point2i(x + 1, y + 1); 
	default: return Point2i(x, y); 
	}
}

void border_tracing() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		imshow("Original image", src);

		//find start point of object
		Point2i startPoint;
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (src.at<uchar>(i, j) == 0) {
					startPoint = Point2i(i, j);
					i = src.rows;
					j = src.cols;
				}
			}
		}

		printf("Starting point: %d, %d\n", startPoint.x, startPoint.y);

		int dir = 7;
		int n = 1;

		std::vector<Point2i> border;
		std::vector<int> chainCode;
		Point2i currentPoint = startPoint;
		border.push_back(startPoint);

		do {
			if (dir % 2 == 0) {
				dir = (dir + 7) % 8;
			}
			else {
				dir = (dir + 6) % 8;
			}

			Point2i nextP = getNextPixel(dir, currentPoint.x, currentPoint.y);

			//what happens if no neighbors are black?
			while (src.at<uchar>(nextP.x, nextP.y) != 0) {
				dir = (dir + 1) % 8;
				nextP = getNextPixel(dir, currentPoint.x, currentPoint.y);
			}

			if (src.at<uchar>(nextP.x, nextP.y) == 0) {
				border.push_back(nextP);
				n = border.size();
				currentPoint = nextP;
				chainCode.push_back(dir);
			}
		} while (border.size() <= 2 || (border.at(0) != border.at(n - 2) && border.at(1) != border.at(n - 1)));

		//build new image
		Mat borderImg(src.rows, src.cols, CV_8UC1);
		for (int i = 0; i < borderImg.rows; i++) {
			for (int j = 0; j < borderImg.cols; j++) {
				borderImg.at<uchar>(i, j) = 255;
			}
		}
		for (int k = 0; k < border.size(); k++) {
			Point2i p = border.at(k);
			borderImg.at<uchar>(p.x, p.y) = 0;
		}

		printf("Chain code: \n", chainCode.size());
		for (int i = 0; i < chainCode.size() - 2; i++) {
			printf("%d ", chainCode.at(i));
		}

		printf("\nDerivative chain code: \n");
		for (int i = 1; i < chainCode.size(); i++) {
			int val = (chainCode.at(i) - chainCode.at(i - 1) + 8) % 8;
			printf("%d ", val);
		}
		imshow("Border image", borderImg);

		waitKey(0);
	}
}

void reconstruct_border() {
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];

	char imN[MAX_PATH];
	strcpy(imN, folderName);
	strcat(imN, "\\gray_background.bmp");

	char txN[MAX_PATH];
	strcpy(txN, folderName);
	strcat(txN, "\\reconstruct.txt");

	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	Mat img = imread(imN, IMREAD_GRAYSCALE);
	cvtColor(img, img, COLOR_GRAY2BGR);

	ifstream file(txN);

	Point p;
	file >> p.x >> p.y;
	int nr;
	file >> nr;
	std::vector<int> chain;


	int buff;
	while (file >> buff) {
		chain.push_back(buff);
	}

	Point point = p;

	img.at<Vec3b>(point.x, point.y)[0] = 255;
	img.at<Vec3b>(point.x, point.y)[1] = 0;
	img.at<Vec3b>(point.x, point.y)[2] = 255;

	for (auto link : chain) {
		point.x += di[link];
		point.y += dj[link];
		img.at<Vec3b>(point.x, point.y)[0] = 255;
		img.at<Vec3b>(point.x, point.y)[1] = 0;
		img.at<Vec3b>(point.x, point.y)[2] = 255;
	}

	imshow("Contour", img);
	waitKey();

}

Mat dilation(Mat src) {
	Mat dst = src.clone();

	for (int i = 1; i < src.rows - 1; i++) 
		for (int j = 1; j < src.cols - 1; j++) 
			if (src.at<uchar>(i, j) == 0) {
				dst.at<uchar>(i - 1, j) = 0; 
				dst.at<uchar>(i + 1, j) = 0;
				dst.at<uchar>(i, j - 1) = 0;
				dst.at<uchar>(i, j + 1) = 0;
			}

	return dst;
}

Mat dilate_n_times(Mat src, int n) {
	Mat dst = src.clone();

	for (int i = 0; i < n; i++) 
		dst = dilation(dst);

	return dst;
}

Mat erosion(Mat src) {
	Mat dst = src.clone();

	for (int i = 1; i < src.rows - 1; i++)
		for (int j = 1; j < src.cols - 1; j++)
			if (src.at<uchar>(i, j) == 0) {
				if (src.at<uchar>(i - 1, j) == 255 ||
					src.at<uchar>(i + 1, j) == 255 ||
					src.at<uchar>(i, j + 1) == 255 ||
					src.at<uchar>(i, j - 1) == 255 || 
					src.at<uchar>(i - 1, j + 1) == 255 ||
					src.at<uchar>(i - 1, j - 1) == 255 ||
					src.at<uchar>(i + 1, j + 1) == 255 || 
					src.at<uchar>(i + 1, j - 1) == 255 )
					dst.at<uchar>(i, j) = 255;
			}

	return dst;
}

Mat erode_n_times(Mat src, int n) {
	Mat dst = src.clone();

	for (int i = 0; i < n; i++)
		dst = erosion(dst);

	return dst;
}

Mat opening(Mat src) {
	Mat dst = src.clone();
	dst = erosion(dst);
	dst = dilation(dst);
	return dst;
}

Mat closing(Mat src) {
	Mat dst = src.clone();
	dst = dilation(dst);
	dst = erosion(dst);
	return dst;
}

Mat boundary_extraction(Mat src) {
	// Convert to grayscale if needed
	if (src.channels() > 1) {
		cvtColor(src, src, COLOR_BGR2GRAY);
	}

	// Binarize the image: anything darker than 128 becomes black (0), else white (255)
	threshold(src, src, 128, 255, THRESH_BINARY);

	Mat copy = src.clone();

	copy = erosion(copy);
	Mat boundary(src.rows, src.cols, CV_8UC1);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			if (src.at<uchar>(i, j) == 0 && copy.at<uchar>(i, j) != 0)
				boundary.at<uchar>(i, j) = 0;
			else
				boundary.at<uchar>(i, j) = 255;
	return boundary;
}


bool areEqual(Mat mat1, Mat mat2) {
	for (int i = 0; i < mat1.rows; i++) {
		for (int j = 0; j < mat1.cols; j++) {
			if (mat1.at<uchar>(i, j) != mat2.at<uchar>(i, j)) {
				return false;
			}
		}
	}
	return true;
}

Mat intersection(Mat mat1, Mat mat2) {
	Mat inters(mat1.rows, mat1.cols, CV_8UC1);
	for (int i = 0; i < mat1.rows; i++) {
		for (int j = 0; j < mat1.cols; j++) {
			if (mat1.at<uchar>(i, j) == mat2.at<uchar>(i, j) && mat1.at<uchar>(i, j) == 0) {
				inters.at<uchar>(i, j) = 0;
			}
			else {
				inters.at<uchar>(i, j) = 255;
			}
		}
	}
	return inters;
}

Mat region_filling(Mat src) {
	Mat complement(src.rows, src.cols, CV_8UC1);
	Mat region(src.rows, src.cols, CV_8UC1);
	complement = 255 - src;

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			region.at<uchar>(i, j) = 255;

	int starti = src.rows / 2;
	int startj = src.cols / 2;
	cout << "Start point: " << starti << " " << startj << endl;
	region.at<uchar>(starti, startj) = 0;

	Mat copy = dilation(region);
	copy = intersection(copy, complement);

	while (!areEqual(copy, region)) {
		region = copy.clone(); 
		copy = dilation(region);
		copy = intersection(copy, complement);
	}

	// combine the filled region with the original image (union)
	Mat finalResult(src.rows, src.cols, CV_8UC1);
	for (int i = 0; i < region.rows; i++) {
		for (int j = 0; j < region.cols; j++) {
			if (region.at<uchar>(i, j) == 0 || src.at<uchar>(i, j) == 0) {
				finalResult.at<uchar>(i, j) = 0;
			}
			else {
				finalResult.at<uchar>(i, j) = 255;
			}
		}
	}

	return finalResult;
}

Mat flip_vertically(Mat src) {
	Mat dst(src.rows, src.cols, CV_8UC3);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst.at<Vec3b>(src.rows - i - 1, j)[2] = src.at<Vec3b>(i, j)[2];
			dst.at<Vec3b>(src.rows - i - 1, j)[1] = src.at<Vec3b>(i, j)[1];
			dst.at<Vec3b>(src.rows - i - 1, j)[0] = src.at<Vec3b>(i, j)[0];
		}
	}
	return dst;
}

Mat applyLowPassFilter(Mat src, int kernel[3][3]) {
	Mat dst = src.clone();
	int kernelSum = 0;

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			kernelSum += kernel[i][j];

	for (int i = 1; i < src.rows - 1; i++)
		for (int j = 1; j < src.cols - 1; j++) {
			int pixel = 0;
			for (int k = -1; k <= 1; k++)
				for (int l = -1; l <= 1; l++)
					pixel += src.at<uchar>(i + k, j + l) * kernel[k + 1][l + 1];
			dst.at<uchar>(i, j) = pixel / kernelSum;
		}
	return dst;
}

Mat applyHighPassFilter(Mat src, int kernel[3][3]) {
	Mat dst = src.clone();
	int sminus = 0, splus = 0, smax = 0;

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			if (kernel[i][j] < 0)
				sminus += kernel[i][j];
			else splus += kernel[i][j];

	smax = max(abs(splus), abs(sminus));

	for (int i = 1; i < src.rows - 1; i++)
		for (int j = 1; j < src.cols - 1; j++) {
			int pixel = 0;
			for (int k = -1; k <= 1; k++)
				for (int l = -1; l <= 1; l++)
					pixel += src.at<uchar>(i + k, j + l) * kernel[k + 1][l + 1];
			pixel = pixel / (2 * smax) + 127;
			dst.at<uchar>(i, j) = pixel;
		}
	return dst;
}

void testFilters() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		imshow("Original image", src);

		int meanKernel[3][3] = {
			{1, 1, 1},
			{1, 1, 1},
			{1, 1, 1}
		};

		Mat mean = applyLowPassFilter(src, meanKernel);
		imshow("Mean Filter", mean);

		int gaussianKernel[3][3] = {
			{1, 2, 1},
			{2, 4, 2},
			{1, 2, 1}
		};
		Mat gaussian = applyLowPassFilter(src, gaussianKernel);
		imshow("Gaussian Filter", gaussian);

		int laplaceKernel[3][3] = {
			{0, -1, 0},
			{-1, 4, -1},
			{0, -1, 0}
		};
		Mat laplace = applyHighPassFilter(src, laplaceKernel);
		imshow("Laplace Filter", laplace);

		int kernel[3][3] = {
			{-1, -1, -1},
			{-1, 9, -1},
			{-1, -1, -1}
		};
		Mat highpass = applyHighPassFilter(src, kernel);
		imshow("High Pass Filter", highpass);

		waitKey(0);
	}
}

void centering_transform(Mat img) {
	//expects floating point image
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
		}
	}
}

Mat generic_frequency_domain_filter(Mat src, int mode, int A, int R) {
	//convert input image to float image
	Mat srcf;
	src.convertTo(srcf, CV_32FC1);
	//centering transformation
	centering_transform(srcf);
	//perform forward transform with complex image output
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);
	//split into real and imaginary channels
	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels); // channels[0] = Re(DFT(I)), channels[1] = Im(DFT(I))
	//calculate magnitude and phase in floating point images mag and phi
	Mat mag, phi;
	magnitude(channels[0], channels[1], mag);
	phase(channels[0], channels[1], phi);
	//display the phase and magnitude images here

	Mat_<float> norm(mag.rows, mag.cols);

	for (int i = 0; i < mag.rows; ++i) {
		for (int j = 0; j < mag.cols; ++j) {
			norm.at<float>(i, j) = log(mag.at<float>(i, j) + 1);
		}
	}

	normalize(norm, norm, 1, 0, NORM_MINMAX, CV_32F); //Might work as well
	imshow("Normalized Mag", norm);

	//insert filtering operations on Fourier coefficients here
	//Gaussian high-pas
	auto filteredMag = mag;
	int H = mag.rows;
	int W = mag.cols;

	if (mode == 1) {
		for (int i = 0; i < mag.rows; ++i) {
			for (int j = 0; j < mag.cols; ++j) {
				filteredMag.at<float>(i, j) = mag.at<float>(i, j) * (1 - (pow(2.718, -(pow((H / 2) - i, 2) + pow((W / 2) - j, 2)) / (A * A))));
			}
		}
	}
	else if (mode == 0) {
		//Gaussian low-pas
		for (int i = 0; i < mag.rows; ++i) {
			for (int j = 0; j < mag.cols; ++j) {
				filteredMag.at<float>(i, j) = mag.at<float>(i, j) * ((pow(2.718, -(pow((H / 2) - i, 2) + pow((W / 2) - j, 2)) / (A * A))));
			}
		}

	}
	else if (mode == 2) {
		//ideal low pass filter
		for (int i = 0; i < mag.rows; ++i) {
			for (int j = 0; j < mag.cols; ++j) {
				auto rez = pow(H / 2 - i, 2) + pow(W / 2 - j, 2);
				if (rez <= R * R) {
					filteredMag.at<float>(i, j) = mag.at<float>(i, j);
				}
				else {
					filteredMag.at<float>(i, j) = 0;
				}
			}
		}
	}
	else if (mode == 3) {
		//ideal high pass filter
		for (int i = 0; i < mag.rows; ++i) {
			for (int j = 0; j < mag.cols; ++j) {
				auto rez = pow(H / 2 - i, 2) + pow(W / 2 - j, 2);
				if (rez > R * R) {
					filteredMag.at<float>(i, j) = mag.at<float>(i, j);
				}
				else {
					filteredMag.at<float>(i, j) = 0;
				}
			}
		}
	}

	//store in real part in channels[0] and imaginary part in channels[1]
	for (int i = 0; i < mag.rows; ++i) {
		for (int j = 0; j < mag.cols; ++j) {
			//real part
			channels[0].at<float>(i, j) = mag.at<float>(i, j) * std::cosf(phi.at<float>(i, j));
			//imaginary part
			channels[1].at<float>(i, j) = mag.at<float>(i, j) * std::sinf(phi.at<float>(i, j));
		}
	}

	//perform inverse transform and put results in dstf
	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
	//inverse centering transformation
	centering_transform(dstf);
	//normalize the result and put in the destination image
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	//Note: normalizing distorts the resut while enhancing the image display in the range [0,255].
	//For exact results (see Practical work 3) the normalization should be replaced with convertion:
	//dstf.convertTo(dst, CV_8UC1);
	return dst;
}

void generic_frequency_domain_filter_test()
{
	int A;
	int R;
	cout << "A=";
	cin >> A;

	cout << "R=";
	cin >> R;

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat originalImage = imread(fname, IMREAD_GRAYSCALE);
		Mat lowPass = generic_frequency_domain_filter(originalImage, 0, A, R);
		Mat highPass = generic_frequency_domain_filter(originalImage, 1, A, R);
		Mat idealLowPass = generic_frequency_domain_filter(originalImage, 2, A, R);
		Mat idealHighPass = generic_frequency_domain_filter(originalImage, 3, A, R);

		imshow("Original", originalImage);
		imshow("Gaussian Low Pass", lowPass);
		imshow("Gaussian High Pass", highPass);
		imshow("Ideal Low Pass", idealLowPass);
		imshow("Ideal High Pass", idealHighPass);

		waitKey();
	}
}

Mat medianFilter(Mat src, int w) {
	Mat dst = src.clone();
	int n = w * w;
	int k = w / 2;
	int* pixel = new int[n];
	int index = 0;

	for (int i = k; i < src.rows - k; i++) {
		for (int j= k; j < src.cols - k; j++) {
			index = 0;
			for (int l = i - k; l <= i + k; l++)
				for (int m = j - k; m <= j + k; m++) {
					pixel[index++] = src.at<uchar>(l, m);
				}
			sort(pixel, pixel + n);
			dst.at<uchar>(i, j) = pixel[n / 2];
		}
	}

	delete[] pixel;
	return dst;
}

Mat applyGaussianFilter1D(Mat src, int w) {
	Mat dst = src.clone();
	int height = src.rows;
	int width = src.cols;

	double sigma = (double) w / 6.0;
	int k = w / 2;
	double* kernel = new double[w];
	double sum = 0.0;

	for (int i = 0; i < w; i++) {
		kernel[i] = exp(-(i - k) * (i- k) / (2 * sigma * sigma));
		sum += kernel[i];
	}
	for (int i = 0; i < w; i++) {
		kernel[i] /= sum;
	}

	for (int i = 0; i < height; i++) {
		for (int j = k; j < width - k; j++) {
			double pixel = 0.0;
			for (int l = j - k; l <= j + k; l++) {
				pixel += src.at<uchar>(i, l) * kernel[l - j + k];
			}
			dst.at<uchar>(i, j) = pixel;
		}
	}

	for (int i = k; i < height - k; i++) {
		for (int j = 0; j < width; j++) {
			double pixel = 0.0;
			for (int l = i - k; l <= i + k; l++) {
				pixel += dst.at<uchar>(l, j) * kernel[l - i + k];
			}
			dst.at<uchar>(i, j) = pixel;
		}
	}

	delete[] kernel;
	return dst;
}

Mat applyGaussianFilter2D(Mat originalImage, int w)
{
	double sigma = (double)w / 6;

	Mat dst = originalImage.clone();

	int k = w / 2;
	double* kernel = new double[w];
	double** kernel2D = new double* [w];
	for (int i = 0; i < w; i++) {
		kernel2D[i] = new double[w];
	}

	// Calculate 1D Gaussian kernel
	double sum = 0.0;
	for (int i = 0; i < w; ++i) {
		kernel[i] = exp(-(i - k) * (i - k) / (2 * sigma * sigma));
		sum += kernel[i];
	}
	for (int i = 0; i < w; ++i) {
		kernel[i] /= sum;
	}

	// Calculate 2D Gaussian kernel
	for (int i = 0; i < w; ++i) {
		for (int j = 0; j < w; ++j) {
			kernel2D[i][j] = kernel[i] * kernel[j];
		}
	}

	int height = originalImage.rows;
	int width = originalImage.cols;

	// Apply filter
	for (int i = k; i < height - k; ++i) {
		for (int j = k; j < width - k; ++j) {
			double sum = 0.0;
			for (int ii = i - k, ki = 0; ii <= i + k; ++ii, ++ki) {
				for (int jj = j - k, kj = 0; jj <= j + k; ++jj, ++kj) {
					sum += kernel2D[ki][kj] * originalImage.at<uchar>(ii, jj);
				}
			}
			dst.at<uchar>(i, j) = static_cast<uchar>(sum);
		}
	}

	for (int i = 0; i < w; i++) {
		delete[] kernel2D[i];
	}
	delete[] kernel;
	delete[] kernel2D;
	return dst;
}

void applyGaussFilters()
{
	int w;

	cout << "w=";
	cin >> w;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		imshow("input image", src);

		double t = (double)getTickCount();

		Mat gauss1d = applyGaussianFilter1D(src, w);

		t = ((double)getTickCount() - t) / getTickFrequency();
		cout << "Time1 [ms]:" << t * 1000 << endl;

		double t2 = (double)getTickCount();

		Mat gauss2d = applyGaussianFilter2D(src, w);

		t2 = ((double)getTickCount() - t2) / getTickFrequency();
		cout << "Time2 [ms]:" << t2 * 1000 << endl;

		imshow("gauss1d filter", gauss1d);
		imshow("gauss2d filter", gauss2d);

		waitKey();
	}
}

// Step 1: Apply Gaussian filter with sigma = 0.5
void applyGaussianFilter(const Mat& src, Mat& dst) {
	// Create a 3x3 Gaussian kernel with sigma = 0.5
	Mat kernel = Mat::zeros(3, 3, CV_32F);

	float sigma = 0.5f;
	float sum = 0.0f;

	// Generate Gaussian kernel
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			kernel.at<float>(i + 1, j + 1) = exp(-(i * i + j * j) / (2 * sigma * sigma));
			sum += kernel.at<float>(i + 1, j + 1);
		}
	}

	// Normalize the kernel
	kernel /= sum;

	// Apply custom convolution with the Gaussian kernel
	dst = Mat(src.rows, src.cols, src.type());

	int kRows = kernel.rows;
	int kCols = kernel.cols;
	int kCenterX = kCols / 2;
	int kCenterY = kRows / 2;

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			float sum = 0.0;

			// Apply convolution
			for (int m = 0; m < kRows; m++) {
				int mm = kRows - 1 - m;  // Flip the kernel

				for (int n = 0; n < kCols; n++) {
					int nn = kCols - 1 - n;  // Flip the kernel

					// Get the corresponding pixel from the input image
					int ii = i + (m - kCenterY);
					int jj = j + (n - kCenterX);

					// Check if pixel is inside the image boundaries
					if (ii >= 0 && ii < src.rows && jj >= 0 && jj < src.cols) {
						sum += src.at<uchar>(ii, jj) * kernel.at<float>(mm, nn);
					}
				}
			}

			dst.at<uchar>(i, j) = saturate_cast<uchar>(sum);
		}
	}
}

// Step 2a: Apply Sobel operators as specified in equation 11.4
void applySobelOperators(const Mat& src, Mat& gradX, Mat& gradY) {
	// Define Sobel kernels as shown in the lecture notes (eq. 11.4)
	Mat sobelX = (Mat_<float>(3, 3) <<
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1);

	Mat sobelY = (Mat_<float>(3, 3) <<
		1, 2, 1,
		0, 0, 0,
		-1, -2, -1);

	// Apply convolution without normalization
	gradX = Mat(src.size(), CV_32F);
	gradY = Mat(src.size(), CV_32F);

	// Apply convolution for each pixel
	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			float sumX = 0.0f;
			float sumY = 0.0f;

			// Apply convolution with Sobel kernels
			for (int m = -1; m <= 1; m++) {
				for (int n = -1; n <= 1; n++) {
					float pixelValue = static_cast<float>(src.at<uchar>(i + m, j + n));
					sumX += pixelValue * sobelX.at<float>(m + 1, n + 1);
					sumY += pixelValue * sobelY.at<float>(m + 1, n + 1);
				}
			}

			// Store the result without normalization
			gradX.at<float>(i, j) = sumX;
			gradY.at<float>(i, j) = sumY;
		}
	}
}

// Step 2b: Calculate gradient magnitude and direction
void calculateGradient(const Mat& gradX, const Mat& gradY, Mat& magnitude, Mat& direction) {
	// Calculate magnitude using eq. 11.6: |∇f(x,y)| = sqrt((∇fx(x,y))² + (∇fy(x,y))²)
	magnitude = Mat(gradX.size(), CV_32F);
	direction = Mat(gradX.size(), CV_32F);

	for (int i = 0; i < gradX.rows; i++) {
		for (int j = 0; j < gradX.cols; j++) {
			float gx = gradX.at<float>(i, j);
			float gy = gradY.at<float>(i, j);

			// Calculate magnitude
			magnitude.at<float>(i, j) = sqrt(gx * gx + gy * gy);

			// Calculate direction
			float theta = atan2(gy, gx);

			// Adjust to [0, 2π] range
			if (theta < 0) {
				theta += 2 * PI;
			}

			direction.at<float>(i, j) = theta;
		}
	}

	// Normalize magnitude to fit [0..255] range by division with 4√2 as specified
	Mat normalizedMagnitude(magnitude.size(), CV_8UC1);
	double normalizationFactor = 4.0 * sqrt(2.0);

	for (int i = 0; i < magnitude.rows; i++) {
		for (int j = 0; j < magnitude.cols; j++) {
			float normalizedValue = magnitude.at<float>(i, j) / normalizationFactor;
			normalizedMagnitude.at<uchar>(i, j) = saturate_cast<uchar>(normalizedValue);
		}
	}

	magnitude = normalizedMagnitude;
}

// Step 3: Non-maxima suppression
void applyNonMaximaSuppression(const Mat& magnitude, const Mat& direction, Mat& result) {
	result = Mat::zeros(magnitude.size(), CV_8UC1);

	// Process the inner part of the image (excluding borders)
	for (int i = 1; i < magnitude.rows - 1; i++) {
		for (int j = 1; j < magnitude.cols - 1; j++) {
			float angle = direction.at<float>(i, j);
			int mag = magnitude.at<uchar>(i, j);

			// Skip processing if magnitude is zero
			if (mag == 0) {
				continue;
			}

			// Map angle to one of the 4 directions (0°, 45°, 90°, 135°)
			int angleDirection;

			// Normalize angle to [0, 180) for edge direction (gradient is perpendicular to edge)
			angle = fmod(angle, PI);

			// Determine which direction the angle corresponds to
			if ((angle >= 0 && angle < PI / 8) || (angle >= 7 * PI / 8 && angle < PI)) {
				angleDirection = 0; // 0° direction - horizontal edge
			}
			else if (angle >= PI / 8 && angle < 3 * PI / 8) {
				angleDirection = 1; // 45° direction - diagonal edge
			}
			else if (angle >= 3 * PI / 8 && angle < 5 * PI / 8) {
				angleDirection = 2; // 90° direction - vertical edge
			}
			else {
				angleDirection = 3; // 135° direction - diagonal edge
			}

			// Get magnitude values of the pixels in the direction of gradient
			int mag1 = 0, mag2 = 0;

			// Check neighbors based on direction
			switch (angleDirection) {
			case 0: // 0° - horizontal - check east/west
				mag1 = magnitude.at<uchar>(i, j + 1);
				mag2 = magnitude.at<uchar>(i, j - 1);
				break;
			case 1: // 45° - check northeast/southwest
				mag1 = magnitude.at<uchar>(i - 1, j + 1);
				mag2 = magnitude.at<uchar>(i + 1, j - 1);
				break;
			case 2: // 90° - vertical - check north/south
				mag1 = magnitude.at<uchar>(i - 1, j);
				mag2 = magnitude.at<uchar>(i + 1, j);
				break;
			case 3: // 135° - check northwest/southeast
				mag1 = magnitude.at<uchar>(i - 1, j - 1);
				mag2 = magnitude.at<uchar>(i + 1, j + 1);
				break;
			}

			// Keep only local maxima
			if (mag >= mag1 && mag >= mag2) {
				result.at<uchar>(i, j) = mag;
			}
		}
	}
}

// Step 4a: Adaptive thresholding
void adaptiveThresholding(const Mat& nonMaxSuppressed, Mat& thresholded) {
	// Create histogram of gradient magnitudes
	int histogram[256] = { 0 };
	int zeroGradientPixels = 0;

	// Calculate histogram and count zero gradient pixels
	for (int i = 1; i < nonMaxSuppressed.rows - 1; i++) {
		for (int j = 1; j < nonMaxSuppressed.cols - 1; j++) {
			int pixelValue = nonMaxSuppressed.at<uchar>(i, j);
			histogram[pixelValue]++;

			if (pixelValue == 0) {
				zeroGradientPixels++;
			}
		}
	}

	// Calculate the number of non-zero gradient pixels
	int totalPixels = (nonMaxSuppressed.rows - 2) * (nonMaxSuppressed.cols - 2);
	int nonZeroPixels = totalPixels - zeroGradientPixels;

	// We assume 10% of the non-zero gradient pixels are edge pixels
	float p = 0.1;
	int nonEdgePixels = (1 - p) * nonZeroPixels;

	// Find ThresholdHigh by counting from the histogram
	int sum = 0;
	int thresholdHigh = 0;

	for (int i = 1; i < 256; i++) {
		sum += histogram[i];
		if (sum > nonEdgePixels) {
			thresholdHigh = i;
			break;
		}
	}

	// Calculate ThresholdLow as 40% of ThresholdHigh
	float k = 0.4;
	int thresholdLow = k * thresholdHigh;
	cout << "Threshold low:  " << thresholdLow << " , Threshold high: " << thresholdHigh << endl;

	// Apply thresholding to categorize pixels
	thresholded = Mat::zeros(nonMaxSuppressed.size(), CV_8UC1);

	// Define edge categories
	const uchar NO_EDGE = 0;
	const uchar WEAK_EDGE = 128;
	const uchar STRONG_EDGE = 255;

	for (int i = 0; i < nonMaxSuppressed.rows; i++) {
		for (int j = 0; j < nonMaxSuppressed.cols; j++) {
			int pixelValue = nonMaxSuppressed.at<uchar>(i, j);

			if (pixelValue < thresholdLow) {
				thresholded.at<uchar>(i, j) = NO_EDGE;
			}
			else if (pixelValue >= thresholdLow && pixelValue <= thresholdHigh) {
				thresholded.at<uchar>(i, j) = WEAK_EDGE;
			}
			else {
				thresholded.at<uchar>(i, j) = STRONG_EDGE;
			}
		}
	}
}

// Step 4b: Edge linking through hysteresis
void edgeLinkingHysteresis(Mat& thresholded) {
	const uchar NO_EDGE = 0;
	const uchar WEAK_EDGE = 128;
	const uchar STRONG_EDGE = 255;

	// Queue for storing STRONG_EDGE points
	queue<Point> edgeQueue;

	// Create a copy of the thresholded image
	Mat result = thresholded.clone();

	// Scan the image to find STRONG_EDGE points
	for (int i = 1; i < thresholded.rows - 1; i++) {
		for (int j = 1; j < thresholded.cols - 1; j++) {
			if (thresholded.at<uchar>(i, j) == STRONG_EDGE) {
				// Found a STRONG_EDGE point, process it
				edgeQueue.push(Point(j, i));

				// Process the queue
				while (!edgeQueue.empty()) {
					Point p = edgeQueue.front();
					edgeQueue.pop();

					// Check all 8 neighbors
					for (int ni = -1; ni <= 1; ni++) {
						for (int nj = -1; nj <= 1; nj++) {
							// Skip the center point
							if (ni == 0 && nj == 0) {
								continue;
							}

							int newY = p.y + ni;
							int newX = p.x + nj;

							// Check if the neighbor is within the image boundaries
							if (newY >= 1 && newY < thresholded.rows - 1 &&
								newX >= 1 && newX < thresholded.cols - 1) {

								// If it's a WEAK_EDGE, convert it to STRONG_EDGE and add to queue
								if (thresholded.at<uchar>(newY, newX) == WEAK_EDGE) {
									thresholded.at<uchar>(newY, newX) = STRONG_EDGE;
									edgeQueue.push(Point(newX, newY));
								}
							}
						}
					}
				}
			}
		}
	}

	// Remove remaining WEAK_EDGE points by setting them to NO_EDGE
	for (int i = 0; i < thresholded.rows; i++) {
		for (int j = 0; j < thresholded.cols; j++) {
			if (thresholded.at<uchar>(i, j) == WEAK_EDGE) {
				thresholded.at<uchar>(i, j) = NO_EDGE;
			}
		}
	}
}

void cannyEdgeDetection() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		imshow("Original Image", src);

		// Step 1: Apply Gaussian filtering with sigma = 0.5
		Mat gaussianFiltered;
		applyGaussianFilter(src, gaussianFiltered);

		// Step 2: Compute gradient using Sobel operators
		Mat gradX, gradY;
		applySobelOperators(gaussianFiltered, gradX, gradY);

		// Calculate gradient magnitude and direction
		Mat gradientMagnitude, gradientDirection;
		calculateGradient(gradX, gradY, gradientMagnitude, gradientDirection);

		// Step 3: Apply non-maxima suppression
		Mat nonMaxSuppressed;
		applyNonMaximaSuppression(gradientMagnitude, gradientDirection, nonMaxSuppressed);

		// Step 4a: Apply adaptive thresholding
		Mat thresholded;
		adaptiveThresholding(nonMaxSuppressed, thresholded);

		// Save a copy for display
		Mat afterThresholding = thresholded.clone();

		// Step 4b: Apply edge linking through hysteresis
		edgeLinkingHysteresis(thresholded);

		// Display results
		imshow("Step 1: Gaussian Filtered", gaussianFiltered);
		imshow("Step 2: Gradient Magnitude", gradientMagnitude);
		imshow("Step 3: Non-Maxima Suppression", nonMaxSuppressed);
		imshow("Step 4a: After Adaptive Thresholding", afterThresholding);
		imshow("Step 4b: Final Edges after Hysteresis", thresholded);

		waitKey(0);
	}
}

int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	projectPath = _wgetcwd(0, 0);

	char fname[MAX_PATH];
	int n, w;

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1  - Open image\n");
		printf(" 2  - Open BMP images from folder\n");
		printf(" 3  - Image negative\n");
		printf(" 4  - Image negative (fast)\n");
		printf(" 5  - BGR->Gray\n");
		printf(" 6  - BGR->Gray (fast, save result to disk) \n");
		printf(" 7  - BGR->HSV\n");
		printf(" 8  - Resize image\n");
		printf(" 9  - Canny edge detection\n");
		printf(" 10 - Edges in a video sequence\n");
		printf(" 11 - Snap frame from live video\n");
		printf(" 12 - Mouse callback demo\n");
		printf(" 13 - Additive factor\n");
		printf(" 14 - Multiplicative factor\n");
		printf(" 15 - Colored squares image\n");
		printf(" 16 - Inverse matrix\n");
		printf(" 17 - Get RGB channels\n");
		printf(" 18 - RGB to grayscale\n");
		printf(" 19 - Grayscale to black and white\n");
		printf(" 20 - RGB to HSV\n");
		printf(" 21 - Display histogram\n");
		printf(" 22 - Display reduced histogram\n");
		printf(" 23 - Compute PDF\n");
		printf(" 24 - Multi-level thresholding\n");
		printf(" 25 - Floyd Steinberg dithering\n");
		printf(" 26 - Mouse callback demo\n");
		printf(" 27 - Processing function\n");
		printf(" 28 - BFS component labeling\n");
		printf(" 29 - Two-pass component labeling\n");
		printf(" 30 - Border tracing\n");
		printf(" 31 - Reconstruct border\n");
		printf(" 32 - Dilation\n");
		printf(" 33 - Dilation applied n times\n");
		printf(" 34 - Erosion\n");
		printf(" 35 - Erosion applied n times\n");
		printf(" 36 - Opening\n");
		printf(" 37 - Closing\n");
		printf(" 38 - Boundary extraction\n");
		printf(" 39 - Region filling\n");
		printf(" 40 - Flip image vertically\n");
		printf(" 41 - The mean and standard deviation + cumulative histogram\n");
		printf(" 42 - Threshold computation\n");
		printf(" 43 - Histogram equalization\n");
		printf(" 44 - Histogram stretching/shrinking\n");
		printf(" 45 - Gamma correction\n");
		printf(" 46 - Modify brightness\n");
		printf(" 47 - License Plate Recognition\n");
		printf(" 48 - Apply filters\n");
		printf(" 49 - Generic frequency domain filter\n");
		printf(" 50 - Median filter\n");
		printf(" 51 - Apply Gaussian filter\n");
		printf(" 52 - Canny Edge Detection\n");
		printf(" 53 - Evaluate plate detection\n");
		printf(" 0  - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testNegativeImage();
			break;
		case 4:
			testNegativeImageFast();
			break;
		case 5:
			testColor2Gray();
			break;
		case 6:
			testImageOpenAndSave();
			break;
		case 7:
			testBGR2HSV();
			break;
		case 8:
			testResize();
			break;
		case 9:
			testCanny();
			break;
		case 10:
			testVideoSequence();
			break;
		case 11:
			testSnap();
			break;
		case 12:
			testMouseClick();
			break;
		case 13:
			int factor;
			printf("Enter the additive factor: ");
			scanf("%d", &factor);
			additive_factor(factor);
			break;
		case 14:
			float factor1;
			printf("Enter the multiplicative factor: ");
			scanf("%f", &factor1);
			multiplicative_factor(factor1);
			break;
		case 15:
			make_squares();
			break;
		case 16:
			inverse_matrix();
			break;
		case 17:
			getRGB();
			break;
		case 18:
			while (openFileDlg(fname)) {
				Mat src = imread(fname, IMREAD_COLOR);
				imshow("Original image", src);
				Mat dst = RGBtoGray(src);
				imshow("Grayscale image", dst);
				waitKey(0);
			}
			break;
		case 19:
			int threshold;
			printf("Enter threshold: ");
			scanf("%d", &threshold);
			while (openFileDlg(fname))
			{
				Mat src = imread(fname, IMREAD_GRAYSCALE);
				imshow("Original image", src);
				Mat dst = gray_to_binary(src, threshold);
				imshow("After grayscale to black and white", dst);
				waitKey(0);
			}
			break;
		case 20:
			RGBtoHSV();
			break;
		case 21:
			showHistogram("Histogram", computeHistogram(imread("Images/cameraman.bmp", IMREAD_GRAYSCALE)), 256, 256);
			waitKey(0);
			break;
		case 22:
			int m;
			printf("Enter the number of bins: ");
			scanf("%d", &m);
			if (m < 0 || m > 256) {
				printf("Invalid number of bins.\n");
				break;
			}
			showHistogram("Histogram", histogramReduced(imread("Images/cameraman.bmp", IMREAD_GRAYSCALE), m), m, 256);
			waitKey(0);
			break;
		case 23:
		{
			float sum = 0.0;
			float* pdf = computePDF(imread("Images/cameraman.bmp", IMREAD_GRAYSCALE));
			for (int i = 0; i < 256; i++) {
				printf("PDF[%d] = %f\n", i, pdf[i]);
				sum += pdf[i];
			}
			printf("Sum of PDF values: %f\n", sum);
			waitKey(3000);
			break;
		}
		case 24:
			MultilevelThresholding();
			break;
		case 25:
			FloydSteinberg();
			break;
		case 26:
			testMouseClick();
			break;
		case 27:
			while (openFileDlg(fname))
			{
				Mat labeledImage = imread(fname);
				if (labeledImage.empty()) {
					printf("Error: Could not open image.\n");
					break;
				}

				double areaThreshold, phiLow, phiHigh;

				// Get user input for area threshold
				printf("Enter area threshold (objects with area < this will be kept): ");
				scanf("%lf", &areaThreshold);

				// Get user input for orientation phi range
				printf("Enter lower bound for orientation phi (in degrees): ");
				scanf("%lf", &phiLow);

				printf("Enter upper bound for orientation phi (in degrees): ");
				scanf("%lf", &phiHigh);

				// Apply filtering
				Mat filteredImage = filterObjectsByAreaAndOrientation(&labeledImage, areaThreshold, phiLow, phiHigh);

				// Display original and filtered images
				imshow("Original Labeled Image", labeledImage);
				imshow("Filtered Objects", filteredImage);
				waitKey(0);
			}
			break;
		case 28:
			bfs();
			break;
		case 29:
			two_pass_labeling();
			break;
		case 30:
			border_tracing();
			break;
		case 31:
			reconstruct_border();
			break;
		case 32:
			while (openFileDlg(fname))
			{
				Mat src = imread(fname, IMREAD_GRAYSCALE);
				imshow("Original image", src);
				Mat dst = dilation(src);
				imshow("After dilation", dst);
				waitKey(0);
			}
			break;
		case 33:
			printf("Enter the number of times to dilate: ");
			scanf("%d", &n);
			while (openFileDlg(fname))
			{
				Mat src = imread(fname, IMREAD_GRAYSCALE);
				imshow("Original image", src);
				Mat dst = dilate_n_times(src, n);
				imshow("After applying dilation n times", dst);
				waitKey(0);
			}
			break;
		case 34:
			while (openFileDlg(fname))
			{
				Mat src = imread(fname, IMREAD_GRAYSCALE);
				imshow("Original image", src);
				Mat dst = erosion(src);
				imshow("After erosion", dst);
				waitKey(0);
			}
			break;
		case 35:
			printf("Enter the number of times to erode: ");
			scanf("%d", &n);
			while (openFileDlg(fname))
			{
				Mat src = imread(fname, IMREAD_GRAYSCALE);
				imshow("Original image", src);
				Mat dst = erode_n_times(src, n);
				imshow("After applying erosion n times", dst);
				waitKey(0);
			}
			break;
		case 36:
			printf("Enter the number of times to apply opening: ");
			scanf("%d", &n);
			while (openFileDlg(fname))
			{
				Mat src = imread(fname, IMREAD_GRAYSCALE);
				imshow("Original image", src);
				Mat dst = opening(src);
				imshow("After opening", dst);
				for (int i = 1; i < n; i++)
					dst = opening(dst);
				imshow("After applying opening n times", dst);
				waitKey(0);
			}
			break;
		case 37:
			printf("Enter the number of times to apply closing: ");
			scanf("%d", &n);
			while (openFileDlg(fname))
			{
				Mat src = imread(fname, IMREAD_GRAYSCALE);
				imshow("Original image", src);
				Mat dst = closing(src);
				imshow("After closing", dst);
				for (int i = 1; i < n; i++)
					dst = closing(dst);
				imshow("After applying closing n times", dst);
				waitKey(0);
			}
			break;
		case 38:
			while (openFileDlg(fname))
			{
				Mat src = imread(fname, IMREAD_GRAYSCALE);
				imshow("Original image", src);
				Mat dst = boundary_extraction(src);
				imshow("After boundary extraction", dst);
				waitKey(0);
			}
			break;
		case 39:
			while (openFileDlg(fname))
			{
				Mat src = imread(fname, IMREAD_GRAYSCALE);
				imshow("Original image", src);
				Mat dst = region_filling(src);
				imshow("After region filling", dst);
				waitKey(0);
			}
			break;
		case 40:
			while (openFileDlg(fname))
			{
				Mat src = imread(fname, IMREAD_COLOR);
				imshow("Original image", src);
				Mat dst = flip_vertically(src);
				imshow("After flipping vertically", dst);
				waitKey(0);
			}
			break;
		case 41:
			while (openFileDlg(fname))
			{
				Mat src = imread(fname, IMREAD_GRAYSCALE);
				imshow("Original image", src);
				showHistogram("Initial histogram", computeHistogram(src), 256, 256);
				double mean = computeMean(src);
				printf("Mean intensity value: %f\n", mean);
				double stddev = computeStandardDeviation(src);
				printf("Standard deviation: %f\n", stddev);
				showHistogram("Cumulative histogram", computeCumulativeHistogram(src), 256, 256);
				waitKey(0);
			}
			break;
		case 42:
			float error;
			printf("Enter the error threshold: ");
			scanf("%f", &error);
			while (openFileDlg(fname))
			{
				Mat src = imread(fname, IMREAD_GRAYSCALE);
				imshow("Original image", src);
				int threshold = computeThreshold(src, error);
				printf("Computed threshold: %d\n", threshold);
				Mat binaryImage = gray_to_binary(src, threshold);
				imshow("Binary image after thresholding", binaryImage);
				waitKey(0);
			}
			break;
		case 43:
			while (openFileDlg(fname))
			{
				Mat src = imread(fname, IMREAD_GRAYSCALE);
				imshow("Original image", src);
				showHistogram("Initial histogram", computeHistogram(src), 256, 256);
				Mat dst = histogramEqualization(src);
				imshow("After histogram equalization", dst);
				showHistogram("Equalized histogram", computeHistogram(dst), 256, 256);
				waitKey(0);
			}
			break;
		case 44:
			int minVal, maxVal;
			printf("Enter the minimum value for histogram stretching/shrinking: ");
			scanf("%d", &minVal);
			printf("Enter the maximum value for histogram stretching/shrinking: ");
			scanf("%d", &maxVal);
			while (openFileDlg(fname))
			{
				Mat src = imread(fname, IMREAD_GRAYSCALE);
				imshow("Original image", src);
				showHistogram("Initial histogram", computeHistogram(src), 256, 256);
				Mat dst = modifyContrast(src, minVal, maxVal);
				imshow("After histogram stretching", dst);
				showHistogram("Stretched histogram", computeHistogram(dst), 256, 256);
				waitKey(0);
			}
			break;
		case 45:
			float gamma;
			printf("Enter the gamma value for gamma correction: ");
			scanf("%f", &gamma);
			while (openFileDlg(fname))
			{
				Mat src = imread(fname, IMREAD_GRAYSCALE);
				imshow("Original image", src);
				showHistogram("Initial histogram", computeHistogram(src), 256, 256);
				Mat dst = gammaCorrection(src, gamma);
				imshow("After gamma correction", dst);
				showHistogram("Gamma corrected histogram", computeHistogram(dst), 256, 256);
				waitKey(0);
			}
			break;
		case 46:
			int brightness;
			printf("Enter the brightness value: ");
			scanf("%d", &brightness);
			while (openFileDlg(fname))
			{
				Mat src = imread(fname, IMREAD_GRAYSCALE);
				imshow("Original image", src);
				showHistogram("Initial histogram", computeHistogram(src), 256, 256);
				Mat dst = modifyBrightness(src, brightness);
				imshow("After brightness modification", dst);
				showHistogram("Brightness modified histogram", computeHistogram(dst), 256, 256);
				waitKey(0);
			}
			break;
		case 47:
			recognizeLicensePlate();
			break;
		case 48:
			testFilters();
			break;
		case 49:
			generic_frequency_domain_filter_test();
			break;
		case 50:
			printf("Enter the size of the filter: ");
			scanf("%d", &w);
			while (openFileDlg(fname))
			{
				Mat src = imread(fname, IMREAD_GRAYSCALE);
				double t = (double)getTickCount();
				Mat dst = medianFilter(src, w);
				t = ((double)getTickCount() - t) / getTickFrequency();
				cout << "Time [ms]:" << t * 1000 << endl;
				imshow("Original image", src);
				imshow("After median filtering", dst);
				waitKey(0);
			}
			break;
		case 51:
			applyGaussFilters();
			break;
		case 52:
			cannyEdgeDetection();
			break;
		case 53:
			cout << "Starting license plate detection evaluation..." << endl;
			evaluatePlateDetection();
			cout << "Evaluation complete." << endl;
			system("pause");
			break;
		}
	} while (op != 0);

	return 0;
}
