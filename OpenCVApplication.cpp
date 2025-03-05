// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
using namespace std;

wchar_t* projectPath;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
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
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
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
void gray_to_binary(int threshold) {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);

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

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("binary image", dst);

		waitKey();
	}
}

/* Create a function that will convert a color RGB24 image (CV_8UC3 type) to 
a grayscale image (CV_8UC1) and display the result image in a destination window. 
*/
void RGBtoGray() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_COLOR);

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

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("grayscale image", dst);

		waitKey();
	}
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
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}


void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
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

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

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
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
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
		Canny(grayFrame,edges,40,100,3);
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
		if (c == 115){ //'s' pressed - snap the image to a file
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

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
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
	for(int i = 0; i < 256; i++)
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

// Implement the multilevel thresholding algorithm. 
vector<int> getLocalMaxima(Mat img, int WH, double TH) {
	double v;
	vector<int> vecMax;
	int height = img.rows;
	int width = img.cols;
	float* pdf = computePDF(img);
	vecMax.push_back(0);

	for (int k = WH; k <= 255 - WH; k++) {
		double v = 0;
		for (int i = k - WH; i <= k + WH; i++) 
			v += pdf[i];
		v = v / (2 * WH + 1);
		if (pdf[k] > v + TH) {
			for (int j = k - WH; j <= k + WH; j++)
				if (pdf[k] >= pdf[j])
					vecMax.push_back(k);
		}
	}
	vecMax.push_back(255);

	return vecMax;
}

Mat multilevelThresholding(Mat img, int WH, double TH)
{
	vector<int> vecMax = getLocalMaxima(img, WH, TH);

	Mat image = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			int closest = 0, smallest = 256;
			for (int i : vecMax) {
				int difference = abs(img.at<uchar>(i, j) - i);
				if (difference <= smallest)
				{
					smallest = difference;
					closest = i;
				}
			}
			image.at<uchar>(i, j) = closest;
		}
	return image;
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
	for (int i = 0; i<hist_cols; i++)
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

int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

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
		printf(" 0  - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
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
				RGBtoGray();
				break;
			case 19:
				int threshold;
				printf("Enter threshold: ");
				scanf("%d", &threshold);
				gray_to_binary(threshold);
				break;
			case 20:
				RGBtoHSV();
				break;
			case 21: 
				showHistogram("Histogram", computeHistogram(imread("Images/cameraman.bmp", IMREAD_GRAYSCALE)), 256, 200);
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
				showHistogram("Histogram", histogramReduced(imread("Images/cameraman.bmp", IMREAD_GRAYSCALE), m), m, 200);
				waitKey(0);
				break;
			case 23:
				float sum = 0.0;
				float* pdf = computePDF(imread("Images/cameraman.bmp", IMREAD_GRAYSCALE));
				for (int i = 0; i < 256; i++) {
					printf("PDF[%d] = %f\n", i, pdf[i]);
					sum += pdf[i];
				}
				printf("Sum of PDF values: %f\n", sum);
				waitKey(3000);
				break;
			case 24:
				Mat img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);
				Mat result = multilevelThresholding(img, 5, 0.0003);
				imshow("Multilevel thresholding", result);
				waitKey(0);
				break;
		}
	}
	while (op!=0);
	return 0;
}