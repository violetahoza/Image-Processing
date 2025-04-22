#pragma once
#define NOGDI
#define NOUSER
#include <windows.h>

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

using namespace cv;
using namespace std;

// Main entry point for license plate recognition
void recognizeLicensePlate();

// Display intermediate processing steps with annotations
//void showStep(const string& windowName, const Mat& img, const string& text, int stepNum, int delay = 2000);

// Image preprocessing function with adaptive enhancements
Mat preprocessImage(const Mat& src, bool showSteps = false);

// Helper function to estimate image noise level
double estimateNoise(const Mat& image);

// License plate detection function (returns rotated rectangles for each detected plate)
vector<RotatedRect> detectLicensePlates(const Mat& preprocessed, const Mat& original, bool showSteps = false);

// Helper function to calculate overlap between two rotated rectangles
float calculateRectOverlap(const RotatedRect& rect1, const RotatedRect& rect2);

// Corrects perspective distortion in license plates
Mat adjustPlatePerspective(const Mat& src, const RotatedRect& plate, bool showSteps = false);

// Character segmentation function
vector<Mat> segmentCharacters(const Mat& plate, bool showSteps = false);

// Template-based character recognition
char recognizeCharacter(const Mat& charImg);

// Helper function to create templates for character recognition
map<char, Mat> createCharacterTemplates();

// Post-processing to improve recognition results
std::string postProcessLicensePlate(const std::string& plateText);

// Validates if a region contains a license plate
bool validatePlateRegion(const Mat& plateRegion);