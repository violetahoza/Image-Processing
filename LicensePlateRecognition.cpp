#define NOGDI
#define NOUSER
#include <windows.h>

#include "stdafx.h"
#include "LicensePlateRecognition.h"
#include "common.h"
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cfloat>
#include <math.h>
//#include <tesseract/baseapi.h>
//#include <leptonica/allheaders.h>

// Global constants for display
const int DISPLAY_WIDTH = 800;
const int DISPLAY_HEIGHT = 600;
const Scalar DISPLAY_COLOR(255, 0, 255);
const int DISPLAY_THICKNESS = 2;

// Constants for plate detection
const double MIN_PLATE_ASPECT_RATIO = 1.0;
const double MAX_PLATE_ASPECT_RATIO = 12.0;
const double MIN_PLATE_AREA_RATIO = 0.0005;
const double MAX_PLATE_AREA_RATIO = 0.05;
const double MIN_CHAR_HEIGHT_RATIO = 0.2;
const double MAX_CHAR_HEIGHT_RATIO = 0.95;
const double MIN_CHAR_ASPECT_RATIO = 0.1;
const double MAX_CHAR_ASPECT_RATIO = 2.0;
const double MIN_PLATE_WIDTH = 60;  // Minimum width in pixels
const int MIN_CHAR_COUNT = 3; // Minimum characters in a valid plate
const int MAX_CHAR_COUNT = 12; // Maximum characters in a valid plate
const float MIN_CHAR_CONFIDENCE = 50.0f;


void showStep(const string& windowName, const Mat& img, const string& text, int stepNum, int delay = 2000) {
	Mat display;
	if (img.channels() == 1) {
		cvtColor(img, display, COLOR_GRAY2BGR);
	}
	else {
		img.copyTo(display);
	}

	putText(display, to_string(stepNum) + ". " + text,
		Point(20, 40), FONT_HERSHEY_SIMPLEX,
		1.0, DISPLAY_COLOR, DISPLAY_THICKNESS);

	imshow(windowName, display);
	waitKey(delay);
}

Mat preprocessImage(const Mat& src, bool showSteps) {
	// Create display window if showing steps
	if (showSteps) {
		namedWindow("Preprocessing Steps", WINDOW_NORMAL);
		resizeWindow("Preprocessing Steps", DISPLAY_WIDTH, DISPLAY_HEIGHT);
		showStep("Preprocessing Steps", src, "Original Image", 0);
	}

	// Convert to grayscale
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	if (showSteps) showStep("Preprocessing Steps", gray, "Grayscale Conversion", 1);

	// Analyze image histogram to determine lighting conditions
	Mat hist;
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	calcHist(&gray, 1, 0, Mat(), hist, 1, &histSize, &histRange);

	// Calculate histogram statistics
	double totalPixels = gray.rows * gray.cols;
	double darkPixels = 0, brightPixels = 0;

	for (int i = 0; i < 64; i++) {
		darkPixels += hist.at<float>(i) / totalPixels;
	}
	for (int i = 192; i < 256; i++) {
		brightPixels += hist.at<float>(i) / totalPixels;
	}

	// Adaptive contrast enhancement based on lighting conditions
	Mat enhancedGray;
	if (darkPixels > 0.4) {
		// Image is too dark - use stronger CLAHE or gamma correction
		Mat temp;
		double gamma = 0.5; // Gamma < 1 brightens dark regions
		gray.convertTo(temp, CV_32F, 1.0 / 255.0);
		pow(temp, gamma, temp);
		temp.convertTo(enhancedGray, CV_8U, 255);

        // Normalize to ensure full range
        normalize(enhancedGray, enhancedGray, 0, 255, NORM_MINMAX);

		if (showSteps) showStep("Preprocessing Steps", enhancedGray, "Gamma Correction for Dark Image", 2);
	}
	else if (brightPixels > 0.4) {
		// Image is too bright - reduce brightness and increase contrast
		Mat temp;
		double alpha = 1.3;  // Higher contrast
		double beta = -25;  // Lower brightness
		gray.convertTo(temp, CV_8U, alpha, beta);
		enhancedGray = temp;

		if (showSteps) showStep("Preprocessing Steps", enhancedGray, "Contrast Adjustment for Bright Image", 2);
	}
	else {
		// Normal lighting - apply standard CLAHE
		Ptr<CLAHE> clahe = createCLAHE(3.5, Size(8, 8));
		clahe->apply(gray, enhancedGray);

		if (showSteps) showStep("Preprocessing Steps", enhancedGray, "CLAHE Enhancement", 2);
	}

	// Analyze image for noise level
	double noiseLevel = estimateNoise(enhancedGray);

	// Apply appropriate noise reduction based on noise level
	Mat denoised;
    //if (noiseLevel > 10) {
    //    // Use stronger filtering but preserve edges
    //    Mat blurred;
    //    bilateralFilter(enhancedGray, blurred, 9, 75, 75);
    //    GaussianBlur(blurred, denoised, Size(3, 3), 0);
    //    if (showSteps) showStep("Preprocessing Steps", denoised, "Noise Reduction", 3);
    //}
	if (noiseLevel > 15) {
		// High noise - use stronger filtering
		GaussianBlur(enhancedGray, denoised, Size(5, 5), 1.8);
		if (showSteps) showStep("Preprocessing Steps", denoised, "Gaussian Blur (High Noise)", 3);
	}
	else if (noiseLevel > 8) {
        // Medium noise - balanced approach
        bilateralFilter(enhancedGray, denoised, 7, 50, 50);
        if (showSteps) showStep("Preprocessing Steps", denoised, "Bilateral Filtering (Medium Noise)", 3);
    }
    else {
        // Low noise - preserve edges
        bilateralFilter(enhancedGray, denoised, 5, 35, 35);
        if (showSteps) showStep("Preprocessing Steps", denoised, "Bilateral Filtering (Low Noise)", 3);
    }

    // Apply edge enhancement for better character detection
    Mat enhanced;
    Mat kernel = (Mat_<float>(3, 3) <<
        -1, -1, -1,
        -1, 9, -1,
        -1, -1, -1);
    filter2D(denoised, enhanced, -1, kernel);
    if (showSteps) showStep("Preprocessing Steps", enhanced, "Edge Enhancement", 4);

	// Apply local adaptive threshold to handle varying lighting conditions
	Mat binary;
	int blockSize = max(7, (int)(min(denoised.rows, denoised.cols) * 0.02) | 1);
	blockSize = blockSize > 31 ? 31 : blockSize; // Cap at reasonable value

	adaptiveThreshold(denoised, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
		THRESH_BINARY_INV, blockSize, 2);
	if (showSteps) showStep("Preprocessing Steps", binary, "Adaptive Thresholding", 4);

    // Morphological operations to clean up the binary image
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat cleaned;

    // Opening to remove noise (erode then dilate)
    morphologyEx(binary, cleaned, MORPH_OPEN, element);
    if (showSteps) showStep("Preprocessing Steps", cleaned, "Morphological Opening", 6);

    // Closing to fill small holes (dilate then erode)
    Mat closingElement = getStructuringElement(MORPH_RECT, Size(5, 1)); // Horizontal emphasis
    morphologyEx(cleaned, cleaned, MORPH_CLOSE, closingElement);
    if (showSteps) showStep("Preprocessing Steps", cleaned, "Morphological Closing", 7);

	return cleaned;
}

// Helper function to estimate noise level in an image
double estimateNoise(const Mat& image) {
	Mat laplacian;
	Laplacian(image, laplacian, CV_64F);

	Scalar mean, stddev;
	meanStdDev(laplacian, mean, stddev);

	// Return standard deviation as noise estimate
	return stddev[0];
}

vector<RotatedRect> detectLicensePlates(const Mat& preprocessed, const Mat& original, bool showSteps) {
    vector<RotatedRect> potentialPlates;
    Mat display = original.clone();

    //// 1. Apply morphological operations to enhance horizontal structures (license plates)
    //Mat morph;
    //Mat kernel = getStructuringElement(MORPH_RECT, Size(17, 3));
    //morphologyEx(preprocessed, morph, MORPH_CLOSE, kernel);

    /*if (showSteps) {
        showStep("Plate Detection", morph, "Morphological Operations", 1);
    }*/

    // 2. Find contours
    vector<vector<Point>> contours;
    findContours(preprocessed.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (showSteps) {
        Mat contourImg = original.clone();
        drawContours(contourImg, contours, -1, Scalar(0, 255, 0), 2);
        showStep("Plate Detection", contourImg, "All Contours Found", 2);
    }

    // 3. Filter contours
    float imgArea = original.rows * original.cols;
    Mat contourMask = Mat::zeros(original.size(), CV_8UC1);

    for (size_t i = 0; i < contours.size(); i++) {
        // Skip small contours
        if (contours[i].size() < 15) continue;

        // Get rotated rectangle
        RotatedRect minRect = minAreaRect(contours[i]);
        float width = minRect.size.width;
        float height = minRect.size.height;

        // Ensure width is always the larger dimension
        if (width < height) {
            swap(width, height);
        }

        float aspectRatio = width / height;
        float area = width * height;

        // Dynamic area constraints based on image size
        float minArea = imgArea * MIN_PLATE_AREA_RATIO;
        float maxArea = imgArea * MAX_PLATE_AREA_RATIO;

        // Check plate characteristics
        if (area > minArea &&
            area < maxArea &&
            aspectRatio > MIN_PLATE_ASPECT_RATIO &&
            aspectRatio < MAX_PLATE_ASPECT_RATIO &&
            width > MIN_PLATE_WIDTH) {

            // Draw the contour on the mask
            vector<Point> hull;
            convexHull(contours[i], hull);
            fillConvexPoly(contourMask, hull, Scalar(255));

            // Additional validation
            Mat plateRegion = adjustPlatePerspective(original, minRect, false);
            if (validatePlateRegion(plateRegion)) {
                potentialPlates.push_back(minRect);

                if (showSteps) {
                    Point2f rectPoints[4];
                    minRect.points(rectPoints);
                    for (int j = 0; j < 4; j++) {
                        line(display, rectPoints[j], rectPoints[(j + 1) % 4], Scalar(0, 255, 0), 3);
                    }
                }
            }
        }
    }

    if (showSteps) {
        showStep("Plate Detection", display, "Potential Plates", 3);
    }

    // Alternative approach: Use Hough Lines to detect plates in case the contour method misses some
    if (potentialPlates.size() < 1) {
        // Edge detection
        Mat edges;
        Canny(preprocessed, edges, 50, 150);

        // Hough Line Detection
        vector<Vec4i> lines;
        HoughLinesP(edges, lines, 1, CV_PI / 180, 80, 30, 10);

        if (showSteps && !lines.empty()) {
            Mat lineImg = original.clone();
            for (size_t i = 0; i < lines.size(); i++) {
                line(lineImg, Point(lines[i][0], lines[i][1]),
                    Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 2);
            }
            showStep("Plate Detection", lineImg, "Hough Lines", 4);
        }

        // Group lines that might form license plates
        // This is a simplified approach - would need more complex logic for production
        if (lines.size() > 4) {
            Mat regionImg = original.clone();
            // Group nearby horizontal lines
            for (size_t i = 0; i < lines.size(); i++) {
                float angle = atan2(lines[i][3] - lines[i][1],
                    lines[i][2] - lines[i][0]) * 180 / CV_PI;
                // Only consider near-horizontal lines
                if (abs(angle) < 20 || abs(angle) > 160) {
                    int y1 = min(lines[i][1], lines[i][3]);
                    int y2 = max(lines[i][1], lines[i][3]);
                    int height = y2 - y1 + 40; // Add padding
                    int width = original.cols * 0.8; // Estimate width

                    int centerX = (lines[i][0] + lines[i][2]) / 2;
                    int centerY = (y1 + y2) / 2;

                    // Create a rotated rectangle
                    RotatedRect rect(Point2f(centerX, centerY),
                        Size2f(width, height),
                        angle);

                    // Validate and add to potential plates
                    Mat region = adjustPlatePerspective(original, rect, false);
                    if (validatePlateRegion(region)) {
                        potentialPlates.push_back(rect);

                        if (showSteps) {
                            Point2f rectPoints[4];
                            rect.points(rectPoints);
                            for (int j = 0; j < 4; j++) {
                                line(regionImg, rectPoints[j], rectPoints[(j + 1) % 4],
                                    Scalar(255, 0, 255), 2);
                            }
                        }
                    }
                }
            }

            if (showSteps && !potentialPlates.empty()) {
                showStep("Plate Detection", regionImg, "Plates from Lines", 5);
            }
        }
    }

    // Apply non-maximum suppression to remove overlapping detections
    vector<RotatedRect> finalPlates;
    if (!potentialPlates.empty()) {
        // Sort by area (largest first)
        sort(potentialPlates.begin(), potentialPlates.end(),
            [](const RotatedRect& a, const RotatedRect& b) {
                return a.size.area() > b.size.area();
            });

        vector<bool> keep(potentialPlates.size(), true);

        for (size_t i = 0; i < potentialPlates.size(); i++) {
            if (!keep[i]) continue;

            for (size_t j = i + 1; j < potentialPlates.size(); j++) {
                // Calculate IoU (Intersection over Union)
                float overlap = calculateRectOverlap(potentialPlates[i], potentialPlates[j]);
                if (overlap > 0.3) { // If more than 30% overlap
                    keep[j] = false; // Suppress this detection
                }
            }
        }

        for (size_t i = 0; i < potentialPlates.size(); i++) {
            if (keep[i]) {
                finalPlates.push_back(potentialPlates[i]);
            }
        }
    }

    if (showSteps && !finalPlates.empty()) {
        Mat finalDisplay = original.clone();
        for (const RotatedRect& plate : finalPlates) {
            Point2f rectPoints[4];
            plate.points(rectPoints);
            for (int j = 0; j < 4; j++) {
                line(finalDisplay, rectPoints[j], rectPoints[(j + 1) % 4],
                    Scalar(0, 0, 255), 3);
            }
        }
        showStep("Plate Detection", finalDisplay, "Final Plates After NMS", 6);
    }

    return finalPlates.empty() ? potentialPlates : finalPlates;
}


// Helper function to calculate overlap between two rotated rectangles
float calculateRectOverlap(const RotatedRect& rect1, const RotatedRect& rect2) {
    // Convert rotated rectangles to contours
    Point2f points1[4];
    rect1.points(points1);
    vector<Point2f> contour1(points1, points1 + 4);

    Point2f points2[4];
    rect2.points(points2);
    vector<Point2f> contour2(points2, points2 + 4);

    // Calculate areas
    float area1 = rect1.size.area();
    float area2 = rect2.size.area();

    // Compute intersection area using contour overlap
    // This is a simplified approach - for accuracy, you'd need
    // a proper polygon intersection algorithm
    Rect boundingRect1 = boundingRect(contour1);
    Rect boundingRect2 = boundingRect(contour2);

    Rect intersection = boundingRect1 & boundingRect2;
    float intersectionArea = intersection.area();

    // Calculate IoU
    float unionArea = area1 + area2 - intersectionArea;
    if (unionArea <= 0) return 0.0f;

    return intersectionArea / unionArea;
}

bool validatePlateRegion(const Mat& plateRegion) {
    if (plateRegion.empty() ||
        plateRegion.rows < 15 ||
        plateRegion.cols < 50 ||
        plateRegion.rows > 200 ||
        plateRegion.cols > 800) {
        return false;
    }

    Mat gray;
    if (plateRegion.channels() > 1) {
        cvtColor(plateRegion, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = plateRegion;
    }

    // Check for minimum brightness and contrast
    Scalar mean, stddev;
    meanStdDev(gray, mean, stddev);
    if (mean[0] < 20 || stddev[0] < 10) {
        return false; // Too dark or low contrast
    }

    // Check for edge density
    Mat edges;
    Canny(gray, edges, 50, 150);
    float edgeRatio = (float)countNonZero(edges) / edges.total();
    if (edgeRatio < 0.03 || edgeRatio > 0.5) {
        return false; // Too few or too many edges
    }

    // Use character segmentation to verify if the region has enough character-like components
    vector<Mat> characters = segmentCharacters(plateRegion, false);
    if (characters.size() < MIN_CHAR_COUNT || characters.size() > MAX_CHAR_COUNT) {
        return false; // Not enough or too many characters
    }

    return true;
}

Mat adjustPlatePerspective(const Mat& src, const RotatedRect& plate, bool showSteps) {
    Mat result;

    // Get the 4 corners of the rotated rectangle
    Point2f vertices[4];
    plate.points(vertices);

    // Sort the vertices in consistent order (top-left, top-right, bottom-right, bottom-left)
    vector<Point2f> sortedVertices(4);

    // Initialize with first vertex
    sortedVertices[0] = vertices[0];
    sortedVertices[1] = vertices[0];
    sortedVertices[2] = vertices[0];

    // Calculate the center of the rectangle
    Point2f center(0, 0);
    for (int i = 0; i < 4; i++) {
        center.x += vertices[i].x;
        center.y += vertices[i].y;
    }
    center.x /= 4;
    center.y /= 4;

    // Sort vertices based on their quadrant relative to the center
    vector<Point2f> topVertices, bottomVertices;
    for (int i = 0; i < 4; i++) {
        if (vertices[i].y < center.y) {
            topVertices.push_back(vertices[i]);
        }
        else {
            bottomVertices.push_back(vertices[i]);
        }
    }

    // Sort top vertices by x-coordinate
    if (topVertices.size() == 2) {
        sortedVertices[0] = topVertices[0].x < topVertices[1].x ? topVertices[0] : topVertices[1]; // top-left
        sortedVertices[1] = topVertices[0].x > topVertices[1].x ? topVertices[0] : topVertices[1]; // top-right
    }

    // Sort bottom vertices by x-coordinate
    if (bottomVertices.size() == 2) {
        sortedVertices[3] = bottomVertices[0].x < bottomVertices[1].x ? bottomVertices[0] : bottomVertices[1]; // bottom-left
        sortedVertices[2] = bottomVertices[0].x > bottomVertices[1].x ? bottomVertices[0] : bottomVertices[1]; // bottom-right
    }

    // Handle cases where sorting might not work perfectly
    if (topVertices.size() != 2 || bottomVertices.size() != 2) {
        // Fallback to original points
        for (int i = 0; i < 4; i++) {
            sortedVertices[i] = vertices[i];
        }
    }

    // Determine the width and height of the output image
    float width = max(
        norm(sortedVertices[0] - sortedVertices[1]),
        norm(sortedVertices[3] - sortedVertices[2])
    );

    float height = max(
        norm(sortedVertices[0] - sortedVertices[3]),
        norm(sortedVertices[1] - sortedVertices[2])
    );

    // If width is less than height, swap them and adjust vertices
    // This ensures license plates are always wider than they are tall
    if (width < height) {
        swap(width, height);
        vector<Point2f> temp = sortedVertices;
        sortedVertices[0] = temp[3]; // top-left becomes old bottom-left
        sortedVertices[1] = temp[0]; // top-right becomes old top-left
        sortedVertices[2] = temp[1]; // bottom-right becomes old top-right
        sortedVertices[3] = temp[2]; // bottom-left becomes old bottom-right
    }

    // Ensure minimum size
    width = max(width, 100.0f);
    height = max(height, 30.0f);

    // Define the destination points for the perspective transform
    vector<Point2f> dstPoints = {
        Point2f(0, 0),                 // top-left
        Point2f(width - 1, 0),         // top-right
        Point2f(width - 1, height - 1), // bottom-right
        Point2f(0, height - 1)          // bottom-left
    };

    // Compute the perspective transform matrix and apply it
    Mat transform = getPerspectiveTransform(sortedVertices, dstPoints);
    warpPerspective(src, result, transform, Size(width, height));

    if (showSteps) {
        Mat display = src.clone();
        for (int i = 0; i < 4; i++) {
            circle(display, sortedVertices[i], 5, Scalar(0, 0, 255), -1);
            line(display, sortedVertices[i], sortedVertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
            putText(display, to_string(i), sortedVertices[i], FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
        }
        showStep("Perspective Correction", display, "Plate Corners", 1);
        showStep("Perspective Correction", result, "Corrected Plate", 2);
    }

    return result;
}

vector<Mat> segmentCharacters(const Mat& plate, bool showSteps) {
    vector<Mat> characters;

	if (plate.empty()) {
		return characters;
	}

    Mat plateGray, plateBinary;

    // Convert to grayscale if needed
    if (plate.channels() == 3) {
        cvtColor(plate, plateGray, COLOR_BGR2GRAY);
    }
    else {
        plateGray = plate.clone();
    }

    if (showSteps) {
        showStep("Character Segmentation", plateGray, "Grayscale Plate Image", 1);
    }

    // Enhance contrast
    Mat enhanced;
    equalizeHist(plateGray, enhanced);

    // Special handling for dark/night images
    double minVal, maxVal;
    minMaxLoc(plateGray, &minVal, &maxVal);
    if (maxVal < 150) { // Dark image
        enhanced.convertTo(enhanced, -1, 2.5, 30);
    }

    if (showSteps) {
        showStep("Character Segmentation", enhanced, "Enhanced Contrast", 2);
    }

    // Adaptive thresholding with different parameters for day/night
    if (maxVal < 150) {
        adaptiveThreshold(plateGray, plateBinary, 255,
            ADAPTIVE_THRESH_GAUSSIAN_C,
            THRESH_BINARY_INV, 15, 10);
    }
    else {
        adaptiveThreshold(plateGray, plateBinary, 255,
            ADAPTIVE_THRESH_GAUSSIAN_C,
            THRESH_BINARY_INV, 13, 2);
    }

    // Clean up the binary image
    Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
    morphologyEx(plateBinary, plateBinary, MORPH_CLOSE, kernel);

    // Close small gaps in characters
    Mat closeKernel = getStructuringElement(MORPH_RECT, Size(1, 3));
    morphologyEx(plateBinary, plateBinary, MORPH_CLOSE, closeKernel);

    if (showSteps) {
        showStep("Character Segmentation", plateBinary, "Binary Plate Image", 3);
    }

    // Find connected components (characters)
    Mat labels, stats, centroids;
    int numLabels = connectedComponentsWithStats(plateBinary, labels, stats, centroids);

    // Filter components by size and position
    vector<Rect> charRects;
    for (int i = 1; i < numLabels; i++) { // Start from 1 to skip background
        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int w = stats.at<int>(i, CC_STAT_WIDTH);
        int h = stats.at<int>(i, CC_STAT_HEIGHT);
        int area = stats.at<int>(i, CC_STAT_AREA);

        // Calculate relative dimensions
        float heightRatio = (float)h / plateBinary.rows;
        float widthRatio = (float)w / plateBinary.cols;
        float aspectRatio = (float)w / h;

        // Filter criteria - more permissive
        if (heightRatio > MIN_CHAR_HEIGHT_RATIO &&
            heightRatio < MAX_CHAR_HEIGHT_RATIO &&
            widthRatio > 0.01 && widthRatio < 0.3 &&
            aspectRatio > MIN_CHAR_ASPECT_RATIO &&
            aspectRatio < MAX_CHAR_ASPECT_RATIO &&
            area > 30) {

            charRects.push_back(Rect(x, y, w, h));
        }
    }

    if (showSteps && !charRects.empty()) {
        Mat visualization = plateGray.clone();
        cvtColor(visualization, visualization, COLOR_GRAY2BGR);
        for (const Rect& r : charRects) {
            rectangle(visualization, r, Scalar(0, 255, 0), 2);
        }
        showStep("Character Segmentation", visualization, "Character Candidates", 4);
    }

    // Sort characters left-to-right
    sort(charRects.begin(), charRects.end(),
        [](const Rect& a, const Rect& b) { return a.x < b.x; });

    // Group characters by vertical position if multiple rows
    if (charRects.size() > 3) {
        // Find the median character height
        vector<int> heights;
        for (const Rect& r : charRects) {
            heights.push_back(r.height);
        }
        sort(heights.begin(), heights.end());
        int medianHeight = heights[heights.size() / 2];

        // Group characters whose vertical centers are within 0.4*medianHeight of each other
        vector<vector<Rect>> charLines;
        for (const Rect& r : charRects) {
            int centerY = r.y + r.height / 2;
            bool added = false;

            for (auto& line : charLines) {
                // Calculate average center Y of this line
                int lineSum = 0;
                for (const Rect& lr : line) {
                    lineSum += lr.y + lr.height / 2;
                }
                int lineCenterY = lineSum / line.size();

                // Check if this character belongs to this line
                if (abs(centerY - lineCenterY) < 0.4 * medianHeight) {
                    line.push_back(r);
                    added = true;
                    break;
                }
            }

            if (!added) {
                charLines.push_back({ r });
            }
        }

        // Sort each line by x-coordinate
        for (auto& line : charLines) {
            sort(line.begin(), line.end(),
                [](const Rect& a, const Rect& b) { return a.x < b.x; });
        }

        // Take the line with most characters
        auto& bestLine = *max_element(charLines.begin(), charLines.end(),
            [](const vector<Rect>& a, const vector<Rect>& b) {
                return a.size() < b.size();
            });

        charRects = bestLine;
    }

    // Extract character images
    for (const Rect& r : charRects) {
        Mat charImg = plateBinary(r);

        // Add padding and resize to standard size for OCR
        int padding = 4;
        Mat paddedChar(r.height + 2 * padding, r.width + 2 * padding, CV_8UC1, Scalar(0));
        charImg.copyTo(paddedChar(Rect(padding, padding, r.width, r.height)));

        // Resize to standard height while preserving aspect ratio
        int standardHeight = 32;
        int standardWidth = (int)(standardHeight * r.width / (float)r.height);
        standardWidth = max(standardWidth, 10); // Ensure minimum width

        Mat resizedChar;
        resize(paddedChar, resizedChar, Size(standardWidth, standardHeight), 0, 0, INTER_CUBIC);

        characters.push_back(resizedChar);
    }

    // Visualize segmented characters
    if (showSteps && !characters.empty()) {
        Mat charDisplay;
        if (characters.size() <= 10) {
            int totalWidth = 0;
            for (const Mat& c : characters) {
                totalWidth += c.cols + 10; // 10 pixels spacing
            }

            charDisplay = Mat(characters[0].rows + 40, totalWidth, CV_8UC3, Scalar(255, 255, 255));

            int xPos = 5;
            for (size_t i = 0; i < characters.size(); i++) {
                Mat colored;
                cvtColor(characters[i], colored, COLOR_GRAY2BGR);

                Rect roi(xPos, 20, colored.cols, colored.rows);
                colored.copyTo(charDisplay(roi));

                // Add character index
                putText(charDisplay, to_string(i), Point(xPos + 5, 15),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);

                xPos += colored.cols + 10;
            }
        }
        else {
            // For many characters, arrange in 2 rows
            int maxCharsPerRow = (characters.size() + 1) / 2;
            int totalRowWidth = 0;
            for (size_t i = 0; i < (maxCharsPerRow < characters.size() ? maxCharsPerRow : characters.size()); i++) {
                totalRowWidth += characters[i].cols + 10;
            }

            charDisplay = Mat(2 * characters[0].rows + 60, totalRowWidth, CV_8UC3, Scalar(255, 255, 255));

            int row1XPos = 5, row2XPos = 5;
            for (size_t i = 0; i < characters.size(); i++) {
                Mat colored;
                cvtColor(characters[i], colored, COLOR_GRAY2BGR);

                if (i < maxCharsPerRow) {
                    Rect roi(row1XPos, 20, colored.cols, colored.rows);
                    colored.copyTo(charDisplay(roi));
                    putText(charDisplay, to_string(i), Point(row1XPos + 5, 15),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
                    row1XPos += colored.cols + 10;
                }
                else {
                    Rect roi(row2XPos, 20 + colored.rows + 20, colored.cols, colored.rows);
                    colored.copyTo(charDisplay(roi));
                    putText(charDisplay, to_string(i), Point(row2XPos + 5, 15 + colored.rows + 20),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
                    row2XPos += colored.cols + 10;
                }
            }
        }

        showStep("Character Segmentation", charDisplay, "Segmented Characters", 5);
    }

    return characters;
}

// Helper function to create templates for character recognition
map<char, Mat> createCharacterTemplates() {
    map<char, Mat> templates;
    string allChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

    // Create template for each character
    for (char c : allChars) {
        Mat templateImg(40, 30, CV_8UC1, Scalar(255));
        string text(1, c);
        int fontFace = FONT_HERSHEY_SIMPLEX;
        double fontScale = 1.2;
        int thickness = 2;
        int baseline = 0;

        Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
        Point textOrg((templateImg.cols - textSize.width) / 2,
            (templateImg.rows + textSize.height) / 2);

        putText(templateImg, text, textOrg, fontFace, fontScale, Scalar(0), thickness);
        templates[c] = templateImg;
    }

    return templates;
}

// Template matching recognition
string recognizeCharacters(const vector<Mat>& characters, bool showSteps) {
    if (characters.empty()) {
        return "";
    }

    static map<char, Mat> templates = createCharacterTemplates();
    string result;

    // Prepare visualization if needed
    Mat charDisplay;
    if (showSteps) {
        // Calculate total width needed for visualization
        int totalWidth = 0;
        int maxHeight = 0;
        for (const Mat& c : characters) {
            totalWidth += min(100, c.cols + 10); // Limit width per character
            maxHeight = max(maxHeight, c.rows);
        }
        totalWidth = min(totalWidth, 800); // Limit overall width

        // Create display with safe dimensions
        charDisplay = Mat(maxHeight + 60, totalWidth, CV_8UC3, Scalar(255, 255, 255));
    }

    int xPos = 5;

    for (size_t i = 0; i < characters.size(); i++) {
        if (characters[i].empty()) {
            continue; // Skip empty characters
        }

        // Resize character to standard size
        Mat resized;
        Size newSize(30, 40);
        try {
            resize(characters[i], resized, newSize, 0, 0, INTER_CUBIC);
        }
        catch (cv::Exception& e) {
            cout << "Warning: Could not resize character " << i << ": " << e.what() << endl;
            continue;
        }

        // Invert if needed (templates have black text on white background)
        Mat processed = 255 - resized;

        char bestChar = '?';
        double bestScore = -1;

        // Compare with all templates
        for (const auto& pair : templates) {
            Mat resultMat;
            try {
                matchTemplate(processed, pair.second, resultMat, TM_CCOEFF_NORMED);
                double score;
                minMaxLoc(resultMat, nullptr, &score);

                if (score > bestScore) {
                    bestScore = score;
                    bestChar = pair.first;
                }
            }
            catch (cv::Exception& e) {
                cout << "Warning: Template matching failed for character " << i << ": " << e.what() << endl;
            }
        }

        // Only add if match is reasonable
        if (bestScore > 0.5) {
            result += bestChar;
        }
        else {
            result += '?';
        }

        // Visualization - be extra careful with dimensions
        if (showSteps) {
            try {
                // Calculate safe ROI
                int dispWidth = min(processed.cols, 100);
                int dispHeight = min(processed.rows, charDisplay.rows - 20);

                // Make sure we don't go beyond display boundaries
                if (xPos + dispWidth <= charDisplay.cols) {
                    Mat colored;
                    cvtColor(processed, colored, COLOR_GRAY2BGR);

                    // Create a safe ROI
                    Rect roi(xPos, 10,
                        min(dispWidth, charDisplay.cols - xPos),
                        min(dispHeight, charDisplay.rows - 20));

                    // Create a correctly sized Mat for copying
                    if (roi.width > 0 && roi.height > 0 &&
                        roi.x + roi.width <= charDisplay.cols &&
                        roi.y + roi.height <= charDisplay.rows) {

                        Mat displayPortion = colored(Rect(0, 0, roi.width, roi.height));
                        displayPortion.copyTo(charDisplay(roi));

                        // Add character index and recognized value if there's room
                        if (xPos + 10 < charDisplay.cols) {
                            putText(charDisplay, to_string(i), Point(xPos + 5, 9),
                                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 255), 1);
                        }

                        string charText = "-> ";
                        charText += bestChar;
                        charText += " (" + to_string(int(bestScore * 100)) + "%)";

                        if (roi.y + roi.height + 15 < charDisplay.rows) {
                            putText(charDisplay, charText,
                                Point(xPos, min(charDisplay.rows - 5, roi.y + roi.height + 15)),
                                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0), 1);
                        }
                    }

                    xPos += dispWidth + 10;
                }
            }
            catch (cv::Exception& e) {
                cout << "Warning: Visualization failed for character " << i << ": " << e.what() << endl;
                // Continue processing even if visualization fails
            }
        }
    }

    if (showSteps && !charDisplay.empty()) {
        showStep("Character Recognition", charDisplay, "Template Matching Results", 1);
    }

    return result;
}

string postProcessLicensePlate(const string& plateText) {
    if (plateText.empty()) {
        return "";
    }

    // Convert to uppercase
    string result = plateText;
    transform(result.begin(), result.end(), result.begin(), ::toupper);

    // Remove any non-alphanumeric characters
    result.erase(remove_if(result.begin(), result.end(),
        [](unsigned char c) { return !isalnum(c); }),
        result.end());

    // Filter by length (most license plates are between 5-8 characters)
    if (result.length() < 3 || result.length() > 12) {
        return "";
    }

    // Apply common OCR correction patterns
    map<char, char> commonMistakes = {
        {'0', 'O'}, {'1', 'I'}, {'8', 'B'},
        {'5', 'S'}, {'2', 'Z'}, {'6', 'G'}
    };

    // Simple heuristic: in most regions, first 2-3 characters are usually letters
    for (int i = 0; i < min(3, (int)result.length()); i++) {
        if (isdigit(result[i])) {
            // Check for common OCR mistakes
            auto it = commonMistakes.find(result[i]);
            if (it != commonMistakes.end()) {
                result[i] = it->second;
            }
        }
    }

    // Simple heuristic: last 3-4 characters are often digits
    for (int i = max(0, (int)result.length() - 4); i < result.length(); i++) {
        if (isalpha(result[i])) {
            // Check for common OCR mistakes (in reverse)
            for (const auto& pair : commonMistakes) {
                if (pair.second == result[i]) {
                    result[i] = pair.first;
                    break;
                }
            }
        }
    }

    return result;
}

void recognizeLicensePlate() {
    char fname[MAX_PATH];
    while (openFileDlg(fname)) {
        Mat original = imread(fname, IMREAD_COLOR);
        if (original.empty()) {
            cout << "Could not open or find the image!" << endl;
            continue;
        }

        cout << "\nProcessing image: " << fname << endl;

        // 1. Preprocess the image
        Mat preprocessed = preprocessImage(original, true);

        // 2. Detect license plates
        vector<RotatedRect> plates = detectLicensePlates(preprocessed, original, true);

        if (plates.empty()) {
            cout << "No license plates found in the image." << endl;
            waitKey(0);
            continue;
        }

        cout << "Found " << plates.size() << " potential license plates.\n" << endl;

        // Create a result image for all detected plates
        Mat resultDisplay = original.clone();

        // Process each detected plate
        for (size_t i = 0; i < plates.size(); i++) {
            cout << "Processing plate " << i + 1 << " of " << plates.size() << endl;

            // 3. Correct perspective and extract the plate
            Mat plate = adjustPlatePerspective(original, plates[i], true);

            // 4. Segment characters
            vector<Mat> characters = segmentCharacters(plate, true);

            if (characters.empty()) {
                cout << "No characters found in this plate." << endl;
                continue;
            }

            cout << "Found " << characters.size() << " potential characters." << endl;

            // 5. Recognize characters
            string plateText = recognizeCharacters(characters, true);

            cout << "Raw OCR result: " << plateText << endl;

            // 6. Post-process the plate text
            string finalPlate = postProcessLicensePlate(plateText);

            if (finalPlate.empty()) {
                cout << "Could not determine a valid license plate number.\n" << endl;
                continue;
            }

            cout << "Detected License Plate: " << finalPlate << "\n" << endl;

            // Draw the detected plate on the result image
            Point2f rectPoints[4];
            plates[i].points(rectPoints);
            for (int j = 0; j < 4; j++) {
                line(resultDisplay, rectPoints[j], rectPoints[(j + 1) % 4], Scalar(0, 255, 0), 3);
            }

            // Calculate position for text
            Point textPos;
            if (rectPoints[0].y > 30) {
                textPos = Point(rectPoints[0].x, rectPoints[0].y - 10);
            }
            else {
                textPos = Point(rectPoints[2].x, rectPoints[2].y + 30);
            }

            // Display the license plate number with background for better visibility
            Size textSize = getTextSize(finalPlate, FONT_HERSHEY_DUPLEX, 1.2, 3, 0);
            rectangle(resultDisplay,
                Point(textPos.x - 5, textPos.y - textSize.height - 5),
                Point(textPos.x + textSize.width + 5, textPos.y + 5),
                Scalar(0, 0, 0), -1);
            putText(resultDisplay, finalPlate, textPos,
                FONT_HERSHEY_DUPLEX, 1.2, Scalar(0, 255, 255), 3);
        }

        // Display final result
        showStep("Final Result", resultDisplay, "License Plate Recognition", 0, 0);

        waitKey(0);
        destroyAllWindows();
    }
}
