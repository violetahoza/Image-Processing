#include "stdafx.h"
#include "common.h"
#include "LicensePlateRecognition.h"
#include <iostream>
#include <algorithm>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <string>
#include <regex>

using namespace cv;
using namespace std;

bool SHOW_STEPS = true;
string TESSDATA_PATH = "C:/dev/tessdata/";

void showStep(const string& windowName, const Mat& img) {
    if (SHOW_STEPS) {
        Mat display = img.clone();
        if (img.cols > 1200 || img.rows > 800) {
            double scale = min(1200.0 / img.cols, 800.0 / img.rows);
            resize(img, display, Size(), scale, scale, INTER_AREA);
        }
        imshow(windowName, display);
        waitKey(0);
    }
}

Mat preprocessImage(const Mat& src) {
    showStep("Original Image", src);
    Mat gray, filtered, edges, dilated;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    bilateralFilter(gray, filtered, 11, 17, 17);
    Canny(filtered, edges, 30, 200);
    dilate(edges, dilated, Mat(), Point(-1, -1), 1);
    showStep("Processed Image", dilated);
    return dilated;
}

Mat preprocessPlate(const Mat& plate) {
    Mat gray;
    if (plate.channels() == 3) {
        cvtColor(plate, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = plate.clone();
    }

    equalizeHist(gray, gray); 
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    Mat enhanced;
    clahe->apply(gray, enhanced);

    Mat blurred;
    GaussianBlur(enhanced, blurred, Size(3, 3), 0);

    Mat binary;
    adaptiveThreshold(blurred, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
        THRESH_BINARY_INV, 31, 10);

    Mat morph;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
    morphologyEx(binary, morph, MORPH_CLOSE, kernel);

    showStep("Plate Binary", morph);
    return morph;
}

Mat extractPlate(const Mat& image, const RotatedRect& rect) {
    float angle = rect.angle;
    Size rect_size = rect.size;

    if (rect_size.width < rect_size.height) {
        swap(rect_size.width, rect_size.height);
        angle += 90.0;
    }

    rect_size.width *= 1.1;
    rect_size.height *= 1.1;

    //if (angle < -45) angle += 90.0;

    Mat M = getRotationMatrix2D(rect.center, angle, 1.0);
    Mat rotated;
    warpAffine(image, rotated, M, image.size(), INTER_CUBIC);

    Mat cropped;
    getRectSubPix(rotated, rect_size, rect.center, cropped);

    int borderX = cropped.cols * 0.05;
    int borderY = cropped.rows * 0.1;
    Rect roi(borderX, borderY, cropped.cols - 2 * borderX, cropped.rows - 2 * borderY);
    if (roi.x >= 0 && roi.y >= 0 && roi.width > 0 && roi.height > 0 &&
        roi.x + roi.width <= cropped.cols && roi.y + roi.height <= cropped.rows) {
        cropped = cropped(roi);
    }

    showStep("Extracted Plate", cropped);
    return cropped;
}

bool isPlateCandidate(const vector<Point>& contour, const Size& imgSize) {
    double area = contourArea(contour);
    Rect rect = boundingRect(contour);
    float width = rect.width;
    float height = rect.height;
    float aspectRatio = width / height;
    float imgArea = imgSize.width * imgSize.height;

    if (area < imgArea * 0.002 || area > imgArea * 0.1) return false;
    if (aspectRatio < 2.0 || aspectRatio > 6.0) return false;
    if (width < imgSize.width * 0.08) return false;

    double rectArea = rect.width * rect.height;
    double extent = area / rectArea;

    vector<Point> hull;
    convexHull(contour, hull);
    double solidity = area / contourArea(hull);

    return (extent > 0.5 && solidity > 0.6);
}

vector<RotatedRect> detectPlates(const Mat& preprocessed, const Mat& original) {
    vector<vector<Point>> contours;
    findContours(preprocessed.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    sort(contours.begin(), contours.end(), [](const vector<Point>& c1, const vector<Point>& c2) {
        return contourArea(c1) > contourArea(c2);
        });

    Mat debugImg = original.clone();
    vector<RotatedRect> plates;

    for (size_t i = 0; i < min(contours.size(), size_t(15)); i++) {
        if (isPlateCandidate(contours[i], original.size())) {
            RotatedRect rect = minAreaRect(contours[i]);
            plates.push_back(rect);

            Point2f pts[4];
            rect.points(pts);
            for (int j = 0; j < 4; j++) {
                line(debugImg, pts[j], pts[(j + 1) % 4], Scalar(0, 255, 0), 2);
            }
        }
    }

    showStep("Detected Plates", debugImg);
    return plates;
}

bool isRomanianPlate(const string& text) {
    string cleaned;
    for (char c : text) {
        if (isalnum(c)) cleaned += toupper(c);
    }

    vector<string> countyCodes = {
        "B", "AB", "AR", "AG", "BC", "BH", "BN", "BT", "BV", "BR",
        "BZ", "CS", "CL", "CJ", "CT", "CV", "DB", "DJ", "GL", "GR",
        "GJ", "HR", "HD", "IL", "IS", "IF", "MM", "MH", "MS", "NT",
        "OT", "PH", "SM", "SJ", "SB", "SV", "TR", "TM", "TL", "VS",
        "VL", "VN"
    };

    for (const auto& code : countyCodes) {
        if (cleaned.substr(0, code.length()) == code) {
            return regex_match(cleaned, regex("^[A-Z]{1,2}[0-9]{1,3}[A-Z0-9]{2,3}$"));
        }
    }
    return false;
}

string fixOcrErrors(const string& text) {
    string cleaned;
    for (char c : text) {
        if (isalnum(c)) cleaned += toupper(c);
    }

    smatch match;
    regex platePattern("([A-Z]{1,2}[0-9]{1,3}[A-Z0-9]{2,3})");
    if (regex_search(cleaned, match, platePattern)) {
        return match[1];
    }

    return "";
}

string recognizePlate(const Mat& plate) {
    tesseract::TessBaseAPI tess;
    string envVar = "TESSDATA_PREFIX=" + TESSDATA_PATH;
    _putenv(envVar.c_str());

    if (tess.Init(NULL, "eng", tesseract::OEM_LSTM_ONLY)) {
        cerr << "Could not initialize tesseract." << endl;
        return "";
    }

    tess.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
    tess.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ");

    Mat processed = preprocessPlate(plate);
    Mat upscaled, inverted;
    resize(processed, upscaled, Size(), 2.0, 2.0, INTER_CUBIC);
    bitwise_not(processed, inverted);

    vector<pair<string, float>> results;
    vector<Mat> versions = { processed, upscaled, inverted };

    for (size_t i = 0; i < versions.size(); ++i) {
        tess.SetImage(versions[i].data, versions[i].cols, versions[i].rows, 1, versions[i].step);
        string text = tess.GetUTF8Text();
        float conf = tess.MeanTextConf() / 100.0f;
        results.push_back({ text, conf });
    }

    string bestText;
    float bestScore = 0;

    for (size_t i = 0; i < results.size(); ++i) {
        string rawText = results[i].first;
        float conf = results[i].second;

        string cleaned = fixOcrErrors(rawText);
        float score = conf;
        if (isRomanianPlate(cleaned)) score += 0.5f;

        if (score > bestScore) {
            bestScore = score;
            bestText = cleaned;
        }
    }

    cout << "Recognized plate: " << bestText << ", confidence: " << bestScore << endl;
    return bestText;
}

void recognizeLicensePlate() {
    char fname[MAX_PATH];

    while (openFileDlg(fname)) {
        Mat image = imread(fname);
        if (image.empty()) {
            cout << "Could not open image." << endl;
            continue;
        }

        cout << "\nProcessing image: " << fname << endl;
        Mat preprocessed = preprocessImage(image);
        vector<RotatedRect> plates = detectPlates(preprocessed, image);

        if (plates.empty()) {
            cout << "No license plates found." << endl;
            continue;
        }

        for (size_t i = 0; i < plates.size(); ++i) {
            Mat plateImg = extractPlate(image, plates[i]);
            if (plateImg.empty()) continue;

            string text = recognizePlate(plateImg);
            if (text.empty()) continue;

            Mat result = image.clone();
            Point2f pts[4];
            plates[i].points(pts);
            for (int j = 0; j < 4; j++) {
                line(result, pts[j], pts[(j + 1) % 4], Scalar(0, 255, 0), 2);
            }

            putText(result, text, pts[1], FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
            showStep("Result Plate " + to_string(i + 1), result);
        }

        destroyAllWindows();
    }
}
