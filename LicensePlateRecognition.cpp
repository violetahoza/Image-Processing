#include "stdafx.h"
#include "common.h"
#include "LicensePlateRecognition.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <string>
#include <regex>
#include <direct.h>

using namespace cv;
using namespace std;

bool SHOW_STEPS = true;
string TESSDATA_PATH = "C:/dev/tessdata/";

void showStep(const string& windowName, const Mat& img) {
    if (SHOW_STEPS) {
        Mat display = img.clone();
        imshow(windowName, display);
        waitKey(0);
    }
}

Mat preprocessImage(const Mat& src) {
    double tTotal = (double)getTickCount();

    showStep("Original Image", src);
    Mat gray, filtered, edges, dilated;

    //double t = (double)getTickCount();
    cvtColor(src, gray, COLOR_BGR2GRAY);
    /*t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: Grayscale conversion took " << t << " seconds" << endl;*/
    showStep("Grayscale Image", gray);

    //t = (double)getTickCount();
    bilateralFilter(gray, filtered, 11, 17, 17);
   /* t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: Bilateral filtering took " << t << " seconds" << endl;*/
    showStep("Filtered Image", filtered);

    //t = (double)getTickCount();
    Canny(filtered, edges, 30, 200);
    /*t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: Canny edge detection took " << t << " seconds" << endl;*/
    showStep("Canny edges", edges);

    //t = (double)getTickCount();
    dilate(edges, dilated, Mat(), Point(-1, -1), 1);
    /*t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: Dilation took " << t << " seconds" << endl;*/
    showStep("Processed Image", dilated);

    tTotal = ((double)getTickCount() - tTotal) / getTickFrequency();
    cout << "TIMING: Total preprocessing took " << tTotal << " seconds" << endl;

    return dilated;
}

Mat preprocessPlate(const Mat& plate) {
    double tTotal = (double)getTickCount();

    Mat gray;
    //double t = (double)getTickCount();
    if (plate.channels() == 3) {
        cvtColor(plate, gray, COLOR_BGR2GRAY);
       /* t = ((double)getTickCount() - t) / getTickFrequency();
        cout << "TIMING: Plate grayscale conversion took " << t << " seconds" << endl;*/
        showStep("Grayscale plate", gray);
    }
    else {
        gray = plate.clone();
    }

    //t = (double)getTickCount();
    equalizeHist(gray, gray);
   /* t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: Histogram equalization took " << t << " seconds" << endl;*/

    //t = (double)getTickCount();
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    Mat enhanced;
    clahe->apply(gray, enhanced);
    /*t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: CLAHE enhancement took " << t << " seconds" << endl;*/
    showStep("Enhanced plate", enhanced);

    //t = (double)getTickCount();
    Mat blurred;
    GaussianBlur(enhanced, blurred, Size(3, 3), 0);
    /*t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: Gaussian blur took " << t << " seconds" << endl;*/
    showStep("Blurred plate", blurred);

    //t = (double)getTickCount();
    Mat binary;
    adaptiveThreshold(blurred, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
        THRESH_BINARY_INV, 31, 10);
    /*t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: Adaptive thresholding took " << t << " seconds" << endl;*/

    //t = (double)getTickCount();
    Mat morph;
    Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
    morphologyEx(binary, morph, MORPH_CLOSE, kernel);
   /* t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: Morphological closing took " << t << " seconds" << endl;*/

    showStep("Plate Binary", morph);

    tTotal = ((double)getTickCount() - tTotal) / getTickFrequency();
    cout << "TIMING: Total plate preprocessing took " << tTotal << " seconds" << endl;

    return morph;
}

Mat extractPlate(const Mat& image, const RotatedRect& rect) {
    double tTotal = (double)getTickCount();

    float angle = rect.angle;
    Size rect_size = rect.size;

    if (angle < -45.0) {
        angle += 90.0;
        swap(rect_size.width, rect_size.height);
    }
    else if (angle > 45.0) {
        angle -= 90.0;
        swap(rect_size.width, rect_size.height);
    }

    rect_size.width *= 1.1;
    rect_size.height *= 1.1;

    //double t = (double)getTickCount();
    Mat M = getRotationMatrix2D(rect.center, angle, 1.0);
    Mat rotated;
    warpAffine(image, rotated, M, image.size(), INTER_CUBIC);
    /*t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: Rotation correction took " << t << " seconds" << endl;*/

    //t = (double)getTickCount();
    Mat plate;
    getRectSubPix(rotated, rect_size, rect.center, plate);
    /*t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: Plate extraction took " << t << " seconds" << endl;*/

    int borderX = plate.cols * 0.05;
    int borderY = plate.rows * 0.1;
    Rect roi(borderX, borderY, plate.cols - 2 * borderX, plate.rows - 2 * borderY);

    //t = (double)getTickCount();
    if (roi.width > 0 && roi.height > 0) {
        plate = plate(roi);
    }
    /*t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: Border removal took " << t << " seconds" << endl;*/

    showStep("Extracted Plate", plate);

    tTotal = ((double)getTickCount() - tTotal) / getTickFrequency();
    cout << "TIMING: Total plate extraction took " << tTotal << " seconds" << endl;

    return plate;
}

bool isPlateCandidate(const vector<Point>& contour, const Size& imgSize) {
    double t = (double)getTickCount();

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

    t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: Contour evaluation took " << t << " seconds" << endl;

    return (extent > 0.5 && solidity > 0.6);
}

vector<RotatedRect> detectPlates(const Mat& preprocessed, const Mat& original) {
    double tTotal = (double)getTickCount();

    vector<vector<Point>> contours;
    //double t = (double)getTickCount();
    findContours(preprocessed.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    /*t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: Finding contours took " << t << " seconds" << endl;*/

    //t = (double)getTickCount();
    sort(contours.begin(), contours.end(), [](const vector<Point>& c1, const vector<Point>& c2) {
        return contourArea(c1) > contourArea(c2);
        });
    /*t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: Sorting contours took " << t << " seconds" << endl;*/

    Mat debugImg = original.clone();
    vector<RotatedRect> plates;

    //t = (double)getTickCount();
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
    /*t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: Filtering contours took " << t << " seconds" << endl;*/

    showStep("Detected Plates", debugImg);

    tTotal = ((double)getTickCount() - tTotal) / getTickFrequency();
    cout << "TIMING: Total plate detection took " << tTotal << " seconds" << endl;

    return plates;
}

bool isRomanianPlate(const string& text) {
    double t = (double)getTickCount();

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

    bool result = false;
    for (const auto& code : countyCodes) {
        if (cleaned.substr(0, code.length()) == code) {
            result = regex_match(cleaned, regex("^[A-Z]{1,2}[0-9]{2,6}[A-Z]{0,3}$"));
            break;
        }
    }

    t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: Plate format validation took " << t << " seconds" << endl;

    return result;
}

string fixOcrErrors(const string& text) {
    double t = (double)getTickCount();

    string cleaned;
    for (char c : text) {
        if (isalnum(c)) cleaned += toupper(c);
    }

    smatch match;
    regex platePattern("([A-Z]{1,2}[0-9]{2,6}[A-Z]{0,3})");
    string result = "";
    if (regex_search(cleaned, match, platePattern)) {
        result = match[1];
    }

    t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: OCR text cleanup took " << t << " seconds" << endl;

    return result;
}

string recognizePlate(const Mat& plate) {
    double tTotal = (double)getTickCount();

    tesseract::TessBaseAPI tess;
    double t = (double)getTickCount();
    string envVar = "TESSDATA_PREFIX=" + TESSDATA_PATH;
    _putenv(envVar.c_str());

    if (tess.Init(NULL, "eng", tesseract::OEM_LSTM_ONLY)) {
        cerr << "Could not initialize tesseract." << endl;
        return "";
    }

    tess.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
    tess.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ");
    /*t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: Tesseract initialization took " << t << " seconds" << endl;*/

    Mat resizedPlate = plate.clone();
    const int OPTIMAL_PLATE_WIDTH = 200;
    //t = (double)getTickCount();
    if (plate.cols < OPTIMAL_PLATE_WIDTH) {
        double scale = static_cast<double>(OPTIMAL_PLATE_WIDTH) / plate.cols;
        resize(plate, resizedPlate, Size(), scale, scale, INTER_CUBIC);
        showStep("Resized Plate", resizedPlate);
    }
    /*t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: Plate resizing took " << t << " seconds" << endl;*/

    //t = (double)getTickCount();
    Mat processed = preprocessPlate(resizedPlate);
    /*t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: OCR plate preprocessing took " << t << " seconds" << endl;*/

    //t = (double)getTickCount();
    Mat upscaled, inverted;
    resize(processed, upscaled, Size(), 2.0, 2.0, INTER_CUBIC);
    bitwise_not(processed, inverted);
    /*t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: Creating OCR variants took " << t << " seconds" << endl;*/

    vector<pair<string, float>> results;
    vector<Mat> versions = { processed, upscaled, inverted };

    t = (double)getTickCount();
    for (size_t i = 0; i < versions.size(); ++i) {
        double tOcr = (double)getTickCount();
        tess.SetImage(versions[i].data, versions[i].cols, versions[i].rows, 1, versions[i].step);
        string text = tess.GetUTF8Text();
        float conf = tess.MeanTextConf() / 100.0f;
        results.push_back({ text, conf });
        /*tOcr = ((double)getTickCount() - tOcr) / getTickFrequency();
        cout << "TIMING: OCR variant " << i << " took " << tOcr << " seconds" << endl;*/
    }
    /*t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "TIMING: All OCR processing took " << t << " seconds" << endl;*/

    string bestText;
    float bestScore = 0;

    //t = (double)getTickCount();
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
    //t = ((double)getTickCount() - t) / getTickFrequency();
    //cout << "TIMING: Result selection took " << t << " seconds" << endl;

    cout << "\nRecognized plate: " << bestText << ", confidence: " << bestScore << endl << endl;

    tTotal = ((double)getTickCount() - tTotal) / getTickFrequency();
    cout << "TIMING: Total OCR process took " << tTotal << " seconds" << endl;

    return bestText;
}

void saveDetectionResult(const string& originalImagePath, const Mat& processedImage,
    const RotatedRect& plateRect, const string& plateNumber) {

    if (plateNumber.empty()) return;

    string baseOutputPath = "C:\\Users\\hozas\\Desktop\\facultate\\Sem2\\IP\\ip-labs\\Images\\license-plates\\results";
    string imagesPath = baseOutputPath + "\\images";
    string annotationsPath = baseOutputPath + "\\annotations";

    _mkdir(baseOutputPath.c_str());
    _mkdir(imagesPath.c_str());
    _mkdir(annotationsPath.c_str());

    string baseFilename = originalImagePath;
    size_t lastSlash = baseFilename.find_last_of("/\\");
    if (lastSlash != string::npos) baseFilename = baseFilename.substr(lastSlash + 1);
    size_t lastDot = baseFilename.find_last_of('.');
    if (lastDot != string::npos) baseFilename = baseFilename.substr(0, lastDot);

    string imageFileName = baseFilename + "_" + plateNumber + ".jpg";
    string xmlFileName = baseFilename + "_" + plateNumber + ".xml";
    string fullImagePath = imagesPath + "\\" + imageFileName;
    string fullXmlPath = annotationsPath + "\\" + xmlFileName;

    imwrite(fullImagePath, processedImage);

    Rect boundingBox = plateRect.boundingRect();

    boundingBox = boundingBox & Rect(0, 0, processedImage.cols, processedImage.rows);

    ofstream xmlFile(fullXmlPath);
    if (xmlFile.is_open()) {
        xmlFile << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << endl;
        xmlFile << "<annotation>" << endl;
        xmlFile << "  <filename>" << imageFileName << "</filename>" << endl;
        xmlFile << "  <size>" << endl;
        xmlFile << "    <width>" << processedImage.cols << "</width>" << endl;
        xmlFile << "    <height>" << processedImage.rows << "</height>" << endl;
        xmlFile << "    <depth>" << processedImage.channels() << "</depth>" << endl;
        xmlFile << "  </size>" << endl;
        xmlFile << "  <object>" << endl;
        xmlFile << "    <name>license-plate</name>" << endl;
        xmlFile << "    <bndbox>" << endl;
        xmlFile << "      <xmin>" << boundingBox.x << "</xmin>" << endl;
        xmlFile << "      <ymin>" << boundingBox.y << "</ymin>" << endl;
        xmlFile << "      <xmax>" << boundingBox.x + boundingBox.width << "</xmax>" << endl;
        xmlFile << "      <ymax>" << boundingBox.y + boundingBox.height << "</ymax>" << endl;
        xmlFile << "    </bndbox>" << endl;
        xmlFile << "    <attributes>" << endl;
        xmlFile << "      <attribute>" << endl;
        xmlFile << "        <name>text</name>" << endl;
        xmlFile << "        <value>" << plateNumber << "</value>" << endl;
        xmlFile << "      </attribute>" << endl;
        xmlFile << "    </attributes>" << endl;
        xmlFile << "  </object>" << endl;
        xmlFile << "</annotation>" << endl;

        xmlFile.close();
        cout << "Saved result to " << fullXmlPath << endl;
    }
}

void recognizeLicensePlate() {
    char fname[MAX_PATH];

    while (openFileDlg(fname)) {
        double tTotal = (double)getTickCount();

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
            destroyAllWindows();
            continue;
        }

        for (size_t i = 0; i < plates.size(); ++i) {
            double tPlate = (double)getTickCount();

            Mat plateImg = extractPlate(image, plates[i]);
            if (plateImg.empty()) continue;

            string text = recognizePlate(plateImg);
            if (text.empty()) continue;

            double t = (double)getTickCount();
            Mat result = image.clone();
            Point2f pts[4];
            plates[i].points(pts);
            for (int j = 0; j < 4; j++) {
                line(result, pts[j], pts[(j + 1) % 4], Scalar(0, 255, 0), 2);
            }

            putText(result, text, pts[1], FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
            t = ((double)getTickCount() - t) / getTickFrequency();
            cout << "TIMING: Result visualization took " << t << " seconds" << endl;

            //saveDetectionResult(fname, result, plates[i], text);
            showStep("Result Plate " + to_string(i + 1), result);

            tPlate = ((double)getTickCount() - tPlate) / getTickFrequency();
            cout << "TIMING: Processing plate " << (i + 1) << " took " << tPlate << " seconds" << endl;
        }

        destroyAllWindows();

        tTotal = ((double)getTickCount() - tTotal) / getTickFrequency();
        cout << "TIMING: Total image processing took " << tTotal << " seconds" << endl;
    }
}
