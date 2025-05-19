#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <Windows.h>

using namespace cv;
using namespace std;

const string PREDICTIONS_DIR = "C:\\Users\\hozas\\Desktop\\facultate\\Sem2\\IP\\ip-labs\\Images\\license-plates\\results\\annotations";
const string GROUND_TRUTH_DIR = "C:\\Users\\hozas\\Desktop\\facultate\\Sem2\\IP\\ip-labs\\Images\\license-plates\\annotations";

const float IOU_THRESHOLD = 0.5;

struct BoundingBox {
    Rect bbox;
    string filename;
};

string readFileContent(const string& path) {
    ifstream file(path);
    if (!file.is_open()) {
        cout << "Failed to open: " << path << endl;
        return "";
    }

    string content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();
    return content;
}

string getFilenameFromXml(const string& xmlContent) {
    size_t filenameStart = xmlContent.find("<filename>");
    size_t filenameEnd = xmlContent.find("</filename>");

    if (filenameStart != string::npos && filenameEnd != string::npos) {
        string filename = xmlContent.substr(filenameStart + 10, filenameEnd - filenameStart - 10);

        size_t lastDot = filename.find_last_of('.');
        if (lastDot != string::npos) {
            filename = filename.substr(0, lastDot);
        }

        size_t underscorePos = filename.find('_');
        if (underscorePos != string::npos) {
            filename = filename.substr(0, underscorePos);
        }

        return filename;
    }
    return "";
}

vector<BoundingBox> loadBoxesFromXml(const string& xmlPath) {
    vector<BoundingBox> boxes;

    string content = readFileContent(xmlPath);
    if (content.empty()) {
        return boxes;
    }

    string filename = getFilenameFromXml(content);
    if (filename.empty()) {
        filename = xmlPath.substr(xmlPath.find_last_of("/\\") + 1);
        filename = filename.substr(0, filename.find_last_of('.'));

        size_t underscorePos = filename.find('_');
        if (underscorePos != string::npos) {
            filename = filename.substr(0, underscorePos);
        }
    }

    size_t pos = 0;
    while ((pos = content.find("<object>", pos)) != string::npos) {
        size_t objEnd = content.find("</object>", pos);
        if (objEnd == string::npos) break;

        string objContent = content.substr(pos, objEnd - pos);
        pos = objEnd + 1;

        size_t bboxStart = objContent.find("<bndbox>");
        size_t bboxEnd = objContent.find("</bndbox>");

        if (bboxStart != string::npos && bboxEnd != string::npos) {
            string bboxContent = objContent.substr(bboxStart, bboxEnd - bboxStart);

            int xmin = 0, ymin = 0, xmax = 0, ymax = 0;

            auto extractValue = [&bboxContent](const string& tag) -> int {
                size_t startPos = bboxContent.find("<" + tag + ">");
                size_t endPos = bboxContent.find("</" + tag + ">");
                if (startPos != string::npos && endPos != string::npos) {
                    string valueStr = bboxContent.substr(startPos + tag.length() + 2,
                        endPos - startPos - tag.length() - 2);
                    try { return stoi(valueStr); }
                    catch (...) { return 0; }
                }
                return 0;
                };

            xmin = extractValue("xmin");
            ymin = extractValue("ymin");
            xmax = extractValue("xmax");
            ymax = extractValue("ymax");

            if (xmax > xmin && ymax > ymin) {
                BoundingBox box;
                box.bbox = Rect(xmin, ymin, xmax - xmin, ymax - ymin);
                box.filename = filename;
                boxes.push_back(box);
            }
        }
    }

    return boxes;
}

float calculateIoU(const Rect& a, const Rect& b) {
    int x1 = max(a.x, b.x);
    int y1 = max(a.y, b.y);
    int x2 = min(a.x + a.width, b.x + b.width);
    int y2 = min(a.y + a.height, b.y + b.height);

    if (x2 < x1 || y2 < y1) return 0.0f;

    float intersectionArea = (x2 - x1) * (y2 - y1);
    float aArea = a.width * a.height;
    float bArea = b.width * b.height;
    float unionArea = aArea + bArea - intersectionArea;

    return intersectionArea / unionArea;
}

void evaluatePlateDetection() {
    map<string, vector<BoundingBox>> groundTruthBoxes;
    map<string, vector<BoundingBox>> predictionBoxes;

    cout << "Starting license plate detection evaluation..." << endl;

    cout << "Loading ground truth files..." << endl;
    WIN32_FIND_DATA findData;
    HANDLE hFind = FindFirstFile((GROUND_TRUTH_DIR + "\\*.xml").c_str(), &findData);

    if (hFind == INVALID_HANDLE_VALUE) {
        cout << "No ground truth files found in " << GROUND_TRUTH_DIR << endl;
        return;
    }

    int gtFileCount = 0;
    int gtBoxCount = 0;

    do {
        string filename = findData.cFileName;
        string path = GROUND_TRUTH_DIR + "\\" + filename;
        vector<BoundingBox> boxes = loadBoxesFromXml(path);

        if (!boxes.empty()) {
            for (auto& box : boxes) {
                groundTruthBoxes[box.filename].push_back(box);
                gtBoxCount++;
            }
            gtFileCount++;
        }
    } while (FindNextFile(hFind, &findData));

    FindClose(hFind);
    cout << "Loaded " << gtFileCount << " ground truth files with " << gtBoxCount << " boxes." << endl;

    hFind = FindFirstFile((PREDICTIONS_DIR + "\\*.xml").c_str(), &findData);

    if (hFind == INVALID_HANDLE_VALUE) {
        cout << "No prediction files found in " << PREDICTIONS_DIR << endl;
        return;
    }

    int predFileCount = 0;
    int predBoxCount = 0;

    do {
        string filename = findData.cFileName;
        string path = PREDICTIONS_DIR + "\\" + filename;
        vector<BoundingBox> boxes = loadBoxesFromXml(path);

        if (!boxes.empty()) {
            for (auto& box : boxes) {
                predictionBoxes[box.filename].push_back(box);
                predBoxCount++;
            }
            predFileCount++;
        }
    } while (FindNextFile(hFind, &findData));

    FindClose(hFind);
    cout << "Loaded " << predFileCount << " prediction files with " << predBoxCount << " boxes." << endl;

    int totalGroundTruth = 0;
    int totalPredictions = 0;
    int correctDetections = 0;

    for (auto& gt_pair : groundTruthBoxes) {
        string baseName = gt_pair.first;
        vector<BoundingBox>& gtBoxes = gt_pair.second;
        totalGroundTruth += gtBoxes.size();

        if (predictionBoxes.find(baseName) != predictionBoxes.end()) {
            vector<BoundingBox>& predBoxes = predictionBoxes[baseName];
            totalPredictions += predBoxes.size();

            vector<bool> predMatched(predBoxes.size(), false);

            for (const auto& gtBox : gtBoxes) {
                float bestIoU = 0.0f;
                int bestMatchIdx = -1;

                for (size_t j = 0; j < predBoxes.size(); j++) {
                    if (predMatched[j]) continue;

                    float iou = calculateIoU(gtBox.bbox, predBoxes[j].bbox);
                    if (iou > bestIoU) {
                        bestIoU = iou;
                        bestMatchIdx = j;
                    }
                }

                if (bestIoU > IOU_THRESHOLD && bestMatchIdx >= 0) {
                    correctDetections++;
                    predMatched[bestMatchIdx] = true;
                }
            }
        }
    }

    for (auto& pred_pair : predictionBoxes) {
        string baseName = pred_pair.first;
        if (groundTruthBoxes.find(baseName) == groundTruthBoxes.end()) {
            totalPredictions += pred_pair.second.size();
        }
    }

    float precision = totalPredictions > 0 ? (float)correctDetections / totalPredictions : 0.0f;
    float recall = totalGroundTruth > 0 ? (float)correctDetections / totalGroundTruth : 0.0f;

    cout << "\n===== LICENSE PLATE DETECTION EVALUATION =====\n";
    cout << "IoU Threshold: " << IOU_THRESHOLD << endl;
    cout << "Total ground truth plates: " << totalGroundTruth << endl;
    cout << "Total predicted plates: " << totalPredictions << endl;
    cout << "Correctly detected plates: " << correctDetections << endl;
    cout << "Precision: " << (precision * 100) << "%" << endl;
    cout << "Recall: " << (recall * 100) << "%" << endl;
}
