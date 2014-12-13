#pragma once
#ifndef __SERIALIZE_H__
#define __SERIALIZE_H__

#include <opencv2\core\core.hpp>
#include <vector>



void SaveVectorPoint3f(std::string filePath, std::vector<cv::Point3f> &vertices, std::string mode = "w");

std::vector<cv::Point3f> LoadVectorPoint3f(std::string filePath, std::string mode = "r");

void SaveVectorPoint2f(std::string filePath, std::vector<cv::Point2f> &vertices, std::string mode = "w");

std::vector<cv::Point2f> LoadVectorPoint2f(std::string filePath, std::string mode = "r");

void SaveVectorVectorInt(std::string filePath, std::vector<std::vector<int>> &data, std::string mode = "w");

std::vector<std::vector<int>> LoadVectorVectorInt(std::string filePath, std::string mode = "r");

#endif