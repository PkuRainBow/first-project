#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "boost/filesystem.hpp"
#include <boost/lexical_cast.hpp>

class util
{
public:
	static std::vector<std::vector<cv::Point>> stringToContors(std::string ss);										//将字符串-->凸包信息
	static std::string contorsToString(std::vector<std::vector<cv::Point>> &contors);
	static void create_path(std::string path);
};

