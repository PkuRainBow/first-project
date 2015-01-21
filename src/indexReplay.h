#pragma once
#include <opencv2\opencv.hpp>
#include "boost/filesystem.hpp"
#include <iostream>
#include <string>
#include "util.h"

struct replayParams{
	std::string replayFileName;
	std::string replayConfigName;
	int video_width;
	int video_height;
	replayParams(std::string _replayFileName, std::string _replayConfigName, int _video_width, int _video_height):replayFileName(_replayFileName), replayConfigName(_replayConfigName), video_width(_video_width), video_height(_video_height){}
	~replayParams(){}
};

class indexReplay
{
	std::string replayFileName, replayConfigName;
public:
	indexReplay(std::string _replayFileName, std::string _replayConfigName):replayFileName(_replayFileName), replayConfigName(_replayConfigName){};
	//文件存取格式：
	//文件名：frame_Num.txt
	//内容：每个mask占一行：“事件号  contorsToString(contors)”

	//将结果视频每一帧中的凸包保存到文件
	//函数调用有先后顺序，后来的事件有可能覆盖先来的事件
	//frame_Num  结果视频序号
	//mask  每帧结果视频由多个事件合成，mask来自其中一个事件。
	//indexOfMask   事件序号

	bool saveEventsParamOfFrameToFile(int frame_Num, int indexOfevent, int bias);
	//从文件中读取contors信息，合成结果视频中frame_Num帧对应的事件Mask信息
	//ISSUE: 文件中读出的事件序号很可能大于255，如何处理？
	//lookupTable: 生成的mask中每个像素有一个0-8的数值，使用数值下标在lookupTable中取出实际的事件序号（1-8有效）

	cv::Mat loadEventsParamToRebuildMask(int frame_Num, int width, int height, std::vector<int>& lookupTable);
	bool restoreMaskOfFrame(cv::Mat& FrameMask, cv::Mat& eventMask, int index);
	std::string loadObjectCube(int bias, std::vector<std::vector<cv::Point>>& contours);									//读取本地存放的运动序列
};
