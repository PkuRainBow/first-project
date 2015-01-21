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
	//�ļ���ȡ��ʽ��
	//�ļ�����frame_Num.txt
	//���ݣ�ÿ��maskռһ�У����¼���  contorsToString(contors)��

	//�������Ƶÿһ֡�е�͹�����浽�ļ�
	//�����������Ⱥ�˳�򣬺������¼��п��ܸ����������¼�
	//frame_Num  �����Ƶ���
	//mask  ÿ֡�����Ƶ�ɶ���¼��ϳɣ�mask��������һ���¼���
	//indexOfMask   �¼����

	bool saveEventsParamOfFrameToFile(int frame_Num, int indexOfevent, int bias);
	//���ļ��ж�ȡcontors��Ϣ���ϳɽ����Ƶ��frame_Num֡��Ӧ���¼�Mask��Ϣ
	//ISSUE: �ļ��ж������¼���źܿ��ܴ���255����δ���
	//lookupTable: ���ɵ�mask��ÿ��������һ��0-8����ֵ��ʹ����ֵ�±���lookupTable��ȡ��ʵ�ʵ��¼���ţ�1-8��Ч��

	cv::Mat loadEventsParamToRebuildMask(int frame_Num, int width, int height, std::vector<int>& lookupTable);
	bool restoreMaskOfFrame(cv::Mat& FrameMask, cv::Mat& eventMask, int index);
	std::string loadObjectCube(int bias, std::vector<std::vector<cv::Point>>& contours);									//��ȡ���ش�ŵ��˶�����
};
