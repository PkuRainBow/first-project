#include "indexReplay.h"

/*
* 事件索引回放功能实现
*/



bool indexReplay::saveEventsParamOfFrameToFile(int frame_Num, int indexOfevent, int bias) {
	//cout<<"frameNum "<<frame_Num<<"   index of mask" << indexOfMask << endl;

	std::ofstream outfile(replayFileName, std::ios::app);
	std::vector<std::vector<cv::Point>> contours;
	//change the way to Get contours ...
	outfile << frame_Num << " " << indexOfevent << " " << bias << std::endl;
	//outfile << indexOfMask << " " << util::loadObjectCube(bias, contours) << std::endl;
	outfile.close();
	return true;
}

cv::Mat indexReplay::loadEventsParamToRebuildMask(int frame_Num, int width, int height, std::vector<int>& lookupTable){
	lookupTable.clear();

	cv::Mat resultMask(height, width, CV_8UC1, cv::Scalar::all(255));
	std::ifstream infile(replayFileName);
	std::string line;
	while (std::getline(infile, line)) 
	{
		if (!line.size())
			continue;
		std::stringstream ss(line);
		int new_frame_Num, indexOfevent, bias;
		ss >> new_frame_Num >> indexOfevent >> bias;
		if (new_frame_Num < frame_Num) {
			continue;
		} else if (new_frame_Num == frame_Num) {
			std::vector<std::vector<cv::Point>> contours;
			loadObjectCube(bias, contours);
			cv::Mat mask(height, width, CV_8UC1, cv::Scalar::all(255));
			cv::drawContours(mask, contours, -1, cv::Scalar(0), -1);
			lookupTable.push_back(indexOfevent);
			restoreMaskOfFrame(resultMask, mask, lookupTable.size() - 1);
		} else {
			break;
		}
	}
	infile.close();
	return resultMask;
}


bool indexReplay::restoreMaskOfFrame(cv::Mat& FrameMask, cv::Mat& eventMask, int index){
	/*imshow("eventMask", eventMask);
	waitKey(0);*/
	int nc = FrameMask.cols;
	int nl = FrameMask.rows;
	if (eventMask.cols != nc || eventMask.rows != nl){
		return false;
	}
	if (FrameMask.isContinuous() && eventMask.isContinuous())
	{
		nc = nc * nl;
		nl = 1;
	}
	for (int j = 0; j < nl; ++j)
	{
		uchar* c_data = eventMask.ptr<uchar>(j);
		uchar* m_data = FrameMask.ptr<uchar>(j);
		for (int i = 0; i < nc; ++i)
		{
			if (*c_data++ == 0) {
				//*m_data++ = index;
				*m_data++ = index;
			} else {
				m_data++;
			}
		}
	}
	return true;
}

std::string indexReplay::loadObjectCube(int bias, std::vector<std::vector<cv::Point>>& contours){ 
	std::ifstream file(replayConfigName);
	std::string temp;
	for(int i=0; i<bias; i++) 
	{
		getline(file, temp, '#');
	}
	contours = util::stringToContors(temp);
	return temp;
}