#pragma once
#include <iostream>
#include <string>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

class UserVideoAbstraction{	
private:
	VideoAbstraction *userVB;
public:
	__declspec(dllexport) UserVideoAbstraction(const char* inputpath, const char* out_path, const char* log_path, const char* config_path,const char* index_path,const char* videoname, const char* midname, int scale);
	__declspec(dllexport) UserVideoAbstraction();
	__declspec(dllexport) ~UserVideoAbstraction();

	__declspec(dllexport) void UserAbstraction(Mat& currentFrame, int frameIndex);							//进行背景减除的函数
	__declspec(dllexport) int UsersaveConfigInfo();														//保存剩余的凸包信息+运动序列的配置信息
	__declspec(dllexport) void Usercompound(int compoundNum, const char* outputpath);							// 视频摘要的函数
	__declspec(dllexport) void Usercompound(int compoundNum, const char* outputpath, int frameCount);			// 视频摘要的函数
	__declspec(dllexport) void UserfreeObject();															// 显式释放内存空间
	__declspec(dllexport) void UsersetGpu(bool isgpu);													// 设置是否使用GPU
	__declspec(dllexport) void UsersetROI(Rect rectroi);											// 设置是否选择感兴趣的区域
	__declspec(dllexport) void UsersetIndex(bool isindex);
	__declspec(dllexport) void UsersetMinArea(int area);
};