//****************************************
//user: PianoCoder
//Create date: 2015/01/01
//Class name:  UserVideoAbstraction(...) class implemention
//Discription: implement the interface for all the video abstraction operation
//Update:      2015/01/17
//****************************************

#include "VideoAbstraction.h"
#include "UserVideoAbstraction.h"

UserVideoAbstraction::UserVideoAbstraction(const char* inputpath, const char* out_path, const char* log_path, const char* config_path,const char* index_path,const char* videoname, const char* midname, float scale)
{
	string t1=inputpath;
	string t2=out_path;
	string t3=log_path;
	string t4=config_path;
	string t5=index_path;
	string t6=videoname;
	string t7=midname;
	userVB =new VideoAbstraction(t1,t2,t3,t4,t5,t6,t7,scale);
}

UserVideoAbstraction::UserVideoAbstraction(){};

UserVideoAbstraction::~UserVideoAbstraction(){
	//userVB->~VideoAbstraction();
	delete userVB;
};

//前背景分离函数
void UserVideoAbstraction::UserAbstraction(Mat& currentFrame, int frameIndex){
	userVB->Abstraction(currentFrame, frameIndex);
}

//保存前背景分离的产生的凸包信息及配置信息/返回有效帧的帧数
int UserVideoAbstraction::UsersaveConfigInfo(){
	//userVB->saveRemainObject();
	userVB->saveConfigInfo();
	return userVB->ObjectCubeNumber;
}

//视频压缩函数 （不能分阶段处理）
void UserVideoAbstraction::Usercompound(int compoundNum, const char* outputpath){
	userVB->motionToCompound=compoundNum;
	string path = outputpath;
	userVB->LoadConfigInfo();
	userVB->compound(path);
}

//视频压缩函数，适用于分阶段处理，frameCount 是视频前背景分离阶段保存的值
void UserVideoAbstraction::Usercompound(int compoundNum, const char* outputpath, int frameCount){
	userVB->motionToCompound=compoundNum;
	string path = outputpath;
	userVB->LoadConfigInfo(frameCount);
	userVB->compound(path);
}

//显式析构函数
void UserVideoAbstraction::UserfreeObject(){
	userVB->freeObject();
}

void UserVideoAbstraction::UsersetGpu(bool isgpu){
	userVB->setGpu(isgpu);
}

void UserVideoAbstraction::UsergetROI(Rect rectroi){	
	userVB->useROI=true;
	userVB->rectROI.x=rectroi.x;
	userVB->rectROI.y=rectroi.y;
	userVB->rectROI.width=rectroi.width;
	userVB->rectROI.height=rectroi.height;
	userVB->setFilter(userVB->roiFilter, userVB->rectROI, userVB->frameHeight*userVB->frameWidth);
}

//void UserVideoAbstraction::UsersetIndex(bool isindex){
//	userVB->setGpu(isindex);
//}

void UserVideoAbstraction::UsersetMinArea(float minarea){
	userVB->thres=minarea;
}

void UserVideoAbstraction::UsersetSingleMinArea(float minarea){
	userVB->objectarea=minarea;
}

void UserVideoAbstraction::UserGetKeyFrame(string filepath, int frameCount){
	userVB->LoadConfigInfo(frameCount);
	userVB->getKeyFrame(filepath);
}

void UserVideoAbstraction::UsersetTimeTag(bool timetag){
	userVB->useTimeFlag=timetag;
}

void UserVideoAbstraction::xincoder_UsersetcompoundNum(int compoundNum)
{
	userVB->motionToCompound=compoundNum;
	cout<<"xincoder_疏密参数设置成功，当前为"<<compoundNum<<"个"<<endl;
}
