#include "VideoAbstraction.h"
#include "UserVideoAbstraction.h"

UserVideoAbstraction::UserVideoAbstraction(const char* inputpath, const char* videoname,const  char* midname)
{
	string t1=inputpath;
	string t2=videoname;
	string t3=midname;
	userVB =new VideoAbstraction(t1,t2,t3);
}

UserVideoAbstraction::UserVideoAbstraction(){};

UserVideoAbstraction::~UserVideoAbstraction(){
	userVB->~VideoAbstraction();
	delete userVB;
};

//ǰ�������뺯��
void UserVideoAbstraction::UserAbstraction(Mat& currentFrame, int frameIndex){
	userVB->Abstraction(currentFrame, frameIndex);
}

//����ǰ��������Ĳ�����͹����Ϣ��������Ϣ/������Ч֡��֡��
int UserVideoAbstraction::UsersaveConfigInfo(){
	//userVB->saveRemainObject();
	userVB->saveConfigInfo();
	return userVB->ObjectCubeNumber;
}

//��Ƶѹ������ �����ֽܷ׶δ���
void UserVideoAbstraction::Usercompound(int compoundNum, const char* outputpath){
	userVB->motionToCompound=compoundNum;
	string path = outputpath;
	userVB->LoadConfigInfo();
	userVB->compound(path);
}

//��Ƶѹ�������������ڷֽ׶δ���frameCount ����Ƶǰ��������׶α����ֵ
void UserVideoAbstraction::Usercompound(int compoundNum, const char* outputpath, int frameCount){
	userVB->motionToCompound=compoundNum;
	string path = outputpath;
	userVB->LoadConfigInfo(frameCount);
	userVB->compound(path);
}

//��ʽ��������
void UserVideoAbstraction::UserfreeObject(){
	userVB->freeObject();
}

void UserVideoAbstraction::UsersetGpu(bool isgpu){
	userVB->setGpu(isgpu);
}

void UserVideoAbstraction::UsersetROI(Rect rectroi){	
	//userVB->setROI(isroi);
	userVB->useROI=true;
	//userVB->frameHeight=rectroi.height;
	//userVB->frameWidth=rectroi.width;
	//userVB->rectROI.width=rectroi.width;
	//userVB->rectROI.height=rectroi.height;
	userVB->rectROI.x=rectroi.x;
	userVB->rectROI.y=rectroi.y;
}

void UserVideoAbstraction::UsersetIndex(bool isindex){
	userVB->setGpu(isindex);
}

void UserVideoAbstraction::UsersetMinArea(int minarea){
	userVB->objectarea=minarea;
}