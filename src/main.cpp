#include "VideoAbstraction.h"
#include "UserVideoAbstraction.h"
#include <time.h>

extern int readFrameLog(string logname);
extern void readAreaLog(string logname, int &base_x, int &base_y);
extern Mat MultiImage(const vector<Mat>& SrcImg_V, Size ImgMax_Size);
extern void video_play(long index);
static void bar_callback(int index,void* userdata);



string testVideoName;
//快照显示
#define DIV_NUMBER 9
VideoCapture capture;
Size SubPlot;
char* window_name="img";
char* window_play="video";
int bar_index;//指示视频播放进度条
vector<long> video_index;//用于记录每个快照对应视频帧数
int small_width;//每一个小图的宽度
int small_height;//每个小图的高度
//感兴趣区域勾选需要的参数
Rect selectarea;
bool select_flag=false;
bool setROI=false;
Mat image,imageRoi,showimage,index_image;
//点击目标播放原始视频需要的参数
int currentFrameIndex=0;
int frame_weidth;
string ori_video;
string index_video;
int EventNumber;
vector<int> event_start;
vector<int> event_end;
vector<int> event_length;
//vector<int> event_count;
int event_count[20];

//兴趣区域选择
void mouseSelect(int mouseEvent,int x,int y,int flags,void* param)  
{  
	Point p1,p2;
	if(mouseEvent==CV_EVENT_LBUTTONDOWN){
		selectarea.x=x;
		selectarea.y=y;
		select_flag=true;
	}
	else if(select_flag && mouseEvent==CV_EVENT_MOUSEMOVE){
		image.copyTo(showimage);
		p1=Point(selectarea.x,selectarea.y);
		p2=Point(x,y);
		rectangle(showimage,p1,p2,Scalar(0,255,0),2);
		imshow("video",showimage);
	}

	else if(select_flag && mouseEvent==CV_EVENT_LBUTTONUP){
		selectarea.width=x-selectarea.x;
		selectarea.height=y-selectarea.y;
		select_flag=false;
	}
	return;  
} 

//回播放原始视频
void mouseRecover(int mouseEvent,int x,int y,int flags,void* param)
{
	Point p1,p2;
	if(mouseEvent==CV_EVENT_LBUTTONDOWN){
		selectarea.x=x;
		selectarea.y=y;
		select_flag=true;
	}
	else if(select_flag && mouseEvent==CV_EVENT_MOUSEMOVE){
		image.copyTo(showimage);
		p1=Point(selectarea.x,selectarea.y);
		p2=Point(x,y);
		rectangle(showimage,p1,p2,Scalar(0,255,0),2);
		imshow("video",showimage);
	}

	else if(select_flag && mouseEvent==CV_EVENT_LBUTTONUP){
		//imshow("check", index_image);
		selectarea.width=x-selectarea.x;
		selectarea.height=y-selectarea.y;
		select_flag=false;
		Mat destmat;
		index_image(selectarea).copyTo(destmat);
		//imshow("check_video", destmat);
		//统计出现次数最多的对应事件序号   Finding the majority Element 找到勾选区域中多数像素所属于的事件序号
		//int ID = (int)index_image.at<uchar>(Point(x,y));
		//int temp = 0;
		//int freq = 1;
		//for(int i=selectarea.x; i<x; i++){
		//	for(int j=selectarea.y; j<y; j++){
		//		temp = (int)index_image.at<uchar>(Point(i,j));
		//		if(temp==ID)	freq++;
		//		else if(freq>=1) freq--;
		//		else if(freq==0) {ID=temp; freq=1;}
		//	}
		//}
		//cout<<ID<<endl;
		//waitKey(1000);
		int ID=0, maxCount=0;
		uchar *p;
		for(int i=0; i<20; i++) event_count[i]=0;
		for(int i=0; i<destmat.rows; i++){
			p=destmat.ptr<uchar>(i);
			for(int j=0; j<destmat.cols; j++){
				//cout<<(int)p[j]<<"-";
				if((int)p[j] > 100)	event_count[255-(int)p[j]]++;
			}
		}
		for(int i=0; i<20; i++){
			//cout<<"ID \t"<<i<<"Count\t"<<event_count[i]<<endl;
			if(event_count[i] > maxCount){
				ID = i;
				maxCount = event_count[i];
			}
		}
		cout<<"ID NUMBER "<<ID<<endl;
		cout<<"Frame Index  "<<currentFrameIndex<<endl;
		int start=event_start[ID];
		int end=event_end[ID];
		cout<<start<<"--->"<<end<<endl;
		//
		VideoCapture vc_read;
		Mat cur_mat;
		vc_read.open(ori_video);
		vc_read.set(CV_CAP_PROP_POS_FRAMES, start);
		namedWindow("索引");
		for(int i=start; i<end; i++){
			vc_read>>cur_mat;
			imshow("索引",cur_mat);	
			waitKey(0);
		}
		destroyWindow("索引");
	}
	return;  
}

//快照功能实现
//鼠标回调函数 
static void mouseSnipShot(int event, int x, int y, int flags, void* userdata)
{  
	switch(event)  
	{  
	case CV_EVENT_LBUTTONDOWN:  //鼠标左键按下
		int heng=y/small_height;//第几行
		int zong=x/small_width;//第几列
		long index=heng*SubPlot.width+zong;//计算快照索引
		video_play(video_index[index]);
		break;  

	}  
	return;  
} 


//************************************************************************/
//*         测试线程                                                      */
//************************************************************************/
void testmultithread(const char* inputpath, const char* videoname, const char* midname, const char* outputname, int frameCount, int CompoundCount){
	testVideoName=videoname;
	UserVideoAbstraction* user=new UserVideoAbstraction(inputpath, videoname, midname);
	user->UsersetGpu(true);
	user->UsersetIndex(false);
	user->UsersetMinArea(60);
	string logpath=inputpath;
	//test = 2 will select the interested area
	int test = 3;
	/************************************************************************/
	/*                    前背景分离阶段测试                                   */
	/************************************************************************/
	if(test==1){
		VideoCapture capture;
		string t1=inputpath,t2=videoname;
		string t3 = t1+t2;
		capture.open(t3);
		ofstream ff(logpath+"FrameLog.txt", ofstream::app);
		ff<<endl<<videoname<<"-总帧数-"<<capture.get(CV_CAP_PROP_FRAME_COUNT);
		int number=0;
		while (capture.read(image))
		{
			if(setROI){
				if(number==0){
					namedWindow("video");
					imshow("video",image);
					setMouseCallback("video",mouseSelect);
					waitKey(0);
					cvDestroyWindow("video");
					user->UsersetROI(selectarea);
					ofstream ff(logpath+"AreaLog.txt", ofstream::app);
					ff<<endl<<selectarea.x<<":"<<selectarea.y;
				}
				number++;
				image(selectarea).copyTo(imageRoi);
				user->UserAbstraction(imageRoi,number);
			}
			else{
				number++;
				user->UserAbstraction(image,number);
			}
		}
		int UsedFrameCount = user->UsersaveConfigInfo();
		user->~UserVideoAbstraction();
		//****   记录不同视频文件的有效帧的数值，视频合成阶段需要传入的参数 frameCount 值   ****//
		ff<<endl<<videoname<<":"<<UsedFrameCount;
		ff.close();
	}
	/************************************************************************/
	/*				视频摘要合成阶段                                        */
	/************************************************************************/
	else if(test==2){
		string t3 = outputname;
		if(setROI){
			int x,y;
			readAreaLog(logpath+"AreaLog.txt", x, y);
			Rect selectRoi(x,y,100,100);
			user->UsersetROI(selectRoi);
		}
		int frCount = readFrameLog(logpath+"FrameLog.txt");
		user->Usercompound(CompoundCount, (char*)t3.data(), frCount);
		user->UserfreeObject();
	}
	else if(test==3){
		int frCount = readFrameLog(logpath+"FrameLog.txt");
		string t1=inputpath,t2=midname,t=videoname;
		ori_video=t1+t;
		string t3 = outputname;
		string temp;
		//读取中间文件获取 event_start event_end event_length 信息
		ifstream file(t1+t2);
		for(int i=0; i<frCount; i++) {		
			getline(file, temp, '#');
		}
		event_start.clear();
		event_end.clear();
		EventNumber=0;
		while(!file.eof()){
			int s,e,len;
			file>>s;
			file>>e;
			len=e-s;
			event_start.push_back(s);
			event_end.push_back(e);
			event_length.push_back(len);
			EventNumber++;
		}
		file.close();
		//index
		VideoCapture abstract_video;
		abstract_video.open(t3);
		currentFrameIndex=0;
		string filepath="D:/summarytest1/indexMat/"+t+"/";
		while(abstract_video.read(image)){
			currentFrameIndex++;	
			string filename=boost::lexical_cast<string>(currentFrameIndex)+".jpg";
			index_image=imread(filepath+filename);
			namedWindow("video");
			imshow("video",image);
			//imshow("index", index_image);
			setMouseCallback("video",mouseRecover);
			int key = waitKey(1); 
			if(key==27)
				waitKey(0);
		}
	}
	else if(test==4){
		namedWindow(window_name);
		setMouseCallback(window_name,mouseSnipShot);//设置鼠标回调函数
		SubPlot=Size(3,3);//最终快照显示图像为3*3 矩阵的九张图像
		string t1=inputpath,t2=videoname;
		string t3 = t1+t2;
		capture.open(t3);
		//检测是否正常打开:成功打开时，isOpened返回ture
		if(!capture.isOpened())
		{
			cout<<"fail to open!"<<endl;
		}
		double totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);//获取整个帧数
		cout<<"整个视频共"<<totalFrameNumber<<"帧"<<endl;
		bar_index=0;//进度条起始位置
		int number_per_div=(int)totalFrameNumber/DIV_NUMBER;//计算两个快照间的间隔帧数
		vector<Mat> images(DIV_NUMBER);//保存快照
		video_index=vector<long>(DIV_NUMBER);
		for (int i=0;i<DIV_NUMBER;i++)
		{
			long now_frame=number_per_div*i;//计算第i张快照 对应的帧数
			video_index[i]=now_frame;

			capture.set(CV_CAP_PROP_POS_FRAMES,now_frame);//设置读取这一帧

			Mat zhong;
			capture.read(zhong);//读取快照
			zhong.copyTo(images[i]);
			zhong.release();
		}
		Mat result_image=MultiImage(images, Size(images[0].cols,images[0].rows));
		imshow(window_name,result_image);
		waitKey(0);
		capture.release();
		result_image.release();
	}
	else{
		VideoCapture videoCapture;
		string Inputpath=inputpath;
		videoCapture.open(Inputpath+"indexVideo.avi");
		videoCapture.set(CV_CAP_PROP_POS_FRAMES,149);
		Mat testframe;
		videoCapture>>testframe;
		//testframe=imread("D:/summarytest1/indexMat/1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		testframe=imread("D:/summarytest1/indexMat/1.jpg");
		//imshow("check",testframe);
		
		int count_check[20];
		for(int i=0; i< 20; i++) count_check[i]=0;			
		for(int ii=0; ii<testframe.rows; ii++)
		{
			uchar* pi=testframe.ptr<uchar>(ii);
			for(int jj=0; jj<testframe.cols;jj++){
				if(pi[jj] > 100)
					count_check[255-pi[jj]]++;
			}
		}
		for(int i=0; i< 20; i++) cout<<";"<<count_check[i];
		waitKey(0);
	}
}


int main(){
	/************************************************************************/
	/*                 利用 boost 测试6个线程的示例                           */
	/************************************************************************/
	time_t start_time,end_time;
	start_time=time(NULL);

	/************************************************************************/
	/*                      Basic  Test !!!                                */
	/************************************************************************/ 
	//boost::thread test1(testmultithread,"F:/VideoAbVideoSet/input/", "1hour_sample.avi", "config-1","f:/VideoAbVideoSet/output/result-1.avi", 44597, 8);
	//boost::thread test1(testmultithread,"F:/VideoAbVideoSet/input/", "AA014303.avi", "config-2","f:/VideoAbVideoSet/output/result-2.avi", 2766, 8);
	//boost::thread test1(testmultithread,"F:/VideoAbVideoSet/input/", "AA014403.avi", "config-3","f:/VideoAbVideoSet/output/result-3.avi", 756, 8);
	//boost::thread test1(testmultithread,"F:/VideoAbVideoSet/input/", "camera2_2013510175011.mp4", "config-4","f:/VideoAbVideoSet/output/result-4.avi", 1306, 8);
	//boost::thread test1(testmultithread,"F:/VideoAbVideoSet/input/", "lena_in_1_clip.avi", "config-5","f:/VideoAbVideoSet/output/result-5.avi", 100, 8);
	//boost::thread test1(testmultithread,"F:/VideoAbVideoSet/input/", "MVI_0678_Compress.avi", "config-6","f:/VideoAbVideoSet/output/result-6.avi", 14006, 8);
	//boost::thread test1(testmultithread,"F:/VideoAbVideoSet/input/", "MVI_0682_Compress_clip.avi", "config-7","f:/VideoAbVideoSet/output/result-7.avi", 4498, 8);
	//boost::thread test1(testmultithread,"F:/VideoAbVideoSet/input/", "MVI_0683_Compress_clip.avi", "config-8","f:/VideoAbVideoSet/output/result-8.avi", 12540, 8);
	//boost::thread test1(testmultithread,"F:/VideoAbVideoSet/input/", "test1.avi", "config-9","f:/VideoAbVideoSet/output/result-9.avi", 400, 8);
	//boost::thread test1(testmultithread,"F:/VideoAbVideoSet/input/", "test2.avi", "config-10","f:/VideoAbVideoSet/output/result-10.avi", 1806, 8);

	//resize the No D1-Video -------> D1-Video
	//boost::thread test1(testmultithread,"F:/VideoAbVideoSet/", "AA012507.mp4", "config-6","f:/VideoAbVideoSet/AA01.avi", 14006, 8);
	//boost::thread test1(testmultithread,"F:/VideoAbVideoSet/", "AA012507.avi", "config-AA012507","f:/VideoAbVideoSet/result-AA012507_ROI.avi", 2332, 8);
	//boost::thread test1(testmultithread,"F:/VideoAbVideoSet/", "AA013102.avi", "config-AA013102","f:/VideoAbVideoSet/result-AA013102.avi", 2332, 8);
	/************************************************************************/
	/*                      Tong Hao Test !!!                               */
	/************************************************************************/ 

	//boost::thread test1(testmultithread,"D:/summarytest1/", "juminxiaoqu.avi", "config-0","D:/summarytest1/result-0.avi", 2337, 8);
	boost::thread test1(testmultithread,"D:/summarytest1/", "testvideo.avi", "config-1","D:/summarytest1/result-1.avi", 1213, 8);
	//boost::thread test1(testmultithread,"D:/summarytest1/", "xiezilou.avi", "config-2","D:/summarytest1/result-2.avi", 2963, 8);
	//boost::thread test1(testmultithread,"D:/summarytest1/", "LOD_CIF_HQ_4_2.avi", "config-3","D:/summarytest1/result-3.avi", 9146, 8);
	//boost::thread test1(testmultithread,"D:/summarytest1/", "gaodangxiaoqu.avi", "config-4","D:/summarytest1/result-4.avi", 2337, 8);  // no video
	//boost::thread test1(testmultithread,"D:/summarytest1/", "road.avi", "config-5","D:/summarytest1/result-5.avi", 6205, 8);
	//boost::thread test1(testmultithread,"D:/summarytest1/", "loumenkou.avi", "config-6","D:/summarytest1/result-6.avi", 26335, 8);
	//boost::thread test1(testmultithread,"D:/summarytest1/", "damenkou.avi", "config-7","D:/summarytest1/result-7.avi", 16930, 8);
	//boost::thread test1(testmultithread,"D:/summarytest1/", "an.avi", "config-8","D:/summarytest1/result-8.avi", 34614, 8);
	//boost::thread test1(testmultithread,"D:/summarytest1/", "20111201_144857.avi", "config-9","D:/summarytest1/result-9.avi", 2337, 8);  //20s length
	//boost::thread test1(testmultithread,"D:/summarytest1/", "20111201_170301.avi", "config-10","D:/summarytest1/result-10.avi", 45460, 8); 
	//boost::thread test1(testmultithread,"D:/summarytest1/", "20111202_082713.avi", "config-11","D:/summarytest1/result-11.avi", 34614, 8);

	test1.join();

	cout<<"finished..."<<endl;
	end_time=time(NULL);
	//ofstream ff("F:/VideoAbVideoSet/TimeLog.txt", ofstream::app);
	ofstream ff("D:/summarytest1/TimeLog.txt", ofstream::app);
	ff<<time(NULL)<<"\t"<<testVideoName<<"\t video abstraction time: "<<end_time-start_time<<endl;
	//ofstream ff("f:/count_config.txt", ofstream::app);
	//ff<<"video abstraction time: "<<end_time-start_time<<endl;
	getchar();
	return 0;
}

int readFrameLog(string logname){
	ifstream fin;
	fin.open(logname);
	if(fin.is_open()) {
		fin.seekg(-1,ios_base::end);                // go to one spot before the EOF
		bool keepLooping = true;
		while(keepLooping) {
			char ch;
			fin.get(ch);                            // Get current byte's data
			if((int)fin.tellg() <= 1) {             // If the data was at or before the 0th byte
				fin.seekg(0);                       // The first line is the last line
				keepLooping = false;                // So stop there
			}
			else if(ch == '\n') {                   // If the data was a newline
				keepLooping = false;                // Stop at the current position.
			}
			else {                                  // If the data was neither a newline nor at the 0 byte
				fin.seekg(-2,ios_base::cur);        // Move to the front of that data, then to the front of the data before it
			}
		}
		string lastLine;            
		getline(fin,lastLine);                      // Read the current line
		cout<< lastLine<<'\n';     // Display it
	    string token = lastLine.substr(lastLine.find(":")+1, lastLine.size());
		return atoi(token.c_str());
		fin.close();
	}
	return 0;
}

void readAreaLog(string logname, int &base_x, int &base_y){
	ifstream fin;
	fin.open(logname);
	if(fin.is_open()) {
		fin.seekg(-1,ios_base::end);                // go to one spot before the EOF
		bool keepLooping = true;
		while(keepLooping) {
			char ch;
			fin.get(ch);                            // Get current byte's data
			if((int)fin.tellg() <= 1) {             // If the data was at or before the 0th byte
				fin.seekg(0);                       // The first line is the last line
				keepLooping = false;                // So stop there
			}
			else if(ch == '\n') {                   // If the data was a newline
				keepLooping = false;                // Stop at the current position.
			}
			else {                                  // If the data was neither a newline nor at the 0 byte
				fin.seekg(-2,ios_base::cur);        // Move to the front of that data, then to the front of the data before it
			}
		}
		string lastLine;            
		getline(fin,lastLine);                      // Read the current line
		cout<< lastLine<<'\n';     // Display it
		string x = lastLine.substr(0,lastLine.find(":"));
		string y = lastLine.substr(lastLine.find(":")+1);
		base_x = atoi(x.c_str());
		base_y = atoi(y.c_str());
		fin.close();
	}
}

Mat MultiImage(const vector<Mat>& SrcImg_V, Size ImgMax_Size)
{
	/*
	*函数功能：	将多张图像拼接到一张图像上
	*SrcImg_v:	需要拼接的图像向量
	*ImgMax_Size:	最总显示的图像的大小
	*/
	Mat return_image(SrcImg_V[0].rows*SubPlot.height,SrcImg_V[0].cols*SubPlot.width,SrcImg_V[0].type());//初始化返回图像

	int heng=SubPlot.height;//最终图像显示多少行
	int zong=SubPlot.width;//最终图像显示多少列

	int width=SrcImg_V[0].cols;
	int height=SrcImg_V[0].rows;

	int num=0;//当前图像的编号
	for (int i=0;i<heng;i++)//纵向循环 每一行
	{
		for (int j=0;j<zong;j++)//横向循环 每一列
		{	
			int start_x=(num%zong)*width;//小图在大图中 左上角的横坐标
			int start_y=(num/zong)*height;//小图在大图中 左上角的纵坐标

			SrcImg_V[num].copyTo(return_image(Rect(start_x,start_y,width,height)));
			num++;//图像编号加一
		}
	}
	resize(return_image,return_image,ImgMax_Size);	
	small_width=ImgMax_Size.width/zong;//计算每一个小图的宽度
	small_height=ImgMax_Size.height/heng;//计算每个小图的高度
	return return_image;
}

static void bar_callback(int index,void* userdata)
{
	capture.set(CV_CAP_PROP_POS_FRAMES,index);
	bar_index=index;  
}

void video_play(long index)
{
	//视频播放函数
	Mat zhong;
	namedWindow(window_play);

	capture.set(CV_CAP_PROP_POS_FRAMES,index);
	double totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);//获取整个帧数
	double fps=capture.get(CV_CAP_PROP_FPS);
	if (totalFrameNumber>0)
	{
		cv::createTrackbar("position",window_play,&bar_index,totalFrameNumber,bar_callback);//创建进度条，bar_callback是回调函数
		bar_index=index;
	}

	while(capture.read(zhong))
	{
		bar_index++;
		setTrackbarPos("position",window_play,bar_index);//
		imshow(window_play,zhong);
		//waitKey(1000.0/fps);//间隔时间可以调整，这样显示会比实际的慢
		waitKey(20);//间隔时间可以调整，这样显示会比实际的慢
	}
	zhong.release();
	cv::destroyWindow(window_play);

}
