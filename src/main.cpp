//****************************************
//user: PianoCoder
//Create date: 2015-01-07
//Class name: Main
//Discription: Test all the function
//Update: none
//****************************************

#include "VideoAbstraction.h"
#include "UserVideoAbstraction.h"
#include "indexReplay.h"
#include "util.h"
#include <time.h>


#define MAX_INDEX_COUNT 100
#define SINAGLE_MIN_AREA 200 //设置单个画面上凸包的最小阈值
#define MIN_AREA_RATE 0.001  //设置单个画面上所有凸包面积之和占比的最小阈值

UserVideoAbstraction* user;
int xincoder_CompoundCount=8;

//declaration
extern int readFrameLog(string logname);
extern void readAreaLog(string logname, int &base_x, int &base_y, int& base_w, int& base_h);
extern Mat MultiImage(const vector<Mat>& SrcImg_V, Size ImgMax_Size);
extern void video_play(long index);
extern string int2string(int _Val);
static void bar_callback(int index,void* userdata);

//variable definition
string testVideoName;
string state;
//ROI-set 
//setROI=false by default 
Rect selectarea;
bool select_flag=false;
bool setROI=false;
bool setTimeTag=false;

Mat image,imageRoi,showimage;
//Video Index
int currentFrameIndex=0;
int frame_weidth;
string ori_video;
string index_video;
int EventNumber;
vector<int> event_start;
vector<int> event_end;
vector<int> event_length;
int event_count[MAX_INDEX_COUNT];

/* calling back event function */
//ROI selection
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
//Replay the seleted object's video shot
void mouseRecover(int mouseEvent,int x,int y,int flags,void* param)
{
	replayParams* replay_params = (replayParams*)param;
	Point p1,p2;
	if(mouseEvent==CV_EVENT_LBUTTONDOWN){
		select_flag=true;
		selectarea.x=x;
		selectarea.y=y;
		
	}
	else if(select_flag && mouseEvent==CV_EVENT_MOUSEMOVE){
		image.copyTo(showimage);
		p1=Point(selectarea.x,selectarea.y);
		p2=Point(x,y);
		rectangle(showimage,p1,p2,Scalar(0,255,0),2);
		imshow("video",showimage);
	}
	else if(select_flag && mouseEvent==CV_EVENT_LBUTTONUP){
		std::vector<int> loopuptable;
		indexReplay replay(replay_params->replayFileName, replay_params->replayConfigName);
		cv::Mat index_image = replay.loadEventsParamToRebuildMask(currentFrameIndex, replay_params->video_width, replay_params->video_height, loopuptable);
		//
		select_flag=false;
		int ID=0, maxCount=0,baseIndex=0;
		uchar *p;
		p=index_image.ptr<uchar>(0);
		//waitKey(0);
		selectarea.width=x-selectarea.x;
		selectarea.height=y-selectarea.y;
		Mat destmat;
		index_image(selectarea).copyTo(destmat);
		for(int i=0; i<MAX_INDEX_COUNT; i++) event_count[i]=0;
		for(int i=0; i<destmat.rows; i++){
			p=destmat.ptr<uchar>(i);
			for(int j=0; j<destmat.cols; j++){
				if ((int)p[j] != 255)//255为mask背景
				{
					if ((int)p[j] >= loopuptable.size())
					{
						cout << "error, p[j] out of boundary : p[j] = " << (int)p[j] << endl;
						return;
					}
					event_count[loopuptable[(int)p[j]]]++;
				}
			}
		}

		for(int i=0; i<MAX_INDEX_COUNT; i++){
			if(event_count[i] > maxCount){
				ID = i;
				maxCount = event_count[i];
			}
		}

		LOG(INFO)<<"selected event No. is "<<ID<<endl;	
		cout<<event_start.size()<<endl;
		int start=event_start[ID];
		int end=event_end[ID];
		LOG(INFO)<<"Info:	frame index start from  "<<start<<"	to	"<<end<<endl;
		//replay the selected video
		VideoCapture vc_read;
		Mat cur_mat;
		vc_read.open(ori_video);
		vc_read.set(CV_CAP_PROP_POS_FRAMES, start);
		namedWindow("Video Contains the Object");
		for(int i=start; i<end; i++){
			vc_read>>cur_mat;
			imshow("Video Contains the Object",cur_mat);	
			waitKey(20);
			//waitKey(0);
		}
		destroyWindow("Video Contains the Object");
	}
	return;  
}


/** test thread **/
/*****************************************************************/
//you can set test = 1,2,3,4 for different test
//test=1: subtract the background and foreground of the input video
//test=2: compound the convex point sequence to produce the abstracted video
//test=3: you can replay the seleted object's event full process
//test=4: you can view 9 snip-shots of the original video 
/*****************************************************************/
void testmultithread(string inputpath, string videoname, string midname, string outputname, int frameCount, int stage, bool readlog){
	time_t start_time,end_time;
	start_time=time(NULL);
	testVideoName=videoname;
	//set all the necessary paths
	string path=inputpath;
	string out_path=path+"OutputVideo/";
	string config_path=path+"Config/";
	string index_path=path+"indexMat/";
	string replay_path=path+"Replay/";
	string keyframe_path=path+"KeyFrame/";
	string log_path=path+"Log/";
	//create the path if not exist
	util::create_path(out_path);
	util::create_path(log_path);
	util::create_path(config_path);
	util::create_path(index_path);
	util::create_path(keyframe_path);
	util::create_path(replay_path);

	VideoCapture capture;
	string t1=inputpath,t2=videoname;
	string t3 = t1+t2;
	capture.open(t3);
	int video_width=capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int video_height=capture.get(CV_CAP_PROP_FRAME_HEIGHT);

	// resize the resolution of input video ...
	float scale=1;
	int basewidth=360;
	if (video_width>basewidth)
	{
		scale=((float)video_width)/basewidth;
	}
	LOG(INFO)<<videoname<<" scale = "<<scale<<endl;

	//create the VideoAbstraction Object and intialize the GPU setting and related threshold ...
	user=new UserVideoAbstraction((char*)path.data(), (char*)out_path.data(), (char*)log_path.data(), (char*)config_path.data(),
														(char*)index_path.data(), (char*)videoname.data(), (char*)midname.data(), scale);
	user->UsersetGpu(false);
	user->UsersetSingleMinArea(SINAGLE_MIN_AREA/(scale*scale));
	user->UsersetMinArea(MIN_AREA_RATE);
	user->UsersetTimeTag(false);
	/*
	*  different choices can excute different step ...
	*/
	int test = stage;
	//record the excution related time information ...
	ofstream ff_time(log_path+"TimeLog.txt", ofstream::app);
	//choice 1:   Background/Foreground Subtraction
	if(test==1)
	{
		state="Background/Foreground Subtraction";
		VideoCapture capture;
		string t1=inputpath,t2=videoname;
		string t3 = t1+t2;
		capture.open(t3);
		//record the frame number related information ...

		ofstream ff_frame(log_path+videoname+"FrameLog.txt", ofstream::app);
		ff_frame<<endl<<videoname<<"\t"<<capture.get(CV_CAP_PROP_FRAME_COUNT);
		int number=0;
		while (capture.read(image))
		{
			//choose the roi if setted 
			if(setROI && number==0){
				namedWindow("video");
				imshow("video",image);
				setMouseCallback("video",mouseSelect);
				waitKey(0);
				cvDestroyWindow("video");
				//record the ROI area related information ...
				ofstream ff_area(log_path+videoname+"AreaLog.txt", ofstream::app);
				ff_area<<endl<<selectarea.x<<":"<<selectarea.y<<":"<<selectarea.width<<":"<<selectarea.height;
				ff_area.close();
			}
			number++;
			user->UserAbstraction(image,number);
		}
		int UsedFrameCount = user->UsersaveConfigInfo();
		frameCount=UsedFrameCount;
		user->~UserVideoAbstraction();
		//record the used frame count information ...
		ff_frame<<"\t"<<UsedFrameCount<<"\t"<<(double)UsedFrameCount/(double)capture.get(CV_CAP_PROP_FRAME_COUNT)<<":"<<UsedFrameCount;
		ff_frame.close();
	}

	//choice 2: compound the result video based on the choice 1
	else if(test==2)
	{
		state="Compound the result video";
		string t3 = out_path+outputname;
		if(setROI){
			int x,y,width,height;
			//get the ROI information recorded in the AreaLog.txt ...
			readAreaLog(log_path+videoname+"AreaLog.txt", x, y, width, height);
			Rect selectRoi(x,y,width,height);
			//Set the ROI information and create the filter 0/1 mat ...
			user->UsergetROI(selectRoi);
			//print the ROI / Original video rate information ...
			float roi_rate=(video_width*video_height)/(width*height);
			LOG(INFO)<<video_width<<" "<<video_height<<endl;
			LOG(INFO)<<width<<" "<<height<<endl;
			LOG(INFO)<<roi_rate<<endl;
		}
		int frCount;
		//get the frameCount info from the FrameLog.txt ...
		if(readlog)
			frCount=readFrameLog(log_path+videoname+"FrameLog.txt");
		else
			frCount=frameCount;
		//excute the compound step to compound the final output video ...
		user->Usercompound(xincoder_CompoundCount, (char*)t3.data(), frCount);
		user->UserfreeObject();
	}

	//choice 3: Test the replay function ...
	else if(test==3)
	{
		state="test the index video function";
		int frCount;
		if(readlog)
			frCount=readFrameLog(log_path+videoname+"FrameLog.txt");
		else
			frCount=frameCount;
		string t1=inputpath,t2=midname,t=videoname;
		ori_video=inputpath+t;
		string t3 = outputname;
		t3=out_path+t3;
		string temp;
		//read the neccerary information from Log info ...
		ifstream file(config_path+t2);
		cout<<t2<<endl;
		cout<<frCount<<endl;
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
		//open the result video ...
		VideoCapture abstract_video;
		abstract_video.open(t3);
		currentFrameIndex=0;
		string filepath=index_path+t+"/";
		namedWindow("video");

		replayParams* replay_params = new replayParams(replay_path+videoname, config_path+midname, video_width / scale, video_height / scale);
		setMouseCallback("video",mouseRecover, (void*)replay_params);
		abstract_video.read(image);
		imshow("video",image);
		waitKey(0);
		abstract_video.open(t3);
		while(abstract_video.read(image)){		
			imshow("video",image);
			int key = waitKey(30); 
			if(key==27)
			{
				waitKey(0);
			}	
			currentFrameIndex++;	
		}
	}
	
	//Test the getting key frames function ...
	else if(test==4)
	{
		string t1=inputpath,t2=midname,t3=videoname,temp;
		int frCount;
		if(readlog)
			frCount=readFrameLog(log_path+videoname+"FrameLog.txt");
		else
			frCount=frameCount;
		user->UserGetKeyFrame(keyframe_path+t3+"/",frCount);
	}

	//others ...
	else
	{
		//check or debug 
	}

	end_time=time(NULL);
	//output the time information to the cmd and timeLog Information ...
	cout<<testVideoName<<"\t"<<state<<"\t"<<"video abstraction time: "<<end_time-start_time<<" s"<<endl;
	ff_time<<testVideoName<<"\t"<<state<<"\t"<<"video abstraction time: "<<end_time-start_time<<endl;
	ff_time.close();
}

void xincoder_thread()
{
	int load_number=0;
	while (cin>>load_number)
	{
		cout<<"输入个数为："<<load_number<<endl;
		user->xincoder_UsersetcompoundNum(load_number);
		xincoder_CompoundCount=load_number;

	}
}

int main(){
	//cout<<"Please input your video path (like C:/TongHaoTestVideo/)"<<endl;
	//string testpath, filename, configname, resultname;
	//int choice;
	//cin>>testpath;
	//cout<<"Please input your test video name (!do not support chinese video name -- like gaodangxiaoqu.avi) : "<<endl;
	//cin>>filename;
	//while(filename!="quit")
	//{
	//	cout<<"********************************************************"<<endl;
	//	cout<<"\t"<<"Using Guidance "<<endl;
	//	cout<<"\t"<<"Please input 1 / 2 "<<endl;
	//	cout<<"\t"<<"1:   abstact & compound the input video"<<endl;
	//	cout<<"\t"<<"2:   you can get the key frames"<<endl;
	//	cout<<"\t"<<"others:   Exit !"<<endl;
	//	cout<<"********************************************************"<<endl;
	//	cout<<"Please input the choice No. : ";
	//	cin>>choice;
	//	if(choice==1)
	//	{
	//		cout<<"Start abstract and compound the final video ..."<<endl;
	//		resultname="result_"+filename;
	//		configname=filename+"_config";
	//		testmultithread(testpath, filename, configname, resultname, 0, 8, 1, true);
	//		testmultithread(testpath, filename, configname, resultname, 0, 8, 2, true);
	//		cout<<"Finished abstact the final video ..."<<endl;
	//	}
	//	else if(choice==2)
	//	{
	//		cout<<"Get the key frames of the original video ..."<<endl;
	//		testmultithread(testpath, filename, configname, resultname, 0, 8, 4, true);
	//	}	
	//}

	string testset1[] = {"20111201_170301.avi", "20111202_082713.avi", "juminxiaoqu.avi", "testvideo.avi", "xiezilou.avi", "LOD_CIF_HQ_4_2.avi",
		"road.avi", "loumenkou.avi", "damenkou.avi", "AA012507.avi", "AA013101.avi", "AA013102.avi", "AA013103.avi", "AA013106.avi", "Cam01.avi", 
		"Cam3.avi", "Cam4.avi"};
	string testset5[] = {"shitang1.avi", "shitang2.avi", "shitang3.avi", "shitang4.avi", "shitang5.avi", "shitang6.avi", "shitang7.avi",
		               "gaodangxiaoqu.avi", "jinrong.avi",  "sanloubangongshi.avi",  "20110915_14-17-35.avi", "20111202_082711.avi", "20111202_101331.avi",  "kakou.avi"};
	string testset3[] = {"M2U00063.avi", "M2U00064.avi", "M2U00066.avi", "M2U00067.avi", "M2U00068.avi", "M2U00s069.avi", 
						"MVI_5612.avi","20111201_170301.avi", "20111202_101331.avi", "20111202_082711.avi", 
						"MVI_5613.avi","che 001.avi"};
	string testset4[] = { "shitang5.avi", "M2U00067.avi", "LOD_CIF_HQ_4_2.avi",  "juminxiaoqu.avi", "loumenkou.avi", "road.avi","20111202_082713.avi"};
	string testset2[] = {"testvideo.avi", "M2U00069.avi", "20111201_170301.avi", "20111202_082711.avi", "20111202_101331.avi", "M2U00064.avi", "M2U00066.avi", "M2U00067.avi", "M2U00068.avi", "M2U00069.avi", "MVI_5612.avi","MVI_5613.avi"};
	
	boost::thread new_thread(xincoder_thread);
	new_thread.start_thread();
	
	for(int i=0; i<1; i++){
	//for(int i=0; i<testset2->size()-1; i++){
		string result_name="new_result_"+testset2[i];
		string config_name=testset2[i]+"_config";
		if(setROI)
		{
			result_name="ROI"+result_name;
			config_name="ROI"+config_name;
		}
		//boost::thread test1(testmultithread,"F:/TongHaoTest4/", testset2[i], config_name, result_name, 0, 8, 1, true);
		//test1.join();
		//testmultithread("F:/TongHaoTest3/", testset2[i], config_name, result_name, 8,  1, true);
		testmultithread("F:/TongHaoTest3/", testset2[i], config_name, result_name, 5, 2, true);
		testmultithread("F:/TongHaoTest3/", testset2[i], config_name, result_name, 5, 3, true);
	}
	//for(int i=0; i<testset1->size(); i++){	
	//	string result_name="result_"+testset1[i];
	//	string config_name="config_"+boost::lexical_cast<string>(i);
	//	boost::thread test1(testmultithread,"F:/TongHaoTest1/", testset1[i], config_name, result_name, 0, 8, 1, true);
	//	test1.join();
	//	cout<<"finished..."<<endl;
	//}
	waitKey(0);
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
		//lastLine
	    string token = lastLine.substr(lastLine.find(":")+1, lastLine.size());
		return atoi(token.c_str());
		fin.close();
	}
	return 0;
}

void readAreaLog(string logname, int &base_x, int &base_y, int& base_w, int& base_h){
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
		vector<string> elem;
		boost::split(elem, lastLine, boost::is_any_of(":"));
		base_x = boost::lexical_cast<int>(elem[0]);
		base_y = boost::lexical_cast<int>(elem[1]);
		base_w = boost::lexical_cast<int>(elem[2]);
		base_h = boost::lexical_cast<int>(elem[3]);
		fin.close();
	}
}

string int2string(int _Val){
	char _Buf[100];
	sprintf(_Buf, "%d", _Val);
	return (string(_Buf));
}
