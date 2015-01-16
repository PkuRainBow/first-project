//****************************************
//user: PianoCoder
//Create date: 2015-01-07
//Class name: Main
//Discription: Test all the function
//Update: none
//****************************************

#include "VideoAbstraction.h"
#include "UserVideoAbstraction.h"
#include <time.h>

#define DIV_NUMBER 9
#define MAX_INDEX_COUNT 300
#define MIN_AREA 100

//declaration
extern int readFrameLog(string logname);
extern void readAreaLog(string logname, int &base_x, int &base_y);
extern Mat MultiImage(const vector<Mat>& SrcImg_V, Size ImgMax_Size);
extern void video_play(long index);
extern void create_path(string path);
extern string int2string(int _Val);
static void bar_callback(int index,void* userdata);

//variable definition
string testVideoName;
string log_path;
string state;
//snap-shot 

VideoCapture capture;
Size SubPlot;
char* window_name="img";
char* window_play="video";
int bar_index;//指示视频播放进度条
vector<long> video_index;//用于记录每个快照对应视频帧数
int small_width;//每一个小图的宽度
int small_height;//每个小图的高度
//ROI-set 
//setROI=false by default 
Rect selectarea;
bool select_flag=false;
bool setROI=false;
Mat image,imageRoi,showimage,index_image;
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
		select_flag=false;
		int ID=0, maxCount=0,baseIndex=0;
		uchar *p;
		p=index_image.ptr<uchar>(0);
		baseIndex=(int)p[0];
		cout<<"base Index"<<baseIndex<<endl;
		//waitKey(0);
		selectarea.width=x-selectarea.x;
		selectarea.height=y-selectarea.y;
		
		Mat destmat;
		index_image(selectarea).copyTo(destmat);
		for(int i=0; i<MAX_INDEX_COUNT; i++) event_count[i]=0;
		for(int i=0; i<destmat.rows; i++){
			p=destmat.ptr<uchar>(i);
			for(int j=0; j<destmat.cols; j++){
				if((int)p[j] > 100)	event_count[255-(int)p[j]]++;
				//if((int)p[j] > 100)	event_count[(int)p[j]]++;
			}
		}
		for(int i=0; i<MAX_INDEX_COUNT; i++){
			if(event_count[i] > maxCount){
				ID = i;
				maxCount = event_count[i];
			}
		}
		//ID=ID+baseIndex*256;
		cout<<"Info:	selected event No. is "<<ID<<endl;	
		cout<<event_start.size()<<endl;
		int start=event_start[ID];
		int end=event_end[ID];
		cout<<"Info:	frame index start from  "<<start<<"	to	"<<end<<endl;
		//replay the selected video
		VideoCapture vc_read;
		Mat cur_mat;
		vc_read.open(ori_video);
		vc_read.set(CV_CAP_PROP_POS_FRAMES, start);
		namedWindow("Video Contains the Object");
		for(int i=start; i<end; i++){
			vc_read>>cur_mat;
			imshow("Video Contains the Object",cur_mat);	
			waitKey(0);
		}
		destroyWindow("Video Contains the Object");
	}
	return;  
}
// snip-shot 
static void mouseSnipShot(int event, int x, int y, int flags, void* userdata)
{  
	switch(event)  
	{  
	case CV_EVENT_LBUTTONDOWN:
		//get the selected video's position
		int heng=y/small_height;
		int zong=x/small_width;
		long index=heng*SubPlot.width+zong;
		video_play(video_index[index]);
		break;
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
void testmultithread(string inputpath, string videoname, string midname, string outputname, int& frameCount, int CompoundCount, int scale, int stage, bool readlog){
//void testmultithread(const char* inputpath, const char* videoname, const char* midname, const char* outputname, int frameCount, int CompoundCount, int scale, int stage){
	time_t start_time,end_time;
	start_time=time(NULL);
	testVideoName=videoname;
	//set all the necessary paths
	string path=inputpath;
	string out_path=path+"OutputVideo/";
	string config_path=path+"Config/";
	string index_path=path+"indexMat/";
	string keyframe_path=path+"KeyFrame/";
	log_path=path+"Log/";
	//create the path if not exist
	create_path(out_path);
	create_path(log_path);
	create_path(config_path);
	create_path(index_path);
	create_path(keyframe_path);
	
	UserVideoAbstraction* user=new UserVideoAbstraction((char*)path.data(), (char*)out_path.data(), (char*)log_path.data(), (char*)config_path.data(),
														(char*)index_path.data(), (char*)videoname.data(), (char*)midname.data(), scale);
	user->UsersetGpu(true);
	user->UsersetIndex(false);
	user->UsersetMinArea(MIN_AREA/(scale*scale));

	int test = stage;
	ofstream ff(log_path+"TimeLog.txt", ofstream::app);
	if(test==1){
		state="Background/Foreground Subtraction";
		VideoCapture capture;
		string t1=inputpath,t2=videoname;
		string t3 = t1+t2;
		capture.open(t3);
		ofstream ff(log_path+"FrameLog.txt", ofstream::app);
		ff<<endl<<videoname<<"\t"<<capture.get(CV_CAP_PROP_FRAME_COUNT);
		int number=0;
		setROI=false;
		while (capture.read(image))
		{
			//if(number==0){
			//	namedWindow("video");
			//	imshow("video",image);
			//	setMouseCallback("video",mouseSelect);
			//	waitKey(0);
			//	cvDestroyWindow("video");
			//	//user->UsersetROI(selectarea);
			//	ofstream ff(log_path+"AreaLog.txt", ofstream::app);
			//	ff<<endl<<selectarea.x<<":"<<selectarea.y;
			//}
			if(setROI){
				if(number==0){
					namedWindow("video");
					imshow("video",image);
					setMouseCallback("video",mouseSelect);
					waitKey(0);
					cvDestroyWindow("video");
					//user->UsersetROI(selectarea);
					ofstream ff(log_path+"AreaLog.txt", ofstream::app);
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
		frameCount=UsedFrameCount;
		user->~UserVideoAbstraction();
		ff<<"\t"<<UsedFrameCount<<"\t"<<(double)UsedFrameCount/(double)capture.get(CV_CAP_PROP_FRAME_COUNT)<<":"<<UsedFrameCount;
		ff.close();
	}
	else if(test==2){
		state="compound the result video";
		string t3 = out_path+outputname;
		if(setROI){
			int x,y;
			readAreaLog(log_path+"AreaLog.txt", x, y);
			Rect selectRoi(x,y,100,100);
			user->UsersetROI(selectRoi);
		}
		int frCount;
		if(readlog)
			frCount=readFrameLog(log_path+"FrameLog.txt");
		else
			frCount=frameCount;
		//int frCount = readFrameLog(log_path+"FrameLog.txt");
		cout<<frCount<<endl;
		//int frCount=frameCount;
		user->Usercompound(CompoundCount, (char*)t3.data(), frCount);
		user->UserfreeObject();
	}
	else if(test==3){
		state="test the index video function";
		int frCount;
		if(readlog)
			frCount=readFrameLog(log_path+"FrameLog.txt");
		else
			frCount=frameCount;
		string t1=inputpath,t2=midname,t=videoname;
		ori_video=inputpath+t;
		string t3 = outputname;
		t3=out_path+t3;
		string temp;
		//读取中间文件获取 event_start event_end event_length 信息
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
		//index
		VideoCapture abstract_video;
		abstract_video.open(t3);
		currentFrameIndex=0;
		string filepath=index_path+t+"/";
		namedWindow("video");
		//setMouseCallback("video",mouseRecover);

		abstract_video.read(image);
		imshow("video",image);
		waitKey(0);
		abstract_video.open(t3);

		vector<int> loopuptable;
		while(abstract_video.read(image)){
			currentFrameIndex++;	
			//string filename=boost::lexical_cast<string>(currentFrameIndex)+".bmp";
			//index_image=imread(filepath+filename);
			imshow("video",image);
			Mat temp = user->userVB->loadContorsOfResultFrameFromFile(currentFrameIndex, image.rows, image.cols, loopuptable);
			imshow("temp", temp);
			waitKey(10);
			//waitKey(30);
			//imshow("index", index_image);
			//setMouseCallback("video",mouseRecover);
			//int key = waitKey(30); 
			//if(key==27)
			//	waitKey(0);
		}
	}
	else if(test==4){
		string t1=inputpath,t2=midname,t3=videoname,temp;
		int frCount;
		if(readlog)
			frCount=readFrameLog(log_path+"FrameLog.txt");
		else
			frCount=frameCount;
		user->UserGetKeyFrame(keyframe_path+t3+"/",frCount);
	}
	else{
		//check or debug 
	}
	end_time=time(NULL);
	cout<<testVideoName<<"\t"<<state<<"\t"<<"video abstraction time: "<<end_time-start_time<<" s"<<endl;
	ff<<testVideoName<<"\t"<<state<<"\t"<<"video abstraction time: "<<end_time-start_time<<endl;
	ff.close();
}


int main(){
	cout<<"***********************************************************************************"<<endl;
	cout<<"\t"<<"Using Guidance "<<endl;
	cout<<"\t"<<"Please input 1 / 2 / 3 / 4"<<endl;
	cout<<"\t"<<"1:   subtract the background and foreground of the input video"<<endl;
	cout<<"\t"<<"2:   compound the convex point sequence to produce the abstracted video"<<endl;
	cout<<"\t"<<"3:   you can replay the seleted object's event full process"<<endl;
	cout<<"\t"<<"4:   you can view 9 snip-shots of the original video"<<endl;
	cout<<"\t"<<"others:   Exit !"<<endl;
	cout<<"*********************************************************************************"<<endl;
	string result_name="result_test.avi";
	string config_name="config_test";
	//boost::thread test1(testmultithread,"test/", "test.avi", config_name, result_name, 1025, 8, 2, 3);

	int choice;
	bool flag=true;
	while(flag){
		cout<<"Please input the choice No. : ";
		cin>>choice;
		if(choice== 1 || choice == 2 || choice == 3 || choice == 4){
			//boost::thread test1(testmultithread,"F:/TongHaoTest2/", "testvideo.avi", config_name, result_name, 0, 8, 1, choice, true);
			boost::thread test1(testmultithread,"F:/TongHaoTest2/", "大门口.avi", config_name, result_name, 0, 8, 1, choice, true);
			test1.join();
			cout<<"finish step "<<choice<<endl;
		}
		else
			return 0;
	}
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

void create_path(string path){
	fstream testfile;
	testfile.open(path, ios::in);
	if(!testfile){
		boost::filesystem::path dir(path);
		boost::filesystem::create_directories(dir);
	}
}

string int2string(int _Val){
	char _Buf[100];
	sprintf(_Buf, "%d", _Val);
	return (string(_Buf));
}