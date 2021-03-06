﻿//****************************************
//user: PianoCoder
//Create date:
//Class name: VideoAbstraction
//Discription:  implement the background/foreground subtraction and the video 255ing
//Update: 2014/01/07
//****************************************
#include "VideoAbstraction.h"

/*
*  自定义的拷贝构造函数
*/
VideoAbstraction::VideoAbstraction(string inputpath, string out_path, string log_path, string config_path, string index_path, string videoname, string midname, float size):replay(inputpath+"Replay/"+videoname, inputpath+"Config/"+midname){
	init();
	scaleSize=size;
	objectarea=100/(scaleSize*scaleSize);
	thres=0.001;
	useGpu=true;
	Inputpath=inputpath;
	Outpath=out_path;
	Logpath=log_path;
	Configpath=config_path;
	Indexpath=index_path;
	InputName=videoname;
	MidName=midname;
	videoCapture.open(inputpath+videoname);
	frameHeight=videoCapture.get(CV_CAP_PROP_FRAME_HEIGHT)/scaleSize;
	frameWidth=videoCapture.get(CV_CAP_PROP_FRAME_WIDTH)/scaleSize;
	framePerSecond=videoCapture.get(CV_CAP_PROP_FPS);
	useROI=false;
}

/*
*	构造函数调用的init()操作
*/
void VideoAbstraction::init(){
	objectarea=60;
	useGpu=true;
	backgroundSubtractionMethod=1;
	LEARNING_RATE=-1;
	ObjectCubeNumber=0;
	sumLength=0;
	loadIndex=0;
	frame_start.clear();
	frame_end.clear();
	currentObject.start=-1;
	detectedMotion=0;
	cacheShift=50;
	motionToCompound=10;
	maxLength=-1;
	maxLengthToSpilt=1000;
	sum=0;
	thres=0.001;
	currentLength=0;
	tempLength=0;
	noObjectCount=0;
	flag=false;
	useROI=false;
}

/*
*  自定义的强制释放内存的函数
*/
void VideoAbstraction::freeObject(){
	videoCapture.~VideoCapture();
	videoWriter.~VideoWriter();
	backgroundImage.release();				
	currentStartIndex.release();
	currentEndIndex.release();
	mog.~BackgroundSubtractorMOG2();
	gFrame.release();			
	gForegroundMask.release();	
	gBackgroundImg.release();	
	currentMask.release();	
	vector<ObjectCube>().swap(partToCompound);
	vector<ObjectCube>().swap(partToCopy);
	vector<ObjectCube>().swap(tempToCompound);
	vector<ObjectCube>().swap(tempToCopy);
	vector<Mat>().swap(compoundResult);
	vector<Mat>().swap(indexs);
	vector<Mat>().swap(indexe);
	vector<int>().swap(frame_start);
	vector<int>().swap(frame_end); 
}

/*
*  自定义的int转变为string
*/
string VideoAbstraction::int2string(int _Val){
	char _Buf[100];
	sprintf(_Buf, "%d", _Val);
	return (string(_Buf));
}

/*
*  对于传入的图片进行后处理可以消除一些额外的噪声影响
*/
void VideoAbstraction::postProc(Mat& frame){
	blur(frame,frame,Size(25,25)); //用于去除噪声 平滑图像 blur（inputArray, outputArray, Size）
	threshold(frame,frame,100,255,THRESH_BINARY);	//对于数组元素进行固定阈值的操作  参数列表：(输入图像，目标图像，阈值，最大的二值value--8对应255, threshold类型)
	dilate(frame,frame,Mat());// 用于膨胀图像 参数列表：(输入图像，目标图像，用于膨胀的结构元素---若为null-则使用3*3的结构元素，膨胀的次数)
}

/*
*  自定义的预处理，提前过滤掉较小的噪声避免小噪声在后面膨胀操作后连通成一个大的凸包
*/
void xincoder_ConnectedComponents(int frameindex, Mat &mask,int thres){  
	Mat ele(2,4,CV_8U,Scalar(1));
	erode(mask,mask,ele);// 默认时，ele 为 cv::Mat() 形式  参数扩展（image， eroded, structure, cv::Point(-1,-1,), 3） 
	//右侧2个参数分别表示 是从矩阵的中间开始，3表示执行3次同样的腐蚀操作
	dilate(mask,mask,ele);
	vector<vector<Point>> contors,newcontors;
	vector<Point> hull;
	findContours(mask,contors,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE); //找到所有的contour 闭包
	vector<vector<Point>>::const_iterator itc=contors.begin();
	//过滤掉过小的闭包，其他闭包全部存放到 newcontors 中
	while(itc!=contors.end()){
		if(contourArea(*itc)<thres){
			//if(itc->size()<thres){
			itc=contors.erase(itc);
		}
		else{
			convexHull(*itc,hull);
			newcontors.push_back(hull);
			itc++;
		}
	}
	mask=0;
	drawContours(mask,newcontors,-1,Scalar(255),-1); // Scalar(255) 表示对应的背景是全部黑色

	vector<vector<Point>>().swap(contors);
	vector<vector<Point>>().swap(newcontors);
}

/*
*  对于传入的求连通分支
*/
void VideoAbstraction::ConnectedComponents(int frameindex, Mat &mask,int thres){  
	Mat ele(2,4,CV_8U,Scalar(1));
	erode(mask,mask,ele);// 默认时，ele 为 cv::Mat() 形式  参数扩展（image， eroded, structure, cv::Point(-1,-1,), 3） 
	//右侧2个参数分别表示 是从矩阵的中间开始，3表示执行3次同样的腐蚀操作
	dilate(mask,mask,ele);
	vector<vector<Point>> contors,newcontors;
	vector<Point> hull;
	findContours(mask,contors,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE); //找到所有的contour 闭包
	vector<vector<Point>>::const_iterator itc=contors.begin();
	//过滤掉过小的闭包，其他闭包全部存放到 newcontors 中
	while(itc!=contors.end()){
		if(contourArea(*itc)<objectarea){
		//if(itc->size()<thres){
			itc=contors.erase(itc);
		}
		else{
			convexHull(*itc,hull);
			newcontors.push_back(hull);
			itc++;
		}
	}
	mask=0;
	drawContours(mask,newcontors,-1,Scalar(255),-1); // Scalar(255) 表示对应的背景是全部黑色
	//xincoder
	dilate(mask,mask,cv::Mat());
	//xincoder
	vector<vector<Point>>().swap(contors);
	vector<vector<Point>>().swap(newcontors);
}

/*
*   将mat变成vector<bool> 凸包内全部是1 其他全部是0
*/
vector<bool> matToVector(Mat &input){
	int step=input.step,step1=input.elemSize();
	uchar* indata=input.data;
	int row=input.rows,col=input.cols;
	vector<bool> ret;
	for(int i=0;i<row;i++)
	{
		for(int j=0;j<col;j++)
		{
			ret.push_back(*(indata+i*step+j*step1));
		}
	}
	return ret;
}

/*
* 将0,1的vector<bool>变成 255/0 组成的mat
*/
Mat vectorToMat(vector<bool> &input,int row,int col){
	Mat re(row,col,CV_8U,Scalar::all(0));
	int step=re.step,step1=re.elemSize();
	for(int i=0;i<input.size();++i)
	{
		*(re.data+i*step1)=(input[i]?255:0);
	}
	return re;
}

/*
*   将input1 和 input2 根据conflictMask的组成来进行虚化或者直接拼接的处理 back是背景的mask,start-end是起始的帧号
*/
void VideoAbstraction::stitch(Mat& conflictMask, Mat &input1,Mat &input2,Mat &output,Mat &back,Mat &mask,int start,int end, int frameno){
	int step10=input1.step,step11=input1.elemSize();
	int step20=input2.step,step21=input2.elemSize();
	int step30=output.step,step31=output.elemSize();
	int stepb1=back.step,stepb2=back.elemSize();
	int stepm1=mask.step,stepm2=mask.elemSize();
	//stitch
	int stepc1=conflictMask.step,stepc2=conflictMask.elemSize();

	int input1sim,input2sim;
	double alpha;
	uchar* indata1,*indata2,*outdata,*mdata,*bdata,*cdata;
	for(int i=0;i<input1.rows;i++)
	{
		for(int j=0;j<input1.cols;j++)
		{
			mdata=mask.data+i*stepm1+j*stepm2;
			cdata=conflictMask.data+i*stepc1+j*stepc2;
			if((*mdata)!=0)
			{
				indata1=input1.data+i*step10+j*step11;
				indata2=input2.data+i*step20+j*step21;
				outdata=output.data+i*step30+j*step31;
				if((*cdata)!=0)
				{
					bdata=back.data+i*stepb1+j*stepb2;
					input1sim=abs(bdata[0]-indata1[0])+abs(bdata[1]-indata1[1])+abs(bdata[2]-indata1[2])+1;
					input2sim=abs(bdata[0]-indata2[0])+abs(bdata[1]-indata2[1])+abs(bdata[2]-indata2[2])+1;
					alpha=input1sim*1.0/(input1sim+input2sim);
					outdata[0]=int(indata1[0]*alpha+indata2[0]*(1-alpha));
					outdata[1]=int(indata1[1]*alpha+indata2[1]*(1-alpha));
					outdata[2]=int(indata1[2]*alpha+indata2[2]*(1-alpha));
				}
				else
				{
					outdata[0]=(int)indata1[0];
					outdata[1]=(int)indata1[1];
					outdata[2]=(int)indata1[2];
				}
				(*cdata)=255;
			}
		}
	}
	//stitch 
	/*
	* put the time tag on all the convex hull
	*/
	start = start/framePerSecond;
	end = end/framePerSecond;
	vector<vector<Point>> m_contours;
	vector<Point> info(0,0);
	if(useTimeFlag)
	{
		findContours(mask,m_contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
		putTextToMat(start, end, output, m_contours);
	}
}

/*
*	计算2个mat之间的冲突程度
*/
int VideoAbstraction::computeMaskCollision(Mat &input1,Mat &input2){
	return countNonZero(input1&input2);
}

/*
*	计算2个存储为vector格式的mat的冲突程度
*/
int VideoAbstraction::computeMaskCollision(vector<bool> &input1,vector<bool> &input2){
	int ret=0;
	if(input1.size()!=input2.size())
	{
		LOG(ERROR)<<"input vector size do not match\t"<<input1.size()<<"\t"<<input2.size()<<"\n";
		return -1;
	}
	for(int i=0;i<input1.size();++i)
	{
		ret+=(input1[i]&&input2[i]);
	}
	return ret;
}

/*
*	计算2个运动序列的冲突数
*/
int VideoAbstraction::computeObjectCollision(ObjectCube &ob1,ObjectCube &ob2,int shift,string path){
	int collision=0;
	int as,bs;
	if(shift>0){
		as=ob1.start+shift;
		bs=ob2.start;
	}
	else{
		as=ob1.start;
		bs=ob2.start-shift;
	}
	Mat amask,bmask;
	if(path==""){
		for(;as<=ob1.end&&bs<=ob2.end;as++,bs++)
		{
			collision+=computeMaskCollision(ob1.objectMask[as-ob1.start],ob2.objectMask[bs-ob2.start]);
		}
	}else{
		for(;as<=ob1.end&&bs<=ob2.end;as++,bs++)
		{
			amask=imread(path+int2string(as)+".pgm",CV_LOAD_IMAGE_GRAYSCALE);
			bmask=imread(path+int2string(bs)+".pgm",CV_LOAD_IMAGE_GRAYSCALE);
			collision+=computeMaskCollision(amask,bmask);
		}
	}
	return collision;
}

/*
*	核心部分：前背景分离并将分离出来的前景保存成指定格式的中间文件
*/
void VideoAbstraction::Abstraction(Mat& currentFrame, int frameIndex){
	//调整处理的mat的大小
	if(scaleSize > 1)
	{
		resize(currentFrame, currentFrame, Size(frameWidth,frameHeight));
	}
	//如果中间文件原来已经存在，则执行清空操作
	if(10==frameIndex)								
	{
		ofstream file_flush(Configpath+MidName, ios::trunc);
	}
	//初始化混合高斯 取前10帧图像来更新背景信息  提示：取值10仅供参考，并非必须是10 因此我们推荐处理的视频长度控制在1分钟以上
	if(frameIndex <= 10)
	{		
		if(useGpu) //使用GPU
		{
			//gpu module
			gpuFrame.upload(currentFrame);
			gpumog(gpuFrame,gpuForegroundMask,LEARNING_RATE);
			gpumog.getBackgroundImage(gpuBackgroundImg);
			gpuBackgroundImg.download(backgroundImage);
		}
		else     //使用CPU
		{
			currentFrame.copyTo(gFrame);				//复制要处理的图像帧到 gFrame 中
			mog(gFrame,gForegroundMask,LEARNING_RATE);	//更新背景模型并且返回前景信息   参数解释： （下一个视频帧， 输出的前景帧信息， 学习速率）
			mog.getBackgroundImage(gBackgroundImg);		//输出的背景信息存储在 gBackgroundImg
			gBackgroundImg.copyTo(backgroundImage);		//保存背景图片到 backgroundImage 中
			//xincoder_start
			erode(gForegroundMask,gForegroundMask,cv::Mat());
			dilate(gForegroundMask,gForegroundMask,cv::Mat());
			//xincoder_end
		}
		imwrite(InputName+"background.jpg",backgroundImage);
	}
	else
	{										//50帧之后的图像需要正常处理
		if(frameIndex%3==0)
		{						//更新前背景信息的频率，表示每5帧做一次前背景分离
			if(useGpu)
			{
				//gpu module
				gpuFrame.upload(currentFrame);
				gpumog(gpuFrame,gpuForegroundMask,LEARNING_RATE);
				gpuForegroundMask.download(currentMask);
				//xincoder_start
				xincoder_ConnectedComponents(frameIndex,currentMask,10);
				dilate(gForegroundMask,gForegroundMask,cv::Mat());
				//xincoder_end
			}
			else
			{
				currentFrame.copyTo(gFrame);
				mog(gFrame,gForegroundMask,LEARNING_RATE);
				//xincoder
				erode(gForegroundMask,gForegroundMask,cv::Mat());
				dilate(gForegroundMask,gForegroundMask,cv::Mat());
				//xincoder
				gForegroundMask.copyTo(currentMask);		//复制运动的凸包序列到 currentMask 中
			}
			ConnectedComponents(frameIndex,currentMask, objectarea);		//计算当前前景信息中的凸包信息，存储在 currentMask 面积大于objectarea的是有效的运动物体，否则过滤掉
			sum=countNonZero(currentMask);			//计算凸包中非0个数
			if((double)sum/(frameHeight*frameWidth)>thres)
			{							//前景包含的点的个数和原始视频帧大小的比值大于0.001，则令flag=true
				flag=true;
			}
			if(useROI && (double)sum/((rectROI.width*rectROI.height)/(scaleSize*scaleSize))>thres)
			{						  //使用感兴趣区域勾选的时候前景包含的点的个数和原始视频帧勾选区域的大小的比值大于0.001，则令flag=true
				flag=true;
			}
		}
		//处理包含有运动序列的帧
		if(flag)	
		{							   
			currentObject.objectMask.push_back(matToVector(currentMask));					//将当前帧添加到运动序列中
			if(currentObject.start<0) currentObject.start=frameIndex;
			//can not abandon any object sequence including very long event ...
			if(currentObject.start>0 && frameIndex-currentObject.start>maxLengthToSpilt*60)  //maxLengthToSplit ~ 1000frames ~ 40s ~ 60*40s ~ 40min
			{
				currentObject.objectMask.clear();
				currentObject.start=-1;
				flag=false;
				noObjectCount=0;
			}
			if(sum<thres)				   //当前图像中无运动序列
			{
				if(noObjectCount>=15)
				{														//已经有连续15帧无运动序列，运动结束  存储运动序列
					currentObject.end=frameIndex-15;
					if(currentObject.end-currentObject.start>30)
					{								//运动序列长度大于 30 才认为是有效运动，否则不认为其是运动的
						detectedMotion++;
						currentLength=currentObject.end-currentObject.start+1;
						if(currentLength>maxLengthToSpilt*60)
						{								//运动序列的长度太长，是无意义的运动序列，直接丢弃
							detectedMotion--;
						} 
						//change split number ...
						else if(currentLength>maxLengthToSpilt*2)
						{							//事件过长 进行切分处理
							LOG(INFO)<<"事件过长:"<<currentLength<<endl;
							int spilt=currentLength/maxLengthToSpilt+1;
							int spiltLength=currentLength/spilt;
							ObjectCube temp;
							for(int i=0;i<spilt;++i)
							{										//保存切分后的运动序列的信息
								vector<vector<bool>>().swap(temp.objectMask);
								temp.start=currentObject.start+i*spiltLength;
								temp.end=temp.start+spiltLength-1;
								tempLength=spiltLength;
								for(int j=0;j<spiltLength;++j)
								{
									temp.objectMask.push_back(currentObject.objectMask[i*spiltLength+j]);
								}
								/*
							     *  filter the NO ROI 
							     */
								if(useROI)
								{
									bool test = checkROI(temp, roiFilter);
									if(test)
									{
										saveObjectCube(temp);
										maxLength=max(tempLength,maxLength);
										LOG(INFO)<<"事件"<<detectedMotion<<"\t开始帧"<<temp.start<<"\t结束帧"<<temp.end<<"\t长度"<<(temp.end-temp.start)*1.0/framePerSecond<<"秒"<<endl;
										detectedMotion++;
									}
								}
								else
								{
									saveObjectCube(temp);
									maxLength=max(tempLength,maxLength);
									LOG(INFO)<<"事件"<<detectedMotion<<"\t开始帧"<<temp.start<<"\t结束帧"<<temp.end<<"\t长度"<<(temp.end-temp.start)*1.0/framePerSecond<<"秒"<<endl;
									detectedMotion++;
								}
							}
							vector<vector<bool>>().swap(temp.objectMask);
							detectedMotion--;
						}
						else
						{														//事件正常长度，直接添加到运动序列中
							maxLength=max(currentLength,maxLength);
							/*
							*  filter the NO ROI 
							*/
							if(useROI)
							{
								bool test = checkROI(currentObject, roiFilter);
								if(test)
								{
									saveObjectCube(currentObject);
									LOG(INFO)<<"事件"<<detectedMotion<<"\t开始帧"<<currentObject.start<<"\t结束帧"<<currentObject.end<<"\t长度"<<(currentObject.end-currentObject.start)*1.0/framePerSecond<<"秒"<<endl;
								}
							}
							else
							{
								saveObjectCube(currentObject);
								LOG(INFO)<<"事件"<<detectedMotion<<"\t开始帧"<<currentObject.start<<"\t结束帧"<<currentObject.end<<"\t长度"<<(currentObject.end-currentObject.start)*1.0/framePerSecond<<"秒"<<endl;
							}	
						}
					}
					vector<vector<bool>>().swap(currentObject.objectMask);
					currentObject.start=-1;
					flag=false;
					noObjectCount=0;
				}
				else noObjectCount++;
			}
			else
			{
				noObjectCount=0;
				flag=true;
			}
		}
		curMaxLength=maxLength;
	}
}

/*
*	保存运动序列到中间文件中
*/
void VideoAbstraction::saveObjectCube(ObjectCube &ob){			//保存运动的凸包序列的函数
	frame_start.push_back(ob.start);						//保存凸包的开始帧号
	frame_end.push_back(ob.end);							//保存凸包的结束帧号
	ofstream ff(Configpath+MidName, ofstream::app);
	for(int i=ob.start,j=0;i<=ob.end;++i,++j)
	{
		Mat tmp=vectorToMat(ob.objectMask[j],frameHeight,frameWidth);
		vector<vector<Point>> contors;
		findContours(tmp,contors,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE); //提取凸包信息
		ff<<util::contorsToString(contors);
		ff<<'#';	
		ObjectCubeNumber++;
	}
	ff.close();
}

/*
*	保存配置信息到中间文件中
*/
void VideoAbstraction::saveConfigInfo(){						//保存所有凸包运动序列的开始和结束帧信息
	ofstream ff(Configpath+MidName, ofstream::app);
	int size = frame_start.size();
	for(int i=0; i<size; i++)
	{
		ff<<endl;
		ff<<frame_start[i]<<endl;
		ff<<frame_end[i];
	}
	ff.close();
}

/*
*	将中间文件中指定偏移量的导入到contours中
*/
string VideoAbstraction::loadObjectCube(int bias, vector<vector<Point>>& contours){ 
	ifstream file(Configpath+MidName);
	string temp;
	for(int i=0; i<bias; i++) 
	{
		getline(file, temp, '#');
	}
	contours = util::stringToContors(temp);
	return temp;
}

/*
*	将中间文件中指定偏移量的运动序列导入内存中
*/
void VideoAbstraction::loadObjectCube(int& currentIndex){
	vector<ObjectCube>().swap(tempToCompound);
	vector<ObjectCube>().swap(tempToCopy);
	tempToCompoundNum=0;
	tempToCopyNum=0;
	tmaxlength=0;
	ifstream file(Configpath+MidName);
	string temp;
	/*
	*filter the first loadIndex Sequence ...
	*/
	for(int i=0; i<loadIndex; i++) 
	{
		getline(file, temp, '#');
	}
	/*
	*  load 8 sequences into the tempToCompound or break if the scene change happens ...
	*/
	int length=0;
	ObjectCube ob;
	vector<vector<Point>> contors;
	bool scene_change=false;
	//cout<<"vector size "<<frame_start.size()<<endl;
	while(tempToCompoundNum < motionToCompound && currentIndex < EventNum){
		int j=currentIndex++;
		int changeSceneNum=0;
		//cout<<"event no. "<<j<<endl;
		cout<<frame_start[j]<<"\t"<<frame_end[j]<<endl;
		ob.start=frame_start[j];
		ob.end=frame_end[j];
		length=frame_end[j]-frame_start[j]+1;
		loadIndex+=length;		

		for(int i=0;i<length;i++)
		{
			vector<vector<Point>>().swap(contors);
			getline(file, temp, '#');
			contors=util::stringToContors(temp);
			Mat bb(frameHeight,frameWidth,CV_8U,Scalar::all(0));
			drawContours(bb,contors,-1,Scalar(255),-1);
			//check whether the view is changed
			int elecount=countNonZero(bb);
			if((double)elecount/(frameWidth*frameHeight) > 0.5)	changeSceneNum+=1;
			ob.objectMask.push_back(matToVector(bb));	
		}
		vector<vector<Point>>().swap(contors);
		//view change
		if(changeSceneNum > 10)
		{
			cout<<"put the event into the tempToCopy ... "<<endl;
			tempToCopy.push_back(ob);
			vector<vector<bool>>().swap(ob.objectMask);
			tempToCopyNum++;
			break;
		}
		else
		{
			//cout<<"put the event into the tempToCompound ... "<<endl;
			tmaxlength=max(length,tmaxlength);
			//curMaxLength=max(length, curMaxLength);
			tempToCompound.push_back(ob);
			vector<vector<bool>>().swap(ob.objectMask);
			tempToCompoundNum++;
		}
	}
    //LOG(INFO)<<tempToCompoundNum<<endl;
	//
}

/*
*  重载的导入指定起始帧之间的运动序列的 loadObjectCube() 函数
*/
void VideoAbstraction::loadObjectCube(int index_start, int index_end){ //将指定事件序列号范围内的运动帧导入 partToCompound 和 partToCopy 中
	partToCompoundNum=0;
	partToCopyNum=0;
	ifstream file(Configpath+MidName);
	string temp;
	for(int i=0; i<loadIndex; i++) 
	{
		getline(file, temp, '#');
	}
	int length=0;
	ObjectCube ob;
	vector<vector<Point>> contors;
	for(int j=index_start; j<=index_end; j++)
	{
		cout<<frame_start[j]<<"\t"<<frame_end[j]<<endl;
		ob.start=frame_start[j];
		ob.end=frame_end[j];
		length=frame_end[j]-frame_start[j]+1;
		int changeSceneNum=0;
		for(int i=0;i<length;++i)
		{
			vector<vector<Point>>().swap(contors);
			getline(file, temp, '#');
			contors=util::stringToContors(temp);
			Mat bb(frameHeight,frameWidth,CV_8U,Scalar::all(0));
			drawContours(bb,contors,-1,Scalar(255),-1);
			//check whether the view is changed
			int elecount=countNonZero(bb);
			if((double)elecount/(frameWidth*frameHeight) > 0.5)	changeSceneNum+=1;
			ob.objectMask.push_back(matToVector(bb));	
		}
		loadIndex+=length;
		vector<vector<Point>>().swap(contors);
		curMaxLength=max(length,curMaxLength);
		//view change
		if(changeSceneNum > 20)
		{
			cout<<"put the event into the partToCopy ... "<<endl;
			partToCopy.push_back(ob);
			vector<vector<bool>>().swap(ob.objectMask);
			partToCopyNum++;
			break;
		}
		else
		{
			//cout<<"put the event into the partToCompound ... "<<endl;
			partToCompound.push_back(ob);
			vector<vector<bool>>().swap(ob.objectMask);
			partToCompoundNum++;
		}
	}
	file.close();
}

/*
*
*/
void  VideoAbstraction::LoadConfigInfo(){		//不能分阶段处理 -- 读取中间文件中的运动起始信息
	EventNum=0;
	ifstream file(Configpath+MidName);
	string temp;
	for(int i=0; i<ObjectCubeNumber; i++) 
	{		
		getline(file, temp, '#');
	}
	frame_start.clear();
	frame_end.clear();
	while(!file.eof())
	{
		int start,end;
		file>>start;
		file>>end;
		frame_start.push_back(start);
		frame_end.push_back(end);
		EventNum++;
	}
	file.close();
}

/*
*
*/
void  VideoAbstraction::LoadConfigInfo(int frameCountUsed){  //用于分阶段处理 ---  需要传入有效帧的帧数信息
	this->ObjectCubeNumber=frameCountUsed;
	this->EventNum=0;
	ifstream file(Configpath+MidName);
	string temp;
	for(int i=0; i<ObjectCubeNumber; i++) 
	{	
		getline(file, temp, '#');
	}
	frame_start.clear();
	frame_end.clear();
	while(!file.eof())
	{
		int start,end;
		file>>start;
		file>>end;
		frame_start.push_back(start);
		frame_end.push_back(end);
		EventNum++;
	}
	file.close();
}

/*
*
*/
int VideoAbstraction::graphCut(vector<int> &shift,vector<ObjectCube> &ob,int step/* =5 */){  //计算所有运动序列的最佳偏移序列组合

	int n=ob.size(),A,B,C,D,label,collision;

	QPBO<int>* q;

	q = new QPBO<int>(n, n*(n-1)/2); // max number of nodes & edges
	q->AddNode(n); // add nodes

	collision=0;
	int mcache=0;

	clock_t starttime=clock();
	for(int i=0;i<n;i++){
		for(int j=i+1;j<n;j++){
			int diff=(shift[j]-shift[i])/step+cacheShift;
			if(cacheCollision[i][j][diff]>=0){
				A=D=cacheCollision[i][j][diff];
				mcache++;
			}
			else{
				A=D=computeObjectCollision(ob[i],ob[j],shift[j]-shift[i]);
				cacheCollision[i][j][diff]=A;
			}
			if(cacheCollision[i][j][diff+1]>=0){
				B=cacheCollision[i][j][diff+1];
				mcache++;
			}
			else{
				B=computeObjectCollision(ob[i],ob[j],shift[j]-shift[i]+step);
				cacheCollision[i][j][diff+1]=B;
			}
			if(cacheCollision[i][j][diff-1]>=0){
				C=cacheCollision[i][j][diff-1];
				mcache++;
			}
			else{
				C=computeObjectCollision(ob[i],ob[j],shift[j]-shift[i]-step);
				cacheCollision[i][j][diff-1]=C;
			}
			q->AddPairwiseTerm(i, j, A, B, C, D);
			collision+=A;
		}
	}
	printf("hit cache %d times\n",mcache);
	printf("current collision:%d\n",collision);
	q->Solve();
	q->ComputeWeakPersistencies();
	q->Improve();
	bool convergence=true;
	for(int i=0;i<n;i++){
		label=q->GetLabel(i);
		if(label>0){
			convergence=false;
			if(shift[i]+ob[i].end-ob[i].start+1 > curMaxLength){
				for(int j=0;j<n;++j){
					cout<<"shift"<<j<<"\t"<<shift[j]<<"\t"<<ob[j].end-ob[j].start+1<<"\t"<<curMaxLength<<endl;
				}
				return -2;
			}

		}
	}
	if(convergence)return -1;
	for(int i=0;i<n;i++){
		label=q->GetLabel(i);
		if(label>0){
			shift[i]+=step;
		}
	}
	return collision;
}

/*
* 视频合成阶段 参数： 包含有结果文件名字的完整路径 eg. compound("F:/input/Test.avi")
*/
void VideoAbstraction::compound(string path){	
	//清空Replay 记录的信息
	ofstream file_flush(Inputpath+"Replay/"+InputName, ios::trunc);
	//create the output video ...
	Outpath=path;	
	cout<<Outpath<<endl;
	backgroundImage=imread(InputName+"background.jpg");
	int ex = static_cast<int>(videoCapture.get(CV_CAP_PROP_FOURCC));
	videoWriter.open(Outpath, ex, videoCapture.get(CV_CAP_PROP_FPS),cv::Size(frameWidth, frameHeight), true);		
	cout<<Outpath<<endl;
	if (!videoWriter.isOpened())
	{
		LOG(ERROR) <<"Can't create output video file: "<<Outpath<<endl;
		return;
	}
	//running 2 thread to load contours and stitch process ...
	LOG(INFO)<<"进入摘要视频合成..."<<endl;
	clock_t starttime = clock();
	loadIndex=0;
	EventNum=frame_start.size();
	load_compound_finish=false;
	mutex_compound.lock();
	boost::thread run_load(&VideoAbstraction::thread_load, this);
	boost::thread run_compound(&VideoAbstraction::thread_compound, this);
	run_load.join();
	run_compound.join();
	//release the final compounded video ...
	videoWriter.release();	
	LOG(INFO)<<"合成结束\n";
	LOG(INFO)<<"合成耗时"<<clock()-starttime<<"ms\n";
	LOG(INFO)<<"总长度"<<sumLength<<endl;
}

/*
*	读取中间文件的线程控制函数
*/
void VideoAbstraction::thread_load()
{
	int currentIndex=0;
	int loadSeqNumber=1;
	while(currentIndex < EventNum)
	{	
		int temp=currentIndex;
		LOG(INFO)<<"*** 第"<<loadSeqNumber++<<"次 ***"<<endl;
		LOG(INFO)<<"load the object cube to compound to the memory ..."<<endl;
		maxLength=0;
		//
		cout<<"start  load  ......."<<endl;
		loadObjectCube(currentIndex);
		cout<<"end    load  ........"<<endl;
		mutex_load.lock();
		curMaxLength=tmaxlength;
		offset=temp;
		partToCompound.swap(tempToCompound);
		partToCompoundNum=tempToCompoundNum;
		partToCopy.swap(tempToCopy);
		partToCopyNum=tempToCopyNum;
		mutex_compound.unlock();
	}
	load_compound_finish=true;
}

void VideoAbstraction::thread_compound()
{
	int testcount=-1;
	while(!load_compound_finish)
	{
		mutex_compound.lock();
		postCompound(testcount, offset, replay);
		mutex_load.unlock();
	}
}

void VideoAbstraction::postCompound(int& testcount, int offset, indexReplay& replay){
		/*
		* 计算需要合成的序列的偏移量
		*/
		LOG(INFO)<<"compute the shift array for the object sequences ..."<<endl;
		LOG(INFO)<<"Compound sequences number: "<<partToCompoundNum<<endl;
		int synopsis=partToCompoundNum;
		vector<int> shift(synopsis,0);
		computeShift(shift, partToCompound);	
		/*
		* 根据求解出来的偏移量进行合成操作
		*/
		LOG(INFO)<<"start to compound the shifted sequences ..."<<endl;	
		Mat currentFrame;
		Mat currentResultFrame;
		Mat tempFrame;
		for(int i=0;i<synopsis;i++)
		{
			cout<<"shift "<<i+1<<"\t"<<shift[i]<<endl;
		}
		int startCompound=INT_MAX;
		for(int i=0;i<synopsis;i++)
		{
			startCompound=std::min(shift[i],startCompound);
		}
		cout<<"start\t"<<startCompound<<endl;
		cout<<"end\t"<<curMaxLength<<endl;
		sumLength+=(curMaxLength-startCompound);	
		for(int j=startCompound;j<curMaxLength;j++)
		{
			//stitch problem
			Mat accumlatedMask;
			//stitch problem
			testcount++;
			bool haveFrame=false;
			Mat resultMask, tempMask;
			Mat indexMat(Size(frameWidth,frameHeight), CV_8U);
			int earliest=INT_MIN,earliestIndex=-1;
			for(int i=0;i<synopsis;i++)
			{	//寻找序列中开始时间最早的作为背景
				if(shift[i]<=j&&shift[i]+partToCompound[i].end-partToCompound[i].start+1>j)
				{
					if(partToCompound[i].end>earliest)
					{
						earliest=partToCompound[i].end;
						earliestIndex=i;
					}
				}
			}
			if(earliestIndex>-1)
			{
				haveFrame=true;
				videoCapture.set(CV_CAP_PROP_POS_FRAMES,partToCompound[earliestIndex].start-1+j-shift[earliestIndex]);
				videoCapture>>currentFrame;
				resize(currentFrame, currentFrame, Size(frameWidth, frameHeight));
				currentResultFrame=currentFrame.clone();
				resultMask=vectorToMat(partToCompound[earliestIndex].objectMask[j-shift[earliestIndex]],frameHeight,frameWidth);
				//stitch problem
				resultMask.copyTo(accumlatedMask);
				//stitch problem
				int offIndex=j-shift[earliestIndex]+1;
				int resultindex=getObjectIndex(earliestIndex+offset, offIndex);
				//index Video setting ...
				replay.saveEventsParamOfFrameToFile(testcount, earliestIndex+offset, resultindex);
			}
			if(!haveFrame)
			{
				cout<<"没有找到最早\n";
				break;
			}

			for(int i=0;i<synopsis;i++)
			{
				if(i==earliestIndex)
				{
					continue;
				}
				if(shift[i]<=j&&shift[i]+partToCompound[i].end-partToCompound[i].start+1>j)
				{
					videoCapture.set(CV_CAP_PROP_POS_FRAMES,partToCompound[i].start-1+j-shift[i]); //设置背景图片
					videoCapture>>currentFrame;
					//pyrDown(currentFrame, currentFrame, Size(frameWidth, frameHeight));
					resize(currentFrame, currentFrame, Size(frameWidth, frameHeight));
					currentMask=vectorToMat(partToCompound[i].objectMask[j-shift[i]],frameHeight,frameWidth);
					//获取偏移后的正确的位移
					int offIndex=j-shift[i]+1;
					int resultindex=getObjectIndex(i+offset, offIndex);
					//index Video setting ...
					replay.saveEventsParamOfFrameToFile(testcount, (i+offset), resultindex);
					stitch(accumlatedMask, currentFrame,currentResultFrame,currentResultFrame,backgroundImage,currentMask,partToCompound[i].start,partToCompound[i].end, j);
					currentMask.release();
				}
			}
			if(earliestIndex>-1)
			{
				int start = partToCompound[earliestIndex].start/framePerSecond;
				int end = partToCompound[earliestIndex].end/framePerSecond;
				vector<Point> info;
				vector<vector<Point>> re_contours;	
				if(useTimeFlag)
				{
					findContours(resultMask,re_contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
					putTextToMat(start, end, currentResultFrame, re_contours);
				}
			}
			rectangle(currentResultFrame,Point(rectROI.x/scaleSize,rectROI.y/scaleSize),
				Point((rectROI.x+rectROI.width)/scaleSize,(rectROI.y+rectROI.height)/scaleSize), CV_RGB(0,255,0),2);
			
			//
			imshow("compound",currentResultFrame);
			waitKey(1);
			videoWriter.write(currentResultFrame);
			
			
		}
		//deal with the scene change cases ...
		if(partToCopyNum>0)
		{
			cout<<"*************    partToCopy part    ***********"<<endl;
			cout<<"Copy sequences number: "<<partToCopyNum<<endl;
			for(int i=0; i<partToCopyNum; i++)
			{
				int start = partToCopy[i].start/framePerSecond;
				int end = partToCopy[i].end/framePerSecond;
				int length = partToCopy[i].end-partToCopy[i].start+1;
				int base_index = partToCopy[i].start;
				sumLength += length;
				for(int j=0; j<length; j++)
				{
					videoCapture.set(CV_CAP_PROP_POS_FRAMES,j+base_index);
					videoCapture>>currentResultFrame;
					resize(currentResultFrame, currentResultFrame, Size(frameWidth, frameHeight));
					vector<Point> info;
					vector<vector<Point>> re_contours;	
					Mat mat1=vectorToMat(partToCopy[i].objectMask[j],frameHeight,frameWidth);
					if(useTimeFlag)
					{
						findContours(mat1,re_contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
						putTextToMat(start, end, currentResultFrame, re_contours);
					}
					//
					int resultindex=getObjectIndex(i+offset+partToCompoundNum, j);
					replay.saveEventsParamOfFrameToFile(testcount, (i+offset+partToCompoundNum), resultindex);
					//
					rectangle(currentResultFrame,Point(rectROI.x/scaleSize,rectROI.y/scaleSize),
						Point((rectROI.x+rectROI.width)/scaleSize,(rectROI.y+rectROI.height)/scaleSize),CV_RGB(0,255,0),2);
					testcount++;
					//
					imshow("compound",currentResultFrame);
					waitKey(1);

					videoWriter.write(currentResultFrame);
					
				}
			}
		}
		currentFrame.release();
		currentResultFrame.release();
}

void VideoAbstraction::setVideoFormat(string Format){	//保存视频的格式
	videoFormat = Format;
}

/*
* 设置是否使用GPU
*/
void VideoAbstraction::setGpu(bool isgpu){
	useGpu=isgpu;
}

/*
* 设置是否设置感兴趣区域勾选
*/
void VideoAbstraction::setROI(bool isroi){
	useROI=isroi;
}
/*
* 用保存图片格式进行索引的方法对于索引的图片上的值进行赋值操作
*/
void VideoAbstraction::writeMask(Mat& input, Mat& output, int index){
	for(int ii=0; ii<input.rows; ii++)
	{
		const uchar* ptr_input=input.ptr<uchar>(ii);
		uchar* ptr_output=output.ptr<uchar>(ii);
		for(int jj=0; jj<input.cols;jj++){
			if(255==ptr_input[jj])
			{
				ptr_output[jj]=255-index;
			}
		}
	}
}

/*
* 获取所有的关键帧及每个关键事件中的一帧信息保存成图片到路径 /KeyFrames/inputname/*
*/
void VideoAbstraction::getKeyFrame(string keyframe_path){
	util::create_path(keyframe_path);
	VideoCapture videoread;
	videoread.open(Inputpath+InputName);
	Mat keyframes;
	Mat keymask;
	int s1,s2,s3,e1,e2,e3,frame_index;
	vector<vector<Point>> keycontors;
	loadIndex=0;
	for(int i=0; i<EventNum; i++)
	{
		//read the key frames
		frame_index=frame_start[i];
		loadIndex+=frame_index-frame_start[i]+1;
		videoread.set(CV_CAP_PROP_POS_FRAMES, frame_index);
		videoread>>keyframes;
		//resiz
		resize(keyframes, keyframes, Size(frameWidth, frameHeight));
		loadObjectCube(loadIndex, keycontors);
		MarkContours(keyframes, keycontors);
		//put the time tag on the frame
		s1=frame_start[i]/3600;
		s2=(frame_start[i]%3600)/60;
		s3=frame_start[i]%60;
		e1=frame_end[i]/3600;
		e2=(frame_end[i]%3600)/60;
		e3=frame_end[i]%60;
		Point mid(keyframes.cols/10, (keyframes.rows*9)/10);
		putText(keyframes,int2string(s1)+":"+int2string(s2)+":"+int2string(s3)+"-"+int2string(e1)+":"+int2string(e2)+":"+int2string(e3),mid,CV_FONT_HERSHEY_COMPLEX,0.6, Scalar(0,255,0),1);
		string filename=boost::lexical_cast<string>(i)+".jpg";
		imwrite(keyframe_path+filename, keyframes);
		loadIndex-=frame_index-frame_start[i]+1;
		loadIndex+=frame_end[i]-frame_start[i]+1;
	}
}

/*
* 在求解出来的关键帧上标记出来运动的物体
*/
void VideoAbstraction::MarkContours(Mat& mat, vector<vector<Point>>& contours){
	for (int i=0;i<contours.size();i++)
	{
		int x_min=9999999;
		int y_min=9999999;
		int x_max=0;
		int y_max=0;
		for (int j=1;j<contours[i].size();j++)
		{
			if (x_min>contours[i][j].x)
			{
				x_min=contours[i][j].x;
			}
			if (y_min>contours[i][j].y)
			{
				y_min=contours[i][j].y;
			}
			if (x_max<contours[i][j].x)
			{
				x_max=contours[i][j].x;
			}
			if (y_max<contours[i][j].y)
			{
				y_max=contours[i][j].y;
			}
		}
		rectangle(mat,Point(x_min,y_min),Point(x_max,y_max),CV_RGB(0,255,0),2);
	}
}

/*
* 参数：指定的事件序号 事件中的偏移量   返回值： 中间config文件中的偏移量（所有运动序列中的偏移量）
*/
int VideoAbstraction::getObjectIndex(int number, int bias){
	int result=0;
	for(int i=0; i<number; i++){
		result+=(frame_end[i]-frame_start[i]+1);
	}
	result+=bias;
	return result;
}

/*
* 将时间标记写到传入的mat上的所有凸包的中间点部分
*/
void VideoAbstraction::putTextToMat(int start, int end, Mat& mat, vector<vector<Point>>& contours){
	vector<Point> info;
	vector<vector<Point>>::const_iterator itc_re=contours.begin();
	while(itc_re!=contours.end()){
		if(contourArea(*itc_re) < objectarea)
		{
			itc_re=contours.erase(itc_re);
		}
		else
		{
			//xincoder_start
			int min_x=999999;
			int max_x=0;
			int min_y=999999;
			for (int xx=0;xx<itc_re->size();xx++)
			{
				int x=(*itc_re)[xx].x;
				int y=(*itc_re)[xx].y;
				if (max_x<x)
				{
					max_x=x;
				}
				if (min_x>x)
				{
					min_x=x;
				}
				if (min_y>y)
				{
					min_y=y;
				}
			}
			Point mid;
			mid.x=(min_x+max_x)/2;
			mid.y=min_y;
			//xincoder_end

			int s1,s2,s3,e1,e2,e3;
			s1=start/3600;
			s2=(start%3600)/60;
			s3=start%60;
			e1=end/3600;
			e2=(end%3600)/60;
			e3=end%60;
			if(useROI)
			{
				if(roiFilter[mid.y*frameWidth+mid.x])
				{
					putText(mat,int2string(s1)+":"+int2string(s2)+":"+int2string(s3)+"-"+int2string(e1)+":"+int2string(e2)+":"+int2string(e3),mid,CV_FONT_HERSHEY_COMPLEX,0.4, Scalar(0,255,0),1);
				}
			}
			else
			{
				putText(mat,int2string(s1)+":"+int2string(s2)+":"+int2string(s3)+"-"+int2string(e1)+":"+int2string(e2)+":"+int2string(e3),mid,CV_FONT_HERSHEY_COMPLEX,0.4, Scalar(0,255,0),1);
			}
			//putText(mat,int2string(s1)+":"+int2string(s2)+":"+int2string(s3)+"-"+int2string(e1)+":"+int2string(e2)+":"+int2string(e3),mid,CV_FONT_HERSHEY_COMPLEX,0.4, Scalar(0,255,0),1);
			itc_re++;
		}
	}
}

/*
* 计算传入需要合成的若干运动序列的偏移量数组
*/
void VideoAbstraction::computeShift(vector<int>& shift, vector<ObjectCube>& pCompound){
	LOG(INFO)<<"开始计算shift"<<endl;
	int min=INT_MAX,cur_collision=0;						
	clock_t starttime=clock();
	vector<int> tmpshift;
	int *tempptr=(int *)cacheCollision;
	int cache_size=sizeof(cacheCollision)/4;
	for(int i=0;i<cache_size;i++)
	{
		tempptr[i]=-1;
	}		
	for(int randtime=0;randtime<1;++randtime)
	{
		LOG(INFO)<<"生成第"<<randtime+1<<"次初始点\n";
		for(int i=0;i<shift.size();i++) //初始化偏移序列
		{					  
			shift[i]=0;
		}
		while(1)                     //计算满足冲突比较少的所有的偏移序列
		{									  
			cur_collision=graphCut(shift,pCompound);
			LOG(INFO)<<"当前碰撞:"<<cur_collision<<endl;
			if(cur_collision<0) break;
			if(cur_collision<min)
			{
				min=cur_collision;
				tmpshift=shift;
			}
		}
	}
	shift=tmpshift;
	LOG(INFO)<<"最小损失"<<min<<endl;
	LOG(INFO)<<"时间偏移计算耗时"<<clock()-starttime<<"豪秒\n";
}

/*
*  根据感兴趣区域生成的过滤窗口判断传入的事件序列是否出现在感兴趣的区域中
*/
bool VideoAbstraction::checkROI(ObjectCube& ob, vector<bool>& filter)
{
	int size = ob.end-ob.start+1;
	vector<bool> check(ob.objectMask[0].size(), false);
	for(int i=0; i<size; i++)
	{
		for(int j=0; j<check.size(); j++)
		{
			check[j] = check[j] | ob.objectMask[i][j];
		}
	}
	for(int i=0; i<check.size(); i++)
	{
		check[i] = check[i] & filter[i];
	}
	vector<vector<Point>> contours;	
	Mat mat = vectorToMat(check, frameHeight, frameWidth);
	findContours(mat, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector<vector<Point>>::const_iterator itc=contours.begin();
	//过滤掉过小的闭包，其他闭包全部存放到 newcontors 中
	while(itc!=contours.end()){
		if(contourArea(*itc) > objectarea){
			return true;
		}
	}
	return false;
}

/*
*  根据之前设置的感兴趣区域来生成用于过滤感兴趣事件的filter
*/
void  VideoAbstraction::setFilter(vector<bool>& filter, Rect& rec, int size)
{
	filter.clear();
	for(int i=0; i<size; i++)
	{
		filter.push_back(false);
	}
	for(int i=(rec.y/scaleSize); i<(rec.y+rec.height)/scaleSize; i++)
	{
		for(int j=(rec.x/scaleSize); j<(rec.x+rec.width)/scaleSize; j++)
		{
			filter[frameWidth*i+j]=true;
		}
	}
}


void VideoAbstraction::writePartToCompound(vector<ObjectCube>& pCompound){}

void VideoAbstraction::writePartToCopy(vector<ObjectCube>& pCopy){}


