//****************************************
//user: PianoCoder
//Create date:
//Class name: VideoAbstraction
//Discription:  implement the background/foreground subtraction and the video 255ing
//Update: 2014/01/07
//****************************************
#include "VideoAbstraction.h"
//
VideoAbstraction::VideoAbstraction(string inputpath, string out_path, string log_path, string config_path, string index_path, string videoname, string midname, float size){
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

VideoAbstraction::VideoAbstraction(){
	init();
}

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
	maxLengthToSpilt=300;
	sum=0;
	thres=0.001;
	currentLength=0;
	tempLength=0;
	noObjectCount=0;
	flag=false;
	useROI=false;
}
//
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
	vector<Mat>().swap(compoundResult);
	vector<Mat>().swap(indexs);
	vector<Mat>().swap(indexe);
	vector<int>().swap(frame_start);
	vector<int>().swap(frame_end); 
}

string VideoAbstraction::int2string(int _Val){
	char _Buf[100];
	sprintf(_Buf, "%d", _Val);
	return (string(_Buf));
}

void VideoAbstraction::postProc(Mat& frame){
	blur(frame,frame,Size(25,25)); //用于去除噪声 平滑图像 blur（inputArray, outputArray, Size）
	threshold(frame,frame,100,255,THRESH_BINARY);	//对于数组元素进行固定阈值的操作  参数列表：(输入图像，目标图像，阈值，最大的二值value--8对应255, threshold类型)
	dilate(frame,frame,Mat());// 用于膨胀图像 参数列表：(输入图像，目标图像，用于膨胀的结构元素---若为null-则使用3*3的结构元素，膨胀的次数)
}


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

Mat vectorToMat(vector<bool> &input,int row,int col){
	Mat re(row,col,CV_8U,Scalar::all(0));
	int step=re.step,step1=re.elemSize();
	for(int i=0;i<input.size();++i)
	{
		*(re.data+i*step1)=(input[i]?255:0);
	}
	return re;
}

void VideoAbstraction::stitch(Mat& conflictMask, Mat &input1,Mat &input2,Mat &output,Mat &back,Mat &mask,int start,int end, int frameno){
//void VideoAbstraction::stitch(Mat &input1,Mat &input2,Mat &output,Mat &back,Mat &mask,int start,int end, vector<vector<Point>>& re_contours, bool& flag){
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
	findContours(mask,m_contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
	putTextToMat(start, end, output, m_contours);
}

int VideoAbstraction::ComponentLable(Mat& fg_mask, vector<Rect>& vComponents_out, int area_threshold)
{
	const double	MAXASPECTRATIO =2;
	const double	MINASPECTRATIO =0.5;
	const double	FILLRATIO	=0.4;
	const int       DISTBTWNPART =81;
	//const int       MINRECTSIZE = hog_des.winSize.height * hog_des.winSize.width;
	const int		AREATHRESHOLD = area_threshold;
	double similar_rects_eps = 0.7;

	const int xdir[8]={-1,0,0,1,1,1,-1,-1};
	const int ydir[8]={0,-1,1,0,1,-1,-1,1};
	const int HEIGHT = fg_mask.rows;
	const int WIDTH = fg_mask.cols;

#define p_lable(x,y)    (p_lable[(y)*(WIDTH)+(x)])

	int x,y,m;
	int cur_pos=0;//start from 0
	int tail_pos=0;
	int cur_x, cur_y;
	int left_pos, right_pos, up_pos, bottom_pos;
	//CvRect rect;
	int lable_count=0;
	int* p_x=NULL;
	int* p_y=NULL;
	int* p_lable=NULL;
	//	int RectSize ;
	//	double FillRatio ;
	//	double AspectRatio ;
	int rWidth;
	int rHeight;

	p_x = new int[HEIGHT *WIDTH ];
	p_y = new int[HEIGHT*WIDTH ];
	p_lable = new int[HEIGHT*WIDTH ];

	memset(p_lable,0,WIDTH*HEIGHT*sizeof(int));
	for(y=0;y<HEIGHT; y++)//y
	{
		for(x=0;x<WIDTH;x++)//x
		{
			if( p_lable(x,y)!=0 || fg_mask.at<uchar>(y,x) != 255 ) //注意只认可255为前景
				continue;
			lable_count++;	//begin a new component
			p_lable(x,y) = lable_count;
			cur_pos = 0;
			tail_pos = 0;
			p_x[tail_pos] = x;
			p_y[tail_pos] = y;
			tail_pos++;
			left_pos = x; right_pos = x;
			up_pos = y; bottom_pos = y;

			while(cur_pos!=tail_pos)
			{
				cur_x = p_x[cur_pos];
				cur_y = p_y[cur_pos];
				cur_pos++;
				for(m=0; m<8; m++)
				{
					if( (cur_y+ydir[m])>=0 && (cur_y+ydir[m])<HEIGHT &&
						(cur_x+xdir[m])>=0 && (cur_x+xdir[m])<WIDTH &&
						fg_mask.at<uchar>(cur_y+ydir[m],cur_x+xdir[m])!=0 &&
						p_lable(cur_x+xdir[m],cur_y+ydir[m])==0 )
					{
						p_x[tail_pos] = cur_x+xdir[m];
						p_y[tail_pos] = cur_y+ydir[m];
						tail_pos++;
						p_lable(cur_x+xdir[m], cur_y+ydir[m]) = lable_count;

						//更新巨型框的坐标（topLeft,bottomRight）
						if(xdir[m]==1 && cur_x+1 > right_pos )
							right_pos = cur_x+1;
						if(xdir[m]==-1 && cur_x-1 < left_pos )
							left_pos = cur_x-1;
						if(ydir[m]==1 && cur_y+1 > bottom_pos)
							bottom_pos = cur_y+1;
						if(ydir[m]==-1 && cur_y-1 < up_pos)
							up_pos = cur_y -1;							
					}
				}
			}

			rWidth=CV_IABS((right_pos-left_pos));
			rHeight=CV_IABS(bottom_pos-up_pos);
			//RectSize =  rWidth*rHeight;
			//FillRatio = (double)cur_pos/(double)RectSize;
			//AspectRatio = (double)rWidth/(double)rHeight;

			if (
				cur_pos<AREATHRESHOLD /*|| 
									  AspectRatio>MAXASPECTRATIO || 
									  AspectRatio<MINASPECTRATIO ||
									  FillRatio<FILLRATIO*/
									  )
			{
				lable_count--;
				for (int i=0;i<tail_pos;i++)
				{
					fg_mask.at<uchar>(p_y[i],p_x[i])=0;
				}
			}
			else
			{
				//Rect r(left_pos,up_pos,rWidth,rHeight);
				Rect r;
				r.x = max(left_pos-cvRound(rWidth * 0.2) ,0);
				r.width = min(cvRound(rWidth * 1.2), WIDTH);
				r.y = max(up_pos-cvRound(rHeight * 0.2), 0);
				r.height = min(cvRound(rHeight * 1.2), HEIGHT);

				if (r.width < 64)
				{
					r.width = 64; 
				}
				if (r.height < 128)
				{
					r.height = 128;
				}
				if (r.x+r.width > WIDTH)
				{
					r.x = WIDTH - r.width;
				}
				if (r.y+r.height > HEIGHT)
				{
					r.y = HEIGHT - r.height;
				}

				bool is_similar_found = false;
				for (vector<Rect>::iterator itr = vComponents_out.begin(); itr != vComponents_out.end(); ++itr)
				{
					if (isSimilarRects(r, *itr, similar_rects_eps))
					{
						is_similar_found = true;
						lable_count--;
						for (int i=0;i<tail_pos;i++)
						{
							fg_mask.at<uchar>(p_y[i],p_x[i])=0;
						}
						break;
					}
				}
				if (!is_similar_found)
				{
					vComponents_out.push_back(r);
				}
			}
		}
	}
	// TODO: 矩形框之间不能有交集，如果有，则合并

	if (p_x!=NULL)
	{
		delete []p_x;
		p_x = NULL;
	}

	if (p_y!=NULL)
	{
		delete []p_y;
		p_y = NULL;
	}

	if (p_lable!=NULL)
	{
		delete []p_lable;
		p_lable = NULL;
	}

	return 0;
}

bool VideoAbstraction::isSimilarRects(const Rect& r1, const Rect& r2, double eps)
{
	return rectsOverlapAreaRate(r1, r2) > eps;
}

double VideoAbstraction::rectsOverlapAreaRate(const Rect& r1, const Rect& r2)
{
	CvRect cr1 = cvRect(r1.x, r1.y, r1.width, r1.height);
	CvRect cr2 = cvRect(r2.x, r2.y, r2.width, r2.height);
	CvRect cr = cvMaxRect(&cr1, &cr2);//返回包含cr1 和 cr2 2个矩阵的最小矩阵信息 然后求解重叠部分的面积
	int w = cr1.width + cr2.width - cr.width;
	int h = cr1.height + cr2.height - cr.height;
	if(w<=0||h<=0)
		return 0;
	else
		return max(double(w*h)/double(cr1.height*cr1.width), double(w * h)/double(cr2.height * cr2.width));
}

double VideoAbstraction::random(double start, double end){
	return start+(end-start)*rand()/(RAND_MAX + 1.0);
}

int VideoAbstraction::computeMaskCollision(Mat &input1,Mat &input2){
	return countNonZero(input1&input2);
}

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


void VideoAbstraction::Abstraction(Mat& currentFrame, int frameIndex){	  //前背景分离函数
	//cout<<objectarea<<"  "<<thres<<endl;
	if(scaleSize > 1)
	{
		//pyrDown(currentFrame, currentFrame, Size(frameWidth,frameHeight));
		resize(currentFrame, currentFrame, Size(frameWidth,frameHeight));
	}
	if(50==frameIndex)								//如果中间文件原来已经存在，则执行清空操作
	{
		ofstream file_flush(Configpath+MidName, ios::trunc);
	}
	if(frameIndex <= 50)
	{										   //初始化混合高斯 取前50帧图像来更新背景信息  提示：取值50仅供参考，并非必须是50
		if(useGpu)
		{
			//gpu module
			gpuFrame.upload(currentFrame);
			gpumog(gpuFrame,gpuForegroundMask,LEARNING_RATE);
			gpumog.getBackgroundImage(gpuBackgroundImg);
			gpuBackgroundImg.download(backgroundImage);
		}
		else
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
		if(frameIndex%2==0)
		{						//更新前背景信息的频率，表示每5帧做一次前背景分离
			if(useGpu)
			{
				//gpu module
				gpuFrame.upload(currentFrame);
				gpumog(gpuFrame,gpuForegroundMask,LEARNING_RATE);
				gpuForegroundMask.download(currentMask);

				//xincoder_start
				xincoder_ConnectedComponents(frameIndex,currentMask,10);
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
			ConnectedComponents(frameIndex,currentMask, objectarea);		//计算当前前景信息中的凸包信息，存储在 currentMask 面积大于objectarea的是有效的运动物体，否则过滤掉 （取值50仅供参考）
			//freopen("exe.txt","a",stdout);
			sum=countNonZero(currentMask);			//计算凸包中非0个数
			if((double)sum/(frameHeight*frameWidth)>thres)
			{							//前景包含的点的个数大于 1000 个 认为是有意义的运动序列（取值1000仅供参考）
				flag=true;
			}
			if(useROI && (double)sum/((rectROI.width*rectROI.height)/(scaleSize*scaleSize))>thres)
			{							//前景包含的点的个数大于 1000 个 认为是有意义的运动序列（取值1000仅供参考）
				//cout<<"points number : "<<sum<<endl;
				flag=true;
			}
		}
		if(flag)
		{							   //判断当前的图像帧是否包含有意义的运动序列信息
			currentObject.objectMask.push_back(matToVector(currentMask));					//将当前帧添加到运动序列中
			if(currentObject.start<0) currentObject.start=frameIndex;
			//can not abandon any object sequence including very long event ...
			if(currentObject.start>0 && frameIndex-currentObject.start>maxLengthToSpilt*10)
			{
				//currentObject.objectMask.clear();
				//currentObject.start=-1;
				//flag=false;
				//noObjectCount=0;
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
						if(currentLength>maxLengthToSpilt*10)
						{								//运动序列的长度太长，是无意义的运动序列，直接丢弃
							//detectedMotion--;
						} 
						//change split number ...
						else if(currentLength>maxLengthToSpilt*8)
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


void VideoAbstraction::loadObjectCube(int& currentIndex){
	partToCompoundNum=0;
	partToCopyNum=0;
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
	*  load 8 sequences into the partToCompound or break if the scene change happens ...
	*/
	int length=0;
	ObjectCube ob;
	vector<vector<Point>> contors;
	bool scene_change=false;
	//cout<<"vector size "<<frame_start.size()<<endl;
	while(partToCompoundNum < motionToCompound && currentIndex < EventNum){
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
}

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
	indexReplay replay(Inputpath+"Replay/"+InputName, Inputpath+"Config/"+MidName);
	int testcount=-1;
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

	int ObjectCount = frame_start.size();				//获取运动序列的个数
	int AverageCount = ObjectCount/motionToCompound;		//每次合成motionToCompound个运动序列的时候，合成的循环的执行次数
	int RemainCount = ObjectCount%motionToCompound;		//多余出来的运动序列的个数

	LOG(INFO)<<"进入摘要视频合成..."<<endl;
	clock_t starttime = clock();
	int currentIndex=0;
	int compoundSeqNumber=1;
	loadIndex=0;
	EventNum=frame_start.size();

	/*
	*  the main loop to fetch 8 event sequences or <8 videos with view change happened until all the event sequences dealt with ...
	*/
	while(currentIndex < EventNum)
	{	
		int synopsis=motionToCompound;
		//bug
		int offset=currentIndex;
		LOG(INFO)<<"*** 第"<<compoundSeqNumber++<<"次 ***"<<endl;
		/*
		* 导入需要进行合并的凸包序列到内存中
		*/
		LOG(INFO)<<"load the object cube to compound to the memory ..."<<endl;
		vector<ObjectCube>().swap(partToCompound);
		vector<ObjectCube>().swap(partToCopy);
		maxLength=0;
		curMaxLength=0;														//正常合成 motionToCompound 个运动序列
		loadObjectCube(currentIndex);	
		/*
		* 计算需要合成的序列的偏移量
		*/
		LOG(INFO)<<"compute the shift array for the object sequences ..."<<endl;
		LOG(INFO)<<"Compound sequences number: "<<partToCompoundNum<<endl;
		synopsis=partToCompoundNum;
		vector<int> shift(synopsis,0);
		computeShift(shift, partToCompound);	
		/*
		* 根据求解出来的偏移量进行合成操作
		*/
		LOG(INFO)<<"start to compound the shifted sequences ..."<<endl;	
		starttime=clock();
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
				findContours(resultMask,re_contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
				putTextToMat(start, end, currentResultFrame, re_contours);
			}
			rectangle(currentResultFrame,Point(rectROI.x/scaleSize,rectROI.y/scaleSize),
				Point((rectROI.x+rectROI.width)/scaleSize,(rectROI.y+rectROI.height)/scaleSize), CV_RGB(0,255,0),2);
			videoWriter.write(currentResultFrame);
			//
			//imshow("compound",currentResultFrame);
			//waitKey(1);
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
					findContours(mat1,re_contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
					putTextToMat(start, end, currentResultFrame, re_contours);
					//
					int resultindex=getObjectIndex(i+offset+partToCompoundNum, j);
					replay.saveEventsParamOfFrameToFile(testcount, (i+offset+partToCompoundNum), resultindex);
					//
					rectangle(currentResultFrame,Point(rectROI.x/scaleSize,rectROI.y/scaleSize),
						Point((rectROI.x+rectROI.width)/scaleSize,(rectROI.y+rectROI.height)/scaleSize),CV_RGB(0,255,0),2);
					testcount++;
					videoWriter.write(currentResultFrame);
					//
					//imshow("compound",currentResultFrame);
					//waitKey(1);
				}
			}
		}
		currentFrame.release();
		currentResultFrame.release();
	}
	videoWriter.release();			//  视频合成结束
	LOG(INFO)<<"合成结束\n";
	LOG(INFO)<<"合成耗时"<<clock()-starttime<<"ms\n";
	LOG(INFO)<<"总长度"<<sumLength<<endl;
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