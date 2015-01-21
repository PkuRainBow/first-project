#include "util.h"


std::string util::contorsToString(std::vector<std::vector<cv::Point>> &contors){
	std::string re="";
	re+=boost::lexical_cast<std::string>(contors.size());
	re+="\t";
	for(int i=0;i<contors.size();++i)
	{
		re+=boost::lexical_cast<std::string>(contors[i].size());
		re+="\t";
		for(int j=0;j<contors[i].size();j++)
		{
			re+=boost::lexical_cast<std::string>(contors[i][j].x);
			re+="\t";
			re+=boost::lexical_cast<std::string>(contors[i][j].y);
			re+="\t";
		}
	}
	re+="\n";
	return re;
}


std::vector<std::vector<cv::Point>> util::stringToContors(std::string ss){
	std::vector<std::vector<cv::Point>> contors;
	int s=0,e=0;
	e=ss.find("\t",s);
	std::string tmp=ss.substr(s,e-s);
	s=e+1;
	int n=boost::lexical_cast<int>(tmp),x,y;
	for(int i=0;i<n;i++)
	{
		std::vector<cv::Point> cur;
		e=ss.find("\t",s);
		tmp=ss.substr(s,e-s);
		s=e+1;
		int nn=boost::lexical_cast<int>(tmp);
		for(int j=0;j<nn;j++)
		{
			e=ss.find("\t",s);
			tmp=ss.substr(s,e-s);
			s=e+1;
			x=boost::lexical_cast<int>(tmp);
			e=ss.find("\t",s);
			tmp=ss.substr(s,e-s);
			s=e+1;
			y=boost::lexical_cast<int>(tmp);
			cur.push_back(cv::Point(x,y));
		}
		contors.push_back(cur);
	}
	return contors;
}



void util::create_path(std::string path){
	std::fstream testfile;
	testfile.open(path, std::ios::in);
	if(!testfile)
	{
		boost::filesystem::path dir(path);
		boost::filesystem::create_directories(dir);
	}
}