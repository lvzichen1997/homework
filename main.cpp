
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <iostream>
#include <time.h>
#include <omp.h>
#include <ctype.h>
#include <stdio.h>
#include <string>

#define DATESET_COUNT 2
#define METHOD_COUNT 3
using namespace cv;
using namespace std;
using namespace xfeatures2d;
 
int main()
{
    vector<string> strDateset(DATESET_COUNT);
    strDateset[0] = "image1";strDateset[1] = "image2";
    vector<string> strMethod(METHOD_COUNT);
    strMethod[0] = "SIFT";strMethod[1]="SURF";;strMethod[2]="ORB";
    ////递归读取目录下全部文件
    Mat descriptors1;  
    std::vector<KeyPoint> keypoints1;
    Mat descriptors2;
    std::vector<KeyPoint> keypoints2;
    std::vector< DMatch > matches;
    std::vector< DMatch > good_matches;
    std::vector< DMatch > best_matches;
    ////用于模型验算
    int innersize = 0;
    Mat img1;
    Mat imgn;
    
    std::cout<<"ＳＩＦＴ、ＳＵＲＦ、ＯＲＢ"<<endl;
    //遍历各种特征点寻找方法
    for (int imethod=0;imethod<METHOD_COUNT;imethod++)
    {
 
        string _strMethod = strMethod[imethod];
        std::cout<<"开始测试"<<_strMethod<<"方法"<<endl;
        //遍历各个路径
        for (int idateset = 1;idateset<=DATESET_COUNT;idateset++)
        {
            //获得测试图片绝对地址
            string path = "/home/bing/projects/CompKeypointExtract/images/image";
	    string resPath = "/home/bing/projects/CompKeypointExtract/images/oResult";
//             std::cout<<"数据集为"<<strDateset[idateset];
            //获得当个数据集中的图片
//             std::cout<<" 共"<<files.size()<<"张图片"<<endl;
                //使用img1对比余下的图片，得出结果  
	    string imageName1=path+to_string(idateset)+".png";
	    string imageName2=path+to_string(idateset)+"_1.png";
                img1 = imread(imageName1);
                imgn = imread(imageName2);
		
                //生成特征点算法及其匹配方法
                Ptr<Feature2D>  extractor;
                BFMatcher matcher;
		
                switch (imethod)
                {
                case 0: //"SIFT"
                    extractor= SIFT::create(1000);
                    matcher = BFMatcher(NORM_L2);    
                    break;
                case 1: //"SURF"
                    extractor= SURF::create(1000);
                    matcher = BFMatcher(NORM_L2);    
                    break;
                case 2: //"ORB"
                    extractor= ORB::create(1000);
                    matcher = BFMatcher(NORM_HAMMING); 
		    break;
               
                }
                clock_t Tstart, Tend;
                try
                {
		  Tstart=clock();
                    extractor->detectAndCompute(img1,Mat(),keypoints1,descriptors1);
		    Tend=clock();
		    double duration_lcd = (double)(Tend - Tstart) / CLOCKS_PER_SEC;
		    duration_lcd=duration_lcd/keypoints1.size()*1000.0;
		    cout<<"时间: "<<duration_lcd<<endl;
                    extractor->detectAndCompute(imgn,Mat(),keypoints2,descriptors2);
		    
                    matcher.match( descriptors1, descriptors2, matches );
                }
                catch(const char* msg)
                {
		  cout<<" 特征点提取时发生错误 "<<endl;
		  cerr << msg << endl;
                    continue;
                }
 
                //对特征点进行粗匹配
                double max_dist = 0; 
                double min_dist = 100;
                for( int a = 0; a < matches.size(); a++ )
                {
                    double dist = matches[a].distance;
                    if( dist < min_dist ) min_dist = dist;
                    if( dist > max_dist ) max_dist = dist;
                }
                for( int a = 0; a < matches.size(); a++ )
                { 
                    if( matches[a].distance <= max(2*min_dist, 0.02) )
                        good_matches.push_back( matches[a]); 
                }
                if (good_matches.size()<4)
                {
                    cout<<" 有效特征点数目小于4个，粗匹配失败 "<<endl;
                    continue;
                }
                //通过RANSAC方法，对现有的特征点对进行“提纯”
                std::vector<Point2f> obj;
                std::vector<Point2f> scene;
		std::vector<int> matchIndex;
                for( int a = 0; a < (int)good_matches.size(); a++ )
                {    
                    //分别将两处的good_matches对应的点对压入向量,只需要压入点的信息就可以
                    obj.push_back( keypoints1[good_matches[a].queryIdx ].pt );
                    scene.push_back( keypoints2[good_matches[a].trainIdx ].pt );
                }
                //计算单应矩阵（在calib3d中)
                Mat H ;
                try
                {
                     H = findHomography( obj, scene, CV_RANSAC );
// 		    H = findHomography( obj, scene, CV_RANSAC );
                }
                catch (const char* msg)
                {
		  cerr << msg << endl;
                    cout<<" findHomography失败 "<<endl;
                    continue;
                }
                if (H.rows < 3)
                {
                    cout<<" findHomography失败 "<<endl;
                    continue;
                }
                //计算内点数目
                Mat matObj;
                Mat matScene;
		Mat* pMat=&H;
//                 CvMat* pcvMat = &(CvMat)H;
		CvMat* pcvMat=(CvMat*)pMat;
                const double* Hmodel = pcvMat->data.db;
                double Htmp = Hmodel[6];
                for( int isize = 0; isize < obj.size(); isize++ )
                {
                    double ww = 1./(Hmodel[6]*obj[isize].x + Hmodel[7]*obj[isize].y + 1.);
                    double dx = (Hmodel[0]*obj[isize].x + Hmodel[1]*obj[isize].y + Hmodel[2])*ww - scene[isize].x;
                    double dy = (Hmodel[3]*obj[isize].x + Hmodel[4]*obj[isize].y + Hmodel[5])*ww - scene[isize].y;
                    float err = (float)(dx*dx + dy*dy); //3个像素之内认为是同一个点
                    if (err< 9)
                    {
                        innersize = innersize+1;
			matchIndex.push_back(isize);
                    }
                }
                cout<<"特征点个数: "<<keypoints1.size()<<" "<<keypoints2.size()<<endl;
                cout<<"匹配点个数： "<<good_matches.size()<<endl;
                //打印内点占全部特征点的比率
                cout<<"内点个数： "<<innersize<<endl;
                float ff = (float)innersize / (float)good_matches.size();
                cout<<ff<<endl;
                
                //如果效果较好，则打印出来
		for(int innerIndex=0;innerIndex<innersize;innerIndex++)
		{
		  best_matches.push_back(good_matches[matchIndex[innerIndex]]);
		}
                Mat matTmp;
//                 if (ff == 1.0)
//                 {
                    drawMatches(img1,keypoints1,imgn,keypoints2,good_matches,matTmp);
                    
                    string strResult = resPath+to_string(idateset)+_strMethod+".png";
                    imwrite(strResult,matTmp);
//                 }
                ff = 0;
                innersize = 0;
                matches.clear();
                good_matches.clear(); 
		best_matches.clear(); 
           
        }
    }
    
    return 0;
 
};