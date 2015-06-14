//#include <OpenNI.h>
#include <iostream>
#include <opencv/highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <FaceTracker/Tracker.h>
#include <math.h>
#include "SRC/utils.hpp"
#include <unistd.h>
#include <fstream>
#include <Python.h>
int _buildarchive()
{

	const string train_sample_list_file= "archive/list_train";
	const size_t n_subject_samples= 2;
	const int sample_width= 80;
	const int sample_height= 100;
	CvSize sample_size= cvSize(sample_width, sample_height);
	const string src_model_file= "../../model/sample.sr";

	vector<string> train_sample_list;

	LoadSampleList(train_sample_list_file, &train_sample_list);
	std::cout<<train_sample_list.size()<<std::endl;
	assert(train_sample_list.size() % n_subject_samples == 0);
	std::cout<<"image file loaded"<<std::endl;

	SRCModel *src= TrainSRCModel(train_sample_list, sample_size, n_subject_samples);

	SaveSRCModel(src, src_model_file);
	ReleaseSRCModel(&src);
	printf("saveing model finished\n");
	return 0;

}

void Draw(cv::Mat &image,cv::Mat &shape,cv::Mat &con,cv::Mat &tri,cv::Mat &visi,std::string& name,std::string & mood)
{
  int i,n = shape.rows/2; cv::Point p1,p2; cv::Scalar c;

  //draw triangulation
  c = CV_RGB(0,0,0);
  for(i = 0; i < tri.rows; i++){
    if(visi.at<int>(tri.at<int>(i,0),0) == 0 ||
       visi.at<int>(tri.at<int>(i,1),0) == 0 ||
       visi.at<int>(tri.at<int>(i,2),0) == 0)continue;
    p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
		   shape.at<double>(tri.at<int>(i,0)+n,0));
    p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
		   shape.at<double>(tri.at<int>(i,1)+n,0));
    cv::line(image,p1,p2,c);
    p1 = cv::Point(shape.at<double>(tri.at<int>(i,0),0),
		   shape.at<double>(tri.at<int>(i,0)+n,0));
    p2 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
		   shape.at<double>(tri.at<int>(i,2)+n,0));
    cv::line(image,p1,p2,c);
    p1 = cv::Point(shape.at<double>(tri.at<int>(i,2),0),
		   shape.at<double>(tri.at<int>(i,2)+n,0));
    p2 = cv::Point(shape.at<double>(tri.at<int>(i,1),0),
		   shape.at<double>(tri.at<int>(i,1)+n,0));
    cv::line(image,p1,p2,c);
  }
  //draw connections
  c = CV_RGB(0,0,255);
  for(i = 0; i < con.cols; i++){
    if(visi.at<int>(con.at<int>(0,i),0) == 0 ||
       visi.at<int>(con.at<int>(1,i),0) == 0)continue;
    p1 = cv::Point(shape.at<double>(con.at<int>(0,i),0),
		   shape.at<double>(con.at<int>(0,i)+n,0));
    p2 = cv::Point(shape.at<double>(con.at<int>(1,i),0),
		   shape.at<double>(con.at<int>(1,i)+n,0));
    cv::line(image,p1,p2,c,1);
  }
  //draw points
  for(i = 0; i < n; i++){
    if(visi.at<int>(i,0) == 0)continue;
    p1 = cv::Point(shape.at<double>(i,0),shape.at<double>(i+n,0));
    c = CV_RGB(255,255,0); cv::circle(image,p1,2,c);
	/*char *text = new char[10];
	sprintf(text,"%d",i);
		cv::putText(image,text,p1,
		CV_FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(255,255,255));
		delete []text;
	*/	
    }return;
}

cv::Point initial[100];


double GetFeature(cv::Mat &image,cv::Mat &shape,cv::Mat &con,cv::Mat &tri,cv::Mat &visi,std::string& name,bool first, vector<int>& features)
{
	bool finish = false;
	int i,n = shape.rows/2; cv::Point p1,p2; cv::Scalar c;
	
	cv::Point p[100];
  //draw triangulation
  c = CV_RGB(0,0,0);
  //draw points
  double energy = 0;
  features.clear();
  for(i = 0; i < n; i++){
	
    if(visi.at<int>(i,0) == 0)continue;
    p[i] = cv::Point(shape.at<double>(i,0),shape.at<double>(i+n,0));
	//if (first) initial[i] = p[i];
    c = CV_RGB(255,255,0); //cv::circle(image,p[i],2,c);
	//energy += (p[i].x-initial[i].x)*(p[i].x-initial[i].x) + (p[i].y-initial[i].y)*(p[i].y-initial[i].y);
	//features.push_back(p[i].x-initial[i].x);
	//features.push_back(p[i].y-initial[i].y);	
	}
	
	//for (int i = 0 ;i<features.size(); i++)
	//	cout<<"in_feature"<<features[i]<<endl;
	
	cout<<"energy:"<<energy<<endl;

  cv::Point left,right,bottom,leftbrow,rightbrow,nose,mouth;
  for (int i = 36;i<42;i++){
	  left.x+=p[i].x;
	  left.y+=p[i].y;
  }
  left.x/=6;
  left.y/=6;

  for (int i = 42;i<48;i++){
	  right.x+=p[i].x;
	  right.y+=p[i].y;
  }
  right.x/=6;
  right.y/=6;

  for (int i = 7;i<10;i++){
	  bottom.x+=p[i].x;
	  bottom.y+=p[i].y;
  }
  bottom.x/=3;
  bottom.y/=3;

  for (int i = 18;i<21;i++){
	  leftbrow.x+=p[i].x;
	  leftbrow.y+=p[i].y;
  }
  leftbrow.x/=3;
  leftbrow.y/=3;

  for (int i = 23;i<26;i++){
	  rightbrow.x+=p[i].x;
	  rightbrow.y+=p[i].y;
  }
  rightbrow.x/=3;
  rightbrow.y/=3;

  for (int i = 31;i<36;i++){
	  nose.x+=p[i].x;
	  nose.y+=p[i].y;
  }
  nose.x/=5;
  nose.y/=5;
  
  for (int i = 50;i<53;i++){
	  mouth.x+=p[i].x;
	  mouth.y+=p[i].y;
  }
  for (int i = 60;i<63;i++){
	  mouth.x+=p[i].x;
	  mouth.y+=p[i].y;
  }
  mouth.x /= 6;
  mouth.y /= 6;
  
  
  cv::circle(image,left,4,c);
  cv::circle(image,right,4,c);
  cv::circle(image,mouth,4,c);
	
	
  cv::Point2f srcTri[3];
  cv::Point2f dstTri[3];

  cv::Mat warp_mat( 2, 3, CV_32FC1 );
  cv::Mat rot_mat( 2, 3, CV_32FC1 );
  cv::Mat  warp_dst, warp_rotate_dst;
  warp_dst = cv::Mat::zeros( image.rows, image.cols, image.type() );
  srcTri[0] = left;
  srcTri[1] = right;
  srcTri[2] = mouth;
  
  int col = 480;
  int row = 480;

  dstTri[0] = cv::Point2f( col*0.4, row*0.5 );
  dstTri[1] = cv::Point2f( col*0.6, row*0.5 );
  dstTri[2] = cv::Point2f( col*0.5, row*0.73 );
  
  warp_mat = getAffineTransform( srcTri, dstTri );
  
  
  std::vector<cv::Point2f> fpoints;
  for( int i = 0; i < 66; i++ )
      fpoints.push_back(cv::Point2f(p[i].x,p[i].y));

  cv::transform(fpoints,fpoints,warp_mat);  
  
  
  for (int i = 0; i < 66; i++)
  {
	  if (first) initial[i] = fpoints[i];
  	  energy += (fpoints[i].x-initial[i].x)*(fpoints[i].x-initial[i].x) + (fpoints[i].y-initial[i].y)*(fpoints[i].y-initial[i].y);
  	  features.push_back(fpoints[i].x-initial[i].x);
  	  features.push_back(fpoints[i].y-initial[i].y);	
  
  }
  

     /// Apply the Affine Transform just found to the src image
     warpAffine( image, warp_dst, warp_mat, warp_dst.size() );

     /** Rotating the image after Warp */

     /// Compute a rotation matrix with respect to the center of the image
	 //cv::imshow("test",warp_dst);
	return energy;
}

void SaveImage(cv::Mat &image,cv::Mat &shape,cv::Mat &con,cv::Mat &tri,cv::Mat &visi,std::string& filename )
{
 	int i,n = shape.rows/2;  cv::Scalar c;
	cv::Point p[100];
  //draw triangulation
  c = CV_RGB(0,0,0);
  //draw points
  for(i = 0; i < n; i++){
    if(visi.at<int>(i,0) == 0)continue;
    p[i] = cv::Point(shape.at<double>(i,0),shape.at<double>(i+n,0));
    c = CV_RGB(255,255,0); //cv::circle(image,p[i],2,c);
	char *text = new char[10];
	sprintf(text,"%d",i);
	//cv::putText(image,text,p[i],
	//CV_FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(255,255,255));
		delete []text;
  }
  cv::Point left,right,bottom,leftbrow,rightbrow,nose;
  for (int i = 36;i<42;i++){
	  left.x+=p[i].x;
	  left.y+=p[i].y;
  }
  left.x/=6;
  left.y/=6;

  for (int i = 42;i<48;i++){
	  right.x+=p[i].x;
	  right.y+=p[i].y;
  }
  right.x/=6;
  right.y/=6;

  for (int i = 7;i<10;i++){
	  bottom.x+=p[i].x;
	  bottom.y+=p[i].y;
  }
  bottom.x/=3;
  bottom.y/=3;

  for (int i = 18;i<21;i++){
	  leftbrow.x+=p[i].x;
	  leftbrow.y+=p[i].y;
  }
  leftbrow.x/=3;
  leftbrow.y/=3;

  for (int i = 23;i<26;i++){
	  rightbrow.x+=p[i].x;
	  rightbrow.y+=p[i].y;
  }
  rightbrow.x/=3;
  rightbrow.y/=3;

  for (int i = 31;i<36;i++){
	  nose.x+=p[i].x;
	  nose.y+=p[i].y;
  }
  nose.x/=5;
  nose.y/=5;


  cv::circle(image,left,4,c);
  cv::circle(image,right,4,c);
  cv::circle(image,bottom,4,c);
  cv::circle(image,leftbrow,4,c);
  cv::circle(image,rightbrow,4,c);
  cv::circle(image,nose,4,c);
  int dist = right.x - left.x;
  //affine
  //cv::Point center = nose;

  cv::Point2f center = cv::Point2f(nose.x, nose.y);  // 旋转中心
  //double angle = 30;  // 旋转角度
  double angle = -atan((double)(right.y-left.y)/(double)(left.x-right.x))/3.1415*180;
  printf("%lf\n",angle);
  double scale = 1.0; // 缩放尺度
  cv::Mat rotateMat;
  rotateMat = cv::getRotationMatrix2D(center, angle, scale);
  cv::Mat rotateImg;
  cv::warpAffine(image, rotateImg, rotateMat, image.size());
  rotateImg=rotateImg(cvRect(nose.x-dist,nose.y-dist*1.3,2*dist,2*dist));
  cv::imshow("rot",rotateImg);

  cv::Point2f srcTri[3];
  cv::Point2f dstTri[3];

  cv::Mat rot_mat( 2, 3, CV_32FC1 );
  cv::Mat warp_mat( 2, 3, CV_32FC1 );
  cv::Mat  warp_dst, warp_rotate_dst;
  warp_dst = cv::Mat::zeros( image.rows, image.cols, image.type() );
  srcTri[0] = left;
  srcTri[1] = right;
  srcTri[2] = bottom;

  dstTri[0] = cv::Point2f( image.cols*0.45, image.rows*0.5 );
  dstTri[1] = cv::Point2f( image.cols*0.55, image.rows*0.5 );
  dstTri[2] = cv::Point2f( image.cols*0.5, image.rows*0.85 );

  warp_mat = getAffineTransform( srcTri, dstTri );

  warpAffine( image, warp_dst, warp_mat, warp_dst.size() );
  cv::Rect region_of_interest = cv::Rect(dstTri[0].x-100,dstTri[0].y-100,dstTri[1].x-dstTri[0].x+200,dstTri[2].y-dstTri[0].y+100);
  cv::Mat image_roi = warp_dst(region_of_interest);

  cv::Mat image_roi_equalization;
  equalizeHist(image_roi,image_roi_equalization);
  imshow( "test", image_roi );
  imshow("test_equ",image_roi_equalization);
  system("mkdir archive");
  imwrite(filename.c_str(),image_roi_equalization);
  return;
}


int main(int argc,  char** argv)
{
	string mood = "unknown";
	
	const char *scriptDirectoryName = "/Users/xiafei/Desktop/FaceTracker/build/bin";
	    Py_Initialize();
	    PyObject *sysPath = PySys_GetObject("path");
	    PyObject *path = PyString_FromString(scriptDirectoryName);
	    int result = PyList_Insert(sysPath, 0, path);
	    PyObject *pModule = PyImport_ImportModule("model");
		if (PyErr_Occurred()) PyErr_Print();
		printf("%p\n", pModule);
		PyObject* pDict = PyModule_GetDict(pModule);
		PyObject* pValue = NULL;

		    // pFunc is also a borrowed reference 
		    PyObject* pFunc = PyDict_GetItemString(pDict, "hello_world");

		    if (PyCallable_Check(pFunc)) 
		    {
		        PyObject_CallObject(pFunc, NULL);
		    } else 
		    {
		        PyErr_Print();
		    }	
			
			
			PyObject* pClass = PyDict_GetItemString(pDict, "Model");
			
			PyObject* pInstance = NULL;
			
		    if (PyCallable_Check(pClass))
		       {
		           pInstance = PyObject_CallObject(pClass, NULL); 
		       }
			   else
			   {
				   PyErr_Print();
			   }
   			PyObject_CallMethod(pInstance, "load_data", NULL);
			PyObject_CallMethod(pInstance, "train_model", NULL);
			PyObject_CallMethod(pInstance, "test_performance", NULL);
		    PyObject_CallMethod(pInstance, "fuck", "[i,i]", 123, 456 );
			

	char *inputvideo = NULL;
	char * outfile = NULL;
	int c;
	bool cam = false;
	while ((c = getopt (argc, argv, "i:o:c")) != -1)
	    switch (c)
	      {
	      case 'i':
	        inputvideo = optarg;
	        break;
  	      case 'o':
  	        outfile = optarg;
  	        break;
		 case 'c':
		 cam = true;
		 break;
	      case '?':
	        if (optopt == 'i')
	          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
	        else if (isprint (optopt))
	          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
	        else
	          fprintf (stderr,
	                   "Unknown option character `\\x%x'.\n",
	                   optopt);
	        return 1;
	      default:
	        abort ();
	      }

	/*openni::Status rc = openni::STATUS_OK;
	openni::Device device;
	openni::VideoStream depth, color;
	const char* deviceURI = openni::ANY_DEVICE;
	if (argc > 1)
	{
		deviceURI = argv[1];
	}

	rc = openni::OpenNI::initialize();

	printf("Hello world, After initialization:\n%s\n", openni::OpenNI::getExtendedError());

	rc = device.open(deviceURI);
	if (rc != openni::STATUS_OK)
	{
		printf("SimpleViewer: Device open failed:\n%s\n", openni::OpenNI::getExtendedError());
		openni::OpenNI::shutdown();
		return 1;
	}

	rc = depth.create(device, openni::SENSOR_DEPTH);
	if (rc == openni::STATUS_OK)
	{
		rc = depth.start();
		if (rc != openni::STATUS_OK)
		{
			printf("SimpleViewer: Couldn't start depth stream:\n%s\n", openni::OpenNI::getExtendedError());
			depth.destroy();
		}
	}
	else
	{
		printf("SimpleViewer: Couldn't find depth stream:\n%s\n", openni::OpenNI::getExtendedError());
	}

	rc = color.create(device, openni::SENSOR_COLOR);
	if (rc == openni::STATUS_OK)
	{
		rc = color.start();
		if (rc != openni::STATUS_OK)
		{
			printf("SimpleViewer: Couldn't start color stream:\n%s\n", openni::OpenNI::getExtendedError());
			color.destroy();
		}
	}
	else
	{
		printf("SimpleViewer: Couldn't find color stream:\n%s\n", openni::OpenNI::getExtendedError());
	}

	if (!depth.isValid() || !color.isValid())
	{
		printf("SimpleViewer: No valid streams. Exiting\n");
		openni::OpenNI::shutdown();
		return 2;
	}

	std::cout<<"colorvalid"<<color.isValid()<<std::endl;

	*/

	const string src_model_file= "../../model/sample.sr";
	const double sci_t=0.2;
	SRCModel *src_model= LoadSRCModel(src_model_file);

	int archiveexist = 0;
	int guest = 0;
    int buildarchive = 1;
	std::vector<int> pnum(5);
	for (int i = 0;i<5;i ++) pnum[i] = 0;

    //CvCapture* camera = cvCreateCameraCapture(CV_CAP_ANY); if(!camera){std::cout<<"cannot access camera"<<std::endl;return -1;}
    cv::VideoCapture cap;
	string videofile = string(inputvideo);
	if (cam) cap.open(0);
	else cap.open(videofile);
    //cap.open("smile.mp4");
	if (!cap.isOpened()) return -1;
    int64 t1,t0 = cvGetTickCount(); int fnum=0; double fps=0;bool show = true;char sss[256]; std::string text; //about drawing frame rate
	 bool fcheck = false; double scale = 1; int fpd = -1;
	 std::vector<int> wSize1(1); wSize1[0] = 7;
	 std::vector<int> wSize2(3); wSize2[0] = 11; wSize2[1] = 9; wSize2[2] = 7;
	  int nIter = 5; double clamp=3,fTol=0.01;

	cvNamedWindow("Face Recognizer");
	cvWaitKey(10);
	cv::Mat frame,im,gray;
	char ftFile[256],conFile[256],triFile[256];
 	strcpy(ftFile,"../../model/face2.tracker");
	std::strcpy(triFile,"../../model/face.tri");
	std::strcpy(conFile,"../../model/face.con");

	FACETRACKER::Tracker model(ftFile);
	cv::Mat tri=FACETRACKER::IO::LoadTri(triFile);
	cv::Mat con=FACETRACKER::IO::LoadCon(conFile);

	FACETRACKER::FDet detector;
	detector.Load("../../model/haarCascade.dat");
	std::vector<cv::Rect*> RectArr;
	bool findface = false;
	bool failed = true;
	std::string name = "unknown";
	int recnum = 0;
	int framenum = 0;
	bool firstframe = true;
    bool lastframe = false;
	cv::Mat nextframe;
	cap >> nextframe;
    std::vector<int> features;
	
	while (1)
	{
		 /*IplImage* I = cvQueryFrame(camera);//query image from camera
		 if (!I) {
			 continue;
		 }
		 frame = I;
		 */
		 //if (framenum++%16 == 0) cap = cv::VideoCapture("smile/S127_004_000000%02d.png.jpeg");
         frame = nextframe;
         im = frame;
		 
		 cap >> nextframe;
		 if (nextframe.empty()) break;
		 while (nextframe.empty()) {
			 cap = cv::VideoCapture(videofile);
	         cap >> nextframe;
	         frame = nextframe;
			 im = frame;
		 }
		 
		 cv::flip(im,im,1);
		 //insert my code here
		 cv::cvtColor(im,gray,CV_BGR2GRAY);//flip and change to grayscale
		if (!findface) {findface = detector.Detect(gray,RectArr);
		std::cout<<"x:"<<RectArr[0]->x<<"y:"<<RectArr[0]->y<<"width:"<<RectArr[0]->width<<"height:"<<RectArr[0]->height<<std::endl;
		std::cout<<"facenum"<<RectArr.size()<<std::endl;
		}

		//sort
		for (int i = 0;i<RectArr.size();i++)
			for (int j=0;j<i;j++)
		{
			if (RectArr[i]->width*RectArr[i]->height>RectArr[j]->width*RectArr[j]->height)
			{
				cv::Rect* tmp = RectArr[i];
				RectArr[i] = RectArr[j];
				RectArr[j] = tmp;
			}
		}

	    for (int i=0;i<RectArr.size();i++)
		{
			rectangle(im,cv::Point(RectArr[i]->x,RectArr[i]->y),cv::Point(RectArr[i]->x+RectArr[i]->width,RectArr[i]->y+RectArr[i]->height),cv::Scalar(255,0,0));
		}

		//std::cout<<findface<<std::endl;
	    //track the image
	    std::vector<int> wSize; if(failed)wSize = wSize2; else wSize = wSize1;
		double energy = 0;
		
		if(model.Track(gray,wSize,cv::Rect(cv::Point(RectArr[0]->x,RectArr[0]->y),cv::Point(RectArr[0]->x+RectArr[0]->width,RectArr[0]->y+RectArr[0]->height)),fpd,nIter,clamp,fTol,fcheck) == 0){
	         int idx = model._clm.GetViewIdx(); failed = false;
	         //im.setTo(cv::Scalar(255,255,255));
			 Draw(im,model._shape,con,tri,model._clm._visi[idx],name,mood);
			energy = GetFeature(im,model._shape,con,tri,model._clm._visi[idx],name,firstframe,features);
			firstframe = false;
	       }
		   else
		   {
			   std::cout<<"track failed"<<std::endl;
			   if(show){cv::Mat R(im,cvRect(0,0,150,50)); R = cv::Scalar(0,0,255);}
			    model.FrameReset(); failed = true;
				firstframe = true;
		   }


		   cout<<energy<<endl;
		   if (energy > 1500)
		   {
		   pValue = PyObject_CallMethod(pInstance, "make_prediction", "[i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i]",features[0],features[1],features[2],features[3],features[4],features[5],features[6],features[7],features[8],features[9],features[10],features[11],features[12],features[13],features[14],features[15],features[16],features[17],features[18],features[19],features[20],features[21],features[22],features[23],features[24],features[25],features[26],features[27],features[28],features[29],features[30],features[31],features[32],features[33],features[34],features[35],features[36],features[37],features[38],features[39],features[40],features[41],features[42],features[43],features[44],features[45],features[46],features[47],features[48],features[49],features[50],features[51],features[52],features[53],features[54],features[55],features[56],features[57],features[58],features[59],features[60],features[61],features[62],features[63],features[64],features[65],features[66],features[67],features[68],features[69],features[70],features[71],features[72],features[73],features[74],features[75],features[76],features[77],features[78],features[79],features[80],features[81],features[82],features[83],features[84],features[85],features[86],features[87],features[88],features[89],features[90],features[91],features[92],features[93],features[94],features[95],features[96],features[97],features[98],features[99],features[100],features[101],features[102],features[103],features[104],features[105],features[106],features[107],features[108],features[109],features[110],features[111],features[112],features[113],features[114],features[115],features[116],features[117],features[118],features[119],features[120],features[121],features[122],features[123],features[124],features[125],features[126],features[127],features[128],features[129],features[130],features[131] );
		   if (pValue != NULL) 
		       {
		           printf("Return of call : %d\n", PyInt_AsLong(pValue));
		           Py_DECREF(pValue);
				   switch (PyInt_AsLong(pValue))
				   {
					   case 1:
					   	mood = "Anger";
					   	break;
 					   case 2:
 					   	mood = "Contempt";
 					   	break;
 					   case 3:
 					   	mood = "Disgust";
 					   	break;
 					   case 4:
 					   	mood = "Fear";
 					   	break;
 					   case 5:
 					   	mood = "Happy";
 					   	break;
 					   case 6:
 					   	mood = "Sadness";
 					   	break;
  					   case 7:
  					   	mood = "Surprise";
  					   	break;
				   }		   
		       }
		       else 
		       {
		           PyErr_Print();
		       }
		   }
		   else
		   {
			   mood = "unknown";
		   }
		   
		   
		   if (energy > 100000)
		   {
			   model.FrameReset();
		    	firstframe = true;
		   }
		   
   

		 //draw the frame rate
	     if(fnum >= 9){
	       t1 = cvGetTickCount();
	       fps = 10.0/((double(t1-t0)/cvGetTickFrequency())/1e+6);
	       t0 = t1; fnum = 0;
	     }else fnum += 1;
	     if(show){
	       sprintf(sss,"%d frames/sec",(int)round(fps)); text = sss;
	       cv::putText(im,text,cv::Point(10,20),
	 		  CV_FONT_HERSHEY_SIMPLEX,0.5,CV_RGB(255,255,255));
	     }



       	  cv::putText(im,mood,cv::Point(50,100),
 		  CV_FONT_HERSHEY_SIMPLEX,2.5,CV_RGB(255,255,255));




		 imshow("Face Recognizer",im);
		 cvNamedWindow("result",CV_WINDOW_AUTOSIZE);
		 cvCreateTrackbar("guest","result",&guest,5);//max six guests
		 cvCreateTrackbar("archiving","result",&buildarchive,1);
		 int c = cvWaitKey(1);
		 if(c == 27)break;
		 if ((c =='w')&&(buildarchive))  {
			 int idx = model._clm.GetViewIdx(); failed = false;
			 std::stringstream ss;
			 std::stringstream ss2;
			 std::string tmp;
			 ss<<guest;
			 ss>>tmp;
			 std::string filename("archive/s");
			 filename+=tmp;
			 filename+="/";
			 std::string tmp2;
			 ss2<<pnum[guest]++;
			 ss2>>tmp2;
			 filename+=tmp2;
			 filename+=".jpg";

			 std::cout<<filename<<std::endl;
			 SaveImage(gray,model._shape,con,tri,model._clm._visi[idx],filename);
		 }
		 if (c =='d')
         {
            findface = false;

         }
        if(c == 'x') {
			model.FrameReset();
		    firstframe = true;
		}
		 if (c=='b') {
			 _buildarchive();
			 buildarchive = 0;
		 }

		 //recognize

		 if ((!buildarchive)&&(c == 'r'))
		 {
			 int idx = model._clm.GetViewIdx(); failed = false;
			 std::stringstream ss2;
			 system("mkdir recognize");
			 std::string filename("recognize/s");
			 std::string tmp2;
			 ss2<<recnum++;
			 ss2>>tmp2;
			 filename+=tmp2;
			 filename+=".jpg";
			 std::cout<<filename<<std::endl;
			 SaveImage(gray,model._shape,con,tri,model._clm._visi[idx],filename);
	 		 CvMat *y= LoadSample(filename, src_model->sample_size_);
			 name= Recognize(src_model, y, sci_t,
	 			(filename+".x").c_str(), (filename+".r").c_str());
			std::cout<<name<<std::endl;
	 }
	}
	
	
	ofstream f;
	f.open(outfile);
	
	for (int i = 0 ;i<features.size(); i++)
		f<<i<<','<<features[i]<<endl;
	f.close();

	printf("Bye\n");
	ReleaseSRCModel(&src_model);
	Py_Finalize();
	return 0;
}

