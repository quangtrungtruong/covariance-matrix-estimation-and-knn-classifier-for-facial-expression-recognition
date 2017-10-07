#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/io/vtk_io.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/bilateral.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/principal_curvatures.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "cv.h" 
#include "highgui.h"
#include "ml.h"
#include <boost/thread/thread.hpp>
#include <vector>
#include <math.h>

#include <XnCppWrapper.h>

using namespace std;

typedef unsigned char       BYTE;
const float depth_limit = 7.0;

/** Global variables */
string face_cascade_name = "haarcascade_frontalface_alt.xml";
string eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
cv::CascadeClassifier face_cascade;
cv::CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
cv::RNG rng(12345);
#define PI 3.14159265

class location_rgbd
{
public:
	float x1, x2, y1, y2;
};

class IntegralImage
{
public:

	cv::Mat _integral;  //integral image
	cv::Mat _sq_integral; //sq integral image
	cv::Mat _image;  //original image
	IntegralImage();

	//function too compute integral image
	void compute(cv::Mat image);
	//function to compute mean value of a patch
	float calcMean(cv::Rect r);

	//function to compute variance of a patch
	float calcVariance(cv::Rect r);

};
IntegralImage::IntegralImage()
{
}

void IntegralImage::compute(cv::Mat image)
{
	image.copyTo(_image);
	cv::integral(_image,_integral,_sq_integral);
}

float IntegralImage::calcMean(cv::Rect r)
{
	int width=_integral.cols;
	int height=_integral.rows;
	unsigned int *ii1 =(unsigned int *)_integral.data;
	int a=r.x+(r.y*width);
	int b=(r.x+r.width)+(r.y*width);
	int c=r.x+((r.y+r.height)*width);
	int d=(r.x+r.width)+(r.y+r.height)*width;
	float mx=ii1[a]+ii1[d]-ii1[b]-ii1[c];
	mx=mx/(r.width*r.height);
	free(ii1);
	return mx;

}

float IntegralImage::calcVariance(cv::Rect r)
{
	int width=_integral.cols;
	int height=_integral.rows;
	int a=r.x+(r.y*width);
	int b=(r.x+r.width)+(r.y*width);
	int c=r.x+((r.y+r.height)*width);
	int d=(r.x+r.width)+(r.y+r.height)*width;
	float mx=calcMean(r);
	double *ii2 = (double *)_sq_integral.data;
	float mx2=ii2[a]+ii2[d]-ii2[b]-ii2[c];
	mx2=mx2/(r.width*r.height);
	mx2=mx2-(mx*mx);
	free(ii2);
	return mx2;
};

class LBPFeatures
{
public:
	/**
	* @brief LBPFeatures : constructor for LBP features
	*/
	LBPFeatures();

	//input image
	cv::Mat image;
	//vector of LBP features
	vector<uchar> features;
	cv::Mat mask;
	//integral image class
	IntegralImage ix;
	//uniform LBP lookup table
	vector<int> lookup;
	int sizeb;
	//class which computes the histogram
	//of LBP features
	//cv::Histogram hist;

	/**
	* @brief countSetBits1 : method to compute the number of set bits
	* used for uniform LBP code
	* @param code
	* @return
	*/
	bool countSetBits1(int code)
	{
		int count=0;
		while(code!=0)
		{
			if(code&&0x01)
				count++;
			code=code>>1;
		}
		return count;
	}

	// Brian Kernighan's method to count set bits
	int countSetBits(int code);


	int rightshift(int num, int shift);

	/**
	* @brief checkUniform : method to check if the LBP
	* pattern is uniform or not
	* @param code : input integer pattern
	* @return
	*/
	bool checkUniform(int code);

	//3x3 neighborhood will have 8 neighborhood pixels
	//all non uniform codes are assigned to 59
	/**
	* @brief initUniform :inititalize the uniform LBP lookup tables
	*/
	void initUniform();


	//f
	//TBD : check if grayscale encoding can be done
	//can be used to encode orientation changes of gradient
	//lbp encodes information about different types of gradients
	/**
	* @brief compute :function to compute LBP image
	* @param image : input image
	* @param dst : output LBP image
	*/

	void computelbp(cv::Mat src, cv::Mat &dst, int radius, int neighbors);
	/**
	* @brief initHistogram : initialize the histogram object
	*/
	//void initHistogram();

	/**
	* @brief computeHistogram : method to compute the spatial histoogram
	* @param cell : input image ROI over which hisogram is computed
	* @return
	*/
	//Mat computeHistogram(Mat cell);

	/**
	* @brief spatialHistogram : computes the spatial histogram ,
	*by dividing image into grids and computing LBP for each grid
	* @param lbpImage : input LBP image
	* @param grid : grid size
	* @return  : vector for LBP histogram features
	*/
	//vector<float> spatialHistogram(Mat lbpImage,Size grid);

	/**
	* @brief computeBlock : divides image into blocks ,each block is assigned LBP
	* value based on mean intensity values over the block.Integral images are used for fast computation
	* @param image : input image
	* @param dst : output LBP image
	* @param block:block size
	*/
};

LBPFeatures::LBPFeatures()
{
	sizeb=58;
}


// Brian Kernighan's method to count set bits
int LBPFeatures::countSetBits(int code)
{
	int count=0;
	int v=code;
	for(count=0;v;count++)
	{
		v&=v-1; //clears the LSB
	}
	return count;
}

int LBPFeatures::rightshift(int num, int shift)
{
	return (num >> shift) | ((num << (8 - shift)&0xFF));
}


bool LBPFeatures::checkUniform(int code)
{
	int b = rightshift(code,1);
	///int d = code << 1;
	int c = code ^ b;
	//d= code ^d;
	int count=countSetBits(c);
	//int count1=countSetBits(d);
	if (count <=2 )
		return true;
	else
		return false;
}


void LBPFeatures::initUniform()
{
	lookup.resize(256);
	int index=0;
	for(int i=0;i<=255;i++)
	{
		bool status=checkUniform(i);
		if(status==true)
		{
			lookup[i]=index;
			index++;
		}
		else
		{
			lookup[i]=58;
		}
	}

	//initHistogram();

}

void LBPFeatures::computelbp(cv::Mat src, cv::Mat &dst, int radius, int neighbors) {
	// allocate memory for result
	dst = cvCreateMat(src.rows-2*radius, src.cols-2*radius, CV_8UC1);
	// zero
	dst.setTo(0);
	for(int n=0; n<neighbors; n++) {
		// sample points
		float x = static_cast<float>(-radius) * sin(2.0*CV_PI*n/static_cast<float>(neighbors));
		float y = static_cast<float>(radius) * cos(2.0*CV_PI*n/static_cast<float>(neighbors));
		// relative indices
		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));
		// fractional part
		float ty = y - fy;
		float tx = x - fx;
		// set interpolation weights
		float w1 = (1 - tx) * (1 - ty);
		float w2 =      tx  * (1 - ty);
		float w3 = (1 - tx) *      ty;
		float w4 =      tx  *      ty;
		// iterate through your data
		for(int i=radius; i < src.rows-radius;i++) {
			for(int j=radius;j < src.cols-radius;j++) {
				// calculate interpolated value
				float t = w1*src.at<uchar>(i+fy,j+fx) + w2*src.at<uchar>(i+fy,j+cx) + w3*src.at<uchar>(i+cy,j+fx) + w4*src.at<uchar>(i+cy,j+cx);
				// floating point precision, so check some machine-dependent epsilon
				dst.at<uchar>(i-radius,j-radius) += ((t > src.at<uchar>(i,j)) || (std::abs(t-src.at<uchar>(i,j)) < std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
}

/* @function detectAndDisplay */
void detectAndDisplay( cv::Mat frame, int &x, int &y, int &width, int &height)
{
	std::vector<cv::Rect> faces;
	cv::Mat frame_gray = frame;

	equalizeHist( frame_gray, frame_gray );

	//-- Detect faces
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

	for( size_t i = 0; i < faces.size(); i++ )
	{
		// Setup a rectangle to define your region of interest
		/*cv::Rect myROI(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
		frame = frame_gray(myROI);*/
		x =faces[i].x;
		y =faces[i].y;
		width =faces[i].width;
		height =faces[i].height;
	}
}

string convertInt(int number)
{
	string num_string;
	char *num;
	if (number<10)
	{
		num = (char *)malloc(2);
		itoa(number, num, 10);
		num_string = num;
		num_string = "0" + num_string;
	}
	else
	{
		num = (char *)malloc(3);
		itoa(number, num, 10);
		num_string = num;
	}
	free(num);
	return num_string;
}

void convertImageToPCL(float &x, float &y, float&z)
{
	int fx = 525.0;  // focal length x
	int fy = 525.0;  // focal length y
	int cx = 319.5;  // optical center x
	int cy = 239.5;  // optical center y
	int factor = 5000;  // 16bit ~ 5000
	z = z *1.0 / factor;
	x = (640 - x - cx) * (z / fx);
	y = (480 - y - cy) * (z / fy);
	return;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr BilaterFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_smoothed  (new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_smoothed1  (new pcl::PointCloud<pcl::PointXYZI>);
	cloud_smoothed->width = cloud->width;
	cloud_smoothed->height = cloud->height;
	cloud_smoothed->resize(cloud_smoothed->width*cloud_smoothed->height);
	/*std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
	cloud_smoothed->resize(cloud_smoothed->height*cloud_smoothed->width);*/
	// Create the filtering object 
	for (int i=0; i<cloud->width;i++)
		for (int j=0; j<cloud->height; j++)
		{
			cloud_smoothed->at(i, j).x = cloud->at(i, j).x;
			cloud_smoothed->at(i, j).y = cloud->at(i, j).y;
			cloud_smoothed->at(i, j).z = cloud->at(i, j).z;
			cloud_smoothed->at(i, j).intensity = cloud->at(i, j).z;
		}
	pcl::BilateralFilter<pcl::PointXYZI> bf;
	bf.setInputCloud(cloud_smoothed); 
	   //bf.setStdDev(1.0f); 
	bf.setHalfSize(0.1f);
	bf.setStdDev(0.1f);
	//bf.applyFilter(*cloud_smoothed1);
	bf.filter(*cloud_smoothed1); 
	for (int i=0; i<cloud->width;i++)
		for (int j=0; j<cloud->height; j++)
			cloud->at(i, j).z = cloud_smoothed1->at(i, j).intensity;
	return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr InputCloudGray(char *input)
{
	cv::Mat matDepthFilterGaussian, matDepth;

	/// Remove noise by blurring with a Gaussian filter
	matDepth = cv::imread(input, CV_LOAD_IMAGE_ANYDEPTH);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	cloud->width = matDepth.cols;
	cloud->height = matDepth.rows;
	cloud->resize(cloud->width*cloud->height);

	cv::Mat element = cv::getStructuringElement( 0, cv::Size( 11, 11), cv::Point( 5, 5) ); // type kernel 0: Rect - 1: Cross - 2: Ellipse
	/// Apply the specified morphology operation
	morphologyEx( matDepth, matDepth, 3, element );
	//cv::bilateralFilter ( matDepth, matDepthFilterGaussian, -1, 0.45, 3 );
	cv::GaussianBlur( matDepth, matDepthFilterGaussian, cv::Size(3,3), 0.8, 0.8);

	//cv::medianBlur(matDepth, matDepthFilterGaussian, 5);
	//short* pixelPtr = (short*)matDepth.data;

	short* pixelPtr = (short*)matDepthFilterGaussian.data;
	int fx = 525.0;  // focal length x
	int fy = 525.0;  // focal length y
	int cx = 319.5;  // optical center x
	int cy = 239.5;  // optical center y
	int factor = 5000;  // 16bit ~ 5000
	float z;
	pcl::PointXYZ point;
	point.z = 0.1;

	for (int i=0; i<matDepth.rows; i++)
		for (int j=0; j<matDepth.cols; j++)
		{
			if ((pixelPtr[i*matDepth.cols+j]>400) && (pixelPtr[i*matDepth.cols+j]<700))
			{
				z = pixelPtr[i*matDepth.cols+j] *1.0 / factor;
				point = pcl::PointXYZ((j - cx) * (z / fx)*1.0, (i - cy) * (z / fy)*1.0, z);
				cloud->at(j, i) = point;
			}
			else
			{
				z = point.z;
				point = pcl::PointXYZ((j - cx) * (z / fx)*1.0, (i - cy) * (z / fy)*1.0, z);
				cloud->at(j, i) = point;
			}

		}
		return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr InputCloudColor(char *input, char *irgb)
{
	cv::Mat matDepthFilterGaussian, matDepth;

	/// Remove noise by blurring with a Gaussian filter
	matDepth = cv::imread(input, CV_LOAD_IMAGE_ANYDEPTH);
	cv::Mat matColor = cv::imread(irgb);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	cloud->width = matDepth.cols;
	cloud->height = matDepth.rows;
	cloud->resize(cloud->width*cloud->height);

	cv::Mat element = cv::getStructuringElement( 0, cv::Size( 7, 7 ), cv::Point( 3, 3 ) ); // type kernel 0: Rect - 1: Cross - 2: Ellipse
	/// Apply the specified morphology operation
	morphologyEx( matDepth, matDepth, 3, element );
	cv::bilateralFilter ( matDepth, matDepthFilterGaussian, 2, 4, 1 );
	//cv::GaussianBlur( matDepth, matDepthFilterGaussian, cv::Size(5,5), 0.4, 0.4);

	short* pixelPtr = (short*)matDepthFilterGaussian.data;
	//short* pixelPtr = (short*)matDepth.data;

	int fx = 525.0;  // focal length x
	int fy = 525.0;  // focal length y
	int cx = 319.5;  // optical center x
	int cy = 239.5;  // optical center y
	int factor = 5000;  // 16bit ~ 5000
	float z;
	pcl::PointXYZRGB point;
	point.z = 0.1;
	for (int i=0; i<matDepth.rows; i++)
		for (int j=0; j<matDepth.cols; j++)
		{
			if ((pixelPtr[i*matDepth.cols+j]>400) && (pixelPtr[i*matDepth.cols+j]<700))
			{
				z = pixelPtr[i*matDepth.cols+j] *1.0 / factor;
				point.r = matColor.at<cv::Vec3b>(i, j)[0];
				point.g = matColor.at<cv::Vec3b>(i, j)[0];
				point.b = matColor.at<cv::Vec3b>(i, j)[0];
				point.x = (j - cx) * (z / fx)*1.0;
				point.y = (i - cy) * (z / fy)*1.0;
				point.z = z;
				cloud->at(j, i) = point;
			}
			else
			{
				z = point.z;
				point.r = matColor.at<cv::Vec3b>(i, j)[0];
				point.g = matColor.at<cv::Vec3b>(i, j)[0];
				point.b = matColor.at<cv::Vec3b>(i, j)[0];
				point.x = (j - cx) * (z / fx)*1.0;
				point.y = (i - cy) * (z / fy)*1.0;
				point.z = z;
				cloud->at(j, i) = point;
			}

		}
		return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr Clone_nan(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_nan)
{
	pcl::PointXYZ point;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i=0; i<cloud_nan->points.size(); i++)
	{
		point = cloud_nan->points.at(i);
		if ((point.z<0.5) && (point.z>0))
			cloud->points.push_back(point);
	}
	return cloud;
}

bool is_stop_icp(Eigen::Matrix4f matrix1, Eigen::Matrix4f matrix2) {
	if ((matrix1(0,1)!=-matrix2(1,0)) || (matrix1(0,2)!=-matrix2(2,0)) || (matrix1(2,1)!=-matrix2(1,2)))// || (matrix1(0,3)!=-matrix2(2,3)))
		return false;
	return true;
}


boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
	// --------------------------------------------
	// -----Open 3D viewer and add point cloud-----
	// --------------------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_in_color_h (cloud, 180, 20, 20);
	viewer->addPointCloud (cloud, cloud_in_color_h, "sample cloud");
	viewer->setBackgroundColor (0, 0, 0);
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
	/*viewer->addCoordinateSystem (1.0, 0);*/
	viewer->setCameraPosition(0, 0 , 0, 0, 0, 0.2, 0);
	viewer->initCameraParameters ();
	return (viewer);
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> GrayVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud/*, pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud1*/)
{
	// --------------------------------------------
	// -----Open 3D viewer and add point cloud-----
	// --------------------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->setBackgroundColor (0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
	viewer->setCameraPosition(0, 0 , 0, 0, 0, 0.2, 0);
	viewer->initCameraParameters ();
	return (viewer);
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> normalsVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
	// --------------------------------------------------------
	// -----Open 3D viewer and add point cloud and normals-----
	// --------------------------------------------------------
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudRGB (new pcl::PointCloud<pcl::PointXYZRGB>);
	uint8_t r(255), g(15), b(15);
	pcl::PointXYZRGB point;
	for (int i=0; i<cloud->size(); i++)
	{
		point.x = cloud->points[i].x;
		point.y = cloud->points[i].y;
		point.z = cloud->points[i].z;
		uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
			static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
		point.rgb = *reinterpret_cast<float*>(&rgb);
		cloudRGB->points.push_back(point);
	}


	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->setBackgroundColor (0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloudRGB);
	viewer->addPointCloud<pcl::PointXYZRGB> (cloudRGB, rgb, "sample cloud");
	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
	viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal> (cloudRGB, normals, 3, 0.002, "normals");
	/*viewer->addCoordinateSystem (1.0, 0);*/
	viewer->setCameraPosition(0, 0 , -0.2, 0, 0, 0, 0);
	//viewer->setCameraPosition(0, 0 , 0, 0, 0, 0.2, 0);
	viewer->initCameraParameters ();
	return (viewer);
}

unsigned int text_id = 0;
void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
	void* viewer_void)
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
	if (event.getKeySym () == "r" && event.keyDown ())
	{
		std::cout << "r was pressed => removing all text" << std::endl;

		char str[512];
		for (unsigned int i = 0; i < text_id; ++i)
		{
			sprintf (str, "text#%03d", i);
			viewer->removeShape (str);
		}
		text_id = 0;
	}
}

void mouseEventOccurred (const pcl::visualization::MouseEvent &event,
	void* viewer_void)
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
	if (event.getButton () == pcl::visualization::MouseEvent::LeftButton &&
		event.getType () == pcl::visualization::MouseEvent::MouseButtonRelease)
	{
		std::cout << "Left mouse button released at position (" << event.getX () << ", " << event.getY () << ")" << std::endl;

		char str[512];
		sprintf (str, "text#%03d", text_id ++);
		viewer->addText ("clicked here", event.getX (), event.getY (), str);
	}
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> interactionCustomizationVis ()
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->setBackgroundColor (0, 0, 0);
	viewer->addCoordinateSystem (1.0, 0);

	viewer->registerKeyboardCallback (keyboardEventOccurred, (void*)&viewer);
	viewer->registerMouseCallback (mouseEventOccurred, (void*)&viewer);

	return (viewer);
}

pcl::PointCloud<pcl::PointNormal>::Ptr calc_img(cv::Mat src, cv::Mat &shape_index_value, cv::Mat & min_curvature, cv::Mat & max_curvature, 
	cv::Mat &mean_curvature, cv::Mat &gauss_curvature, cv::Mat &rcurvedness, cv::Mat &grad, cv::Mat &abs_grad_x, cv::Mat &abs_grad_xx, cv::Mat &abs_grad_y, 
	cv::Mat &abs_grad_yy, cv::Mat &edge_orientation, cv::Mat*& mat_lbp, char* file_name1)
{
	// read cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = InputCloudGray(file_name1); 

	//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	////viewer = normalsVis(cloud, normals);
	//viewer = simpleVis(cloud);
	//while (!viewer->wasStopped ())
	//{
	//	viewer->spinOnce (100);
	//	boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	//}

	cv::Mat depth_img = cv::imread(file_name1, CV_LOAD_IMAGE_ANYDEPTH);
	// hien thi phap tuyen
	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud (cloud);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
	ne.setSearchMethod (tree);
	//ne.setRadiusSearch (0.0005);
	ne.setKSearch(20);
	ne.compute (*normals);

	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields(*cloud, *normals, *cloud_normals);

	// Setup the principal curvatures computation
	pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> principal_curvatures_estimation;

	// Provide the original point cloud (without normals)
	principal_curvatures_estimation.setInputCloud (cloud);

	// Provide the point cloud with normals
	principal_curvatures_estimation.setInputNormals (normals);

	// Use the same KdTree from the normal estimation
	principal_curvatures_estimation.setSearchMethod (tree);
	principal_curvatures_estimation.setKSearch(50);
	pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principal_curvatures (new pcl::PointCloud<pcl::PrincipalCurvatures> ());
	principal_curvatures_estimation.compute(*principal_curvatures);
	shape_index_value = cvCreateMat(src.rows, src.cols, CV_64FC1);
	cv::Mat shape_index_value1 = cvCreateMat(src.rows, src.cols, CV_8UC1);
	cv::Mat normalX8UC1 = cvCreateMat(src.rows, src.cols, CV_8UC1);
	cv::Mat normalY8UC1 = cvCreateMat(src.rows, src.cols, CV_8UC1);
	cv::Mat normalZ8UC1 = cvCreateMat(src.rows, src.cols, CV_8UC1);
	min_curvature = cvCreateMat(src.rows, src.cols, CV_64FC1);
	max_curvature = cvCreateMat(src.rows, src.cols, CV_64FC1);
	mean_curvature = cvCreateMat(src.rows, src.cols, CV_64FC1);
	gauss_curvature = cvCreateMat(src.rows, src.cols, CV_64FC1);
	rcurvedness = cvCreateMat(src.rows, src.cols, CV_64FC1);
	double k1, k2, s;
	for (int i=0; i<src.rows; i++)
		for (int j=0; j<src.cols; j++)
		{
			k1 = principal_curvatures->at(j,i).pc1;
			k2 = principal_curvatures->at(j,i).pc2;
			s = 0.5-(1/PI)* atan(double((k1+k2)/(k1-k2)));
			/*if (k1==k2)
			{
				s = 0;
			}
			else
			{
				s = 0.5-(1/PI)* atan(double((k1+k2)/(k1-k2)));
			}*/
			
			shape_index_value.at<double>(i, j) = s;
			s = (int)(s*255);
			shape_index_value1.at<uchar>(i, j) = s;
			normalX8UC1.at<uchar>(i, j) = (int)(cloud_normals->at(j, i).normal_x*255);
			normalY8UC1.at<uchar>(i, j) = (int)(cloud_normals->at(j, i).normal_y*255);
			normalZ8UC1.at<uchar>(i, j) = (int)(cloud_normals->at(j, i).normal_z*255);
			min_curvature.at<double>(i, j) = k2;
			max_curvature.at<double>(i, j) = k1;
			mean_curvature.at<double>(i, j) = (k1+k2)/2;
			gauss_curvature.at<double>(i, j) = k1*k2;
			rcurvedness.at<double>(i, j) = sqrt(k1*k1+k2*k2)/2;
		}

		/*cv::imshow("sss",src);
		cv::imshow("sds", depth_img);*/
		string sa = file_name1, sa1, sa2, sa3;
		
		sa1 = sa + "_NORMAL_X.png";
		sa2 = sa + "_NORMAL_Y.png";
		sa3 = sa + "_NORMAL_Z.png";
		sa = sa + "_INDEX.png";
		cv::imwrite(sa, shape_index_value1);
		cv::imwrite(sa1, normalX8UC1);
		cv::imwrite(sa2, normalY8UC1);
		cv::imwrite(sa3, normalZ8UC1);
		//cv::waitKey();
	// equal hist
	//equalizeHist(src, src);
	// gray
	grad = cvCreateMat(src.rows, src.cols, CV_8UC1);
	edge_orientation = cvCreateMat(src.rows, src.cols, CV_8UC1);
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	/// Generate grad_x and grad_y
	cv::Mat grad_x, grad_xx, grad_y, grad_yy;

	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel( src, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );

	/// Gradient XX
	//Scharr( gray_x, grad_xx, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel( grad_x, grad_xx, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( grad_xx, abs_grad_xx );

	/// Gradient Y
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel( src, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );

	/// Gradient YY
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel( grad_y, grad_yy, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( grad_yy, abs_grad_yy );

	/// Total Gradient (approximate)
	// edge orientation
	double a;
	for (int i=0; i<src.rows; i++)
		for (int j=0; j<src.cols; j++)
		{
			if (abs_grad_y.at<uchar>(i, j)!=0)
				a = abs_grad_x.at<uchar>(i, j) / abs_grad_y.at<uchar>(i, j);
			else
				a= 1;
			edge_orientation.at<uchar>(i, j) = atan(a);
			a = abs_grad_x.at<uchar>(i,j)*abs_grad_x.at<uchar>(i,j)+abs_grad_y.at<uchar>(i, j)*abs_grad_y.at<uchar>(i, j);
			grad.at<uchar>(i, j) = sqrt(a);
		}

		for (int i=0; i<src.rows; i++)
			for (int j=0; j<src.cols; j++)
				grad.at<uchar>(i,j) = src.at<uchar>(i,j);

		// LBP 
		LBPFeatures *lbp = new LBPFeatures();
		// multi scale			
		int nLBP = 5;
		mat_lbp = new cv::Mat[nLBP];
		for (int i=0; i<nLBP; i++)
			mat_lbp[i] = cvCreateMat(src.rows, src.cols, 0);
		lbp->initUniform();
		/*lbp->computeMultiScale(src, mat_lbp);*/
		lbp->computelbp(src, mat_lbp[0], 1, 8);
		lbp->computelbp(src, mat_lbp[1], 2, 8);
		lbp->computelbp(src, mat_lbp[2], 2, 16);
		lbp->computelbp(src, mat_lbp[3], 3, 16);
		lbp->computelbp(src, mat_lbp[4], 4, 16);
		return cloud_normals;
}

////
void descritor(cv::Mat src, cv::Mat depth_img, cv::Mat shape_index_value, cv::Mat min_curvature, cv::Mat max_curvature, 
	cv::Mat mean_curvature, cv::Mat gauss_curvature, cv::Mat rcurvedness, cv::Rect myRect, cv::Mat &cov, string file0, 
	cv::Mat grad, cv::Mat abs_grad_x, cv::Mat abs_grad_xx, cv::Mat abs_grad_y, cv::Mat abs_grad_yy, cv::Mat edge_orientation, 
	cv::Mat *mat_lbp, pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals, char *file_name1)
{
	/*cloud_smoothed->width = cloud->width;
	cloud_smoothed->height = cloud->height;
	cloud_smoothed->resize(cloud_smoothed->width*cloud_smoothed->height);*/
	/*std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
	cloud_smoothed->resize(cloud_smoothed->height*cloud_smoothed->width);*/
	// Create the filtering object 
	//for (int i=0; i<cloud->width;i++)
	//	for (int j=0; j<cloud->height; j++)
	//	{
	//		cloud_smoothed->at(i, j).x = cloud->at(i, j).x;
	//		cloud_smoothed->at(i, j).y = cloud->at(i, j).y;
	//		cloud_smoothed->at(i, j).z = cloud->at(i, j).z;
	//		cloud_smoothed->at(i, j).intensity = cloud->at(i, j).z;
	//	}
	//pcl::BilateralFilter<pcl::PointXYZI> bf;
	//bf.setInputCloud(cloud_smoothed); 
	//   //bf.setStdDev(1.0f); 
	//bf.setHalfSize(5.0f);
	//bf.setStdDev(0.05f);
	//   bf.filter(*cloud_smoothed); 
	//for (int i=0; i<cloud->width;i++)
	//	for (int j=0; j<cloud->height; j++)
	//		cloud->at(i, j).z = cloud_smoothed->at(i, j).intensity;

	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	//pcl::PointXYZ point;
	//for (int i=myRect.y; i<(myRect.y+myRect.height); i++)
	//	for (int j=myRect.x; j<(myRect.x+myRect.width); j++)
	//	{
	//		point.x = cloud_normals->at(j, i).x;
	//		point.y = cloud_normals->at(j, i).y;
	//		point.z = cloud_normals->at(j, i).z;
	//		cloud->push_back(point);
	//	}

	//	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	//	//viewer = normalsVis(cloud, normals);
	//	viewer = simpleVis(cloud);
	//	while (!viewer->wasStopped ())
	//	{
	//	viewer->spinOnce (100);
	//	boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	//	}

	/// Total Gradient (approximate)
	// edge orientation

	string sa = file_name1, sa1, sa2, sa3;
		
	sa1 = sa + "_NORMAL_X.png";
	sa2 = sa + "_NORMAL_Y.png";
	sa3 = sa + "_NORMAL_Z.png";

	cv::Mat grad_x_normalX, grad_y_normalX, grad_x_normalY, grad_y_normalY, grad_x_normalZ, grad_y_normalZ, normalX, normalY, normalZ;

	normalX = cv::imread(sa1, 1);
	normalY = cv::imread(sa2, 1);
	normalZ = cv::imread(sa3, 1);

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

		/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel( normalX, grad_x_normalX, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( grad_x_normalX, grad_x_normalX);
	Sobel( normalX, grad_y_normalX, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( grad_y_normalX, grad_y_normalX );

	Sobel( normalY, grad_x_normalY, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( grad_x_normalY, grad_x_normalY);
	Sobel( normalY, grad_y_normalY, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( grad_y_normalY, grad_y_normalY );

	Sobel( normalZ, grad_x_normalZ, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( grad_x_normalZ, grad_x_normalZ);
	Sobel( normalZ, grad_y_normalZ, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( grad_y_normalZ, grad_y_normalZ );

	cv::Mat sample(myRect.width *myRect.height, 9, cv::DataType<float>::type);
	cv::Mat mu;
	int count = 0;
	ofstream f1("out.txt", ios::app);
	for (int i=myRect.y; i<(myRect.y+myRect.height); i++)
	{
		for (int j=myRect.x; j<(myRect.x+myRect.width); j++)
		{
			/*sample.at<float>(count, 0) = i-myRect.y;
			sample.at<float>(count, 1) = j-myRect.x;*/
			//sample.at<float>(count, 0) = cloud_normals->at(j, i).z;
			//sample.at<float>(count, 2) = src.at<uchar>(i, j);
			/*sample.at<float>(count, 2) = img.at<cv::Vec3b>(i, j)[0];
			sample.at<float>(count, 3) = img.at<cv::Vec3b>(i, j)[1];
			sample.at<float>(count, 4) = img.at<cv::Vec3b>(i, j)[2];*/
			/*sample.at<float>(count, 2) = abs_grad_x.at<uchar>(i, j);
			sample.at<float>(count, 3) = abs_grad_y.at<uchar>(i, j);*/
			/*sample.at<float>(count, 4) = abs_grad_xx.at<uchar>(i, j);
			sample.at<float>(count, 5) = abs_grad_yy.at<uchar>(i, j);
			sample.at<float>(count, 6) = grad.at<uchar>(i, j);*/
			//sample.at<float>(count, 3) = edge_orientation.at<uchar>(i, j);
			//sample.at<float>(count, 5) = mat_lbp[0].at<char>(i, j);
			//sample.at<float>(count, 3) = mat_lbp[1].at<char>(i, j);
			/*sample.at<float>(count, 7) = mat_lbp[2].at<char>(i, j);
			sample.at<float>(count, 8) = mat_lbp[3].at<char>(i, j);
			sample.at<float>(count, 9) = mat_lbp[4].at<char>(i, j);
			sample.at<float>(count, 10) = mat_lbp[5].at<char>(i, j);
			sample.at<float>(count, 11) = mat_lbp[6].at<char>(i, j);
			sample.at<float>(count, 12) = mat_lbp[7].at<char>(i, j);*/
			//sample.at<float>(count, 9) = mat_lbp[0].at<uchar>(i, j);
			//sample.at<float>(count, 3) = mat_lbp[1].at<uchar>(i, j);
			/*sample.at<float>(count, 4) = mat_lbp[2].at<uchar>(i, j);
			sample.at<float>(count, 5) = mat_lbp[3].at<uchar>(i, j);
			sample.at<float>(count, 6) = mat_lbp[4].at<uchar>(i, j);*/
			//sample.at<float>(count, 6) = shape_index_value.at<double>(i, j);
			//sample.at<float>(count, 6) = mean_curvature.at<double>(i, j);
			//sample.at<float>(count, 6) = gauss_curvature.at<double>(i, j);
			/*sample.at<float>(count, 6) = max_curvature.at<double>(i, j);
			sample.at<float>(count, 7) = min_curvature.at<double>(i, j);*/
			sample.at<float>(count, 0) = cloud_normals->at(j, i).normal_x;
			sample.at<float>(count, 1) = cloud_normals->at(j, i).normal_y;
			sample.at<float>(count, 2) = cloud_normals->at(j, i).normal_z;
			//sample.at<float>(count, 6) = cloud_normals->at(j, i).curvature;
			//f1 << cloud_normals->at(j, i) << " ";
			sample.at<float>(count, 3) = grad_x_normalX.at<uchar>(i, j);
			sample.at<float>(count, 4) = grad_y_normalX.at<uchar>(i, j);
			sample.at<float>(count, 5) = grad_x_normalY.at<uchar>(i, j);
			sample.at<float>(count, 6) = grad_y_normalY.at<uchar>(i, j);
			sample.at<float>(count, 7) = grad_x_normalZ.at<uchar>(i, j);
			sample.at<float>(count, 8) = grad_y_normalZ.at<uchar>(i, j);
			count++;
		}		
		f1 << endl;
	}
	calcCovarMatrix(sample, cov, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE); 

	/*ofstream fo1("samples.txt", ios::app);
		for (int i=0; i<sample.rows; i++)
		{
			for (int j=0; j<sample.cols; j++)
				fo1 << sample.at<float>(i, j) << " ";
			fo1 << endl;
		}
		fo1 << "--------------------" << endl;
		ofstream fo("mu.txt", ios::app);
		for (int i=0; i<mu.rows; i++)
		{
			for (int j=0; j<mu.cols; j++)
				fo << mu.at<double>(i, j) << " ";
			fo << endl;
		}	*/
	return;
}
////

void descritor1(cv::Mat src, cv::Mat depth_img, cv::Mat shape_index_value, cv::Mat min_curvature, cv::Mat max_curvature, 
	cv::Mat mean_curvature, cv::Mat gauss_curvature, cv::Mat rcurvedness, cv::Rect myRect, cv::Mat &cov, string file0, 
	cv::Mat grad, cv::Mat abs_grad_x, cv::Mat abs_grad_xx, cv::Mat abs_grad_y, cv::Mat abs_grad_yy, cv::Mat edge_orientation, 
	cv::Mat *mat_lbp, pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals, char *file_name1)
{
	/*cloud_smoothed->width = cloud->width;
	cloud_smoothed->height = cloud->height;
	cloud_smoothed->resize(cloud_smoothed->width*cloud_smoothed->height);*/
	/*std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
	cloud_smoothed->resize(cloud_smoothed->height*cloud_smoothed->width);*/
	// Create the filtering object 
	//for (int i=0; i<cloud->width;i++)
	//	for (int j=0; j<cloud->height; j++)
	//	{
	//		cloud_smoothed->at(i, j).x = cloud->at(i, j).x;
	//		cloud_smoothed->at(i, j).y = cloud->at(i, j).y;
	//		cloud_smoothed->at(i, j).z = cloud->at(i, j).z;
	//		cloud_smoothed->at(i, j).intensity = cloud->at(i, j).z;
	//	}
	//pcl::BilateralFilter<pcl::PointXYZI> bf;
	//bf.setInputCloud(cloud_smoothed); 
	//   //bf.setStdDev(1.0f); 
	//bf.setHalfSize(5.0f);
	//bf.setStdDev(0.05f);
	//   bf.filter(*cloud_smoothed); 
	//for (int i=0; i<cloud->width;i++)
	//	for (int j=0; j<cloud->height; j++)
	//		cloud->at(i, j).z = cloud_smoothed->at(i, j).intensity;

	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	//pcl::PointXYZ point;
	//for (int i=myRect.y; i<(myRect.y+myRect.height); i++)
	//	for (int j=myRect.x; j<(myRect.x+myRect.width); j++)
	//	{
	//		point.x = cloud_normals->at(j, i).x;
	//		point.y = cloud_normals->at(j, i).y;
	//		point.z = cloud_normals->at(j, i).z;
	//		cloud->push_back(point);
	//	}

	//	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	//	//viewer = normalsVis(cloud, normals);
	//	viewer = simpleVis(cloud);
	//	while (!viewer->wasStopped ())
	//	{
	//	viewer->spinOnce (100);
	//	boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	//	}

	/// Total Gradient (approximate)
	// edge orientation

	string sa = file_name1, sa1, sa2, sa3;
		
	sa1 = sa + "_NORMAL_X.png";
	sa2 = sa + "_NORMAL_Y.png";
	sa3 = sa + "_NORMAL_Z.png";

	cv::Mat grad_x_normalX, grad_y_normalX, grad_x_normalY, grad_y_normalY, grad_x_normalZ, grad_y_normalZ, normalX, normalY, normalZ;

	normalX = cv::imread(sa1, 1);
	normalY = cv::imread(sa2, 1);
	normalZ = cv::imread(sa3, 1);

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

		/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel( normalX, grad_x_normalX, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( grad_x_normalX, grad_x_normalX);
	Sobel( normalX, grad_y_normalX, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( grad_y_normalX, grad_y_normalX );

	Sobel( normalY, grad_x_normalY, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( grad_x_normalY, grad_x_normalY);
	Sobel( normalY, grad_y_normalY, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( grad_y_normalY, grad_y_normalY );

	Sobel( normalZ, grad_x_normalZ, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( grad_x_normalZ, grad_x_normalZ);
	Sobel( normalZ, grad_y_normalZ, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
	convertScaleAbs( grad_y_normalZ, grad_y_normalZ );

	cv::Mat sample(myRect.width *myRect.height, 11, cv::DataType<float>::type);
	cv::Mat mu;
	int count = 0;
	ofstream f1("out.txt", ios::app);
	for (int i=myRect.y; i<(myRect.y+myRect.height); i++)
	{
		for (int j=myRect.x; j<(myRect.x+myRect.width); j++)
		{
			sample.at<float>(count, 0) = i-myRect.y;
			sample.at<float>(count, 1) = j-myRect.x;
			//sample.at<float>(count, 0) = cloud_normals->at(j, i).z;
			//sample.at<float>(count, 2) = src.at<uchar>(i, j);
			/*sample.at<float>(count, 2) = img.at<cv::Vec3b>(i, j)[0];
			sample.at<float>(count, 3) = img.at<cv::Vec3b>(i, j)[1];
			sample.at<float>(count, 4) = img.at<cv::Vec3b>(i, j)[2];*/
			/*sample.at<float>(count, 2) = abs_grad_x.at<uchar>(i, j);
			sample.at<float>(count, 3) = abs_grad_y.at<uchar>(i, j);
			sample.at<float>(count, 4) = abs_grad_xx.at<uchar>(i, j);
			sample.at<float>(count, 5) = abs_grad_yy.at<uchar>(i, j);
			sample.at<float>(count, 6) = grad.at<uchar>(i, j);
			sample.at<float>(count, 7) = edge_orientation.at<uchar>(i, j);*/
			//sample.at<float>(count, 5) = mat_lbp[0].at<char>(i, j);
			//sample.at<float>(count, 3) = mat_lbp[1].at<char>(i, j);
			/*sample.at<float>(count, 7) = mat_lbp[2].at<char>(i, j);
			sample.at<float>(count, 8) = mat_lbp[3].at<char>(i, j);
			sample.at<float>(count, 9) = mat_lbp[4].at<char>(i, j);
			sample.at<float>(count, 10) = mat_lbp[5].at<char>(i, j);
			sample.at<float>(count, 11) = mat_lbp[6].at<char>(i, j);
			sample.at<float>(count, 12) = mat_lbp[7].at<char>(i, j);*/
			//sample.at<float>(count, 9) = mat_lbp[0].at<uchar>(i, j);
			/*sample.at<float>(count, 8) = mat_lbp[1].at<uchar>(i, j);
			sample.at<float>(count, 9) = mat_lbp[2].at<uchar>(i, j);
			sample.at<float>(count, 10) = mat_lbp[3].at<uchar>(i, j);
			sample.at<float>(count, 11) = mat_lbp[4].at<uchar>(i, j);*/
			//sample.at<float>(count, 2) = shape_index_value.at<double>(i, j);
			//sample.at<float>(count, 1) = mean_curvature.at<double>(i, j);
			//sample.at<float>(count, 2) = gauss_curvature.at<double>(i, j);
			//sample.at<float>(count, 3) = max_curvature.at<double>(i, j);
			//sample.at<float>(count, 4) = min_curvature.at<double>(i, j);
			//sample.at<float>(count, 2) = rcurvedness.at<double>(i, j);
			sample.at<float>(count, 2) = cloud_normals->at(j, i).normal_x;
			sample.at<float>(count, 3) = cloud_normals->at(j, i).normal_y;
			sample.at<float>(count, 4) = cloud_normals->at(j, i).normal_z;
			//sample.at<float>(count, 7) = cloud_normals->at(j, i).curvature;
			sample.at<float>(count, 5) = grad_x_normalX.at<uchar>(i, j);
			sample.at<float>(count, 6) = grad_y_normalX.at<uchar>(i, j);
			sample.at<float>(count, 7) = grad_x_normalY.at<uchar>(i, j);
			sample.at<float>(count, 8) = grad_y_normalY.at<uchar>(i, j);
			sample.at<float>(count, 9) = grad_x_normalZ.at<uchar>(i, j);
			sample.at<float>(count, 10) = grad_y_normalZ.at<uchar>(i, j);

			//f1 << cloud_normals->at(j, i) << " ";
			count++;
		}
		f1 << endl;
	}
	calcCovarMatrix(sample, cov, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE); 
	/*cv::imshow("gradient", grad_x_normalX);
	cvWaitKey(0);*/

	/*ofstream fo1("samples.txt", ios::app);
		for (int i=0; i<sample.rows; i++)
		{
			for (int j=0; j<sample.cols; j++)
				fo1 << sample.at<float>(i, j) << " ";
			fo1 << endl;
		}
		fo1 << "--------------------" << endl;
		ofstream fo("mu.txt", ios::app);
		for (int i=0; i<mu.rows; i++)
		{
			for (int j=0; j<mu.cols; j++)
				fo << mu.at<double>(i, j) << " ";
			fo << endl;
		}	*/

	return;
}


void output_covariance(string file0, string file1, string file2, string file3, int index)
{
	// file0 anh rgb
	// file1 cloud
	// file2 cloud target ailgn
	// file3 asm landmark eye, nose, mouth
	cv::Mat cov, cov1;
	//-- 2. Read input
	if (_access(file0.c_str(), 0) == -1)
		return;
	IplImage *gray = cvLoadImage(file0.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	/*if (gray=='\0')
	return;*/

	cvReleaseImage(&gray);

	//-- 3. Detect face
	int x,y,width,height;

	char *file_name0, *file_name1, *file_name2, *file_name3;

	file_name0 = (char *)malloc(file0.size() + 1);
	memcpy(file_name0, file0.c_str(), file0.size() + 1);

	file_name1 = (char *)malloc(file1.size() + 1);
	memcpy(file_name1, file1.c_str(), file1.size() + 1);

	file_name2 = (char *)malloc(file2.size() + 1);
	memcpy(file_name2, file2.c_str(), file2.size() + 1);

	file_name3 = (char *)malloc(file3.size() + 1);
	memcpy(file_name3, file3.c_str(), file3.size() + 1);

	cv::Mat in = cv::imread(file0, CV_LOAD_IMAGE_GRAYSCALE);
	equalizeHist(in, in);
	cv::Mat abs_grad_x, abs_grad_xx, abs_grad_y, abs_grad_yy, edge_orientation, grad, shape_index_value, min_curvature, max_curvature, 
		mean_curvature, gauss_curvature, rcurvature, normalX;
	cv::Mat depth_img = cv::imread(file_name1, CV_LOAD_IMAGE_ANYDEPTH);
	cv::Mat *mat_lbp;  
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals = calc_img(in, shape_index_value, min_curvature, max_curvature, mean_curvature, 
		gauss_curvature, rcurvature,grad, abs_grad_x, abs_grad_xx, abs_grad_y, abs_grad_yy, edge_orientation, mat_lbp, file_name1);	
	//string f = file0 + ".png";
	//string f_directionOfNormalVector = file0 + "_normal.png";
	//cv::imwrite(f, shape_index_value);
	//cv::imwrite(f_directionOfNormalVector, cloud_normals->;
	ofstream out_covariance("covariance.txt", ios::app);
	ofstream out_covariance1("covariance1.txt", ios::app);
	ofstream out_label("label.txt", ios::app);
	ifstream in_asm(file3);
	location_rgbd lo;
	int nRegion = 1;
	cv::Rect myRect;
	int dx1, dx2, dy1, dy2;
	for (int i=0; i<3; i++)
	{
		in_asm >> dy1 >> dy2 >> dx1 >> dx2;
		/*dx1=20;
		dy1 = 25;
		dy2 = in.rows-7;
		dx2 = in.cols-20;
		myRect.x=dx1;
		myRect.y=dy1;
		myRect.width=dx2-dx1;
		myRect.height=dy2-dy1;*/
		/*cv::imshow("ss", in(myRect));
		cv::waitKey();*/
		int w1 = (dx2-dx1)/nRegion, h1 =  (dy2-dy1)/nRegion;
		for (int k=0; k<nRegion; k++)
			for (int l=0; l<nRegion; l++)
			{
				myRect.x = k*w1+dx1;
				myRect.y = l*h1+dy1;
				myRect.width = w1;
				myRect.height = h1;
				descritor(in, depth_img, shape_index_value, min_curvature, max_curvature, mean_curvature, gauss_curvature, rcurvature, 
					myRect, cov, file0, grad, abs_grad_x, abs_grad_xx, abs_grad_y, abs_grad_yy, edge_orientation, mat_lbp, cloud_normals, file_name1);	
				for (int k=0; k<cov.rows; k++)
					for (int j=0; j<cov.cols; j++)
						out_covariance << cov.at<double>(k,j) << " ";

				descritor1(in, depth_img, shape_index_value, min_curvature, max_curvature, mean_curvature, gauss_curvature, rcurvature, 
					myRect, cov1, file0, grad, abs_grad_x, abs_grad_xx, abs_grad_y, abs_grad_yy, edge_orientation, mat_lbp, cloud_normals, file_name1);	
				for (int k=0; k<cov1.rows; k++)
					for (int j=0; j<cov1.cols; j++)
						out_covariance1 << cov1.at<double>(k,j) << " ";
			}
	}

	out_label << index << endl;
	out_covariance << endl;
	out_covariance.close();
	out_covariance1 << endl;
	out_covariance1.close();
	out_label.close();
	 
	//delete mat_lbp;
	delete file_name0;
	delete file_name1;
	delete file_name2;
	delete file_name3;
}

int main (int argc, char** argv)
{  
	//-- 1. Load the cascades
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading 1\n"); return -1; };
	if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading 2\n"); return -1; };

	string path_local = "D:\\subject\\Luan van\\DATABASE\\FaceWarehouse\\custom\\db_align\\";
	//string path_local = "db_align\\";
	string path_covarance_pre = "covariance", path_covarance;
	path_covarance = path_covarance_pre + ".txt";
	string write;
	string path_rgb, path_cloud, path_asm;
	for (int i=1; i<=6; i++)
		for (int j=1; j<=21; j++)
		{
			path_rgb = path_local + convertInt(i) + "\\" + convertInt(j) + ".png";
			path_cloud = path_local + convertInt(i) + "\\" + convertInt(j) + "_depth.png";
			path_asm = path_local + convertInt(i) + "\\" + convertInt(j) + ".png.txt";
			output_covariance(path_rgb, path_cloud, path_cloud, path_asm, 2);
			cout << path_cloud << endl;
		}
		return (0);
}