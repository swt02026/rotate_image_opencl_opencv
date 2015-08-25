#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <boost/compute.hpp>
#include <string>
#include <bitset>

using namespace std;

string sourceStr = BOOST_COMPUTE_STRINGIZE_SOURCE(

		__kernel void img_rotate(
				__global int* dst,
				__global int* src,
				int W, int H,
				int D,
				float sinTheta,float cosTheta){

			int x = get_global_id(0);
			int y = get_global_id(1);
			float xpos = x * cosTheta + y * sinTheta;
			float ypos = -1. * x * sinTheta + y * cosTheta + (W /2. );
			if(xpos >= 0 && xpos < D && 
			ypos >= 0 && ypos < D){
				dst[(int)(ypos)*D + (int)(xpos)] = src[y*W + x];
			}
			

		}
);

int main(){

	vector<cl::Platform> platforms;

	cl::Platform::get(&platforms);

	vector<cl::Device> devices;

	platforms[0].getDevices(CL_DEVICE_TYPE_ALL,&devices);

	cl::Context context(devices);

	cl::CommandQueue queue(context,devices[0]);

	cv::Mat image = cv::imread("/home/opencv-3.0.0/data/detect_blob.png",-1);

	if(!image.data){

		return -1;
	}


	cv::namedWindow("testcv",CV_WINDOW_AUTOSIZE);
	cl_uint* in = (cl_uint*)image.data;

	cl_uint diagonal = std::sqrt(image.rows * image.rows + image.cols * image.cols);

 	vector<cl_uint> clear(diagonal * diagonal,0xFFFFFFFF);

	cl::Buffer input(context,CL_MEM_READ_ONLY,image.total()*sizeof(cl_uint));
	cl::Buffer output(context,CL_MEM_READ_WRITE,diagonal * diagonal *sizeof(cl_uint));
	//write data
	queue.enqueueWriteBuffer(input,CL_TRUE,0,image.total()*sizeof(cl_uint),in);
	//clear buffer memory
	queue.enqueueWriteBuffer(output,CL_TRUE,0,clear.size()*sizeof(cl_uint),clear.data());


	cl::Program::Sources source;
	source.push_back({sourceStr.c_str(),sourceStr.length()+1});
	cl::Program program(context,source);

	int eb = program.build(devices);
//get kernel
	cl::Kernel kernel(program,"img_rotate");
	cout << eb << endl;

	float cos_theta=(cos(45));
	float sin_theta=(sin(45));


	kernel.setArg(0,output);
	kernel.setArg(1,input);
	kernel.setArg(2,image.cols);
	kernel.setArg(3,image.rows);
	kernel.setArg(4,diagonal);
	kernel.setArg(5,cos_theta);
	kernel.setArg(6,sin_theta);
//run kernel
    int err = queue.enqueueNDRangeKernel(kernel,cl::NullRange,cl::NDRange(image.cols,image.rows),cl::NullRange);

	cout << err << endl;
	
 	vector<cl_uint> out(diagonal * diagonal,0xFFFFFFFF);

//get result data
	queue.enqueueReadBuffer(output,CL_TRUE,0,diagonal * diagonal*sizeof(cl_uint),out.data());
	cv::Mat result(diagonal,diagonal,CV_8UC4,out.data());

	cv::imshow("Display windows",result);
	cv::waitKey(0);

	cout << image.rows << endl << image.cols << endl << image.elemSize();
	cout << endl << image.total();


}
