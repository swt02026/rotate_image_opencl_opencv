#include <boost/compute.hpp>
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>
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

	cv::Mat image = cv::imread("./detect_blob.png",-1);

	if(!image.data){
		exit(-1);
	}	

	cl_uint* in = (cl_uint*)image.data;

	auto device = boost::compute::system::default_device();

	boost::compute::context context(device);

	boost::compute::command_queue queue(context, device);

	auto program = 
			boost::compute::program::build_with_source(sourceStr.c_str(),context);

	auto kernel = program.create_kernel("img_rotate");

	boost::compute::vector<cl_uint> in_gpu_vec(
					in,
					in + image.total(),
				   	queue);

	
	int diagonal = sqrt(image.rows * image.rows + image.cols * image.cols);

	boost::compute::vector<cl_uint> out_gpu_vec(
					diagonal * diagonal,
					0xFFFFFFFF,
					queue
					);

	kernel.set_arg(0, out_gpu_vec);
	kernel.set_arg(1, in_gpu_vec);
	kernel.set_arg(2, image.cols);
	kernel.set_arg(3, image.rows);
	kernel.set_arg(4, diagonal);
	kernel.set_arg(5, (cl_float)cos(45));
	kernel.set_arg(6, (cl_float)sin(45));

	const size_t global_size[2]={(unsigned)image.cols,(unsigned)image.rows};

	queue.enqueue_nd_range_kernel(
					kernel,
					2,
					0,
					global_size,
					0
					);
			/*
	queue.enqueue_nd_range_kernel(
					kernel,
					boost::compute::extents<2>{0,0},
					boost::compute::extents<2>{(unsigned)image.cols,(unsigned)image.rows},
					boost::compute::extents<2>{0,0}
					);	
*/
	vector<cl_uint> out_vec;

	boost::compute::copy(
					out_gpu_vec.begin(),
					out_gpu_vec.end(),
					std::back_inserter(out_vec),
					queue
					);

	cv::Mat outImg(diagonal, diagonal, CV_8UC4, out_vec.data());

	cv::imshow("boost compute rotate", outImg);

	cv::waitKey(0);

}
