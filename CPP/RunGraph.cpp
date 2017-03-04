
#include <unordered_set>

#include "MakePair.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/saved_model/loader.h"



int main(int argc, char** argv) {

	//initial declarations
	int ImageWidth = 75;
	int ImageHeigth = 50;
	cv::Mat Image1;
	cv::Mat Image2; 
	cv::Mat Image1In;
	cv::Mat Image2In;
	cv::Mat Image1float;
	cv::Mat Image2float;
	cv::Mat Image1Resize;
	cv::Mat Image2Resize;
	cv::Mat dst;
	cv::Mat dst_RGB;
	int ii = 0;
	char key;
	bool Exit = false;

	std::string PathGraph = "/home/daniel/Documents/TensorflowCPP/frozen_graph.pb";
	
	//Setup Inputs and Outputs
	tensorflow::Tensor InputPair(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, ImageHeigth,ImageWidth,2}));
	tensorflow::Tensor Label(tensorflow::DT_INT32,tensorflow::TensorShape());
	tensorflow::Tensor keep_prob1(tensorflow::DT_FLOAT,tensorflow::TensorShape());
	//tensorflow::Tensor keep_prob2(tensorflow::DT_FlOAT,tensorflow::TensorShape());
	//tensorflow::Tensor keep_prob3(tensorflow::DT_FlOAT,tensorflow::TensorShape());
	tensorflow::Tensor learning_rate(tensorflow::DT_FLOAT,tensorflow::TensorShape());
	auto InputPair_mapped = InputPair.tensor<float, 4>();
	std::vector<tensorflow::Tensor> outputs;


	Label.scalar<int>()() = 1;
	keep_prob1.scalar<float>()() = 1.0;
	//keep_prob2.scalar<float>()() = 1.0;
	//keep_prob3.scalar<float>()() = 1.0;
	learning_rate.scalar<float>()() = 0.0;

	//initial declaration Tensorflow

	tensorflow::Session* session;
	 tensorflow::Status status;
	 status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
	  if (!status.ok()) {
   		 std::cout << status.ToString() << "\n";
      	 return 1;
     }
  	
  	tensorflow::SavedModelBundle bundle;
  	tensorflow::SessionOptions session_options;
  	tensorflow::RunOptions run_options;
  	std::unordered_set<std::string> Test={"serve"};
  	
  	//"/home/daniel/Result/ResultsTensorflow_withoutScaling/model.ckpt-847.meta";
  	const std::string export_dir = "/media/daniel/MAYER/ResultsTensorflow/CPP";
    //	  io::JoinPath(testing::TensorFlowSrcRoot(), kTestDataSharded);
  	//status = tensorflow::LoadSavedModel(session_options, run_options, export_dir,Test, &bundle);
  	// CheckSavedModelBundle(export_dir, bundle);
  	//std::cout<<status.ToString()<<"\n";
	tensorflow::GraphDef graph_def;
	 status = ReadBinaryProto(tensorflow::Env::Default(),PathGraph, &graph_def);
	
	 if (!status.ok()) {
    		 std::cout << status.ToString() << "\n";
     	 return 1;
   	}

   	// Add the graph to the session
  	 status = session->Create(graph_def);
    	if (!status.ok()) {
     		 std::cout << status.ToString() << "\n";
     	 return 1;
    	}

	//makeFileList ind Loop over List
	std::vector<std::string> FileList = make_List("/home/daniel/List/TrainingsList.txt"); 
	while(ii<FileList.size()){
		
		Image1In = read_Picture1(FileList[ii]);
		Image2In = read_Picture2(FileList[ii]);
		Image1 = resize_Image(Image1In,500,300);
		Image2 = resize_Image(Image2In,500,300);
		Image1Resize = resize_Image(Image1,ImageWidth,ImageHeigth);
		Image2Resize = resize_Image(Image2,ImageWidth,ImageHeigth);
		//std::cout<<"Debug"<<std::endl;
	//	
		//File Tensorflow Tensor with Image
		
		Image1Resize.convertTo(Image1float, CV_32FC1);
		Image2Resize.convertTo(Image2float,CV_32FC1);

		//cv::equalizeHist(Image2,ImagePair);
		const float* source_data1 =(float*) Image1float.data;
		//copying Image into Tensor
		//Picture1
		for(int y=0;y<ImageHeigth;y++){
			const float* source_row1 = source_data1 + (y * ImageWidth);
			for(int x=0;x<ImageWidth;x++){
				const float* source_pixel1 = source_row1 + x;

				InputPair_mapped(0,y,x,0) = *source_pixel1;
				//std::cout<<*source_pixel<<std::endl;
			}

		}
		const float* source_data2 =(float*) Image2float.data;
		//Picture 2
		for(int y=0;y<ImageHeigth;y++){
			const float* source_row2= source_data2 + (y* ImageWidth);
			for(int x=0;x<ImageWidth;x++){
				const float* source_pixel2 = source_row2 + x;

				InputPair_mapped(0,y,x-ImageWidth+1,1) = *source_pixel2;
				//std::cout<<*source_pixel<<std::endl;
			}

		}

		std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
   			 { "Images:0", InputPair},
    		 //{ "labels:0", Label},
    		 { "keep_prob1:0", keep_prob1},
    		 { "keep_prob2:0", keep_prob1},
    		 { "keep_prob3:0", keep_prob1},
    		 //{ "learning_rate:0", learning_rate},
  		};
		status = session->Run(inputs, {"SoftMax/Output"},{}, &outputs);
		if (!status.ok()) {
   		 std::cout << status.ToString() << "\n";
   		return 1;
  		}
		auto output_soft = outputs[0].matrix<float>();
		//std::cout << output_soft() << "\n";
		int dstWidth = Image1.cols*2;
  		int dstHeight = Image1.rows;
   		cv::Mat dst = cv::Mat(dstHeight, dstWidth, CV_8UC1, cv::Scalar(0));
   		cv::Rect roi(cv::Rect(0,0,Image1.cols, Image1.rows));
   		cv::Mat targetROI = dst(roi);
   		Image1.copyTo(targetROI);
   		targetROI = dst(cv::Rect(Image1.cols,0,Image1.cols, Image1.rows));
   		Image2.copyTo(targetROI);
		cv::cvtColor(dst,dst_RGB,CV_GRAY2RGB);

   		putText(dst_RGB, std::to_string(output_soft(0,1)), cv::Point(450,50),
			 cv::FONT_HERSHEY_PLAIN, 3.0, CV_RGB(0,0,255), 3.0);
   		// create image window named "My Image"
   		cv::namedWindow("OpenCV Window");
  		  // show the image on window
   		cv::imshow("OpenCV Window", dst_RGB);
  		  // wait key for 5000 ms
   		
   		while(true){
			key = cv::waitKey(30);
			if(key==45){
		   		if(ii>0){
		   			ii = ii-1;
		   			break;
		   		}
		   	}
		   	if(key==43){
		   		ii = ii+1;
		   		break;
		   	}
		   	if(key==27){
		   		Exit = true;
		   		break;
		   	}
	   	}
		if(Exit == true){
			break;
		}
   }



	return 0;
}