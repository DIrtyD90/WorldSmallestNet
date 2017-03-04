#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"

int main(int argc, char** argv) {

	std::string PathGraph = "/home/daniel/tensorflow/tensorflow/WorldSmallestNet/Python/SaveFiles/frozen_graph.pb";

	//Setup Inputs and Outputs
	tensorflow::Tensor Input1(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,1}));
	tensorflow::Tensor Input0(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,1}));

	std::vector<tensorflow::Tensor> output;
	Input1.scalar<float>()() = 1.0;
	Input0.scalar<float>()() = 0.0;

	//initial declaration Tensorflow
	tensorflow::Session* session;
	tensorflow::Status status;
	status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
	if (!status.ok()) {
   		std::cout << status.ToString() << "\n";
    	return 1;
    }
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
 

	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
   		 { "Input:0", Input0},
    	};
		status = session->Run(inputs, {"Layer2/Output"},{}, &output);
		if (!status.ok()) {
   		 std::cout << status.ToString() << "\n";
   		return 1;
  		}
		auto Result = output[0].matrix<float>();
		std::cout << "Input: 0 | Output: "<< Result(0,0) << std::endl;
	
	inputs = {
   		 { "Input:0", Input1},
    	};
		status = session->Run(inputs, {"Layer2/Output"},{}, &output);
		if (!status.ok()) {
   		 std::cout << status.ToString() << "\n";
   		return 1;
  		}
		auto Result1 = output[0].matrix<float>();
		std::cout << "Input: 1 | Output: "<< Result1(0,0) << std::endl;	
		

}