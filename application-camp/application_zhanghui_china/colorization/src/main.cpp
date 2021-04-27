#include <iostream>
#include <stdlib.h>
#include <vector>
#include <memory>

#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include "opencv2/imgproc/types_c.h"

#include "acl/acl.h"
#include "atlasutil/atlas_model.h"

using namespace std;

//定义ColorizeProcess类
class ColorizeProcess{

public:
	int32_t 	deviceId_;
	aclrtContext 	context_;
	aclrtStream 	stream_;
	void*		inputBuf_;
	uint32_t	modelWidth_;	//模型宽度 固定为 224
	uint32_t	modelHeight_;	//模型高度 固定为 224
	bool		isInited_;  	//是否已经初始化

	const char* 	modelPath_;	//模型路径 resource目录
	uint32_t	inputDataSize_;
	std::vector<InferenceOutput> inferenceOutput;
	AtlasModel 	model_;
	AtlasError 	ret;
	aclrtRunMode	runMode_;


	//构造函数
	ColorizeProcess(const char* modelPath, uint32_t modelWidth, uint32_t modelHeight)
		:deviceId_(0), context_(nullptr), stream_(nullptr), inputBuf_(nullptr), modelWidth_(modelWidth), modelHeight_(modelHeight), isInited_(false){
		modelPath_ = modelPath;
	}

	//析构函数
	~ColorizeProcess(){
		destroy_resource();
	}

	//预处理
	int preprocess( const string & imageFile){

		//读取图片
		cv::Mat mat = cv::imread(imageFile, CV_LOAD_IMAGE_COLOR);

		//resize图片到 224，224
		cv::Mat resizeMat;
		cv::resize(mat,resizeMat,cv::Size( modelWidth_, modelHeight_));

		//图片增强
		//转换为32位浮点数
		resizeMat.convertTo(resizeMat, CV_32FC3);

		//归一化
		resizeMat = 1.0 * resizeMat /255;

		//色域转换
		cv::cvtColor(resizeMat, resizeMat, CV_BGR2Lab);

		//拆通道
		std::vector<cv::Mat> channels;
		cv::split(resizeMat, channels);

		//取通道0，减均值
		cv::Mat resizeMatL = channels[0] - 50;

		//拷贝到device
		if (runMode_ == ACL_HOST) {
        		ret = aclrtMemcpy(inputBuf_, inputDataSize_, resizeMatL.ptr<uint8_t>(), inputDataSize_, ACL_MEMCPY_HOST_TO_DEVICE);
        		if (ret != ACL_ERROR_NONE) {
            			cout <<"Copy resized image data to device failed.\n";
				return -1;
        		}
    		} else {
        		memcpy(inputBuf_, resizeMatL.ptr<uint8_t>(), inputDataSize_);
		}

		return 0;
	};

	int inference(){
		ret = model_.Execute(inferenceOutput);
    		if (ret != ATLAS_OK) {
        		cout <<"Execute model inference failed\n";
		}
		return 0;
	}

	int postprocess ( const string & imageFile ){

		uint32_t dataSize = 0;


		//获取output的device结果
    		void* data = inferenceOutput[0].data.get();
    		if (data == nullptr) 
    		{
			return -1;
    		}

    		dataSize = inferenceOutput[0].size;

    		uint32_t size = static_cast<uint32_t>(dataSize) / sizeof(float);

    		//取AB分量
    		cv::Mat mat_a(56, 56, CV_32FC1, const_cast<float*>((float*)data));
    		cv::Mat mat_b(56, 56, CV_32FC1, const_cast<float*>((float*)data + size / 2));

    		//从原图中取出L分量
    		cv::Mat mat = cv::imread(imageFile, CV_LOAD_IMAGE_COLOR);
    		mat.convertTo(mat, CV_32FC3);
    		mat = 1.0 * mat / 255;
    		cv::cvtColor(mat, mat, CV_BGR2Lab);
    		std::vector<cv::Mat> channels;
    		cv::split(mat, channels);

    		//将图片resize成原来图片的大小
    		int r = mat.rows;
    		int c = mat.cols;
    		cv::Mat mat_a_up(r, c, CV_32FC1);
    		cv::Mat mat_b_up(r, c, CV_32FC1);
    		cv::resize(mat_a, mat_a_up, cv::Size(c, r));
    		cv::resize(mat_b, mat_b_up, cv::Size(c, r));

    		//将三个通道合并形成LAB图片格式
    		cv::Mat newChannels[3] = { channels[0], mat_a_up, mat_b_up };
    		cv::Mat resultImage;
    		cv::merge(newChannels, 3, resultImage);

    		//将LAB图片转换为BGR格式
    		cv::cvtColor(resultImage, resultImage, CV_Lab2BGR);
    		resultImage = resultImage * 255;

		//将图片保存
    		save_pic(imageFile, resultImage);

		return 0;
	};


	//保存图片
	void save_pic(const std::string& origImageFile, cv::Mat& image){

		int pos = origImageFile.find_last_of("/");

    		string filename(origImageFile.substr(pos + 1));
    		stringstream sstream;
    		sstream.str("");
    		sstream << "../output/out_" << filename;

    		string outputPath = sstream.str();

		cout << "OUTPUT filename " << outputPath <<"\n";
    		cv::imwrite(outputPath, image);

		return ;
	};

	//清理资源
	void destroy_resource(){
		model_.DestroyInput();
    		model_.DestroyResource();

    		aclrtFree(inputBuf_);
    		inputBuf_ = nullptr;

    		if (stream_ != nullptr) {
        		ret = aclrtDestroyStream(stream_);
        		if (ret != ACL_ERROR_NONE) {
            			cout << "destroy stream failed\n";
        		}
        		stream_ = nullptr;
    		}
    		cout <<"end to destroy stream\n";

    		if (context_ != nullptr) {
        		ret = aclrtDestroyContext(context_);
        		if (ret != ACL_ERROR_NONE) {
            			cout <<"destroy context failed\n";
        		}
        		context_ = nullptr;
    		}
    		cout <<"end to destroy context\n";

     		ret = aclrtResetDevice(deviceId_);
    		if (ret != ACL_ERROR_NONE) {
       			cout <<"reset device failed\n";
    		}
    		cout <<"end to reset device is " << deviceId_ << "\n";

    		ret = aclFinalize();
    		if (ret != ACL_ERROR_NONE) {
        		cout <<"finalize acl failed\n";
    		}
    		cout <<"end to finalize acl\n";

		return ;
	};

	//初始化
	int init(){

		//判断是否初始化过
		if(isInited_){
			cout << "already inited\n";
			return -1;
		}

		//初始化ACL环境
    		const char *aclConfigPath = "../etc/acl.json";
    		ret = aclInit(aclConfigPath);
    		if (ret != ACL_ERROR_NONE) {
        		cout <<"Acl init failed\n";
			return -1;
    		}

		//打开设备,默认0
		ret = aclrtSetDevice(deviceId_);
		if (ret != ACL_ERROR_NONE) {
        		cout << "Acl open device " << deviceId_ << "failed\n";
			return -1;
    		}

		//获取运行模式 EP模式：区分EP和RC模式，EP模式是device+host,RC模式是直接在device侧启动应用
		ret = aclrtGetRunMode(&runMode_);
		if (ret != ACL_ERROR_NONE) {
			cout <<"acl get run mode failed\n";
			return -1;
    		}

		//模型初始化(output buffer在这里处理)
		ret = model_.Init(modelPath_);
    		if (ret != ATLAS_OK) {
        		cout <<"Init model failed\n";
			return -1;
    		}

		inputDataSize_ = model_.GetModelInputSize(0);

		//申请Device内存
		aclrtMalloc(&inputBuf_, (size_t)(inputDataSize_), ACL_MEM_MALLOC_HUGE_FIRST);
    		if (inputBuf_ == nullptr) {
        		cout <<"Acl malloc image buffer failed.\n";
			return -1;
    		}

		//创建input buffer
   		ret = model_.CreateInput(inputBuf_, inputDataSize_);
    		if (ret != ATLAS_OK) {
        		cout <<"Create mode input dataset failed\n";
			return -1;
    		}


		isInited_ = true;


		return 0;
	}

};

int main(int argc, char **argv){

	int ret;

	//由于水平有限，先只能做一张图片的上色 
  	if(argc!=2){
	  	cout << "Usage: ./bin/main image_filename\n";
	  	return -1;
  	}

	cout << argv[1] <<"\n";
	//初始化实例对象
	ColorizeProcess colorize("../model/colorization.om", 224, 224);

	//初始化
	ret = colorize.init();
	if(ret){
		cout<<"init error\n";
		colorize.destroy_resource();
		exit(-1);
	}
	cout<<"init success\n";

	//预处理
	ret = colorize.preprocess(argv[1]);
	if(ret){
		cout<<"preprocess error\n";
		colorize.destroy_resource();
		exit(-1);
	}
	cout<<"preprocess " << argv[1] <<" success\n";

	//推理
	ret = colorize.inference();
	if(ret){
		cout<<"inference error\n";
		colorize.destroy_resource();
		exit(-1);
	}
	cout<<"inference success\n";

	//后处理
	ret = colorize.postprocess(argv[1]);
	if(ret){
		cout<<"postprocess error\n";
		colorize.destroy_resource();
		exit(-1);
	}
	cout<<"postprocess " << argv[1] <<" success\n";

	return 0;
	
}

