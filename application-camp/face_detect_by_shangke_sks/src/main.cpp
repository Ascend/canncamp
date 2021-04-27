#include <stdio.h>
#include <unistd.h>
#include "acl/acl.h"
#include "atlas_model.h"
#include "commdef.h"
#include <memory>
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include "opencv2/imgproc/types_c.h"

using namespace std;

#define BATCH_COUNT 1
#define INPUT_SIZE 640

// 读取文件
// 返回值：
//   < 0 : 文件读取失败；
//   >= 0 : 文件的实际长度
int64_t ReadFile(vector<uint8_t> &data, const char *szFilePath);

// 加载图片文件，resize，传出对象，分解颜色，复制到指定的内存中
bool load_img( const char* file_path,
               cv::Mat& img,
               float* host_input_buffer,
               const int input_channel,
               const int input_wh );

// 从推理结果获得人脸框
void output_2_face_rect( vector<vector<float>>& rects, const vector<vector<float>>& infer_outputs, int input_wh );

int main( int argc, const char* argv[] )
{
    if ( argc <= 1 )
    {
        printf( "Usage:\n%s model.om rgb.bin", argv[0] );
        return 1;
    }

    void* input_buf = nullptr;
    aclError ret = 0;
    do
    {
        ret = aclInit("");

        BREAK_ON_FALSE( ret == 0 );
        ret = aclrtSetDevice(0);
        BREAK_ON_FALSE( ret == 0 );

        aclrtRunMode run_mode;
        ret = aclrtGetRunMode(&run_mode);
        BREAK_ON_FALSE( ret == 0 );

        vector<uint8_t> model_data;
        ReadFile( model_data, argv[1] );
        BREAK_ON_FALSE( model_data.size() > 0 );

        AtlasModel model;
        ret = model.Init( model_data );
        BREAK_ON_FALSE( ret == 0 );
        uint32_t input_size = model.GetModelInputSize(0);

        aclrtMalloc(&input_buf, (size_t)(input_size), ACL_MEM_MALLOC_HUGE_FIRST);
        BREAK_ON_NULL( input_buf );

        ret = model.CreateInput( input_buf, input_size);

        shared_ptr<float> data_tmp( (float*)new uint8_t[input_size] );
        BREAK_ON_NULL( data_tmp.get() );

        // 这里需要找个方法得到输入层的宽高————好像没办法
        cv::Mat img;
        BREAK_ON_FALSE( load_img( argv[2], img, data_tmp.get(), 3, INPUT_SIZE ) );

        if (run_mode == ACL_HOST)
        {
            ret = aclrtMemcpy( input_buf,
                               input_size,
                               data_tmp.get(),
                               input_size,
                               ACL_MEMCPY_HOST_TO_DEVICE);
            BREAK_ON_FALSE( ret == 0 );
        } else {
            memcpy(input_buf, data_tmp.get(), input_size);
        }

        vector<InferenceOutput> infer_outputs;
        ret = model.Execute(infer_outputs);
        BREAK_ON_FALSE( ret == 0 );

        BREAK_ON_FALSE( infer_outputs.size() == 9 );

        vector<vector<float>> infer_outputs_copy;
        for ( const auto& output_layer : infer_outputs )
        {
            const float* data = (const float*)output_layer.data.get();
            size_t size = output_layer.size / sizeof(float);
            infer_outputs_copy.emplace_back( data, data + size );
        }

        vector<vector<float>> face_rects;
        output_2_face_rect( face_rects, infer_outputs_copy, INPUT_SIZE );

        // 在图像上画框
        if ( face_rects.size() > 0 )
        {
            printf( "             x   y   w   h\n" );
        }
        for ( const auto& rect : face_rects )
        {
            CONTINUE_ON_FALSE( rect.size() == 4 );
            printf( "face rect: %3d %3d %3d %3d\n", (int)rect.at(0), (int)rect.at(1), (int)rect.at(2), (int)rect.at(3) );
            cv::Rect rc( rect.at(0), rect.at(1), rect.at(2), rect.at(3) );
            cv::rectangle( img, rc, cv::Scalar( 255, 0, 0 ), 2);
        }
        cv::imwrite( "./output.png", img );

    } while ( false );

    ret = aclrtResetDevice(0);
    ret = aclFinalize();

    return 0;
}

// 加载图片文件，resize，传出对象，分解颜色，复制到指定的内存中
bool load_img( const char* file_path,
               cv::Mat& img,
               float* host_input_buffer,
               const int input_channel,
               const int input_wh )
{
    bool re = false;
    do
    {
        cv::Mat imgSrc = cv::imread(file_path, CV_LOAD_IMAGE_COLOR);
        BREAK_ON_NULL( imgSrc.data );
        BREAK_ON_FALSE( imgSrc.channels() == input_channel );

        int nW = imgSrc.cols;
        int nH = imgSrc.rows;

        int border[2] = {0};
        float scale = 1.0f;

        if (nW > nH)
        {
            border[0] = 0;
            border[1] = (int)((nW - nH) / 2);
            cv::copyMakeBorder(imgSrc, img, border[1], border[1], border[0], border[0], cv::BORDER_CONSTANT);
        }
        if (nW < nH)
        {
            border[0] = (int)((nH - nW) / 2);
            border[1] = 0;
            cv::copyMakeBorder(imgSrc, img, border[1], border[1], border[0], border[0], cv::BORDER_CONSTANT);
        }

        if (nW == nH)
        {
            border[0] = 0;
            border[1] = 0;
            imgSrc.copyTo(img);
        }

        int square_size = img.cols > img.rows ? img.cols : img.rows;

        scale = input_wh * 1.0f / square_size;

        cv::resize(img, img, cv::Size(input_wh, input_wh));

        cv::Mat img_float;

        img.convertTo(img_float, input_channel == 3 ? CV_32FC3 : CV_32FC1 );

        vector<cv::Mat> color_channels;
        cv::split(img_float, color_channels);

        int page_size = input_wh * input_wh;

        for (int j = (int)color_channels.size() - 1; j >= 0 ; j--)
        {
            memcpy(host_input_buffer, color_channels[j].data, page_size * sizeof(float));
            host_input_buffer += page_size;
        }
        re = true;
    } while (false);
    return re;
}

// 读取文件
// 返回值：
//   < 0 : 文件读取失败；
//   >= 0 : 文件的实际长度
int64_t ReadFile(vector<uint8_t> &data, const char *szFilePath)
{
    int64_t re = -1;
    FILE *pFile = NULL;
    do
    {
        int64_t lFileSize = 0;

        pFile = fopen(szFilePath, "rb");
        if (!pFile)
        {
            break;
        }

        //获取当前读取文件的位置 进行保存
        unsigned int current_read_position = ftell(pFile);
        fseek(pFile, 0, SEEK_END);
        //获取文件的大小
        lFileSize = ftell(pFile);
        //恢复文件原来读取的位置
        fseek(pFile, current_read_position, SEEK_SET);

        data.resize(lFileSize);

        int64_t lRead = (long long)fread(&data[0], 1, (int)lFileSize, pFile);

        if (lRead != lFileSize)
        {
            data.clear();
        }
        else
        {
            re = lRead;
        }

    } while (false);

    if (pFile)
    {
        fclose(pFile);
    }
    return re;
}
