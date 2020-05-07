#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio>
#include <vector>

int main(int argc, char** argv)
{
    cv::VideoCapture cap(0);
    if ( !cap.isOpened()){
        std::cout << "No camera found";
        return -1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);         //Upto your resolution
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    const int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const float aspect = float(frameWidth) / frameHeight;
    const int inHeight = 480;                       //Upto your machine performance
    const int inWidth = ceil(inHeight * aspect);

    cv::Mat frame, rFrame;
    
    cv::dnn::Net net = cv::dnn::readNetFromTorch("models/normed/candy.ts"); //Modify for your interest
    const cv::Scalar mean = cv::Scalar(103.939, 116.779, 123.680);

    while(1){
        cap >> frame;
        
        cv::resize(frame,rFrame,cv::Size(inWidth, inHeight));

        cv::Mat inpBlob =
                cv::dnn::blobFromImage(rFrame, 1.0,
                                        cv::Size(inWidth, inHeight),
                                        mean, false, false);

        net.setInput(inpBlob);
        cv::Mat outBlob = net.forward();

        std::vector<cv::Mat> results;
        cv::dnn::imagesFromBlob(outBlob,results);    //Suppose there will be 1 item
        
        assert(results.empty());
        
        results[0] += mean;
        results[0] /= 255.0;
        
        cv::imshow("Neural-Style-Transfer", results[0]);
        if ( 0 < cv::waitKey(10)){
            break;
        }
    }

    cap.release();
    return 0;
}
