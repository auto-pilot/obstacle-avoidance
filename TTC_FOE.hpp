#ifndef TTC_FOE_HPP
#define TTC_FOE_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <list>
#include <vector>


class TTC_FOE {
    double PI = 3.1415926;
    
    public:
        // calculation Time to Collision 
        double TTC(cv::Point2f keypoint, cv::Mat FOE, cv::Point2f px_flow);

        // calculation Focus of Expansion
        cv::Mat FOE(std::vector<cv::Point2f> optical_flow, std::list< cv::Point2f > keypoints);

        double AB(cv::Point2f px_flow);
        
        void draw_arrow(cv::Mat& img, cv::Point2f pStart, cv::Point2f pEnd, int len, int alpha, cv::Scalar& color, int thickness, int lineType);

};

#endif