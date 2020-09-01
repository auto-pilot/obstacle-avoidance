#include "TTC_FOE.hpp"

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <list>
#include <vector>
#include <math.h>

double TTC_FOE::TTC(cv::Point2f keypoint, cv::Mat FOE, cv::Point2f px_flow)
{
     return sqrt ( 
                    ((keypoint.x - FOE.at<double>(0,0)) * (keypoint.x - FOE.at<double>(0,0)) + 
                     (keypoint.y - FOE.at<double>(1,0)) * (keypoint.y - FOE.at<double>(1, 0))) 
                    / 
                    (px_flow.x * px_flow.x + px_flow.y * px_flow.y)
                );
}

/*
    Least squares solution
    https://www.dgp.toronto.edu/~donovan/stabilization/opticalflow.pdf (p 13-14)
*/
cv::Mat TTC_FOE::FOE(std::vector<cv::Point2f> optical_flow, std::list< cv::Point2f > keypoints)
{
    cv::Mat A(optical_flow.size(), 2, CV_64F);
    cv::Mat b(optical_flow.size(), 1, CV_64F);
    cv::Mat foe(2, 1, CV_64F);
    
    auto kp = keypoints.begin();

    for (int i=0; i< A.rows; i++)
    {
        auto px_A = A.ptr<double>(i);
        auto px_b = b.ptr<double>(i);

        double opt_x = optical_flow[i].x, opt_y = optical_flow[i].y;
        px_A[0] = opt_y;
        px_A[1] = -opt_x;
        px_b[0] = kp->x * opt_y - kp->y * opt_x;
        kp++;
    }

    cv::solve(A, b, foe, cv::DECOMP_QR);

    return foe;
}

/*
    Distance Between 2 Points
    https://www.mathsisfun.com/algebra/distance-2-points.html
*/
double TTC_FOE::AB(cv::Point2f px_flow)
{
    return sqrt((px_flow.y * px_flow.y) + (px_flow.x * px_flow.x));
}

void TTC_FOE::draw_arrow(cv::Mat& img, cv::Point2f pStart, cv::Point2f pEnd, int len, int alpha, cv::Scalar& color, int thickness, int lineType)
{
    cv::Point2f arrow;
    double angle = std::atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));
    cv::line(img, pStart, pEnd, color, thickness, lineType);
    arrow.x = pEnd.x + len * cos(angle + PI * alpha / 180);
    arrow.y = pEnd.y + len * sin(angle + PI * alpha / 180);
    cv::line(img, pEnd, arrow, color, thickness, lineType);
    arrow.x = pEnd.x + len * cos(angle - PI * alpha / 180);
    arrow.y = pEnd.y + len * sin(angle - PI * alpha / 180);
    cv::line(img, pEnd, arrow, color, thickness, lineType);
}