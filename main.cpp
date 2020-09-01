#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <math.h>
#include <numeric>

#include "TTC_FOE.hpp"

double PI = 3.1415926;
const int BLOCK_X = 16;
const int BLOCK_Y = 16;
const int LIMIT = 0.5;


int main(int argc, char **argv) 
{
    TTC_FOE ttc_foe;
    std::list<cv::Point2f> keypoints;

    cv::Mat frame, prev_frame;
    cv::VideoCapture capture(0);

    cv::Size camera_width_heigh = cv::Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH), (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter output_video;
    output_video.open("output.avi", CV_FOURCC('M', 'P', '4', '2'), 25.0, camera_width_heigh);

    int iterator = 0;
    while(true)
    {
        capture >> frame;

        int img_width = frame.cols;
        int img_height = frame.rows;

        // detecting corners per 50 frames
        if(iterator % 50 == 0)
        {
            std::vector<cv::KeyPoint> kps;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect(frame, kps);

            for(auto kp:kps)
            {
                keypoints.push_back(kp.pt);
            }

            prev_frame = frame.clone();
            iterator++;
            continue;
        }

        /*
            Calculating optical flow with Lucas-Kanade algorithm
            resource: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
        */
        std::vector<cv::Point2f> next_keypoints;
        std::vector<cv::Point2f> prev_keypoints;

        std::vector<unsigned char> status;
        std::vector<float> err;

        for (auto kp: keypoints)
        {
            prev_keypoints.push_back(kp);
        }

        // Lucas-Kanade
        cv::calcOpticalFlowPyrLK(prev_frame, frame, prev_keypoints, next_keypoints, status, err);

        /*
            erase the lost corners
        */
        int i = 0;
        for (auto iter = keypoints.begin(); iter != keypoints.end(); i++)
        {
            if(status[i] == 0)
            {
                iter = keypoints.erase(iter);
                continue;
            }
            *iter = next_keypoints[i];
            iter++;
        }

        // erase the lost previous corners
        i = 0;
        for (auto iter = prev_keypoints.begin(); iter != prev_keypoints.end(); i++)
        {
            if (status[i] == 0)
            {
                iter = prev_keypoints.erase(iter);
                continue;
            }
            iter++;
        }

        // If all key points are lost, the program is shutdown
        if(keypoints.size() == 0)
        {
            std::cout<<"lost\n";
            break;
        }

        std::vector<cv::Point2f> optical_flow;
        cv::Point2f point;
        i = 0;

        // Calculation of the change in x and y position between the previous position of the optical flow and the current position.
        for (auto kp: keypoints)
        {
            point.x = kp.x - prev_keypoints[i].x; 
            point.y = kp.y - prev_keypoints[i].y; 
            optical_flow.push_back(point);
            i++;
        }

        // Focus of Expansion
        cv::Mat foe = ttc_foe.FOE(optical_flow, keypoints);


        double A_ttc_sum[BLOCK_X][BLOCK_Y] = {0};
        double b_num[BLOCK_X][BLOCK_Y] = {0};
        double optical_flow_sum[BLOCK_X][BLOCK_Y] = {0};

        /*
            Calculating collision time for each corner
        */
        i = 0;
        for(auto kp: keypoints)
        {
            int block_x = (int)(kp.x * BLOCK_X / img_width);
            int block_y = (int)(kp.y * BLOCK_Y / img_height);
            b_num[block_x][block_y]++;

            // Collecting collision times on matrix points
            A_ttc_sum[block_x][block_y] += ttc_foe.TTC(kp, foe, optical_flow[i]);

            optical_flow_sum[block_x][block_y] += ttc_foe.AB(optical_flow[i]);
            i++;
        }

        // Balance Strategy
        double ttc_block_average[BLOCK_X][BLOCK_Y] = {0};
        double optical_flow_block_average[BLOCK_X][BLOCK_Y] = {0};
     
        double left_side_ttc = 0, right_side_ttc = 0;
        double left_side_opt = 0, right_side_opt = 0;
        
        for(int i=0; i<BLOCK_X; i++){
            for(int j=0; j<BLOCK_Y; j++){
                    
                if(b_num[i][j])
                {
                    // Average collision time at each matrix point
                    ttc_block_average[i][j] = A_ttc_sum[i][j] / b_num[i][j];

                    // Average optical flow at each matrix point
                    optical_flow_block_average[i][j] = optical_flow_sum[i][j] / b_num[i][j];

                    // frame's left side
                    if(i <= BLOCK_X / 2)
                    {    
                        left_side_ttc += ttc_block_average[i][j];
                        left_side_opt += optical_flow_block_average[i][j];
                    }
                    // frame's right side
                    else
                    {
                        right_side_ttc += ttc_block_average[i][j];
                        right_side_opt += optical_flow_block_average[i][j];
                    }
                
                }
                
            }
        }

        // Calculating robot car left and right side's forces
        // https://journals.sagepub.com/doi/pdf/10.5772/5715
        double balance_strategy_ttc = std::abs((left_side_ttc - right_side_ttc))/(left_side_ttc + right_side_ttc);

        // balance strategy with optical flow
        double balance_strategy_opt = std::abs((left_side_opt - right_side_opt))/(left_side_opt + right_side_opt);

        cv::Mat img_show = frame.clone();
        
        // draw Focus of Expansion point
        cv::circle(img_show, cv::Point(int(foe.at<double>(0,0)), int(foe.at<double>(1,0))), 5, cv::Scalar(0, 0, 240), 3); 
        
        // draw optical flow vectors
        cv::Scalar vector_color;
        i = 0;
        for ( auto opt: optical_flow)
        {
            int theta_to_scale = int( atan( abs(opt.y/ opt.x) ) /PI*2*255 );
            vector_color = cv::Scalar(theta_to_scale, 160, 160);
            ttc_foe.draw_arrow(img_show, prev_keypoints[i], prev_keypoints[i] + opt, 2, 15, vector_color, 1, 8);
            i++;
        }

        // draw block matrix
        for (int i=0; i<BLOCK_X; i++)
        {
            for (int j=0; j<BLOCK_Y; j++)
            {
                if (b_num[i][j])
                {
                    char str[25];
                    sprintf(str, "%.1lf", ttc_block_average[i][j]);
                    std::string text = str;
                    cv::Point origin = cv::Point(img_width/BLOCK_X * i, img_height/BLOCK_Y * j);

                    cv::putText(img_show, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.2, cv::Scalar(0,0,255), 1, 8);
                }
            }
        }

        /*
            Make a decision with balance strategy
            This decision mechanism also works with optical flow forces results.
            If you want to test with optical flow forces, you should change the last characters of the variable names to "ttc-> opt".
        */
        if(balance_strategy_ttc > LIMIT)
        {
            // Which side of the robot vehicle has low collision time, turn to the other side.
            if(right_side_ttc > left_side_ttc)
            {                    
                cv::putText(img_show, "TURN RIGHT!!!", cv::Point(50,100), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 4, 14);
            }
            else
            {
                cv::putText(img_show, "TURN LEFT!!!", cv::Point(50,100), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 4, 14);
            }
        }
        // If the difference between the two forces lower than the specified limit, go forward.
        else
        {
            cv::putText(img_show, "FORWARD", cv::Point(50,100), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 4, 14);
        }
        
        /*
        char leftstr[25];
        sprintf(leftstr, "%.1lf", left_side_ttc);
        std::string leftstrtext = leftstr;
        char rightstr[25];
        sprintf(rightstr, "%.1lf", right_side_ttc);
        std::string rightstrtext = rightstr;
        char balance[25];
        sprintf(balance, "%.1lf", balance_strategy_opt);
        std::string balancetext = balance;
        cv::putText(img_show, "Left TTC : ", cv::Point(300,300), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2, 2);
        cv::putText(img_show, leftstrtext, cv::Point(400,300), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2, 2);

        cv::putText(img_show, "Right TTC: ", cv::Point(300,320), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2, 2);
        cv::putText(img_show, rightstrtext, cv::Point(400,320), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2, 2);

        cv::putText(img_show, "F DIFF   : ", cv::Point(300,340), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2, 2);
        cv::putText(img_show, balancetext, cv::Point(400,340), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2, 2);
        */

        cv::imshow("corners", img_show);
        cv::waitKey(1);
        output_video << img_show;
        prev_frame = frame.clone();
        iterator++;
    }

    output_video.release();
    return 0;
}