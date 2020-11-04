/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <list>
#include <map>
#include <set>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;
using std::begin, std::end; 


// Comparison function for sorting the 
// set by increasing order of its pair's 
// second value 
struct comp { 
    template <typename T> 
  
    // Comparator function 
    bool operator()(const T& l, 
                    const T& r) const
    { 
        if (l.second != r.second) { 
            return l.second > r.second; 
        } 
        return l.first > r.first; 
    } 
}; 


// Function to sort the map according 
// to value in a (key-value) pairs 
void sort(map<string, float>& M) 
{ 
  
    // Declare set of pairs and insert 
    // pairs according to the comparator 
    // function comp() 
    set<pair<string, float>, comp> S(M.begin(), 
                                   M.end()); 
  
    // Print the sorted value 
    for (auto& it : S) { 
        cout << it.first << it.second << " | " << endl; 
    } 
} 

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "./";
    vector<cv::KeyPoint> allKeypoints;
    vector<cv::DMatch> allMatches;
    bool log = false;
    std::map<string,float> eff;

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    list<string> detectorsList {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
    list<string> descriptorsList {"BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"}; // "BRISK" not working

    /* MAIN LOOP OVER ALL IMAGES */

    for (string detector : detectorsList)
    {
        for (string descriptor : descriptorsList)
        {   
            if (detector.compare("SIFT") == 0 && descriptor.compare("BRIEF") == 0 ||
                detector.compare("SIFT") == 0 && descriptor.compare("ORB") == 0 ||
                detector.compare("SIFT") == 0 && descriptor.compare("FREAK") == 0 ||
                detector.compare("SIFT") == 0 && descriptor.compare("AKAZE") == 0 ||
                detector.compare("SIFT") == 0 && descriptor.compare("SIFT") == 0)
            {
                cout << " | " << detector << "/" << descriptor;
                cout << " | Out of memory | N/A |"  << endl;
                continue;
            } 
            else if (descriptor.compare("FREAK") == 0 ||
                     descriptor.compare("AKAZE") == 0 ||
                     descriptor.compare("SIFT") == 0)
            {
                cout << " | " << detector << "/" << descriptor;
                cout << " | Assertion failed | N/A |"  << endl;
                continue;
            }

            double t = (double)cv::getTickCount();

            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
            {
                /* LOAD IMAGE INTO BUFFER */

                // assemble filenames for current index
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

                // load image from file and convert to grayscale
                cv::Mat img, imgGray;
                img = cv::imread(imgFullFilename);
                cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

                //// STUDENT ASSIGNMENT
                //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

                // push image into data frame buffer
                DataFrame frame;
                frame.cameraImg = imgGray;
                if (dataBuffer.size() > 1)
                    dataBuffer.erase(dataBuffer.begin());
                dataBuffer.push_back(frame);
                //// EOF STUDENT ASSIGNMENT

                if (log)
                    cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl; 

                /* DETECT IMAGE KEYPOINTS */

                // extract 2D keypoints from current image
                vector<cv::KeyPoint> keypoints, roiKeypoints; // create empty feature list for current image
                string detectorType = detector; //// -> SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

                //// STUDENT ASSIGNMENT
                //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
                //// -> SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

                if (detectorType.compare("SHITOMASI") == 0)
                {
                    detKeypointsShiTomasi(keypoints, imgGray, false);
                }
                else 
                {
                    if (detectorType.compare("HARRIS") == 0)
                    {
                        detKeypointsHarris(keypoints, imgGray, false);
                    }
                    else
                    {
                        detKeypointsModern(keypoints, imgGray, detectorType, false);
                    }
                    
                }
                //// EOF STUDENT ASSIGNMENT

                //// STUDENT ASSIGNMENT
                //// TASK MP.3 -> only keep keypoints on the preceding vehicle

                // only keep keypoints on the preceding vehicle
                bool bFocusOnVehicle = true;
                cv::Rect vehicleRect(535, 180, 180, 150);
                if (bFocusOnVehicle)
                {
                    for (vector<cv::KeyPoint>::iterator it = keypoints.begin(); it != keypoints.end(); ++it)
                    {
                        if (vehicleRect.contains(it->pt))
                            roiKeypoints.push_back(*it);
                    }
                    keypoints = roiKeypoints;
                    allKeypoints.insert(end(allKeypoints), begin(keypoints), end(keypoints));
                }

                //// EOF STUDENT ASSIGNMENT

                // optional : limit number of keypoints (helpful for debugging and learning)
                bool bLimitKpts = false;
                if (bLimitKpts)
                {
                    int maxKeypoints = 50;

                    if (detectorType.compare("SHITOMASI") == 0)
                    { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                    }
                    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);

                    if (log)
                        cout << " NOTE: Keypoints have been limited!" << endl;
                }

                // push keypoints and descriptor for current frame to end of data buffer
                (dataBuffer.end() - 1)->keypoints = keypoints;

                if (log)
                    cout << "#2 : DETECT KEYPOINTS done" << endl;

                /* EXTRACT KEYPOINT DESCRIPTORS */

                //// STUDENT ASSIGNMENT
                //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
                //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

                cv::Mat descriptors;
                string descriptorType = descriptor; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
                descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
                //// EOF STUDENT ASSIGNMENT

                // push descriptors for current frame to end of data buffer
                (dataBuffer.end() - 1)->descriptors = descriptors;

                if (log)
                    cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

                if (dataBuffer.size() > 1) // wait until at least two images have been processed
                {

                    /* MATCH KEYPOINT DESCRIPTORS */

                    vector<cv::DMatch> matches;
                    string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
                    string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
                    string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN

                    //// STUDENT ASSIGNMENT
                    //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
                    //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

                    matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                    (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                    matches, descriptorType, matcherType, selectorType);

                    //// EOF STUDENT ASSIGNMENT

                    // store matches in current data frame
                    (dataBuffer.end() - 1)->kptMatches = matches;

                    if (log)
                        cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                    // visualize matches between current and previous image
                    bVis = false;
                    if (bVis)
                    {
                        cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                        cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                        (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                        matches, matchImg,
                                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                                        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                        string windowName = "Matching keypoints between two camera images";
                        cv::namedWindow(windowName, 7);
                        cv::imshow(windowName, matchImg);
                        cout << "Press key to continue to next image" << endl;
                        cv::waitKey(0); // wait for key to be pressed
                    }
                    bVis = false;
                    allMatches.insert(end(allMatches), begin(matches), end(matches));
                }
            } // eof loop over all images

            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency() *1000;

            float kpi = allMatches.size()/(t*1000);

            cout << " | " << detector << "/" << descriptor;
            cout << " | " << allMatches.size() << " | "  << " "<< t << " | ";
            cout << kpi << " | " << endl;
             
            eff.insert(std::pair<string,float>( " | " + detector + "/" + descriptor + " | " , kpi*1000));

            allMatches.clear();
        }
    }

    sort(eff);
    cout << "All keypoints: " << allKeypoints.size() << endl;

    return 0;
}
