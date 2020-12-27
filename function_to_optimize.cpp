#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

int get_derivatives(int argc, char** argv );
    {
          /// Declare variables
    Mat src, u_derivative, v_derivative;
    int ufilter[] = {-1, 8, 0, -8, 1}/12;    
    int vfliter[] = {{-1}, {8}, {0}, {-8}, {1}}/12;
    Point anchor;
    double delta;
    int ddepth;
    int kernel_size;
    char* window_name = "filter2D Demo";

    int c;

    /// Load an image
    src = imread( argv[1] );

    if( !src.data )
    { return -1; }


    /// Initialize arguments for the filter
    anchor = Point( -1, -1 );
    delta = 0;
    ddepth = -1;
    /// Apply filter
    u_derivative = filter2D(src, ddepth , ufilter, anchor, delta, BORDER_DEFAULT);
    v_derivative = filter2D(src, ddepth , vfilter, anchor, delta, BORDER_DEFAULT);
    
    return u_derivative, v_derivative;
    }













