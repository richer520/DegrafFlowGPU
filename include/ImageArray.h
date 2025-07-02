// File description
/*!
  Copyright \htmlonly &copy \endhtmlonly 2008-2011 Cranfield University
  \file ImageArray.h
  \brief ImageArray class header
  \author Ioannis Katramados
*/

// Pragmas
#pragma once

// Include Files
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h> // For legacy support
#include <vector>
// #include "VisionerLibTypes.h"

//! A class of image array functions
class ImageArray
{
private:
  // Private variables
  cv::Size image_size;

public:
  // Public variables
  bool init_flag;
  int array_length;
  IplImage **image; // Compatibility interface

  // Modern storage (for internal use)
  std::vector<cv::Mat> image_mats;
  std::vector<IplImage *> temp_ipl_images; // IplImage wrappers

  // Public functions
  ImageArray();
  ~ImageArray();
  void InitArray(IplImage *p_src, uint p_length);
  void ReleaseArray(void);
};

// Function Prototypes
void CopyImageArray(ImageArray *p_src_array, ImageArray *p_dest_array);