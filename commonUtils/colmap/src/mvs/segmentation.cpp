/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Segmentation.cpp
 * Author: mariap
 * 
 * Created on April 10, 2018, 11:04 AM
 */
#include "base/warp.h"
#include "segmentation.h"
#include "mvs/mat.h"
#include "util/bitmap.h"
#include <FreeImage.h>
namespace colmap {
namespace mvs {
Segmentation::Segmentation() : Mat<float>(0, 0, 1) {}

Segmentation::Segmentation(const size_t width, const size_t height): Mat<float>(width, height, 1) {}


Segmentation::Segmentation(const Mat<float>& mat): Mat<float>(mat.GetWidth(), mat.GetHeight(), mat.GetDepth()) {
  CHECK_EQ(mat.GetDepth(), 1);
  data_ = mat.GetData();
}

void Segmentation::Rescale(const float factor){
  if (width_ * height_ == 0) {
    return;
  }

  const size_t new_width = std::round(width_ * factor);
  const size_t new_height = std::round(height_ * factor);
  std::vector<float> new_data(new_width * new_height);
  DownsampleImage(data_.data(), height_, width_, new_height, new_width,
                  new_data.data());

  data_ = new_data;
  width_ = new_width;
  height_ = new_height;

  data_.shrink_to_fit();
}

void Segmentation::Downsize(const size_t max_width, const size_t max_height){
  if (height_ <= max_height && width_ <= max_width) {
    return;
  }
  const float factor_x = static_cast<float>(max_width) / width_;
  const float factor_y = static_cast<float>(max_height) / height_;
  Rescale(std::min(factor_x, factor_y));
}


Bitmap Segmentation::ToBitmap() const{
  CHECK_GT(width_, 0);
  CHECK_GT(height_, 0);
  
  Bitmap bitmap;
  bitmap.Allocate(width_, height_, true);
  
  
  int colors[19][3]= {{128, 64, 128}, {244, 35, 232}, {70, 70, 70}, {102, 102, 156}, {190, 153, 153}, {153, 153, 153},
                    {250, 170, 30}, {220, 220, 0}, {107, 142, 35}, {152, 251, 152}, {70, 130, 180}, {220, 20, 60},
                    {255, 0, 0}, {0, 0, 142}, {0, 0, 70}, {0, 60, 100}, {0, 80, 100}, {0, 0, 230}, {119, 11, 32}}; 
  
  for (size_t y = 0; y < height_; ++y) {
    for (size_t x = 0; x < width_; ++x) {
      const int label = Get(y, x);
      if (label > 0) {
       
        const BitmapColor<float> color(255  * JetColormap::Red(colors[label-1][0]/255.0),
                                       255  * JetColormap::Green(colors[label-1][0]/255.0),
                                       255  * JetColormap::Blue(colors[label-1][0]/255.0));
        bitmap.SetPixel(x, y, color.Cast<uint8_t>());
      } else {
        bitmap.SetPixel(x, y, BitmapColor<uint8_t>(0));
      }
    }
  }

  return bitmap;
}


// Change this!!!
void Segmentation::Read(const std::string& path) {
  std::fstream text_file(path, std::ios::in | std::ios::binary);
  CHECK(text_file.is_open()) << path;

  char unused_char;
  text_file >> width_ >> unused_char >> height_ >> unused_char >> depth_ >>
      unused_char;
  std::streampos pos = text_file.tellg();
  text_file.close();

  CHECK_GT(width_, 0);
  CHECK_GT(height_, 0);
  CHECK_GT(depth_, 0);
  data_.resize(width_ * height_ * depth_);


  

  const FREE_IMAGE_FORMAT format = FreeImage_GetFileType(path.c_str(), 0);

  //if (format == FIF_UNKNOWN) {}

  FIBITMAP* fi_bitmap = FreeImage_Load(format, path.c_str());
  //if (fi_bitmap == nullptr) {}

  data_ = FIBitmapPtr(fi_bitmap, &FreeImage_Unload);

 
FIBITMAP* converted_bitmap = FreeImage_ConvertToGreyscale(fi_bitmap);
 data_ = FIBitmapPtr(converted_bitmap, &FreeImage_Unload);

}
}
