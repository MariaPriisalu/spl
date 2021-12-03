/*
 * `colmap::mvs::Segmentation::Segmentation()'
workspace.cc:(.text+0x329a): undefined reference to `colmap::mvs::Segmentation::Read(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)'
workspace.cc:(.text+0x3304): undefined reference to `colmap::mvs::Segmentation::Downsize(unsigned long, unsigned long)'

 */

/* 
 * File:   Segmentation.h
 * Author: mariap
 *
 * Created on April 10, 2018, 11:04 AM
 */

#include <string>
#include <vector>
#include <memory>

#include "mvs/mat.h"
#include "util/bitmap.h"

#include <FreeImage.h>

#ifndef SEGMENTATION_H
#define SEGMENTATION_H
namespace colmap {
namespace mvs {
class Segmentation : public Mat<float> {
public:
    Segmentation();
    //Segmentation(const Segmentation& orig);
    Segmentation(const size_t width, const size_t height);
    explicit Segmentation(const Mat<float>& mat);

    void Rescale(const float factor);
    void Downsize(const size_t max_width, const size_t max_height);
    void Read(const std::string& path);
    Bitmap ToBitmap() const;
private:
    typedef std::unique_ptr<FIBITMAP, decltype(&FreeImage_Unload)> FIBitmapPtr;


};
}
}
#endif /* SEGMENTATION_H */

