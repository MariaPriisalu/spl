// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "mvs/workspace.h"

#include <numeric>
#include <string>
#include <iostream>
#include "util/misc.h"

namespace colmap {
namespace mvs {

Workspace::CachedImage::CachedImage() {}

Workspace::CachedImage::CachedImage(CachedImage&& other) {
  num_bytes = other.num_bytes;
  bitmap = std::move(other.bitmap);
  depth_map = std::move(other.depth_map);
  normal_map = std::move(other.normal_map);
  segmentation = std::move(other.segmentation);
}

Workspace::CachedImage& Workspace::CachedImage::operator=(CachedImage&& other) {
  if (this != &other) {
    num_bytes = other.num_bytes;
    bitmap = std::move(other.bitmap);
    depth_map = std::move(other.depth_map);
    normal_map = std::move(other.normal_map);
    segmentation = std::move(other.segmentation);
  }
  return *this;
}

size_t Workspace::CachedImage::NumBytes() const { return num_bytes; }

Workspace::Workspace(const Options& options)
    : options_(options),
      cache_(1024 * 1024 * 1024 * options_.cache_size,
             [](const int) { return CachedImage(); }) {
  StringToLower(&options_.input_type);
  model_.Read(options_.workspace_path, options_.workspace_format);
  if (options_.max_image_size > 0) {
    for (auto& image : model_.images) {
      image.Downsize(options_.max_image_size, options_.max_image_size);
    }
  }

  depth_map_path_ = EnsureTrailingSlash(
      JoinPaths(options_.workspace_path, options_.stereo_folder, "depth_maps"));
  normal_map_path_ = EnsureTrailingSlash(JoinPaths(
      options_.workspace_path, options_.stereo_folder, "normal_maps"));
  seg_map_path_ = EnsureTrailingSlash(JoinPaths(
      options_.workspace_path,"segmentation"));
  std::string dense_("dense/");
  if (seg_map_path_.find(dense_)!= std::string::npos) {
    seg_map_path_.replace(seg_map_path_.find(dense_), dense_.size(), "");
  }
  //EnsureTrailingSlash(options.seg_map_path);
  //seg_map_path_right_  =EnsureTrailingSlash(options.seg_map_path);
}

void Workspace::ClearCache() { cache_.Clear(); }

const Workspace::Options& Workspace::GetOptions() const { return options_; }

const Model& Workspace::GetModel() const { return model_; }

const Bitmap& Workspace::GetBitmap(const int image_id) {
  auto& cached_image = cache_.GetMutable(image_id);
  if (!cached_image.bitmap) {
    cached_image.bitmap.reset(new Bitmap());
    cached_image.bitmap->Read(GetBitmapPath(image_id), options_.image_as_rgb);
    if (options_.max_image_size > 0) {
      cached_image.bitmap->Rescale(model_.images.at(image_id).GetWidth(),
                                   model_.images.at(image_id).GetHeight());
    }
    cached_image.num_bytes += cached_image.bitmap->NumBytes();
    cache_.UpdateNumBytes(image_id);
  }
  return *cached_image.bitmap;
}

const DepthMap& Workspace::GetDepthMap(const int image_id) {
  auto& cached_image = cache_.GetMutable(image_id);
  if (!cached_image.depth_map) {
    cached_image.depth_map.reset(new DepthMap());
    cached_image.depth_map->Read(GetDepthMapPath(image_id));
    if (options_.max_image_size > 0) {
      cached_image.depth_map->Downsize(model_.images.at(image_id).GetWidth(),
                                       model_.images.at(image_id).GetHeight());
    }
    cached_image.num_bytes += cached_image.depth_map->GetNumBytes();
    cache_.UpdateNumBytes(image_id);
  }
  return *cached_image.depth_map;
}

const NormalMap& Workspace::GetNormalMap(const int image_id) {
  auto& cached_image = cache_.GetMutable(image_id);
  if (!cached_image.normal_map) {
    cached_image.normal_map.reset(new NormalMap());
    cached_image.normal_map->Read(GetNormalMapPath(image_id));
    if (options_.max_image_size > 0) {
      cached_image.normal_map->Downsize(model_.images.at(image_id).GetWidth(),
                                        model_.images.at(image_id).GetHeight());
    }
    cached_image.num_bytes += cached_image.normal_map->GetNumBytes();
    cache_.UpdateNumBytes(image_id);
  }
  return *cached_image.normal_map;
}

const Bitmap& Workspace::GetSegmentation(const int image_id) {
   auto& cached_image = cache_.GetMutable(image_id);
  if (!cached_image.segmentation) {
    cached_image.segmentation.reset(new Bitmap());
    cached_image.segmentation->Read(GetSegmentationPath(image_id), false);

    if (options_.max_image_size > 0) {
      cached_image.segmentation->Rescale(model_.images.at(image_id).GetWidth(),
                                   model_.images.at(image_id).GetHeight());
    }
    cached_image.num_bytes += cached_image.bitmap->NumBytes();
    cache_.UpdateNumBytes(image_id);
  }

  return *cached_image.segmentation;
}

std::string Workspace::GetBitmapPath(const int image_id) const {
  return model_.images.at(image_id).GetPath();
}

std::string Workspace::GetDepthMapPath(const int image_id) const {
  return depth_map_path_ + GetFileName(image_id);
}

std::string Workspace::GetNormalMapPath(const int image_id) const {
  return normal_map_path_ + GetFileName(image_id);
}

std::string Workspace::GetSegmentationPath(const int image_id) const {
    //std::string str2 ("right");
    /*
    if (GetFileName(image_id).find(str2) != std::string::npos) {
         std::cout<<seg_map_path_right_ + GetFileName(image_id);
         return seg_map_path_right_ + GetFileName(image_id);
    } else{
    */
        std::string str2("_rightImg8bit.png.geometric.bin"); // replace with png
        std::string str("_leftImg8bit.png.geometric.bin"); //replace with png
        std::string name=GetFileName(image_id);
        if (name.find(str2)!= std::string::npos) {
          name.replace(name.find(str2), str2.size(), ".png");
        }
        if (name.find(str)!= std::string::npos) {
          name.replace(name.find(str), str.size(), ".png");
        }
        std::cout<<seg_map_path_ + name<<std::endl;
        return seg_map_path_ + name;
    //}

}


bool Workspace::HasBitmap(const int image_id) const {
  return ExistsFile(GetBitmapPath(image_id));
}

bool Workspace::HasDepthMap(const int image_id) const {
  return ExistsFile(GetDepthMapPath(image_id));
}

bool Workspace::HasNormalMap(const int image_id) const {
  return ExistsFile(GetNormalMapPath(image_id));
}

bool Workspace::HasSegmentation(const int image_id) const {
  return ExistsFile(GetSegmentationPath(image_id));
}

std::string Workspace::GetFileName(const int image_id) const {
  const auto& image_name = model_.GetImageName(image_id);
  return StringPrintf("%s.%s.bin", image_name.c_str(),
                      options_.input_type.c_str());
}

void ImportPMVSWorkspace(const Workspace& workspace,
                         const std::string& option_name) {
  const std::string& workspace_path = workspace.GetOptions().workspace_path;
  const std::string& stereo_folder = workspace.GetOptions().stereo_folder;

  CreateDirIfNotExists(JoinPaths(workspace_path, stereo_folder));
  CreateDirIfNotExists(JoinPaths(workspace_path, stereo_folder, "depth_maps"));
  CreateDirIfNotExists(JoinPaths(workspace_path, stereo_folder, "normal_maps"));
  CreateDirIfNotExists(
      JoinPaths(workspace_path, stereo_folder, "consistency_graphs"));

  const auto option_lines =
      ReadTextFileLines(JoinPaths(workspace_path, option_name));
  for (const auto& line : option_lines) {
    if (!StringStartsWith(line, "timages")) {
      continue;
    }

    const auto elems = StringSplit(line, " ");
    int num_images = std::stoull(elems[1]);

    std::vector<int> image_ids;
    if (num_images == -1) {
      CHECK_EQ(elems.size(), 4);
      const int range_lower = std::stoull(elems[2]);
      const int range_upper = std::stoull(elems[3]);
      CHECK_LT(range_lower, range_upper);
      num_images = range_upper - range_lower;
      image_ids.resize(num_images);
      std::iota(image_ids.begin(), image_ids.end(), range_lower);
    } else {
      CHECK_EQ(num_images + 2, elems.size());
      image_ids.reserve(num_images);
      for (size_t i = 2; i < elems.size(); ++i) {
        const int image_id = std::stoull(elems[i]);
        image_ids.push_back(image_id);
      }
    }

    std::vector<std::string> image_names;
    image_names.reserve(num_images);
    for (const auto image_id : image_ids) {
      const std::string image_name =
          workspace.GetModel().GetImageName(image_id);
      image_names.push_back(image_name);
    }

    const auto& overlapping_images =
        workspace.GetModel().GetMaxOverlappingImagesFromPMVS();

    const auto patch_match_path =
        JoinPaths(workspace_path, stereo_folder, "patch-match.cfg");
    const auto fusion_path =
        JoinPaths(workspace_path, stereo_folder, "fusion.cfg");
    std::ofstream patch_match_file(patch_match_path, std::ios::trunc);
    std::ofstream fusion_file(fusion_path, std::ios::trunc);
    CHECK(patch_match_file.is_open()) << patch_match_path;
    CHECK(fusion_file.is_open()) << fusion_path;
    for (size_t i = 0; i < image_names.size(); ++i) {
      const auto& ref_image_name = image_names[i];
      patch_match_file << ref_image_name << std::endl;
      if (overlapping_images.empty()) {
        patch_match_file << "__auto__, 20" << std::endl;
      } else {
        for (const int image_id : overlapping_images[i]) {
          patch_match_file << workspace.GetModel().GetImageName(image_id)
                           << ", ";
        }
        patch_match_file << std::endl;
      }
      fusion_file << ref_image_name << std::endl;
    }
  }
}

}  // namespace mvs
}  // namespace colmap
