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

#ifndef COLMAP_SRC_UI_PROJECT_WIDGET_H_
#define COLMAP_SRC_UI_PROJECT_WIDGET_H_

#include <QtCore>
#include <QtWidgets>

#include "util/misc.h"
#include "util/option_manager.h"

namespace colmap {

class ProjectWidget : public QWidget {
 public:
  ProjectWidget(QWidget* parent, OptionManager* options);

  bool IsValid() const;
  void Reset();

  std::string GetDatabasePath() const;
  std::string GetImagePath() const;
  void SetDatabasePath(const std::string& path);
  void SetImagePath(const std::string& path);

 private:
  void Save();
  void SelectNewDatabasePath();
  void SelectExistingDatabasePath();
  void SelectImagePath();
  QString DefaultDirectory();

  OptionManager* options_;

  // Whether file dialog was opened previously.
  bool prev_selected_;

  // Text boxes that hold the currently selected paths.
  QLineEdit* database_path_text_;
  QLineEdit* image_path_text_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_PROJECT_WIDGET_H_
