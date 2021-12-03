# Install script for directory: /home/mariap/Packages/colmap/colmap/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/mariap/Packages/colmap/colmap/build/src/base/cmake_install.cmake")
  include("/home/mariap/Packages/colmap/colmap/build/src/controllers/cmake_install.cmake")
  include("/home/mariap/Packages/colmap/colmap/build/src/estimators/cmake_install.cmake")
  include("/home/mariap/Packages/colmap/colmap/build/src/exe/cmake_install.cmake")
  include("/home/mariap/Packages/colmap/colmap/build/src/ext/cmake_install.cmake")
  include("/home/mariap/Packages/colmap/colmap/build/src/feature/cmake_install.cmake")
  include("/home/mariap/Packages/colmap/colmap/build/src/mvs/cmake_install.cmake")
  include("/home/mariap/Packages/colmap/colmap/build/src/optim/cmake_install.cmake")
  include("/home/mariap/Packages/colmap/colmap/build/src/retrieval/cmake_install.cmake")
  include("/home/mariap/Packages/colmap/colmap/build/src/sfm/cmake_install.cmake")
  include("/home/mariap/Packages/colmap/colmap/build/src/tools/cmake_install.cmake")
  include("/home/mariap/Packages/colmap/colmap/build/src/ui/cmake_install.cmake")
  include("/home/mariap/Packages/colmap/colmap/build/src/util/cmake_install.cmake")

endif()

