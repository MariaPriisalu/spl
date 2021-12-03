# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mariap/Packages/colmap/colmap

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mariap/Packages/colmap/colmap/build

# Include any dependencies generated for this target.
include src/estimators/CMakeFiles/estimators.dir/depend.make

# Include the progress variables for this target.
include src/estimators/CMakeFiles/estimators.dir/progress.make

# Include the compile flags for this target's objects.
include src/estimators/CMakeFiles/estimators.dir/flags.make

src/estimators/CMakeFiles/estimators.dir/absolute_pose.cc.o: src/estimators/CMakeFiles/estimators.dir/flags.make
src/estimators/CMakeFiles/estimators.dir/absolute_pose.cc.o: ../src/estimators/absolute_pose.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mariap/Packages/colmap/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/estimators/CMakeFiles/estimators.dir/absolute_pose.cc.o"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/estimators.dir/absolute_pose.cc.o -c /home/mariap/Packages/colmap/colmap/src/estimators/absolute_pose.cc

src/estimators/CMakeFiles/estimators.dir/absolute_pose.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/estimators.dir/absolute_pose.cc.i"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mariap/Packages/colmap/colmap/src/estimators/absolute_pose.cc > CMakeFiles/estimators.dir/absolute_pose.cc.i

src/estimators/CMakeFiles/estimators.dir/absolute_pose.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/estimators.dir/absolute_pose.cc.s"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mariap/Packages/colmap/colmap/src/estimators/absolute_pose.cc -o CMakeFiles/estimators.dir/absolute_pose.cc.s

src/estimators/CMakeFiles/estimators.dir/absolute_pose.cc.o.requires:

.PHONY : src/estimators/CMakeFiles/estimators.dir/absolute_pose.cc.o.requires

src/estimators/CMakeFiles/estimators.dir/absolute_pose.cc.o.provides: src/estimators/CMakeFiles/estimators.dir/absolute_pose.cc.o.requires
	$(MAKE) -f src/estimators/CMakeFiles/estimators.dir/build.make src/estimators/CMakeFiles/estimators.dir/absolute_pose.cc.o.provides.build
.PHONY : src/estimators/CMakeFiles/estimators.dir/absolute_pose.cc.o.provides

src/estimators/CMakeFiles/estimators.dir/absolute_pose.cc.o.provides.build: src/estimators/CMakeFiles/estimators.dir/absolute_pose.cc.o


src/estimators/CMakeFiles/estimators.dir/affine_transform.cc.o: src/estimators/CMakeFiles/estimators.dir/flags.make
src/estimators/CMakeFiles/estimators.dir/affine_transform.cc.o: ../src/estimators/affine_transform.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mariap/Packages/colmap/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/estimators/CMakeFiles/estimators.dir/affine_transform.cc.o"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/estimators.dir/affine_transform.cc.o -c /home/mariap/Packages/colmap/colmap/src/estimators/affine_transform.cc

src/estimators/CMakeFiles/estimators.dir/affine_transform.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/estimators.dir/affine_transform.cc.i"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mariap/Packages/colmap/colmap/src/estimators/affine_transform.cc > CMakeFiles/estimators.dir/affine_transform.cc.i

src/estimators/CMakeFiles/estimators.dir/affine_transform.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/estimators.dir/affine_transform.cc.s"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mariap/Packages/colmap/colmap/src/estimators/affine_transform.cc -o CMakeFiles/estimators.dir/affine_transform.cc.s

src/estimators/CMakeFiles/estimators.dir/affine_transform.cc.o.requires:

.PHONY : src/estimators/CMakeFiles/estimators.dir/affine_transform.cc.o.requires

src/estimators/CMakeFiles/estimators.dir/affine_transform.cc.o.provides: src/estimators/CMakeFiles/estimators.dir/affine_transform.cc.o.requires
	$(MAKE) -f src/estimators/CMakeFiles/estimators.dir/build.make src/estimators/CMakeFiles/estimators.dir/affine_transform.cc.o.provides.build
.PHONY : src/estimators/CMakeFiles/estimators.dir/affine_transform.cc.o.provides

src/estimators/CMakeFiles/estimators.dir/affine_transform.cc.o.provides.build: src/estimators/CMakeFiles/estimators.dir/affine_transform.cc.o


src/estimators/CMakeFiles/estimators.dir/coordinate_frame.cc.o: src/estimators/CMakeFiles/estimators.dir/flags.make
src/estimators/CMakeFiles/estimators.dir/coordinate_frame.cc.o: ../src/estimators/coordinate_frame.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mariap/Packages/colmap/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/estimators/CMakeFiles/estimators.dir/coordinate_frame.cc.o"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/estimators.dir/coordinate_frame.cc.o -c /home/mariap/Packages/colmap/colmap/src/estimators/coordinate_frame.cc

src/estimators/CMakeFiles/estimators.dir/coordinate_frame.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/estimators.dir/coordinate_frame.cc.i"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mariap/Packages/colmap/colmap/src/estimators/coordinate_frame.cc > CMakeFiles/estimators.dir/coordinate_frame.cc.i

src/estimators/CMakeFiles/estimators.dir/coordinate_frame.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/estimators.dir/coordinate_frame.cc.s"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mariap/Packages/colmap/colmap/src/estimators/coordinate_frame.cc -o CMakeFiles/estimators.dir/coordinate_frame.cc.s

src/estimators/CMakeFiles/estimators.dir/coordinate_frame.cc.o.requires:

.PHONY : src/estimators/CMakeFiles/estimators.dir/coordinate_frame.cc.o.requires

src/estimators/CMakeFiles/estimators.dir/coordinate_frame.cc.o.provides: src/estimators/CMakeFiles/estimators.dir/coordinate_frame.cc.o.requires
	$(MAKE) -f src/estimators/CMakeFiles/estimators.dir/build.make src/estimators/CMakeFiles/estimators.dir/coordinate_frame.cc.o.provides.build
.PHONY : src/estimators/CMakeFiles/estimators.dir/coordinate_frame.cc.o.provides

src/estimators/CMakeFiles/estimators.dir/coordinate_frame.cc.o.provides.build: src/estimators/CMakeFiles/estimators.dir/coordinate_frame.cc.o


src/estimators/CMakeFiles/estimators.dir/essential_matrix.cc.o: src/estimators/CMakeFiles/estimators.dir/flags.make
src/estimators/CMakeFiles/estimators.dir/essential_matrix.cc.o: ../src/estimators/essential_matrix.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mariap/Packages/colmap/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/estimators/CMakeFiles/estimators.dir/essential_matrix.cc.o"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/estimators.dir/essential_matrix.cc.o -c /home/mariap/Packages/colmap/colmap/src/estimators/essential_matrix.cc

src/estimators/CMakeFiles/estimators.dir/essential_matrix.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/estimators.dir/essential_matrix.cc.i"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mariap/Packages/colmap/colmap/src/estimators/essential_matrix.cc > CMakeFiles/estimators.dir/essential_matrix.cc.i

src/estimators/CMakeFiles/estimators.dir/essential_matrix.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/estimators.dir/essential_matrix.cc.s"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mariap/Packages/colmap/colmap/src/estimators/essential_matrix.cc -o CMakeFiles/estimators.dir/essential_matrix.cc.s

src/estimators/CMakeFiles/estimators.dir/essential_matrix.cc.o.requires:

.PHONY : src/estimators/CMakeFiles/estimators.dir/essential_matrix.cc.o.requires

src/estimators/CMakeFiles/estimators.dir/essential_matrix.cc.o.provides: src/estimators/CMakeFiles/estimators.dir/essential_matrix.cc.o.requires
	$(MAKE) -f src/estimators/CMakeFiles/estimators.dir/build.make src/estimators/CMakeFiles/estimators.dir/essential_matrix.cc.o.provides.build
.PHONY : src/estimators/CMakeFiles/estimators.dir/essential_matrix.cc.o.provides

src/estimators/CMakeFiles/estimators.dir/essential_matrix.cc.o.provides.build: src/estimators/CMakeFiles/estimators.dir/essential_matrix.cc.o


src/estimators/CMakeFiles/estimators.dir/fundamental_matrix.cc.o: src/estimators/CMakeFiles/estimators.dir/flags.make
src/estimators/CMakeFiles/estimators.dir/fundamental_matrix.cc.o: ../src/estimators/fundamental_matrix.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mariap/Packages/colmap/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/estimators/CMakeFiles/estimators.dir/fundamental_matrix.cc.o"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/estimators.dir/fundamental_matrix.cc.o -c /home/mariap/Packages/colmap/colmap/src/estimators/fundamental_matrix.cc

src/estimators/CMakeFiles/estimators.dir/fundamental_matrix.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/estimators.dir/fundamental_matrix.cc.i"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mariap/Packages/colmap/colmap/src/estimators/fundamental_matrix.cc > CMakeFiles/estimators.dir/fundamental_matrix.cc.i

src/estimators/CMakeFiles/estimators.dir/fundamental_matrix.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/estimators.dir/fundamental_matrix.cc.s"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mariap/Packages/colmap/colmap/src/estimators/fundamental_matrix.cc -o CMakeFiles/estimators.dir/fundamental_matrix.cc.s

src/estimators/CMakeFiles/estimators.dir/fundamental_matrix.cc.o.requires:

.PHONY : src/estimators/CMakeFiles/estimators.dir/fundamental_matrix.cc.o.requires

src/estimators/CMakeFiles/estimators.dir/fundamental_matrix.cc.o.provides: src/estimators/CMakeFiles/estimators.dir/fundamental_matrix.cc.o.requires
	$(MAKE) -f src/estimators/CMakeFiles/estimators.dir/build.make src/estimators/CMakeFiles/estimators.dir/fundamental_matrix.cc.o.provides.build
.PHONY : src/estimators/CMakeFiles/estimators.dir/fundamental_matrix.cc.o.provides

src/estimators/CMakeFiles/estimators.dir/fundamental_matrix.cc.o.provides.build: src/estimators/CMakeFiles/estimators.dir/fundamental_matrix.cc.o


src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose.cc.o: src/estimators/CMakeFiles/estimators.dir/flags.make
src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose.cc.o: ../src/estimators/generalized_absolute_pose.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mariap/Packages/colmap/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose.cc.o"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/estimators.dir/generalized_absolute_pose.cc.o -c /home/mariap/Packages/colmap/colmap/src/estimators/generalized_absolute_pose.cc

src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/estimators.dir/generalized_absolute_pose.cc.i"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mariap/Packages/colmap/colmap/src/estimators/generalized_absolute_pose.cc > CMakeFiles/estimators.dir/generalized_absolute_pose.cc.i

src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/estimators.dir/generalized_absolute_pose.cc.s"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mariap/Packages/colmap/colmap/src/estimators/generalized_absolute_pose.cc -o CMakeFiles/estimators.dir/generalized_absolute_pose.cc.s

src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose.cc.o.requires:

.PHONY : src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose.cc.o.requires

src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose.cc.o.provides: src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose.cc.o.requires
	$(MAKE) -f src/estimators/CMakeFiles/estimators.dir/build.make src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose.cc.o.provides.build
.PHONY : src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose.cc.o.provides

src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose.cc.o.provides.build: src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose.cc.o


src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose_coeffs.cc.o: src/estimators/CMakeFiles/estimators.dir/flags.make
src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose_coeffs.cc.o: ../src/estimators/generalized_absolute_pose_coeffs.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mariap/Packages/colmap/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose_coeffs.cc.o"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/estimators.dir/generalized_absolute_pose_coeffs.cc.o -c /home/mariap/Packages/colmap/colmap/src/estimators/generalized_absolute_pose_coeffs.cc

src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose_coeffs.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/estimators.dir/generalized_absolute_pose_coeffs.cc.i"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mariap/Packages/colmap/colmap/src/estimators/generalized_absolute_pose_coeffs.cc > CMakeFiles/estimators.dir/generalized_absolute_pose_coeffs.cc.i

src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose_coeffs.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/estimators.dir/generalized_absolute_pose_coeffs.cc.s"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mariap/Packages/colmap/colmap/src/estimators/generalized_absolute_pose_coeffs.cc -o CMakeFiles/estimators.dir/generalized_absolute_pose_coeffs.cc.s

src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose_coeffs.cc.o.requires:

.PHONY : src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose_coeffs.cc.o.requires

src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose_coeffs.cc.o.provides: src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose_coeffs.cc.o.requires
	$(MAKE) -f src/estimators/CMakeFiles/estimators.dir/build.make src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose_coeffs.cc.o.provides.build
.PHONY : src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose_coeffs.cc.o.provides

src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose_coeffs.cc.o.provides.build: src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose_coeffs.cc.o


src/estimators/CMakeFiles/estimators.dir/generalized_relative_pose.cc.o: src/estimators/CMakeFiles/estimators.dir/flags.make
src/estimators/CMakeFiles/estimators.dir/generalized_relative_pose.cc.o: ../src/estimators/generalized_relative_pose.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mariap/Packages/colmap/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object src/estimators/CMakeFiles/estimators.dir/generalized_relative_pose.cc.o"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/estimators.dir/generalized_relative_pose.cc.o -c /home/mariap/Packages/colmap/colmap/src/estimators/generalized_relative_pose.cc

src/estimators/CMakeFiles/estimators.dir/generalized_relative_pose.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/estimators.dir/generalized_relative_pose.cc.i"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mariap/Packages/colmap/colmap/src/estimators/generalized_relative_pose.cc > CMakeFiles/estimators.dir/generalized_relative_pose.cc.i

src/estimators/CMakeFiles/estimators.dir/generalized_relative_pose.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/estimators.dir/generalized_relative_pose.cc.s"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mariap/Packages/colmap/colmap/src/estimators/generalized_relative_pose.cc -o CMakeFiles/estimators.dir/generalized_relative_pose.cc.s

src/estimators/CMakeFiles/estimators.dir/generalized_relative_pose.cc.o.requires:

.PHONY : src/estimators/CMakeFiles/estimators.dir/generalized_relative_pose.cc.o.requires

src/estimators/CMakeFiles/estimators.dir/generalized_relative_pose.cc.o.provides: src/estimators/CMakeFiles/estimators.dir/generalized_relative_pose.cc.o.requires
	$(MAKE) -f src/estimators/CMakeFiles/estimators.dir/build.make src/estimators/CMakeFiles/estimators.dir/generalized_relative_pose.cc.o.provides.build
.PHONY : src/estimators/CMakeFiles/estimators.dir/generalized_relative_pose.cc.o.provides

src/estimators/CMakeFiles/estimators.dir/generalized_relative_pose.cc.o.provides.build: src/estimators/CMakeFiles/estimators.dir/generalized_relative_pose.cc.o


src/estimators/CMakeFiles/estimators.dir/homography_matrix.cc.o: src/estimators/CMakeFiles/estimators.dir/flags.make
src/estimators/CMakeFiles/estimators.dir/homography_matrix.cc.o: ../src/estimators/homography_matrix.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mariap/Packages/colmap/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object src/estimators/CMakeFiles/estimators.dir/homography_matrix.cc.o"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/estimators.dir/homography_matrix.cc.o -c /home/mariap/Packages/colmap/colmap/src/estimators/homography_matrix.cc

src/estimators/CMakeFiles/estimators.dir/homography_matrix.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/estimators.dir/homography_matrix.cc.i"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mariap/Packages/colmap/colmap/src/estimators/homography_matrix.cc > CMakeFiles/estimators.dir/homography_matrix.cc.i

src/estimators/CMakeFiles/estimators.dir/homography_matrix.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/estimators.dir/homography_matrix.cc.s"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mariap/Packages/colmap/colmap/src/estimators/homography_matrix.cc -o CMakeFiles/estimators.dir/homography_matrix.cc.s

src/estimators/CMakeFiles/estimators.dir/homography_matrix.cc.o.requires:

.PHONY : src/estimators/CMakeFiles/estimators.dir/homography_matrix.cc.o.requires

src/estimators/CMakeFiles/estimators.dir/homography_matrix.cc.o.provides: src/estimators/CMakeFiles/estimators.dir/homography_matrix.cc.o.requires
	$(MAKE) -f src/estimators/CMakeFiles/estimators.dir/build.make src/estimators/CMakeFiles/estimators.dir/homography_matrix.cc.o.provides.build
.PHONY : src/estimators/CMakeFiles/estimators.dir/homography_matrix.cc.o.provides

src/estimators/CMakeFiles/estimators.dir/homography_matrix.cc.o.provides.build: src/estimators/CMakeFiles/estimators.dir/homography_matrix.cc.o


src/estimators/CMakeFiles/estimators.dir/pose.cc.o: src/estimators/CMakeFiles/estimators.dir/flags.make
src/estimators/CMakeFiles/estimators.dir/pose.cc.o: ../src/estimators/pose.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mariap/Packages/colmap/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object src/estimators/CMakeFiles/estimators.dir/pose.cc.o"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/estimators.dir/pose.cc.o -c /home/mariap/Packages/colmap/colmap/src/estimators/pose.cc

src/estimators/CMakeFiles/estimators.dir/pose.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/estimators.dir/pose.cc.i"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mariap/Packages/colmap/colmap/src/estimators/pose.cc > CMakeFiles/estimators.dir/pose.cc.i

src/estimators/CMakeFiles/estimators.dir/pose.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/estimators.dir/pose.cc.s"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mariap/Packages/colmap/colmap/src/estimators/pose.cc -o CMakeFiles/estimators.dir/pose.cc.s

src/estimators/CMakeFiles/estimators.dir/pose.cc.o.requires:

.PHONY : src/estimators/CMakeFiles/estimators.dir/pose.cc.o.requires

src/estimators/CMakeFiles/estimators.dir/pose.cc.o.provides: src/estimators/CMakeFiles/estimators.dir/pose.cc.o.requires
	$(MAKE) -f src/estimators/CMakeFiles/estimators.dir/build.make src/estimators/CMakeFiles/estimators.dir/pose.cc.o.provides.build
.PHONY : src/estimators/CMakeFiles/estimators.dir/pose.cc.o.provides

src/estimators/CMakeFiles/estimators.dir/pose.cc.o.provides.build: src/estimators/CMakeFiles/estimators.dir/pose.cc.o


src/estimators/CMakeFiles/estimators.dir/triangulation.cc.o: src/estimators/CMakeFiles/estimators.dir/flags.make
src/estimators/CMakeFiles/estimators.dir/triangulation.cc.o: ../src/estimators/triangulation.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mariap/Packages/colmap/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object src/estimators/CMakeFiles/estimators.dir/triangulation.cc.o"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/estimators.dir/triangulation.cc.o -c /home/mariap/Packages/colmap/colmap/src/estimators/triangulation.cc

src/estimators/CMakeFiles/estimators.dir/triangulation.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/estimators.dir/triangulation.cc.i"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mariap/Packages/colmap/colmap/src/estimators/triangulation.cc > CMakeFiles/estimators.dir/triangulation.cc.i

src/estimators/CMakeFiles/estimators.dir/triangulation.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/estimators.dir/triangulation.cc.s"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mariap/Packages/colmap/colmap/src/estimators/triangulation.cc -o CMakeFiles/estimators.dir/triangulation.cc.s

src/estimators/CMakeFiles/estimators.dir/triangulation.cc.o.requires:

.PHONY : src/estimators/CMakeFiles/estimators.dir/triangulation.cc.o.requires

src/estimators/CMakeFiles/estimators.dir/triangulation.cc.o.provides: src/estimators/CMakeFiles/estimators.dir/triangulation.cc.o.requires
	$(MAKE) -f src/estimators/CMakeFiles/estimators.dir/build.make src/estimators/CMakeFiles/estimators.dir/triangulation.cc.o.provides.build
.PHONY : src/estimators/CMakeFiles/estimators.dir/triangulation.cc.o.provides

src/estimators/CMakeFiles/estimators.dir/triangulation.cc.o.provides.build: src/estimators/CMakeFiles/estimators.dir/triangulation.cc.o


src/estimators/CMakeFiles/estimators.dir/two_view_geometry.cc.o: src/estimators/CMakeFiles/estimators.dir/flags.make
src/estimators/CMakeFiles/estimators.dir/two_view_geometry.cc.o: ../src/estimators/two_view_geometry.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mariap/Packages/colmap/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object src/estimators/CMakeFiles/estimators.dir/two_view_geometry.cc.o"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/estimators.dir/two_view_geometry.cc.o -c /home/mariap/Packages/colmap/colmap/src/estimators/two_view_geometry.cc

src/estimators/CMakeFiles/estimators.dir/two_view_geometry.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/estimators.dir/two_view_geometry.cc.i"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mariap/Packages/colmap/colmap/src/estimators/two_view_geometry.cc > CMakeFiles/estimators.dir/two_view_geometry.cc.i

src/estimators/CMakeFiles/estimators.dir/two_view_geometry.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/estimators.dir/two_view_geometry.cc.s"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mariap/Packages/colmap/colmap/src/estimators/two_view_geometry.cc -o CMakeFiles/estimators.dir/two_view_geometry.cc.s

src/estimators/CMakeFiles/estimators.dir/two_view_geometry.cc.o.requires:

.PHONY : src/estimators/CMakeFiles/estimators.dir/two_view_geometry.cc.o.requires

src/estimators/CMakeFiles/estimators.dir/two_view_geometry.cc.o.provides: src/estimators/CMakeFiles/estimators.dir/two_view_geometry.cc.o.requires
	$(MAKE) -f src/estimators/CMakeFiles/estimators.dir/build.make src/estimators/CMakeFiles/estimators.dir/two_view_geometry.cc.o.provides.build
.PHONY : src/estimators/CMakeFiles/estimators.dir/two_view_geometry.cc.o.provides

src/estimators/CMakeFiles/estimators.dir/two_view_geometry.cc.o.provides.build: src/estimators/CMakeFiles/estimators.dir/two_view_geometry.cc.o


src/estimators/CMakeFiles/estimators.dir/utils.cc.o: src/estimators/CMakeFiles/estimators.dir/flags.make
src/estimators/CMakeFiles/estimators.dir/utils.cc.o: ../src/estimators/utils.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mariap/Packages/colmap/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object src/estimators/CMakeFiles/estimators.dir/utils.cc.o"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/estimators.dir/utils.cc.o -c /home/mariap/Packages/colmap/colmap/src/estimators/utils.cc

src/estimators/CMakeFiles/estimators.dir/utils.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/estimators.dir/utils.cc.i"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mariap/Packages/colmap/colmap/src/estimators/utils.cc > CMakeFiles/estimators.dir/utils.cc.i

src/estimators/CMakeFiles/estimators.dir/utils.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/estimators.dir/utils.cc.s"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mariap/Packages/colmap/colmap/src/estimators/utils.cc -o CMakeFiles/estimators.dir/utils.cc.s

src/estimators/CMakeFiles/estimators.dir/utils.cc.o.requires:

.PHONY : src/estimators/CMakeFiles/estimators.dir/utils.cc.o.requires

src/estimators/CMakeFiles/estimators.dir/utils.cc.o.provides: src/estimators/CMakeFiles/estimators.dir/utils.cc.o.requires
	$(MAKE) -f src/estimators/CMakeFiles/estimators.dir/build.make src/estimators/CMakeFiles/estimators.dir/utils.cc.o.provides.build
.PHONY : src/estimators/CMakeFiles/estimators.dir/utils.cc.o.provides

src/estimators/CMakeFiles/estimators.dir/utils.cc.o.provides.build: src/estimators/CMakeFiles/estimators.dir/utils.cc.o


# Object files for target estimators
estimators_OBJECTS = \
"CMakeFiles/estimators.dir/absolute_pose.cc.o" \
"CMakeFiles/estimators.dir/affine_transform.cc.o" \
"CMakeFiles/estimators.dir/coordinate_frame.cc.o" \
"CMakeFiles/estimators.dir/essential_matrix.cc.o" \
"CMakeFiles/estimators.dir/fundamental_matrix.cc.o" \
"CMakeFiles/estimators.dir/generalized_absolute_pose.cc.o" \
"CMakeFiles/estimators.dir/generalized_absolute_pose_coeffs.cc.o" \
"CMakeFiles/estimators.dir/generalized_relative_pose.cc.o" \
"CMakeFiles/estimators.dir/homography_matrix.cc.o" \
"CMakeFiles/estimators.dir/pose.cc.o" \
"CMakeFiles/estimators.dir/triangulation.cc.o" \
"CMakeFiles/estimators.dir/two_view_geometry.cc.o" \
"CMakeFiles/estimators.dir/utils.cc.o"

# External object files for target estimators
estimators_EXTERNAL_OBJECTS =

src/estimators/libestimators.a: src/estimators/CMakeFiles/estimators.dir/absolute_pose.cc.o
src/estimators/libestimators.a: src/estimators/CMakeFiles/estimators.dir/affine_transform.cc.o
src/estimators/libestimators.a: src/estimators/CMakeFiles/estimators.dir/coordinate_frame.cc.o
src/estimators/libestimators.a: src/estimators/CMakeFiles/estimators.dir/essential_matrix.cc.o
src/estimators/libestimators.a: src/estimators/CMakeFiles/estimators.dir/fundamental_matrix.cc.o
src/estimators/libestimators.a: src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose.cc.o
src/estimators/libestimators.a: src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose_coeffs.cc.o
src/estimators/libestimators.a: src/estimators/CMakeFiles/estimators.dir/generalized_relative_pose.cc.o
src/estimators/libestimators.a: src/estimators/CMakeFiles/estimators.dir/homography_matrix.cc.o
src/estimators/libestimators.a: src/estimators/CMakeFiles/estimators.dir/pose.cc.o
src/estimators/libestimators.a: src/estimators/CMakeFiles/estimators.dir/triangulation.cc.o
src/estimators/libestimators.a: src/estimators/CMakeFiles/estimators.dir/two_view_geometry.cc.o
src/estimators/libestimators.a: src/estimators/CMakeFiles/estimators.dir/utils.cc.o
src/estimators/libestimators.a: src/estimators/CMakeFiles/estimators.dir/build.make
src/estimators/libestimators.a: src/estimators/CMakeFiles/estimators.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mariap/Packages/colmap/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Linking CXX static library libestimators.a"
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && $(CMAKE_COMMAND) -P CMakeFiles/estimators.dir/cmake_clean_target.cmake
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/estimators.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/estimators/CMakeFiles/estimators.dir/build: src/estimators/libestimators.a

.PHONY : src/estimators/CMakeFiles/estimators.dir/build

src/estimators/CMakeFiles/estimators.dir/requires: src/estimators/CMakeFiles/estimators.dir/absolute_pose.cc.o.requires
src/estimators/CMakeFiles/estimators.dir/requires: src/estimators/CMakeFiles/estimators.dir/affine_transform.cc.o.requires
src/estimators/CMakeFiles/estimators.dir/requires: src/estimators/CMakeFiles/estimators.dir/coordinate_frame.cc.o.requires
src/estimators/CMakeFiles/estimators.dir/requires: src/estimators/CMakeFiles/estimators.dir/essential_matrix.cc.o.requires
src/estimators/CMakeFiles/estimators.dir/requires: src/estimators/CMakeFiles/estimators.dir/fundamental_matrix.cc.o.requires
src/estimators/CMakeFiles/estimators.dir/requires: src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose.cc.o.requires
src/estimators/CMakeFiles/estimators.dir/requires: src/estimators/CMakeFiles/estimators.dir/generalized_absolute_pose_coeffs.cc.o.requires
src/estimators/CMakeFiles/estimators.dir/requires: src/estimators/CMakeFiles/estimators.dir/generalized_relative_pose.cc.o.requires
src/estimators/CMakeFiles/estimators.dir/requires: src/estimators/CMakeFiles/estimators.dir/homography_matrix.cc.o.requires
src/estimators/CMakeFiles/estimators.dir/requires: src/estimators/CMakeFiles/estimators.dir/pose.cc.o.requires
src/estimators/CMakeFiles/estimators.dir/requires: src/estimators/CMakeFiles/estimators.dir/triangulation.cc.o.requires
src/estimators/CMakeFiles/estimators.dir/requires: src/estimators/CMakeFiles/estimators.dir/two_view_geometry.cc.o.requires
src/estimators/CMakeFiles/estimators.dir/requires: src/estimators/CMakeFiles/estimators.dir/utils.cc.o.requires

.PHONY : src/estimators/CMakeFiles/estimators.dir/requires

src/estimators/CMakeFiles/estimators.dir/clean:
	cd /home/mariap/Packages/colmap/colmap/build/src/estimators && $(CMAKE_COMMAND) -P CMakeFiles/estimators.dir/cmake_clean.cmake
.PHONY : src/estimators/CMakeFiles/estimators.dir/clean

src/estimators/CMakeFiles/estimators.dir/depend:
	cd /home/mariap/Packages/colmap/colmap/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mariap/Packages/colmap/colmap /home/mariap/Packages/colmap/colmap/src/estimators /home/mariap/Packages/colmap/colmap/build /home/mariap/Packages/colmap/colmap/build/src/estimators /home/mariap/Packages/colmap/colmap/build/src/estimators/CMakeFiles/estimators.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/estimators/CMakeFiles/estimators.dir/depend

