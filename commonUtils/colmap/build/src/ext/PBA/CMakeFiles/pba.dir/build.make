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
include src/ext/PBA/CMakeFiles/pba.dir/depend.make

# Include the progress variables for this target.
include src/ext/PBA/CMakeFiles/pba.dir/progress.make

# Include the compile flags for this target's objects.
include src/ext/PBA/CMakeFiles/pba.dir/flags.make

src/ext/PBA/CMakeFiles/pba.dir/pba_generated_ProgramCU.cu.o: src/ext/PBA/CMakeFiles/pba.dir/pba_generated_ProgramCU.cu.o.depend
src/ext/PBA/CMakeFiles/pba.dir/pba_generated_ProgramCU.cu.o: src/ext/PBA/CMakeFiles/pba.dir/pba_generated_ProgramCU.cu.o.cmake
src/ext/PBA/CMakeFiles/pba.dir/pba_generated_ProgramCU.cu.o: ../src/ext/PBA/ProgramCU.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/mariap/Packages/colmap/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object src/ext/PBA/CMakeFiles/pba.dir/pba_generated_ProgramCU.cu.o"
	cd /home/mariap/Packages/colmap/colmap/build/src/ext/PBA/CMakeFiles/pba.dir && /usr/bin/cmake -E make_directory /home/mariap/Packages/colmap/colmap/build/src/ext/PBA/CMakeFiles/pba.dir//.
	cd /home/mariap/Packages/colmap/colmap/build/src/ext/PBA/CMakeFiles/pba.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING=Release -D generated_file:STRING=/home/mariap/Packages/colmap/colmap/build/src/ext/PBA/CMakeFiles/pba.dir//./pba_generated_ProgramCU.cu.o -D generated_cubin_file:STRING=/home/mariap/Packages/colmap/colmap/build/src/ext/PBA/CMakeFiles/pba.dir//./pba_generated_ProgramCU.cu.o.cubin.txt -P /home/mariap/Packages/colmap/colmap/build/src/ext/PBA/CMakeFiles/pba.dir//pba_generated_ProgramCU.cu.o.cmake

src/ext/PBA/CMakeFiles/pba.dir/ConfigBA.cpp.o: src/ext/PBA/CMakeFiles/pba.dir/flags.make
src/ext/PBA/CMakeFiles/pba.dir/ConfigBA.cpp.o: ../src/ext/PBA/ConfigBA.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mariap/Packages/colmap/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/ext/PBA/CMakeFiles/pba.dir/ConfigBA.cpp.o"
	cd /home/mariap/Packages/colmap/colmap/build/src/ext/PBA && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pba.dir/ConfigBA.cpp.o -c /home/mariap/Packages/colmap/colmap/src/ext/PBA/ConfigBA.cpp

src/ext/PBA/CMakeFiles/pba.dir/ConfigBA.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pba.dir/ConfigBA.cpp.i"
	cd /home/mariap/Packages/colmap/colmap/build/src/ext/PBA && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mariap/Packages/colmap/colmap/src/ext/PBA/ConfigBA.cpp > CMakeFiles/pba.dir/ConfigBA.cpp.i

src/ext/PBA/CMakeFiles/pba.dir/ConfigBA.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pba.dir/ConfigBA.cpp.s"
	cd /home/mariap/Packages/colmap/colmap/build/src/ext/PBA && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mariap/Packages/colmap/colmap/src/ext/PBA/ConfigBA.cpp -o CMakeFiles/pba.dir/ConfigBA.cpp.s

src/ext/PBA/CMakeFiles/pba.dir/ConfigBA.cpp.o.requires:

.PHONY : src/ext/PBA/CMakeFiles/pba.dir/ConfigBA.cpp.o.requires

src/ext/PBA/CMakeFiles/pba.dir/ConfigBA.cpp.o.provides: src/ext/PBA/CMakeFiles/pba.dir/ConfigBA.cpp.o.requires
	$(MAKE) -f src/ext/PBA/CMakeFiles/pba.dir/build.make src/ext/PBA/CMakeFiles/pba.dir/ConfigBA.cpp.o.provides.build
.PHONY : src/ext/PBA/CMakeFiles/pba.dir/ConfigBA.cpp.o.provides

src/ext/PBA/CMakeFiles/pba.dir/ConfigBA.cpp.o.provides.build: src/ext/PBA/CMakeFiles/pba.dir/ConfigBA.cpp.o


src/ext/PBA/CMakeFiles/pba.dir/CuTexImage.cpp.o: src/ext/PBA/CMakeFiles/pba.dir/flags.make
src/ext/PBA/CMakeFiles/pba.dir/CuTexImage.cpp.o: ../src/ext/PBA/CuTexImage.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mariap/Packages/colmap/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/ext/PBA/CMakeFiles/pba.dir/CuTexImage.cpp.o"
	cd /home/mariap/Packages/colmap/colmap/build/src/ext/PBA && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pba.dir/CuTexImage.cpp.o -c /home/mariap/Packages/colmap/colmap/src/ext/PBA/CuTexImage.cpp

src/ext/PBA/CMakeFiles/pba.dir/CuTexImage.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pba.dir/CuTexImage.cpp.i"
	cd /home/mariap/Packages/colmap/colmap/build/src/ext/PBA && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mariap/Packages/colmap/colmap/src/ext/PBA/CuTexImage.cpp > CMakeFiles/pba.dir/CuTexImage.cpp.i

src/ext/PBA/CMakeFiles/pba.dir/CuTexImage.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pba.dir/CuTexImage.cpp.s"
	cd /home/mariap/Packages/colmap/colmap/build/src/ext/PBA && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mariap/Packages/colmap/colmap/src/ext/PBA/CuTexImage.cpp -o CMakeFiles/pba.dir/CuTexImage.cpp.s

src/ext/PBA/CMakeFiles/pba.dir/CuTexImage.cpp.o.requires:

.PHONY : src/ext/PBA/CMakeFiles/pba.dir/CuTexImage.cpp.o.requires

src/ext/PBA/CMakeFiles/pba.dir/CuTexImage.cpp.o.provides: src/ext/PBA/CMakeFiles/pba.dir/CuTexImage.cpp.o.requires
	$(MAKE) -f src/ext/PBA/CMakeFiles/pba.dir/build.make src/ext/PBA/CMakeFiles/pba.dir/CuTexImage.cpp.o.provides.build
.PHONY : src/ext/PBA/CMakeFiles/pba.dir/CuTexImage.cpp.o.provides

src/ext/PBA/CMakeFiles/pba.dir/CuTexImage.cpp.o.provides.build: src/ext/PBA/CMakeFiles/pba.dir/CuTexImage.cpp.o


src/ext/PBA/CMakeFiles/pba.dir/pba.cpp.o: src/ext/PBA/CMakeFiles/pba.dir/flags.make
src/ext/PBA/CMakeFiles/pba.dir/pba.cpp.o: ../src/ext/PBA/pba.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mariap/Packages/colmap/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/ext/PBA/CMakeFiles/pba.dir/pba.cpp.o"
	cd /home/mariap/Packages/colmap/colmap/build/src/ext/PBA && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pba.dir/pba.cpp.o -c /home/mariap/Packages/colmap/colmap/src/ext/PBA/pba.cpp

src/ext/PBA/CMakeFiles/pba.dir/pba.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pba.dir/pba.cpp.i"
	cd /home/mariap/Packages/colmap/colmap/build/src/ext/PBA && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mariap/Packages/colmap/colmap/src/ext/PBA/pba.cpp > CMakeFiles/pba.dir/pba.cpp.i

src/ext/PBA/CMakeFiles/pba.dir/pba.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pba.dir/pba.cpp.s"
	cd /home/mariap/Packages/colmap/colmap/build/src/ext/PBA && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mariap/Packages/colmap/colmap/src/ext/PBA/pba.cpp -o CMakeFiles/pba.dir/pba.cpp.s

src/ext/PBA/CMakeFiles/pba.dir/pba.cpp.o.requires:

.PHONY : src/ext/PBA/CMakeFiles/pba.dir/pba.cpp.o.requires

src/ext/PBA/CMakeFiles/pba.dir/pba.cpp.o.provides: src/ext/PBA/CMakeFiles/pba.dir/pba.cpp.o.requires
	$(MAKE) -f src/ext/PBA/CMakeFiles/pba.dir/build.make src/ext/PBA/CMakeFiles/pba.dir/pba.cpp.o.provides.build
.PHONY : src/ext/PBA/CMakeFiles/pba.dir/pba.cpp.o.provides

src/ext/PBA/CMakeFiles/pba.dir/pba.cpp.o.provides.build: src/ext/PBA/CMakeFiles/pba.dir/pba.cpp.o


src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCPU.cpp.o: src/ext/PBA/CMakeFiles/pba.dir/flags.make
src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCPU.cpp.o: ../src/ext/PBA/SparseBundleCPU.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mariap/Packages/colmap/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCPU.cpp.o"
	cd /home/mariap/Packages/colmap/colmap/build/src/ext/PBA && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pba.dir/SparseBundleCPU.cpp.o -c /home/mariap/Packages/colmap/colmap/src/ext/PBA/SparseBundleCPU.cpp

src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCPU.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pba.dir/SparseBundleCPU.cpp.i"
	cd /home/mariap/Packages/colmap/colmap/build/src/ext/PBA && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mariap/Packages/colmap/colmap/src/ext/PBA/SparseBundleCPU.cpp > CMakeFiles/pba.dir/SparseBundleCPU.cpp.i

src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCPU.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pba.dir/SparseBundleCPU.cpp.s"
	cd /home/mariap/Packages/colmap/colmap/build/src/ext/PBA && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mariap/Packages/colmap/colmap/src/ext/PBA/SparseBundleCPU.cpp -o CMakeFiles/pba.dir/SparseBundleCPU.cpp.s

src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCPU.cpp.o.requires:

.PHONY : src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCPU.cpp.o.requires

src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCPU.cpp.o.provides: src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCPU.cpp.o.requires
	$(MAKE) -f src/ext/PBA/CMakeFiles/pba.dir/build.make src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCPU.cpp.o.provides.build
.PHONY : src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCPU.cpp.o.provides

src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCPU.cpp.o.provides.build: src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCPU.cpp.o


src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCU.cpp.o: src/ext/PBA/CMakeFiles/pba.dir/flags.make
src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCU.cpp.o: ../src/ext/PBA/SparseBundleCU.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mariap/Packages/colmap/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCU.cpp.o"
	cd /home/mariap/Packages/colmap/colmap/build/src/ext/PBA && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pba.dir/SparseBundleCU.cpp.o -c /home/mariap/Packages/colmap/colmap/src/ext/PBA/SparseBundleCU.cpp

src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCU.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pba.dir/SparseBundleCU.cpp.i"
	cd /home/mariap/Packages/colmap/colmap/build/src/ext/PBA && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mariap/Packages/colmap/colmap/src/ext/PBA/SparseBundleCU.cpp > CMakeFiles/pba.dir/SparseBundleCU.cpp.i

src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCU.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pba.dir/SparseBundleCU.cpp.s"
	cd /home/mariap/Packages/colmap/colmap/build/src/ext/PBA && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mariap/Packages/colmap/colmap/src/ext/PBA/SparseBundleCU.cpp -o CMakeFiles/pba.dir/SparseBundleCU.cpp.s

src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCU.cpp.o.requires:

.PHONY : src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCU.cpp.o.requires

src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCU.cpp.o.provides: src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCU.cpp.o.requires
	$(MAKE) -f src/ext/PBA/CMakeFiles/pba.dir/build.make src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCU.cpp.o.provides.build
.PHONY : src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCU.cpp.o.provides

src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCU.cpp.o.provides.build: src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCU.cpp.o


# Object files for target pba
pba_OBJECTS = \
"CMakeFiles/pba.dir/ConfigBA.cpp.o" \
"CMakeFiles/pba.dir/CuTexImage.cpp.o" \
"CMakeFiles/pba.dir/pba.cpp.o" \
"CMakeFiles/pba.dir/SparseBundleCPU.cpp.o" \
"CMakeFiles/pba.dir/SparseBundleCU.cpp.o"

# External object files for target pba
pba_EXTERNAL_OBJECTS = \
"/home/mariap/Packages/colmap/colmap/build/src/ext/PBA/CMakeFiles/pba.dir/pba_generated_ProgramCU.cu.o"

src/ext/PBA/libpba.a: src/ext/PBA/CMakeFiles/pba.dir/ConfigBA.cpp.o
src/ext/PBA/libpba.a: src/ext/PBA/CMakeFiles/pba.dir/CuTexImage.cpp.o
src/ext/PBA/libpba.a: src/ext/PBA/CMakeFiles/pba.dir/pba.cpp.o
src/ext/PBA/libpba.a: src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCPU.cpp.o
src/ext/PBA/libpba.a: src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCU.cpp.o
src/ext/PBA/libpba.a: src/ext/PBA/CMakeFiles/pba.dir/pba_generated_ProgramCU.cu.o
src/ext/PBA/libpba.a: src/ext/PBA/CMakeFiles/pba.dir/build.make
src/ext/PBA/libpba.a: src/ext/PBA/CMakeFiles/pba.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mariap/Packages/colmap/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX static library libpba.a"
	cd /home/mariap/Packages/colmap/colmap/build/src/ext/PBA && $(CMAKE_COMMAND) -P CMakeFiles/pba.dir/cmake_clean_target.cmake
	cd /home/mariap/Packages/colmap/colmap/build/src/ext/PBA && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pba.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/ext/PBA/CMakeFiles/pba.dir/build: src/ext/PBA/libpba.a

.PHONY : src/ext/PBA/CMakeFiles/pba.dir/build

src/ext/PBA/CMakeFiles/pba.dir/requires: src/ext/PBA/CMakeFiles/pba.dir/ConfigBA.cpp.o.requires
src/ext/PBA/CMakeFiles/pba.dir/requires: src/ext/PBA/CMakeFiles/pba.dir/CuTexImage.cpp.o.requires
src/ext/PBA/CMakeFiles/pba.dir/requires: src/ext/PBA/CMakeFiles/pba.dir/pba.cpp.o.requires
src/ext/PBA/CMakeFiles/pba.dir/requires: src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCPU.cpp.o.requires
src/ext/PBA/CMakeFiles/pba.dir/requires: src/ext/PBA/CMakeFiles/pba.dir/SparseBundleCU.cpp.o.requires

.PHONY : src/ext/PBA/CMakeFiles/pba.dir/requires

src/ext/PBA/CMakeFiles/pba.dir/clean:
	cd /home/mariap/Packages/colmap/colmap/build/src/ext/PBA && $(CMAKE_COMMAND) -P CMakeFiles/pba.dir/cmake_clean.cmake
.PHONY : src/ext/PBA/CMakeFiles/pba.dir/clean

src/ext/PBA/CMakeFiles/pba.dir/depend: src/ext/PBA/CMakeFiles/pba.dir/pba_generated_ProgramCU.cu.o
	cd /home/mariap/Packages/colmap/colmap/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mariap/Packages/colmap/colmap /home/mariap/Packages/colmap/colmap/src/ext/PBA /home/mariap/Packages/colmap/colmap/build /home/mariap/Packages/colmap/colmap/build/src/ext/PBA /home/mariap/Packages/colmap/colmap/build/src/ext/PBA/CMakeFiles/pba.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/ext/PBA/CMakeFiles/pba.dir/depend

