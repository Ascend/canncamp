# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ma-user/AscendProjects/Sub

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ma-user/AscendProjects/Sub/build/cmake

# Include any dependencies generated for this target.
include framework/tf_plugin/CMakeFiles/cust_tf_parsers.dir/depend.make

# Include the progress variables for this target.
include framework/tf_plugin/CMakeFiles/cust_tf_parsers.dir/progress.make

# Include the compile flags for this target's objects.
include framework/tf_plugin/CMakeFiles/cust_tf_parsers.dir/flags.make

framework/tf_plugin/CMakeFiles/cust_tf_parsers.dir/tensorflow_sub_plugin.cc.o: framework/tf_plugin/CMakeFiles/cust_tf_parsers.dir/flags.make
framework/tf_plugin/CMakeFiles/cust_tf_parsers.dir/tensorflow_sub_plugin.cc.o: ../../framework/tf_plugin/tensorflow_sub_plugin.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ma-user/AscendProjects/Sub/build/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object framework/tf_plugin/CMakeFiles/cust_tf_parsers.dir/tensorflow_sub_plugin.cc.o"
	cd /home/ma-user/AscendProjects/Sub/build/cmake/framework/tf_plugin && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cust_tf_parsers.dir/tensorflow_sub_plugin.cc.o -c /home/ma-user/AscendProjects/Sub/framework/tf_plugin/tensorflow_sub_plugin.cc

framework/tf_plugin/CMakeFiles/cust_tf_parsers.dir/tensorflow_sub_plugin.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cust_tf_parsers.dir/tensorflow_sub_plugin.cc.i"
	cd /home/ma-user/AscendProjects/Sub/build/cmake/framework/tf_plugin && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ma-user/AscendProjects/Sub/framework/tf_plugin/tensorflow_sub_plugin.cc > CMakeFiles/cust_tf_parsers.dir/tensorflow_sub_plugin.cc.i

framework/tf_plugin/CMakeFiles/cust_tf_parsers.dir/tensorflow_sub_plugin.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cust_tf_parsers.dir/tensorflow_sub_plugin.cc.s"
	cd /home/ma-user/AscendProjects/Sub/build/cmake/framework/tf_plugin && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ma-user/AscendProjects/Sub/framework/tf_plugin/tensorflow_sub_plugin.cc -o CMakeFiles/cust_tf_parsers.dir/tensorflow_sub_plugin.cc.s

# Object files for target cust_tf_parsers
cust_tf_parsers_OBJECTS = \
"CMakeFiles/cust_tf_parsers.dir/tensorflow_sub_plugin.cc.o"

# External object files for target cust_tf_parsers
cust_tf_parsers_EXTERNAL_OBJECTS =

makepkg/packages/framework/custom/tensorflow/libcust_tf_parsers.so: framework/tf_plugin/CMakeFiles/cust_tf_parsers.dir/tensorflow_sub_plugin.cc.o
makepkg/packages/framework/custom/tensorflow/libcust_tf_parsers.so: framework/tf_plugin/CMakeFiles/cust_tf_parsers.dir/build.make
makepkg/packages/framework/custom/tensorflow/libcust_tf_parsers.so: /home/ma-user/Ascend/ascend-toolkit/5.0.2.1/atc/include/../lib64/libgraph.so
makepkg/packages/framework/custom/tensorflow/libcust_tf_parsers.so: framework/tf_plugin/CMakeFiles/cust_tf_parsers.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ma-user/AscendProjects/Sub/build/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library ../../makepkg/packages/framework/custom/tensorflow/libcust_tf_parsers.so"
	cd /home/ma-user/AscendProjects/Sub/build/cmake/framework/tf_plugin && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cust_tf_parsers.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
framework/tf_plugin/CMakeFiles/cust_tf_parsers.dir/build: makepkg/packages/framework/custom/tensorflow/libcust_tf_parsers.so

.PHONY : framework/tf_plugin/CMakeFiles/cust_tf_parsers.dir/build

framework/tf_plugin/CMakeFiles/cust_tf_parsers.dir/clean:
	cd /home/ma-user/AscendProjects/Sub/build/cmake/framework/tf_plugin && $(CMAKE_COMMAND) -P CMakeFiles/cust_tf_parsers.dir/cmake_clean.cmake
.PHONY : framework/tf_plugin/CMakeFiles/cust_tf_parsers.dir/clean

framework/tf_plugin/CMakeFiles/cust_tf_parsers.dir/depend:
	cd /home/ma-user/AscendProjects/Sub/build/cmake && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ma-user/AscendProjects/Sub /home/ma-user/AscendProjects/Sub/framework/tf_plugin /home/ma-user/AscendProjects/Sub/build/cmake /home/ma-user/AscendProjects/Sub/build/cmake/framework/tf_plugin /home/ma-user/AscendProjects/Sub/build/cmake/framework/tf_plugin/CMakeFiles/cust_tf_parsers.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : framework/tf_plugin/CMakeFiles/cust_tf_parsers.dir/depend

