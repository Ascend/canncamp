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
CMAKE_SOURCE_DIR = /home/ma-user/AscendProjects/MyOperator

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ma-user/AscendProjects/MyOperator/cmake-build

# Include any dependencies generated for this target.
include op_proto/CMakeFiles/cust_op_proto.dir/depend.make

# Include the progress variables for this target.
include op_proto/CMakeFiles/cust_op_proto.dir/progress.make

# Include the compile flags for this target's objects.
include op_proto/CMakeFiles/cust_op_proto.dir/flags.make

op_proto/CMakeFiles/cust_op_proto.dir/sqrt.cc.o: op_proto/CMakeFiles/cust_op_proto.dir/flags.make
op_proto/CMakeFiles/cust_op_proto.dir/sqrt.cc.o: ../op_proto/sqrt.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ma-user/AscendProjects/MyOperator/cmake-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object op_proto/CMakeFiles/cust_op_proto.dir/sqrt.cc.o"
	cd /home/ma-user/AscendProjects/MyOperator/cmake-build/op_proto && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cust_op_proto.dir/sqrt.cc.o -c /home/ma-user/AscendProjects/MyOperator/op_proto/sqrt.cc

op_proto/CMakeFiles/cust_op_proto.dir/sqrt.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cust_op_proto.dir/sqrt.cc.i"
	cd /home/ma-user/AscendProjects/MyOperator/cmake-build/op_proto && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ma-user/AscendProjects/MyOperator/op_proto/sqrt.cc > CMakeFiles/cust_op_proto.dir/sqrt.cc.i

op_proto/CMakeFiles/cust_op_proto.dir/sqrt.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cust_op_proto.dir/sqrt.cc.s"
	cd /home/ma-user/AscendProjects/MyOperator/cmake-build/op_proto && g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ma-user/AscendProjects/MyOperator/op_proto/sqrt.cc -o CMakeFiles/cust_op_proto.dir/sqrt.cc.s

# Object files for target cust_op_proto
cust_op_proto_OBJECTS = \
"CMakeFiles/cust_op_proto.dir/sqrt.cc.o"

# External object files for target cust_op_proto
cust_op_proto_EXTERNAL_OBJECTS =

makepkg/packages/op_proto/custom/libcust_op_proto.so: op_proto/CMakeFiles/cust_op_proto.dir/sqrt.cc.o
makepkg/packages/op_proto/custom/libcust_op_proto.so: op_proto/CMakeFiles/cust_op_proto.dir/build.make
makepkg/packages/op_proto/custom/libcust_op_proto.so: /home/ma-user/Ascend/ascend-toolkit/5.0.2.1/atc/include/../lib64/libgraph.so
makepkg/packages/op_proto/custom/libcust_op_proto.so: op_proto/CMakeFiles/cust_op_proto.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ma-user/AscendProjects/MyOperator/cmake-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library ../makepkg/packages/op_proto/custom/libcust_op_proto.so"
	cd /home/ma-user/AscendProjects/MyOperator/cmake-build/op_proto && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cust_op_proto.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
op_proto/CMakeFiles/cust_op_proto.dir/build: makepkg/packages/op_proto/custom/libcust_op_proto.so

.PHONY : op_proto/CMakeFiles/cust_op_proto.dir/build

op_proto/CMakeFiles/cust_op_proto.dir/clean:
	cd /home/ma-user/AscendProjects/MyOperator/cmake-build/op_proto && $(CMAKE_COMMAND) -P CMakeFiles/cust_op_proto.dir/cmake_clean.cmake
.PHONY : op_proto/CMakeFiles/cust_op_proto.dir/clean

op_proto/CMakeFiles/cust_op_proto.dir/depend:
	cd /home/ma-user/AscendProjects/MyOperator/cmake-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ma-user/AscendProjects/MyOperator /home/ma-user/AscendProjects/MyOperator/op_proto /home/ma-user/AscendProjects/MyOperator/cmake-build /home/ma-user/AscendProjects/MyOperator/cmake-build/op_proto /home/ma-user/AscendProjects/MyOperator/cmake-build/op_proto/CMakeFiles/cust_op_proto.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : op_proto/CMakeFiles/cust_op_proto.dir/depend
