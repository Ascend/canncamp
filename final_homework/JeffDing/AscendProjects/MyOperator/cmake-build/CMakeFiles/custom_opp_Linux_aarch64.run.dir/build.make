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

# Utility rule file for custom_opp_Linux_aarch64.run.

# Include the progress variables for this target.
include CMakeFiles/custom_opp_Linux_aarch64.run.dir/progress.make

CMakeFiles/custom_opp_Linux_aarch64.run: makepkg/packages/op_proto/custom/libcust_op_proto.so
CMakeFiles/custom_opp_Linux_aarch64.run: makepkg/packages/framework/custom/tensorflow/libcust_tf_parsers.so


custom_opp_Linux_aarch64.run: CMakeFiles/custom_opp_Linux_aarch64.run
custom_opp_Linux_aarch64.run: CMakeFiles/custom_opp_Linux_aarch64.run.dir/build.make
	mkdir -p ./makepkg/packages/fusion_rules/custom/
	mkdir -p ./makepkg/packages/op_impl/custom/ai_core/tbe/custom_impl
	mkdir -p ./makepkg/packages/op_impl/custom/vector_core/tbe/custom_impl
	mkdir -p ./makepkg/packages/op_impl/custom/cpu/aicpu_kernel/custom_impl
	bash /home/ma-user/AscendProjects/MyOperator/cmake/util/gen_impl_and_mrege_json.sh /home/ma-user/AscendProjects/MyOperator /home/ma-user/AscendProjects/MyOperator/cmake-build
	cp /home/ma-user/AscendProjects/MyOperator/scripts/install.sh ./makepkg/
	cp /home/ma-user/AscendProjects/MyOperator/scripts/upgrade.sh ./makepkg/
	cp /home/ma-user/AscendProjects/MyOperator/scripts/uninstall.sh ./makepkg/
	cp /home/ma-user/AscendProjects/MyOperator/scripts/help.sh ./makepkg/
	chmod u+x ./makepkg/*.sh
	../cmake/util/makeself/makeself.sh --gzip --complevel 4 --nomd5 --sha256 ./makepkg custom_opp_Linux_aarch64.run version:1.0 ./install.sh
.PHONY : custom_opp_Linux_aarch64.run

# Rule to build all files generated by this target.
CMakeFiles/custom_opp_Linux_aarch64.run.dir/build: custom_opp_Linux_aarch64.run

.PHONY : CMakeFiles/custom_opp_Linux_aarch64.run.dir/build

CMakeFiles/custom_opp_Linux_aarch64.run.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/custom_opp_Linux_aarch64.run.dir/cmake_clean.cmake
.PHONY : CMakeFiles/custom_opp_Linux_aarch64.run.dir/clean

CMakeFiles/custom_opp_Linux_aarch64.run.dir/depend:
	cd /home/ma-user/AscendProjects/MyOperator/cmake-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ma-user/AscendProjects/MyOperator /home/ma-user/AscendProjects/MyOperator /home/ma-user/AscendProjects/MyOperator/cmake-build /home/ma-user/AscendProjects/MyOperator/cmake-build /home/ma-user/AscendProjects/MyOperator/cmake-build/CMakeFiles/custom_opp_Linux_aarch64.run.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/custom_opp_Linux_aarch64.run.dir/depend

