# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /u/sw/toolchains/gcc-glibc/11.2.0/base/bin/cmake

# The command to remove a file.
RM = /u/sw/toolchains/gcc-glibc/11.2.0/base/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/giacomo/Documenti/PACS_Andena_Bottacini/Codici/Validation_pn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/giacomo/Documenti/PACS_Andena_Bottacini/Codici/Validation_pn/build

# Utility rule file for info.

# Include any custom commands dependencies for this target.
include CMakeFiles/info.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/info.dir/progress.make

CMakeFiles/info:
	/u/sw/toolchains/gcc-glibc/11.2.0/base/bin/cmake -P /home/giacomo/Documenti/PACS_Andena_Bottacini/Codici/Validation_pn/build/CMakeFiles/print_usage.cmake

info: CMakeFiles/info
info: CMakeFiles/info.dir/build.make
.PHONY : info

# Rule to build all files generated by this target.
CMakeFiles/info.dir/build: info
.PHONY : CMakeFiles/info.dir/build

CMakeFiles/info.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/info.dir/cmake_clean.cmake
.PHONY : CMakeFiles/info.dir/clean

CMakeFiles/info.dir/depend:
	cd /home/giacomo/Documenti/PACS_Andena_Bottacini/Codici/Validation_pn/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/giacomo/Documenti/PACS_Andena_Bottacini/Codici/Validation_pn /home/giacomo/Documenti/PACS_Andena_Bottacini/Codici/Validation_pn /home/giacomo/Documenti/PACS_Andena_Bottacini/Codici/Validation_pn/build /home/giacomo/Documenti/PACS_Andena_Bottacini/Codici/Validation_pn/build /home/giacomo/Documenti/PACS_Andena_Bottacini/Codici/Validation_pn/build/CMakeFiles/info.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/info.dir/depend

