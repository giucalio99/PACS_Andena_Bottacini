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
CMAKE_SOURCE_DIR = /home/giacomo/Documenti/PACS_Andena_Bottacini/TimeDependentNS

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/giacomo/Documenti/PACS_Andena_Bottacini/TimeDependentNS/build

# Utility rule file for strip_comments.

# Include any custom commands dependencies for this target.
include CMakeFiles/strip_comments.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/strip_comments.dir/progress.make

CMakeFiles/strip_comments:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/giacomo/Documenti/PACS_Andena_Bottacini/TimeDependentNS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "strip comments"
	/u/sw/toolchains/gcc-glibc/11.2.0/base/bin/perl -pi -e 's#^[ \t]*//.*\n##g;' MyDataStruct.hpp Time.hpp BoundaryValues.hpp BlockSchurPreconditioner.hpp BlockSchurPreconditioner.cpp InsIMEX.hpp InsIMEX.cpp time_dependent_navier_stokes.cc

strip_comments: CMakeFiles/strip_comments
strip_comments: CMakeFiles/strip_comments.dir/build.make
.PHONY : strip_comments

# Rule to build all files generated by this target.
CMakeFiles/strip_comments.dir/build: strip_comments
.PHONY : CMakeFiles/strip_comments.dir/build

CMakeFiles/strip_comments.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/strip_comments.dir/cmake_clean.cmake
.PHONY : CMakeFiles/strip_comments.dir/clean

CMakeFiles/strip_comments.dir/depend:
	cd /home/giacomo/Documenti/PACS_Andena_Bottacini/TimeDependentNS/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/giacomo/Documenti/PACS_Andena_Bottacini/TimeDependentNS /home/giacomo/Documenti/PACS_Andena_Bottacini/TimeDependentNS /home/giacomo/Documenti/PACS_Andena_Bottacini/TimeDependentNS/build /home/giacomo/Documenti/PACS_Andena_Bottacini/TimeDependentNS/build /home/giacomo/Documenti/PACS_Andena_Bottacini/TimeDependentNS/build/CMakeFiles/strip_comments.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/strip_comments.dir/depend

