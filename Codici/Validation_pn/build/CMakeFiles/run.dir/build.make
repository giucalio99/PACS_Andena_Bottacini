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
CMAKE_SOURCE_DIR = /mnt/d/IPROP_PACS_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/d/IPROP_PACS_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn/build3

# Utility rule file for run.

# Include any custom commands dependencies for this target.
include CMakeFiles/run.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/run.dir/progress.make

CMakeFiles/run: PN
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/mnt/d/IPROP_PACS_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn/build3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Run PN with Debug configuration"
	./PN

run: CMakeFiles/run
run: CMakeFiles/run.dir/build.make
.PHONY : run

# Rule to build all files generated by this target.
CMakeFiles/run.dir/build: run
.PHONY : CMakeFiles/run.dir/build

CMakeFiles/run.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/run.dir/cmake_clean.cmake
.PHONY : CMakeFiles/run.dir/clean

CMakeFiles/run.dir/depend:
	cd /mnt/d/IPROP_PACS_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn/build3 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/d/IPROP_PACS_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn /mnt/d/IPROP_PACS_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn /mnt/d/IPROP_PACS_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn/build3 /mnt/d/IPROP_PACS_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn/build3 /mnt/d/IPROP_PACS_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn/build3/CMakeFiles/run.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/run.dir/depend

