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
CMAKE_SOURCE_DIR = /home/giacomo/Documenti/PACS_Andena_Bottacini/Codici

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/giacomo/Documenti/PACS_Andena_Bottacini/Codici/build

# Utility rule file for debug.

# Include any custom commands dependencies for this target.
include CMakeFiles/debug.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/debug.dir/progress.make

CMakeFiles/debug:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/giacomo/Documenti/PACS_Andena_Bottacini/Codici/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Switching CMAKE_BUILD_TYPE to Debug"
	/u/sw/toolchains/gcc-glibc/11.2.0/base/bin/cmake -DCMAKE_BUILD_TYPE=Debug /home/giacomo/Documenti/PACS_Andena_Bottacini/Codici
	/u/sw/toolchains/gcc-glibc/11.2.0/base/bin/cmake -E echo "***"
	/u/sw/toolchains/gcc-glibc/11.2.0/base/bin/cmake -E echo "*** Switched to Debug mode. Now recompile with:  \$$ make"
	/u/sw/toolchains/gcc-glibc/11.2.0/base/bin/cmake -E echo "***"

debug: CMakeFiles/debug
debug: CMakeFiles/debug.dir/build.make
.PHONY : debug

# Rule to build all files generated by this target.
CMakeFiles/debug.dir/build: debug
.PHONY : CMakeFiles/debug.dir/build

CMakeFiles/debug.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/debug.dir/cmake_clean.cmake
.PHONY : CMakeFiles/debug.dir/clean

CMakeFiles/debug.dir/depend:
	cd /home/giacomo/Documenti/PACS_Andena_Bottacini/Codici/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/giacomo/Documenti/PACS_Andena_Bottacini/Codici /home/giacomo/Documenti/PACS_Andena_Bottacini/Codici /home/giacomo/Documenti/PACS_Andena_Bottacini/Codici/build /home/giacomo/Documenti/PACS_Andena_Bottacini/Codici/build /home/giacomo/Documenti/PACS_Andena_Bottacini/Codici/build/CMakeFiles/debug.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/debug.dir/depend

