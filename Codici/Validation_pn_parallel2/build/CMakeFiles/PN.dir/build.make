# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

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
CMAKE_COMMAND = /opt/mox/mk/toolchains/gcc-glibc/11.2.0/base/bin/cmake

# The command to remove a file.
RM = /opt/mox/mk/toolchains/gcc-glibc/11.2.0/base/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/d/IPROP_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn_parallel2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/d/IPROP_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn_parallel2/build

# Include any dependencies generated for this target.
include CMakeFiles/PN.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/PN.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/PN.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/PN.dir/flags.make

CMakeFiles/PN.dir/Validation_pn.cc.o: CMakeFiles/PN.dir/flags.make
CMakeFiles/PN.dir/Validation_pn.cc.o: /mnt/d/IPROP_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn_parallel2/Validation_pn.cc
CMakeFiles/PN.dir/Validation_pn.cc.o: CMakeFiles/PN.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/mnt/d/IPROP_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn_parallel2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/PN.dir/Validation_pn.cc.o"
	/opt/mox/mk/toolchains/gcc-glibc/11.2.0/prefix/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/PN.dir/Validation_pn.cc.o -MF CMakeFiles/PN.dir/Validation_pn.cc.o.d -o CMakeFiles/PN.dir/Validation_pn.cc.o -c /mnt/d/IPROP_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn_parallel2/Validation_pn.cc

CMakeFiles/PN.dir/Validation_pn.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/PN.dir/Validation_pn.cc.i"
	/opt/mox/mk/toolchains/gcc-glibc/11.2.0/prefix/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/IPROP_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn_parallel2/Validation_pn.cc > CMakeFiles/PN.dir/Validation_pn.cc.i

CMakeFiles/PN.dir/Validation_pn.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/PN.dir/Validation_pn.cc.s"
	/opt/mox/mk/toolchains/gcc-glibc/11.2.0/prefix/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/IPROP_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn_parallel2/Validation_pn.cc -o CMakeFiles/PN.dir/Validation_pn.cc.s

# Object files for target PN
PN_OBJECTS = \
"CMakeFiles/PN.dir/Validation_pn.cc.o"

# External object files for target PN
PN_EXTERNAL_OBJECTS =

PN: CMakeFiles/PN.dir/Validation_pn.cc.o
PN: CMakeFiles/PN.dir/build.make
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.5.1/lib/libdeal_II.g.so.9.5.1
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/boost/1.76.0/lib/libboost_iostreams.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/boost/1.76.0/lib/libboost_serialization.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/boost/1.76.0/lib/libboost_system.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/boost/1.76.0/lib/libboost_thread.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/boost/1.76.0/lib/libboost_regex.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/boost/1.76.0/lib/libboost_chrono.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/boost/1.76.0/lib/libboost_date_time.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/boost/1.76.0/lib/libboost_atomic.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libkokkossimd.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libkokkosalgorithms.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libkokkoscontainers.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libkokkoscore.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/librol.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libmuelu.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libmuelu-adapters.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libnox.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libnoxlapack.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libnoxepetra.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libloca.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/liblocalapack.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/liblocaepetra.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/liblocathyra.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libstratimikosifpack.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libstratimikosml.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libstratimikosamesos.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libstratimikosaztecoo.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libstratimikosbelos.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libstratimikos.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libanasazi.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libanasaziepetra.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libModeLaplace.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libanasazitpetra.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libbelos.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libbelosepetra.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libbelostpetra.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libbelosxpetra.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libml.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libifpack.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libamesos.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libgaleri-epetra.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libgaleri-xpetra.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libaztecoo.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libisorropia.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libxpetra.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libxpetra-sup.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libthyracore.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libthyraepetra.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libthyraepetraext.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libthyratpetra.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libtrilinosss.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libkokkostsqr.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libtpetraclassic.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libtpetra.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libtpetrainout.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libtpetraext.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libepetraext.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libtriutils.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libzoltan.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libepetra.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libsacado.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/librtop.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libkokkoskernels.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libteuchoscore.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libteuchosparser.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libteuchosparameterlist.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libteuchoscomm.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libteuchosnumerics.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libteuchosremainder.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libteuchoskokkoscompat.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/lib64/libteuchoskokkoscomm.so.15.0
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/prefix/lib/libpthread.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/base/lib/libhwloc.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/tbb/2021.4.0/lib64/libtbb.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/blacs/1.1/lib/libblacs.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/blacs/1.1/lib/libblacsF77init.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/mumps/5.4.0/lib/libzmumps.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/mumps/5.4.0/lib/libcmumps.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/mumps/5.4.0/lib/libsmumps.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/suitesparse/5.10.1/lib/libccolamd.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/suitesparse/5.10.1/lib/libcolamd.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/suitesparse/5.10.1/lib/libcamd.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/suitesparse/5.10.1/lib/libsuitesparseconfig.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/adol-c/2.7.2/lib64/libadolc.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/arpack/3.8.0/lib/libarpack.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/gsl/2.7/lib/libgsl.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/gsl/2.7/lib/libgslcblas.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/prefix/lib/libdl.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/prefix/lib/libm.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/petsc/3.21.0/lib/libslepc.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/petsc/3.21.0/lib/libpetsc.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/hypre/2.22.0/lib/libHYPRE.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/suitesparse/5.10.1/lib/libspqr.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/suitesparse/5.10.1/lib/libumfpack.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/suitesparse/5.10.1/lib/libklu.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/suitesparse/5.10.1/lib/libcholmod.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/suitesparse/5.10.1/lib/libamd.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/mumps/5.4.0/lib/libdmumps.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/mumps/5.4.0/lib/libmumps_common.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/mumps/5.4.0/lib/libpord.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/scalapack/2.1.0/lib/libscalapack.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/fftw/3.3.9/lib/libfftw3_mpi.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/fftw/3.3.9/lib/libfftw3.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/p4est/2.8.6/lib/libp4est.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/p4est/2.8.6/lib/libsc.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/openblas/0.3.15/lib/libopenblas.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/scotch/7.0.4/lib/libptesmumps.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/scotch/7.0.4/lib/libptscotchparmetisv3.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/scotch/7.0.4/lib/libptscotch.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/scotch/7.0.4/lib/libptscotcherr.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/scotch/7.0.4/lib/libesmumps.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/metis/5.1.0/lib/libmetis.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/scotch/7.0.4/lib/libscotch.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/scotch/7.0.4/lib/libscotcherr.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/metis/5.1.0/lib/libparmetis.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/netcdf/4.9.2/lib/libnetcdf.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/hdf5/1.12.0/lib/libhdf5_hl.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/pkgs/hdf5/1.12.0/lib/libhdf5.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/base/lib/libz.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/base/lib/libbz2.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/base/lib/libmpi_usempif08.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/base/lib/libmpi_usempi_ignore_tkr.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/base/lib/libmpi_mpifh.so
PN: /opt/mox/mk/toolchains/gcc-glibc/11.2.0/base/lib/libmpi.so
PN: CMakeFiles/PN.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/mnt/d/IPROP_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn_parallel2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable PN"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/PN.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/PN.dir/build: PN
.PHONY : CMakeFiles/PN.dir/build

CMakeFiles/PN.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/PN.dir/cmake_clean.cmake
.PHONY : CMakeFiles/PN.dir/clean

CMakeFiles/PN.dir/depend:
	cd /mnt/d/IPROP_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn_parallel2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/d/IPROP_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn_parallel2 /mnt/d/IPROP_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn_parallel2 /mnt/d/IPROP_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn_parallel2/build /mnt/d/IPROP_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn_parallel2/build /mnt/d/IPROP_PROJECT/PACS_Andena_Bottacini/Codici/Validation_pn_parallel2/build/CMakeFiles/PN.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/PN.dir/depend

