SOURCES = BlockSchurPreconditioner.cpp InsIMEX.cpp time_dependent_navier_stokes.cc
HEADERS = BlockSchurPreconditioner.hpp InsIMEX.hpp MyDataStruct.hpp Time.hpp BoundaryValiues.hpp
EXEC = time_dependent_navier_stokes

# Percorso delle librerie e dei file di intestazione di deal.II
DEAL_II_INCLUDE = /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.3.1/include
DEAL_II_LIB = /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.3.1/lib

ADOL_C_INCLUDE = /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/adol-c/2.7.2/include
ADOL_C_LIB = /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/adol-c/2.7.2/lib

.PHONY= clean

$(EXEC) : $(SOURCES) $(HEADERS)
	g++ -I$(DEAL_II_INCLUDE) -I$(ADOL_C_INCLUDE) $(SOURCES) -o $(EXEC) -L$(DEAL_II_LIB) -L$(ADOL_C_LIB) -ldeal_II
	./$(EXEC)

clean :
	$(RM) -f $(EXEC)
