#this makefile has two command: main that create the output file
#and clean that remove all the resulting files
SOURCES = main.cpp Build_Geometry.cpp Point.cpp
OBJECTS = main.o Build_Geometry.o Point.o
HEADERS = json.hpp Build_Geometry.hpp Point.hpp MyDataStruct.hpp
EXEC = main

.PHONY= clean

$(EXEC) : $(OBJECTS)
	g++ $(OBJECTS) -o $(EXEC)
	./main

clean :
	$(RM) -f $(EXEC) $(OBJECTS) *.geo