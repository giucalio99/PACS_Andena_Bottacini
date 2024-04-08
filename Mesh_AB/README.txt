Original code written by Tommaso Andena and Giacomo Bottacini. Politecnico di Milano a.a 2023/2024.

vedere il file:"MyDataStruct.hpp" per il significato dei parameteri specificati nel file .json

comandi del make file: 1) make -f makefile.mak main         per lanciare il codice 
                       2) make -f makefile.mak clean        per cancellare tutti gli output .o , executible and .geo file

TODO: - dati coerenti con la corda o funzione con corda diversa? (!!) funzione per corda diversa !!
      - per adesso gli input della parabola non sono parametrizzati
      - algoritmo dei quads va bene (?). algo cambiato, con questi setting va bene ma se cambiano alcuni parametri fallisce
      - ricordarsi modificare geometri raggio ala per simu
      - i parametri vengono stampati, ora bisogna modificare le function in modo tale che scrivano in funzione del parametro che si vede nel .geo
      - BL anche per emitter
      - BL fix punto di attacco refinement e Tedge taglia (prova)