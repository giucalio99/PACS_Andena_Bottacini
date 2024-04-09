Original code written by Tommaso Andena and Giacomo Bottacini. Politecnico di Milano a.a 2023/2024.

vedere il file:"MyDataStruct.hpp" per il significato dei parameteri specificati nel file .json

comandi del make file: 1) make -f makefile.mak main         per lanciare il codice 
                       2) make -f makefile.mak clean        per cancellare tutti gli output .o , executible and .geo file

TODO: - dati coerenti con la corda o funzione con corda diversa? (!!) funzione per corda diversa !!
      - per adesso gli input della parabola non sono parametrizzati (quelli che calcolano i pesi)
      - algoritmo dei quads a volte fallisce se non piacciano i parametri in input
      - ricordarsi modificare geometri raggio ala per simu
      - BL fix punto di attacco refinement e Tedge taglia (prova) NB: bisognerebbe cambaire quasi tutto il codice per impelementarlo!
        bisognerebbe tipo mettere una if condition in ogni funzione per tenere conto se si vuole tagliare la coda o no