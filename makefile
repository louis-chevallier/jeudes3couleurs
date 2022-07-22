

BASE=$(shell which python)
$(warning ${BASE})

removetrailingslash = ${1:/=}
u=$(dir $(call removetrailingslash,${1}))
BB=$(call u,$(call u,$(BASE)))
$(warning ${BB})
start :
	time python jeudes3couleurs.py


xxx:
	cython --embed jeudes3couleurs.py 
	g++  jeudes3couleurs.c $(shell python3-config --cflags) $(shell python3-config --ldflags) -lpython3.10 -o jeudes3couleurs
#-I${BB}/include/python3.10 -L${BB}/lib -lpython3
	time ./jeudes3couleurs



