all: l63_for l96_for

l63_for: 
	gfortran -O3 -c $@.f90
	f2py -c $@.f90 -m $@

l96_for:
	gfortran -O3 -c $@.f90
	f2py -c $@.f90 -m $@ only: tinteg1scl emtinteg1scl

clean:
	rm -rf __pycache__ *.pyc *.o *.mod *.so.* *.so build
