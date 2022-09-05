# PSyclone xDSL transformations

Needs to have ftn xDSL dialect in _PYTHONPATH_

./psy-opt -p "ftn-ast-to-ftn-dag" output.xdsl

./psy-opt -p "ftn-ast-to-ftn-dag" output.xdsl -t fortran

./psy-opt -p "ftn-ast-to-ftn-dag,bind-procedures,apply-gpu-analysis" output.xdsl -t fortran-openacc 

./psy-opt -p "ftn-ast-to-ftn-dag,bind-procedures,apply-pgas-analysis" output.xdsl -t fortran
