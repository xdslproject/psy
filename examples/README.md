# Readme for PSy examples

## General PSyclone build

python3.10 ~/projects/xdsl/PSyclone/bin/psyclone -api nemo -s ./xdsl_backends_transform.py code_sample/gauss_seidel.F90

This outputs the IR, and Fortran code, but also saves the IR into a file `psy_output.mlir`

## Non-MPI compilation flow

To compile, for non-MPI:

./psy-opt -p apply-stencil-analysis,lower-psy-ir,extract-stencil,rewrite-fir-to-standard psy_output.mlir

Then copy the module 0 file out to the xdsl tools directory, and run

./xdsl-opt -p stencil-shape-inference,convert-stencil-to-ll-mlir module_0.mlir -o stencil.mlir

Then copy stencil.mlir and module_1.mlir (in generated directory) to your compile directory.

Issue:

mlir-opt --pass-pipeline="builtin.module(canonicalize, cse, loop-invariant-code-motion, canonicalize, cse, loop-invariant-code-motion,cse,canonicalize,fold-memref-alias-ops,lower-affine,finalize-memref-to-llvm{use-opaque-pointers=false},loop-invariant-code-motion,canonicalize,cse,convert-scf-to-openmp,finalize-memref-to-llvm{use-opaque-pointers=false},convert-scf-to-cf,convert-openmp-to-llvm,convert-math-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts,canonicalize,cse)" stencil.mlir | mlir-translate --mlir-to-llvmir -o stencil.bc

clang -g -c stencil.bc flang-new -fc1 -emit-obj module_1.mlir flang-new -fopenmp -o stencil stencil.o module_1.o

And run the executable

## MPI compilation flow

To compile, for MPI:

./psy-opt -p apply-stencil-analysis,lower-psy-ir,lower-mpi,extract-stencil,rewrite-fir-to-standard psy_output.mlir

Then copy the module 0 file out to the xdsl tools directory, and run

./xdsl-opt -p apply-mpi,stencil-shape-inference,convert-stencil-to-ll-mlir,lower-mpi module_0.mlir -o stencil.mlir

Then copy stencil.mlir and module_1.mlir (in generated directory) to your compile directory.

Issue:

mlir-opt --pass-pipeline="builtin.module(canonicalize, cse, loop-invariant-code-motion, canonicalize, cse, loop-invariant-code-motion,cse,canonicalize,fold-memref-alias-ops,lower-affine,finalize-memref-to-llvm{use-opaque-pointers=false},loop-invariant-code-motion,canonicalize,cse,convert-scf-to-openmp,finalize-memref-to-llvm{use-opaque-pointers=false},convert-scf-to-cf,convert-openmp-to-llvm,convert-math-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts,canonicalize,cse)" stencil.mlir | mlir-translate --mlir-to-llvmir -o stencil.bc

clang -g -c stencil.bc flang-new -fc1 -emit-obj module_1.mlir flang-new -o stencil stencil.o module_1.o -lmpi

And run the executable using mpiexec

## GPU compilation flow

To compile, for GPU:

./psy-opt -p apply-stencil-analysis,lower-mpi,extract-stencil,rewrite-fir-to-standard psy_output.mlir

Then copy the module 0 file out to the xdsl tools directory, and run

./xdsl-opt -p stencil-shape-inference,convert-stencil-to-gpu module_0.mlir -o stencil-gpu.mlir

Then copy stencil.mlir and module_1.mlir (in generated directory) to your compile directory.

Issue (the mlir-opt needs to be on GPU node of Cirrus):

(this pass pipeline is now incorrect)

bin/mlir-opt --pass-pipeline="builtin.module(test-math-algebraic-simplification,scf-parallel-loop-tiling{parallel-loop-tile-sizes=1024,1,1}, canonicalize, func.func(gpu-map-parallel-loops), convert-parallel-loops-to-gpu, lower-affine, gpu-kernel-outlining,func.func(gpu-async-region),canonicalize,convert-arith-to-llvm{index-bitwidth=64},convert-memref-to-llvm{index-bitwidth=64},convert-scf-to-cf,convert-cf-to-llvm{index-bitwidth=64},gpu.module(convert-gpu-to-nvvm,reconcile-unrealized-casts,canonicalize,gpu-to-cubin),gpu-to-llvm,canonicalize)" stencil-gpu.mlir | bin/mlir-translate --mlir-to-llvmir -o stencil.bc

clang -g -c stencil.bc flang-new -fc1 -emit-obj module_1.mlir flang-new -o stencil stencil.o module_1.o -lmpi

And run the executable
