from abc import ABC
from typing import TypeVar, cast
from dataclasses import dataclass
from ftn.dialects import fir
from xdsl.utils.hints import isa
from xdsl.dialects.memref import MemRefType
from xdsl.ir import Operation, SSAValue, OpResult, Attribute, MLContext, Block, Region

from xdsl.pattern_rewriter import (RewritePattern, PatternRewriter,
                                   op_type_rewrite_pattern,
                                   PatternRewriteWalker,
                                   GreedyRewritePatternApplier)
from xdsl.dialects import builtin, func, llvm, arith
from xdsl.dialects.experimental import stencil


@dataclass
class _StencilExtractorRewriteBase(RewritePattern, ABC):
  bridgedFunctions: dict[str, Operation]
  bridge_id: int

  def __init__(self):
    self.bridgedFunctions={}
    self.bridge_id=0

  def get_nested_type(self, in_type, search_type):
    if isinstance(in_type, search_type): return in_type
    return self.get_nested_type(in_type.type, search_type)

  @staticmethod
  def wrap_stencil_in_func(name, input_types, operations):
    """
    Wraps the extracted stencil operations for
    """

    input_types_translated=[]
    for typ in input_types:
      if isinstance(typ, fir.LLVMPointerType):
        input_types_translated.append(llvm.LLVMPointerType.typed(typ.type))
      else:
        input_types_translated.append(typ)

    block = Block.from_arg_types(input_types_translated)
    block.add_ops(operations)
    body=Region()
    body.add_block(block)

    return func.FuncOp.from_region(name, input_types_translated, [], body)

class ExtractStencilOps(_StencilExtractorRewriteBase):
    """
    This will extract the stencil operations and replace them with a function call, converting the
    input array into fir.LLVMPointer type. Note currently this is very limited in terms of assuming
    one input only and a statically allocated array - it will get more complex as we add support
    for other things around this.

    Note that the logic here is fairly simple, we assume all the stencil operations are between the external
    load and the external store. Hence we grab all of those as a chunk and move them out
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.ExternalLoadOp, rewriter: PatternRewriter, /):
        parent_op=op.parent
        idx = parent_op.ops.index(op)
        stencil_ops=[]
        ops_to_load_in=[]
        for i in range(idx, len(op.parent.ops)):
          stencil_ops.append(op.parent.ops[i])
          if isinstance(op.parent.ops[i], stencil.ExternalLoadOp):
            ops_to_load_in.append(op)
          if isinstance(op.parent.ops[i], stencil.ExternalStoreOp):
            break

        pass_ops=[]
        op_types=[]
        for sop in ops_to_load_in:
          nt=self.get_nested_type(sop.field.typ, fir.ArrayType)
          ptr_type=fir.LLVMPointerType([nt.type])
          op_types.append(ptr_type)
          pass_ops.append(fir.Convert.create(operands=[sop.field],
                      result_types=[ptr_type]))

        function_name="_InternalBridgeStencil_"+str(self.bridge_id)
        self.bridge_id+=1

        call_stencil=fir.Call.create(attributes={"callee": builtin.SymbolRefAttr(function_name)}, operands=[el.results[0] for el in pass_ops], result_types=[])
        parent_op.insert_op(pass_ops+[call_stencil], idx+1)
        for sop in stencil_ops:
          parent_op.detach_op(sop)
        stencil_bridge_fn=_StencilExtractorRewriteBase.wrap_stencil_in_func(function_name, op_types, stencil_ops)
        self.bridgedFunctions[function_name]=stencil_bridge_fn

class AddExternalFuncDefs(RewritePattern):
    """
    Will add in function signatures of externally bridged functions so that the Fortran code
    can call this
    """
    bridge_functions = list[str]

    def __init__(self, bridge_functions):
      self.bridge_functions=bridge_functions

    @op_type_rewrite_pattern
    def match_and_rewrite(self, module: builtin.ModuleOp,
                          rewriter: PatternRewriter, /):
        # collect all func calls to MPI functions
        funcs_to_emit: dict[str, tuple[list[Attribute],
                                       list[Attribute]]] = dict()

        def walker(op: Operation):
            if not isinstance(op, fir.Call):
                return
            if op.callee.string_value() not in self.bridge_functions:
                return
            funcs_to_emit[op.callee.string_value()] = (
                [arg.typ for arg in op.args],
                [res.typ for res in op.results],
            )

        module.walk(walker)

        # for each func found, add a FuncOp to the top of the module.
        for name, types in funcs_to_emit.items():
            arg, res = types
            rewriter.insert_op_at_pos(func.FuncOp.external(name, arg, res),
                                      module.body.blocks[0],
                                      len(module.body.blocks[0].ops))

class ConnectExternalLoadToFunctionInput(RewritePattern):

  def count_instances_preceeding(ops, to_index, typ):
    instances=0
    for idx, op in enumerate(ops):
      if idx >= to_index: break
      if isinstance(op, typ): instances+=1
    return instances

  """
  Connects external load at the start of the bridged function with the
  input argument (note we will need to load this into a memref here)
  """
  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: stencil.ExternalLoadOp, rewriter: PatternRewriter, /):
    """
    idx = op.parent.ops.index(op)
    if idx != 0: return
    ptr_type=op.parent.args[0].typ
    array_typ=llvm.LLVMArrayType.from_type_and_size(builtin.i64, builtin.IntAttr(1))
    struct_type=llvm.LLVMStructType.from_type_list([ptr_type, ptr_type, builtin.i64, array_typ, array_typ])

    undef_memref_struct=llvm.LLVMMLIRUndef.create(result_types=[struct_type])
    insert_alloc_ptr_op=llvm.LLVMInsertValue.create(attributes={"position":  builtin.DenseArrayBase.from_list(builtin.i32, [0])},
      operands=[undef_memref_struct.results[0], op.parent.args[0]], result_types=[struct_type])
    insert_aligned_ptr_op=llvm.LLVMInsertValue.create(attributes={"position":  builtin.DenseArrayBase.from_list(builtin.i32, [1])},
      operands=[insert_alloc_ptr_op.results[0], op.parent.args[0]], result_types=[struct_type])

    offset_op=arith.Constant.from_int_and_width(0, 32)
    insert_offset_op=llvm.LLVMInsertValue.create(attributes={"position":  builtin.DenseArrayBase.from_list(builtin.i32, [2])},
      operands=[insert_aligned_ptr_op.results[0], offset_op.results[0]], result_types=[struct_type])

    size_op=arith.Constant.from_int_and_width(10, 32)
    insert_size_op=llvm.LLVMInsertValue.create(attributes={"position":  builtin.DenseArrayBase.from_list(builtin.i32, [3])},
      operands=[insert_offset_op.results[0], size_op.results[0]], result_types=[struct_type])

    stride_op=arith.Constant.from_int_and_width(1, 32)
    insert_stride_op=llvm.LLVMInsertValue.create(attributes={"position":  builtin.DenseArrayBase.from_list(builtin.i32, [4])},
      operands=[insert_size_op.results[0], stride_op.results[0]], result_types=[struct_type])

    ops_to_add=[undef_memref_struct, insert_alloc_ptr_op, insert_aligned_ptr_op, offset_op, insert_offset_op, size_op, insert_size_op, stride_op, insert_stride_op]

    block=op.parent
    block.insert_op(ops_to_add, 0)
"""
    idx = op.parent.ops.index(op)
    num_prev_external_loads=(ConnectExternalLoadToFunctionInput.count_instances_preceeding(op.parent.ops, idx-1, stencil.ExternalLoadOp))
    op.operands=[op.parent.args[num_prev_external_loads]]

class ConnectExternalStoreToFunctionInput(RewritePattern):
  """
  Connects the external store at the end of the bridged function with the
  input argument (note we will need to load this into a memref and then an llvm pointer)
  """
  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: stencil.ExternalStoreOp, rewriter: PatternRewriter, /):
    # Look up the external load and then external store to that operand
    op.operands=[op.temp, op.temp.op.field]

def extract_stencil(ctx: MLContext, module: builtin.ModuleOp):
    # First extract the stencil aspects and replace with a function call
    extractStencil=ExtractStencilOps()
    walker1 = PatternRewriteWalker(GreedyRewritePatternApplier([
              extractStencil,
    ]),
                                   apply_recursively=True)
    walker1.rewrite_module(module)

    # Now add in external function signature for new bridged functions
    bridged_fn_names=[v for v in extractStencil.bridgedFunctions.keys()]
    walker2 = PatternRewriteWalker(AddExternalFuncDefs(bridged_fn_names))
    walker2.rewrite_module(module)

    # Create a new module with all the stencil functions as part of it
    bridged_functions=[v for v in extractStencil.bridgedFunctions.values()]
    new_module=builtin.ModuleOp.from_region_or_ops(bridged_functions)

    # Rewrite the stencil functions to hook up the function/block arguments
    walker3 = PatternRewriteWalker(GreedyRewritePatternApplier([
              ConnectExternalLoadToFunctionInput(),
              ConnectExternalStoreToFunctionInput()
    ]), apply_recursively=False)
    walker3.rewrite_module(new_module)

    # Now we want to have two modules packaged together
    containing_mod=builtin.ModuleOp.from_region_or_ops([])
    module.regions[0].move_blocks(containing_mod.regions[0])

    block = Block()
    block.add_ops([new_module, containing_mod])
    module.regions[0].add_block(block)

