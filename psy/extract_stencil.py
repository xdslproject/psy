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
from xdsl.passes import ModulePass
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
    block.add_op(func.Return.get())
    body=Region()
    body.add_block(block)

    return func.FuncOp.from_region(name, input_types_translated, [], body)

class ExtractStencilOps(_StencilExtractorRewriteBase):

    def has_nested_type(in_type, search_type):
      if isinstance(in_type, search_type): return True
      if getattr(in_type, "type", None) is None: return False
      return ExtractStencilOps.has_nested_type(in_type.type, search_type)

    """
    This will extract the stencil operations and replace them with a function call, converting the
    input array into fir.LLVMPointer type. Note currently this is very limited in terms of assuming
    one input only and a statically allocated array - it will get more complex as we add support
    for other things around this.

    Note that the logic here is fairly simple, we assume all the stencil operations are between the external
    load and the external store. Hence we grab all of those as a chunk and move them out
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, external_load_op: stencil.ExternalLoadOp, rewriter: PatternRewriter, /):
        parent_op=external_load_op.parent
        idx = None
        op_list=[op for op in parent_op.ops]
        for index, test_op in enumerate(parent_op.ops):
          if test_op == external_load_op:
            idx=index
            break
        assert idx is not None
        #parent_op.ops.index(op)
        if idx > 0 and isinstance(op_list[idx-1], builtin.UnrealizedConversionCastOp):
          # We do this to catch any unrealized conversion cast that preceeds the first external load
          # this will be detached later on, but needs to be in the extracted function to do so
          idx-=1
        stencil_ops=[]
        ops_to_load_in=[]
        for index, op in enumerate(parent_op.ops): #range(idx, len(op.parent.ops)):
          if index < idx: continue
          stencil_ops.append(op_list[index])
          if isinstance(op_list[index], stencil.ExternalLoadOp):
            ops_to_load_in.append(op_list[index])
          if isinstance(op_list[index], stencil.ExternalStoreOp):
            break

        arg_ops=[]
        op_types=[]
        ops_to_add=[]
        for sop in ops_to_load_in:
          nt=self.get_nested_type(sop.field.typ, fir.ArrayType)
          ptr_type=fir.LLVMPointerType([nt.type])
          op_types.append(ptr_type)
          if isinstance(sop.field.owner, builtin.UnrealizedConversionCastOp):
            # This is deferred type, but just check the type is what we expect
            # Should be !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x?x!f64>>>>
            # With any number of array dimensions
            # In this case need to load and debox it, that gives us a heap
            # we can then convert to an llvm pointer
            data_type=sop.field.owner.inputs[0].typ
            assert isinstance(data_type, fir.ReferenceType)
            assert ExtractStencilOps.has_nested_type(data_type, fir.BoxType)
            assert ExtractStencilOps.has_nested_type(data_type, fir.HeapType)

            box_type=self.get_nested_type(data_type, fir.BoxType)
            heap_type=self.get_nested_type(data_type, fir.HeapType)

            load_op=fir.Load.create(operands=[sop.field.owner.inputs[0]], result_types=[box_type])
            box_addr_op=fir.BoxAddr.create(operands=[load_op.results[0]], result_types=[heap_type])
            convert_op=fir.Convert.create(operands=[box_addr_op.results[0]], result_types=[ptr_type])
            arg_ops.append(convert_op)
            ops_to_add+=[load_op, box_addr_op, convert_op]
          else:
            convert_op=fir.Convert.create(operands=[sop.field], result_types=[ptr_type])
            arg_ops.append(convert_op)
            ops_to_add+=[convert_op]

        function_name="_InternalBridgeStencil_"+str(self.bridge_id)
        self.bridge_id+=1

        call_stencil=fir.Call.create(attributes={"callee": builtin.SymbolRefAttr(function_name)}, operands=[el.results[0] for el in arg_ops], result_types=[])
        #parent_op.insert_op(ops_to_add+[call_stencil], idx+1)
        #rewriter.insert_op_before(ops_to_add+[call_stencil], external_load_op)
        parent_op.insert_ops_before(ops_to_add+[call_stencil], external_load_op)
        for index, sop in enumerate(stencil_ops):
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
            rewriter.insert_op_at_end(func.FuncOp.external(name, arg, res),
                                      module.body.blocks[0])

class ConnectExternalLoadToFunctionInput(RewritePattern):

  def count_instances_preceeding(ops, to_index, typ):
    instances=0
    for idx, op in enumerate(ops):
      if idx >= to_index: break
      if isinstance(op, typ): instances+=1
    return instances

  def get_nested_type(in_type, search_type):
    if isinstance(in_type, search_type): return in_type
    return ConnectExternalLoadToFunctionInput.get_nested_type(in_type.type, search_type)

  def get_array_dimension_size(array_type, dim):
    # We need to access shape of array type in reverse order, as indexing
    # is different between C and Fortran
    shape=array_type.shape.data
    access_dim=len(shape) - dim - 1
    return shape[access_dim].value.data

  def get_c_style_array_shape(array_type):
    shape=array_type.shape.data
    c_style=[]
    for i in range(len(shape)):
      c_style.insert(0, builtin.IntegerAttr.from_index_int_value(shape[i].value.data))
    return c_style

  """
  Connects external load at the start of the bridged function with the
  input argument (note we will need to load this into a memref here)
  """
  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: stencil.ExternalLoadOp, rewriter: PatternRewriter, /):
    # If this already accepts a memref then don't need to wrap pointer
    if isinstance(op.field.typ, MemRefType): return

    array_type=ConnectExternalLoadToFunctionInput.get_nested_type(op.field.typ, fir.ArrayType)
    number_dims=len(array_type.shape.data)

    op_list=[sop for sop in op.parent.ops]
    idx = None
    for index, sop in enumerate(op_list):
      if sop == op:
        idx=index
        break

    assert idx is not None

    num_prev_external_loads=ConnectExternalLoadToFunctionInput.count_instances_preceeding(op.parent.ops, idx-1, stencil.ExternalLoadOp)

    ptr_type=op.parent.args[num_prev_external_loads].typ
    array_typ=llvm.LLVMArrayType.from_size_and_type(builtin.IntAttr(number_dims), builtin.i64)
    struct_type=llvm.LLVMStructType.from_type_list([ptr_type, ptr_type, builtin.i64, array_typ, array_typ])

    undef_memref_struct=llvm.LLVMMLIRUndef.create(result_types=[struct_type])
    insert_alloc_ptr_op=llvm.LLVMInsertValue.create(attributes={"position":  builtin.DenseArrayBase.from_list(builtin.i64, [0])},
      operands=[undef_memref_struct.results[0], op.parent.args[num_prev_external_loads]], result_types=[struct_type])
    insert_aligned_ptr_op=llvm.LLVMInsertValue.create(attributes={"position":  builtin.DenseArrayBase.from_list(builtin.i64, [1])},
      operands=[insert_alloc_ptr_op.results[0], op.parent.args[num_prev_external_loads]], result_types=[struct_type])

    offset_op=arith.Constant.from_int_and_width(0, 64)
    insert_offset_op=llvm.LLVMInsertValue.create(attributes={"position":  builtin.DenseArrayBase.from_list(builtin.i64, [2])},
      operands=[insert_aligned_ptr_op.results[0], offset_op.results[0]], result_types=[struct_type])

    ops_to_add=[undef_memref_struct, insert_alloc_ptr_op, insert_aligned_ptr_op, offset_op, insert_offset_op]

    for dim in range(number_dims):
      dim_size=ConnectExternalLoadToFunctionInput.get_array_dimension_size(array_type, dim)
      size_op=arith.Constant.from_int_and_width(dim_size, 64)
      insert_size_op=llvm.LLVMInsertValue.create(attributes={"position":  builtin.DenseArrayBase.from_list(builtin.i64, [3, dim])},
        operands=[ops_to_add[-1].results[0], size_op.results[0]], result_types=[struct_type])

      # One for dimension stride
      stride_op=arith.Constant.from_int_and_width(1, 64)
      insert_stride_op=llvm.LLVMInsertValue.create(attributes={"position":  builtin.DenseArrayBase.from_list(builtin.i64, [4, dim])},
        operands=[insert_size_op.results[0], stride_op.results[0]], result_types=[struct_type])

      ops_to_add+=[size_op, insert_size_op, stride_op, insert_stride_op]


    target_memref_type=MemRefType.from_element_type_and_shape(ptr_type.type, ConnectExternalLoadToFunctionInput.get_c_style_array_shape(array_type))

    unrealised_conv_cast_op=builtin.UnrealizedConversionCastOp.create(operands=[insert_stride_op.results[0]], result_types=[target_memref_type])
    ops_to_add.append(unrealised_conv_cast_op)

    if idx > 0 and isinstance(op_list[idx-1], builtin.UnrealizedConversionCastOp):
      # Remove the unrealized conversion cast as it is no longer needed
      op_list[idx-1].detach()
      idx-=1

    block=op.parent
    block.insert_ops_before(ops_to_add, op_list[idx])

    op.operands=[unrealised_conv_cast_op.results[0]]

class ConnectExternalStoreToFunctionInput(RewritePattern):
  """
  Connects the external store at the end of the bridged function with the
  input argument (note we will need to load this into a memref and then an llvm pointer)
  """
  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: stencil.ExternalStoreOp, rewriter: PatternRewriter, /):
    # Look up the external load and then external store to that operand
    op.operands=[op.temp, op.temp.op.field]

@dataclass
class ExtractStencil(ModulePass):
  """
  This is the entry point for the transformation pass which will then apply the rewriter
  """
  name = 'extract-stencil'

  def apply(self, ctx: MLContext, module: builtin.ModuleOp):
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
    new_module=builtin.ModuleOp(bridged_functions)

    # Rewrite the stencil functions to hook up the function/block arguments
    walker3 = PatternRewriteWalker(GreedyRewritePatternApplier([
              ConnectExternalLoadToFunctionInput(),
              ConnectExternalStoreToFunctionInput()
    ]), apply_recursively=False)
    walker3.rewrite_module(new_module)

    # Now we want to have two modules packaged together
    containing_mod=builtin.ModuleOp([])
    module.regions[0].move_blocks(containing_mod.regions[0])

    block = Block()
    block.add_ops([new_module, containing_mod])
    module.regions[0].add_block(block)
