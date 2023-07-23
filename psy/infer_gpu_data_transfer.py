from abc import ABC
from dataclasses import dataclass
from xdsl.utils.hints import isa
from xdsl.ir import Operation, SSAValue, Region, Block, MLContext
from xdsl.pattern_rewriter import (RewritePattern, PatternRewriter,
                                   op_type_rewrite_pattern,
                                   PatternRewriteWalker,
                                   GreedyRewritePatternApplier)
from xdsl.passes import ModulePass
from xdsl.dialects import builtin, func, llvm, arith, memref, gpu
from xdsl.dialects.experimental import fir
from util.visitor import Visitor

class GPU_Stencil_Invocation():
  def __init__(self, name, call_op, arg_names, arg_ssas):
    self.name=name
    self.call_op=call_op
    self.arg_names=arg_names
    self.arg_ssas=arg_ssas

class FindFirstStencilBridgedFunction(Visitor):
  def __init__(self):
    self.first_bridged_fn=None

  def traverse_call(self, call_op:fir.Call):
    fn_name=call_op.callee.root_reference.data
    if "InternalBridgeStencil" in fn_name and self.first_bridged_fn is None:
      self.first_bridged_fn=call_op

class GatherStencilBridgedFunctions(Visitor):
  def __init__(self):
    self.stencil_bridge_calls={}
    self.stencil_bridge_fn_issues=[]
    self.gpu_data_symbols=[]
    self.stencil_invokes=[]

  def get_nested_type(in_type, search_type):
    if isinstance(in_type, search_type): return in_type
    return GatherStencilBridgedFunctions.get_nested_type(in_type.type, search_type)

  def get_symbol(token):
    if isa(token, fir.Convert):
      return GatherStencilBridgedFunctions.get_symbol(token.value.owner)
    elif isa(token, fir.BoxAddr):
      return GatherStencilBridgedFunctions.get_symbol(token.val.owner)
    elif isa(token, fir.Load):
      return GatherStencilBridgedFunctions.get_symbol(token.memref.owner)
    elif isa(token, fir.AddressOf):
      array_type=GatherStencilBridgedFunctions.get_nested_type(token.results[0].typ, fir.SequenceType)
      return (token.symbol, array_type.type)
    else:
      assert False

  def traverse_call(self, call_op:fir.Call):
    fn_name=call_op.callee.root_reference.data
    if "InternalBridgeStencil" in fn_name:
      self.stencil_bridge_calls[fn_name]=call_op
      self.stencil_bridge_fn_issues.append(fn_name)
      arg_names=[]
      arg_ssas=[]
      for op in call_op.args:
        # This is a convert, now walk backwards to grab the symbol
        assert isa(op.owner, fir.Convert)
        data_symbol=GatherStencilBridgedFunctions.get_symbol(op.owner)
        self.gpu_data_symbols.append(data_symbol)
        arg_names.append(data_symbol[0].root_reference.data)
        arg_ssas.append(op)

      self.stencil_invokes.append(GPU_Stencil_Invocation(fn_name, call_op, arg_names, arg_ssas))

class GetScalarAssignedValue(Visitor):
  def __init__(self, scalar_name):
    self.scalar_name=scalar_name
    self.value=None

  def traverse_store(self, store_op:fir.Store):
    if isa(store_op.memref.owner, fir.Alloca) and store_op.memref.owner.uniq_name.data == self.scalar_name:
      assert isa(store_op.value.owner, arith.Constant)
      assert isa(store_op.value.owner.value, builtin.IntegerAttr)
      self.value=store_op.value.owner.value.value.data

class DetermineArraySizeAndDims(Visitor):
  def __init__(self, symbol_name, module):
    self.sizes=[]
    self.symbol_name=symbol_name.root_reference.data
    self.module=module

  def getScalarVariableName(token):
    if isa(token, fir.Convert):
      return DetermineArraySizeAndDims.getScalarVariableName(token.value.owner)
    elif isa(token, fir.Load):
      return DetermineArraySizeAndDims.getScalarVariableName(token.memref.owner)
    elif isa(token, fir.Alloca):
      return token.uniq_name.data
    else:
      return None

  def traverse_embox(self, embox_op:fir.Embox):
    for use in embox_op.results[0].uses:
      if isa(use.operation, fir.Store):
        assert isa(use.operation.memref.owner, fir.AddressOf)
        data_symbol_name=use.operation.memref.owner.symbol.root_reference.data
        if data_symbol_name == self.symbol_name:
          assert isa(embox_op.shape.owner, fir.Shape)
          for extent in embox_op.shape.owner.extents:
            name=DetermineArraySizeAndDims.getScalarVariableName(extent.owner)
            if name is None: return
            v=GetScalarAssignedValue(name)
            v.traverse(self.module)
            assert(v.value is not None)
            self.sizes.append(v.value)


class GenerateSymbolGPUAllocations():

  def generate_GPU_allocation(stencil_bridged_functions, driver_module, stencils_module):
    alloc_ops=[]
    return_ops=[]
    return_types=[]
    for data_symbol, data_type in stencil_bridged_functions.gpu_data_symbols:
      array_size_dims_visitor=DetermineArraySizeAndDims(data_symbol, driver_module)
      array_size_dims_visitor.traverse(driver_module)

      memref_type=memref.MemRefType.from_element_type_and_shape(data_type, array_size_dims_visitor.sizes)
      data_alloc_op=gpu.AllocOp(memref_type)
      extract_aligned_ptr_op=memref.ExtractAlignedPointerAsIndexOp.get(data_alloc_op)
      index_cast_op=arith.IndexCastOp.get(extract_aligned_ptr_op, builtin.i64)
      build_llvm_ptr_op=llvm.IntToPtrOp.get(index_cast_op, data_type)
      alloc_ops+=[data_alloc_op, extract_aligned_ptr_op, index_cast_op, build_llvm_ptr_op]
      return_ops.append(build_llvm_ptr_op.results[0])
      return_types.append(llvm.LLVMPointerType.typed(data_type))

    alloc_ops.append(func.Return(*return_ops))

    block = Block()
    block.add_ops(alloc_ops)
    body=Region()
    body.add_block(block)

    alloc_func=func.FuncOp.from_region("GPU_allocation", [], return_types, body)
    call_op=fir.Call.create(attributes={"callee": builtin.SymbolRefAttr("GPU_allocation")}, operands=[], result_types=return_types)
    external_alloc_func_def=func.FuncOp.external("GPU_allocation", [], return_types)

    return alloc_func, call_op, external_alloc_func_def

@dataclass
class InferGPUDataTransfer(ModulePass):
  """
  This is the entry point for the transformation pass which will then apply the rewriter
  """
  name = 'infer-gpu-data-transfer'

  def find_array_index(array_name, stencil_bridged_functions):
    for index, (data_symbol, data_type) in enumerate(stencil_bridged_functions.gpu_data_symbols):
      if data_symbol.root_reference.data == array_name: return index
    return None

  def erase_unused_data_ops(op):
    op.parent.erase_op(op)
    if isa(op, fir.Convert):
      InferGPUDataTransfer.erase_unused_data_ops(op.value.owner)
    elif isa(op, fir.BoxAddr):
      InferGPUDataTransfer.erase_unused_data_ops(op.val.owner)

  def apply(self, ctx: MLContext, module: builtin.ModuleOp):
    stencils_module=list(module.ops)[0]
    driver_module=list(module.ops)[1]
    stencil_bridged_functions=GatherStencilBridgedFunctions()
    stencil_bridged_functions.traverse(driver_module)
    alloc_func, call_op, external_alloc_func_def=GenerateSymbolGPUAllocations.generate_GPU_allocation(stencil_bridged_functions, driver_module, stencils_module)
    for stencil_invoke in stencil_bridged_functions.stencil_invokes:
      for idx, arg_name in enumerate(stencil_invoke.arg_names):
        ssa_index=InferGPUDataTransfer.find_array_index(arg_name, stencil_bridged_functions)
        stencil_invoke.arg_ssas[idx]=call_op.results[ssa_index]

      new_call_op=fir.Call.create(attributes={"callee": builtin.SymbolRefAttr(stencil_invoke.name)}, operands=stencil_invoke.arg_ssas, result_types=[])
      stencil_invoke.call_op.parent.insert_op_after(new_call_op, stencil_invoke.call_op)
      stencil_invoke.call_op.parent.erase_op(stencil_invoke.call_op)
      # Now erase all data loading for the origional call op
      for op in stencil_invoke.call_op.args:
        InferGPUDataTransfer.erase_unused_data_ops(op.owner)

    stencils_module.regions[0].block.add_op(alloc_func)
    driver_module.regions[0].block.add_op(external_alloc_func_def)

    v=FindFirstStencilBridgedFunction()
    v.traverse(driver_module)
    assert v.first_bridged_fn is not None
    v.first_bridged_fn.parent.insert_op_before(call_op, v.first_bridged_fn)

