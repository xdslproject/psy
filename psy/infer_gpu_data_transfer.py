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

class FindAddressOfForArray(Visitor):
  def __init__(self, array_name):
    self.array_name=array_name
    self.addressof_symbol=None

  def traverse_address_of(self, addressof_op:fir.AddressOf):
    if addressof_op.symbol.root_reference.data == self.array_name:
      self.addressof_symbol=addressof_op

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

class FindLastTimerStopFunction(Visitor):
  def __init__(self):
    self.last_timer_stop_fn=None

  def traverse_call(self, call_op:fir.Call):
    fn_name=call_op.callee.root_reference.data
    if "timerPtimer_stop" in fn_name:
      self.last_timer_stop_fn=call_op

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
        if isa(op.owner, fir.Convert):
          # This is a convert, now walk backwards to grab the symbol
          # An array, we care about this!
          data_symbol=GatherStencilBridgedFunctions.get_symbol(op.owner)
          self.gpu_data_symbols.append(data_symbol)
          arg_names.append(data_symbol[0].root_reference.data)
        elif isa(op.owner, fir.Load):
          # A scalar, we do not care, add a placeholder to ignore
          arg_names.append(None)
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

  class GPUAllocDescriptor:
    def __init__(self,sizes,basetype):
        self.sizes = sizes
        self.basetype = basetype

    def __hash__(self):
        return hash(str(len(self.sizes))+"-"+' '.join(str(i) for i in self.sizes)+self.basetype.name)

    def __eq__(self, other):
        if len(self.sizes) != len(other.sizes): return False
        for s, t in zip(self.sizes, other.sizes):
          if s != t: return False
        return self.basetype == other.basetype

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)

  def generate_GPU_allocation(stencil_bridged_functions, driver_module, stencils_module):

    function_defs={}
    call_ops=[]
    external_call_defs={}

    handled_symbol_types={}

    for data_symbol, data_type in stencil_bridged_functions.gpu_data_symbols:
      if data_symbol.root_reference.data in handled_symbol_types.keys(): continue

      array_size_dims_visitor=DetermineArraySizeAndDims(data_symbol, driver_module)
      array_size_dims_visitor.traverse(driver_module)

      handled_symbol_types[data_symbol.root_reference.data]=(data_type, array_size_dims_visitor.sizes)
      memref_type=memref.MemRefType.from_element_type_and_shape(data_type, array_size_dims_visitor.sizes)
      allocDescriptor=GenerateSymbolGPUAllocations.GPUAllocDescriptor(array_size_dims_visitor.sizes, data_type)
      if allocDescriptor not in function_defs:
        assert allocDescriptor not in external_call_defs
        data_alloc_op=gpu.AllocOp(memref_type)
        extract_aligned_ptr_op=memref.ExtractAlignedPointerAsIndexOp.get(data_alloc_op)
        index_cast_op=arith.IndexCastOp.get(extract_aligned_ptr_op, builtin.i64)
        build_llvm_ptr_op=llvm.IntToPtrOp.get(index_cast_op, data_type)

        block = Block()
        block.add_ops([data_alloc_op, extract_aligned_ptr_op, index_cast_op, build_llvm_ptr_op, func.Return(build_llvm_ptr_op)])
        body=Region()
        body.add_block(block)

        fn_name="GPU_allocation_"+str(len(function_defs))

        return_types=[llvm.LLVMPointerType.typed(data_type)]

        new_func=func.FuncOp.from_region(fn_name, [], return_types, body)
        function_defs[allocDescriptor]=new_func
        external_fn_def=func.FuncOp.external(fn_name, [], return_types)
        external_call_defs[allocDescriptor]=external_fn_def


      target_fn_name=function_defs[allocDescriptor].sym_name.data

      call_op=fir.Call.create(attributes={"callee": builtin.SymbolRefAttr(target_fn_name)}, operands=[], result_types=return_types)
      call_ops.append(call_op)

    return function_defs.values(), call_ops, external_call_defs.values(), handled_symbol_types

class GenerateDataCopyBack():
  def construct_memref_creation(dim_sizes, base_type, llvm_pointer):
    construction_ops=[]

    number_dims=len(dim_sizes)
    ptr_type=llvm.LLVMPointerType.typed(base_type)

    array_typ=llvm.LLVMArrayType.from_size_and_type(builtin.IntAttr(number_dims), builtin.i64)
    struct_type=llvm.LLVMStructType.from_type_list([ptr_type, ptr_type, builtin.i64, array_typ, array_typ])

    undef_memref_struct=llvm.LLVMMLIRUndef.create(result_types=[struct_type])
    insert_alloc_ptr_op=llvm.LLVMInsertValue.create(attributes={"position":  builtin.DenseArrayBase.from_list(builtin.i64, [0])},
      operands=[undef_memref_struct.results[0], llvm_pointer], result_types=[struct_type])
    insert_aligned_ptr_op=llvm.LLVMInsertValue.create(attributes={"position":  builtin.DenseArrayBase.from_list(builtin.i64, [1])},
      operands=[insert_alloc_ptr_op.results[0], llvm_pointer], result_types=[struct_type])

    offset_op=arith.Constant.from_int_and_width(0, 64)
    insert_offset_op=llvm.LLVMInsertValue.create(attributes={"position":  builtin.DenseArrayBase.from_list(builtin.i64, [2])},
      operands=[insert_aligned_ptr_op.results[0], offset_op.results[0]], result_types=[struct_type])

    construction_ops=[undef_memref_struct, insert_alloc_ptr_op, insert_aligned_ptr_op, offset_op, insert_offset_op]

    for dim, dim_size in enumerate(dim_sizes):
      size_op=arith.Constant.from_int_and_width(dim_size, 64)
      insert_size_op=llvm.LLVMInsertValue.create(attributes={"position":  builtin.DenseArrayBase.from_list(builtin.i64, [3, dim])},
        operands=[construction_ops[-1].results[0], size_op.results[0]], result_types=[struct_type])

      # One for dimension stride
      stride_op=arith.Constant.from_int_and_width(1, 64)
      insert_stride_op=llvm.LLVMInsertValue.create(attributes={"position":  builtin.DenseArrayBase.from_list(builtin.i64, [4, dim])},
        operands=[insert_size_op.results[0], stride_op.results[0]], result_types=[struct_type])

      construction_ops+=[size_op, insert_size_op, stride_op, insert_stride_op]

    target_memref_type=memref.MemRefType.from_element_type_and_shape(ptr_type.type, dim_sizes)

    unrealised_conv_cast_op=builtin.UnrealizedConversionCastOp.create(operands=[insert_stride_op.results[0]], result_types=[target_memref_type])
    construction_ops.append(unrealised_conv_cast_op)

    return construction_ops, unrealised_conv_cast_op.results[0]

  def construct_FIR_array_ptr_extract(array_full_data_type, address_of_ssa, base_type):
    ptr_type=fir.LLVMPointerType([base_type])
    result_ptr_type=llvm.LLVMPointerType.typed(base_type)
    box_type=GatherStencilBridgedFunctions.get_nested_type(array_full_data_type, fir.BoxType)
    heap_type=GatherStencilBridgedFunctions.get_nested_type(array_full_data_type, fir.HeapType)

    load_op=fir.Load.create(operands=[address_of_ssa], result_types=[box_type])
    box_addr_op=fir.BoxAddr.create(operands=[load_op.results[0]], result_types=[heap_type])
    convert_op=fir.Convert.create(operands=[box_addr_op.results[0]], result_types=[ptr_type])

    return [load_op, box_addr_op, convert_op], convert_op.results[0]

  def generate_copy_back(named_gpu_pointers, handled_symbol_types, driver_module, field_wildcards=None):
    argument_types=[]
    external_argument_types=[]

    data_ssas=[]
    call_data_convert_ops=[]

    if field_wildcards is None:
      gpu_pointer_names=named_gpu_pointers.keys()
    else:
      gpu_pointer_names=[]
      for e in named_gpu_pointers.keys():
        for w in field_wildcards:
          if w in e:
            gpu_pointer_names.append(e)
            break

    for symbol_name in gpu_pointer_names:
      # Two arg types per key, both llvm pointer
      arg_type=llvm.LLVMPointerType.typed(handled_symbol_types[symbol_name][0])
      fir_arg_type=fir.LLVMPointerType([handled_symbol_types[symbol_name][0]])
      argument_types+=[arg_type, arg_type]
      external_argument_types+=[fir_arg_type, arg_type]
      v=FindAddressOfForArray(symbol_name)
      v.traverse(driver_module)
      assert v.addressof_symbol is not None
      ops, ssa=GenerateDataCopyBack.construct_FIR_array_ptr_extract(v.addressof_symbol.results[0].typ, v.addressof_symbol.results[0], handled_symbol_types[symbol_name][0])
      data_ssas.append(ssa)
      data_ssas.append(named_gpu_pointers[symbol_name])
      call_data_convert_ops+=ops

    call_op=fir.Call.create(attributes={"callee": builtin.SymbolRefAttr("GPU_copyback")}, operands=data_ssas, result_types=[])
    call_data_convert_ops.append(call_op)

    block = Block(arg_types=argument_types)

    fn_ops=[]

    for idx, symbol_name in enumerate(gpu_pointer_names):
      base_arg=idx*2
      memref_host_data_ops, memref_host_data_ssa=GenerateDataCopyBack.construct_memref_creation(handled_symbol_types[symbol_name][1], handled_symbol_types[symbol_name][0], block.args[base_arg])
      memref_gpu_data_ops, memref_gpu_data_ssa=GenerateDataCopyBack.construct_memref_creation(handled_symbol_types[symbol_name][1], handled_symbol_types[symbol_name][0], block.args[base_arg+1])
      gpu_copy_back=gpu.MemcpyOp(memref_gpu_data_ssa, memref_host_data_ssa)
      fn_ops.extend(memref_host_data_ops)
      fn_ops.extend(memref_gpu_data_ops)
      fn_ops.append(gpu_copy_back)

    block.add_ops(fn_ops)
    block.add_op(func.Return())
    body=Region()
    body.add_block(block)

    copy_back_fn=func.FuncOp.from_region("GPU_copyback", argument_types, [], body)
    external_fn_def=func.FuncOp.external("GPU_copyback", external_argument_types, [])

    return copy_back_fn, call_data_convert_ops, external_fn_def


@dataclass(frozen=True)
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
    alloc_funcs, call_ops, external_alloc_func_defs, handled_symbol_types=GenerateSymbolGPUAllocations.generate_GPU_allocation(stencil_bridged_functions, driver_module, stencils_module)

    named_gpu_llvm_ptr={}
    for index, (data_symbol, data_type) in enumerate(stencil_bridged_functions.gpu_data_symbols):
      if data_symbol.root_reference.data not in named_gpu_llvm_ptr.keys():
        named_gpu_llvm_ptr[data_symbol.root_reference.data]=call_ops[index].results[0]

    for stencil_invoke in stencil_bridged_functions.stencil_invokes:
      for idx, arg_name in enumerate(stencil_invoke.arg_names):
        if arg_name is not None:
          ssa_index=InferGPUDataTransfer.find_array_index(arg_name, stencil_bridged_functions)
          stencil_invoke.arg_ssas[idx]=call_ops[ssa_index].results[0]

      new_call_op=fir.Call.create(attributes={"callee": builtin.SymbolRefAttr(stencil_invoke.name)}, operands=stencil_invoke.arg_ssas, result_types=[])
      stencil_invoke.call_op.parent.insert_op_after(new_call_op, stencil_invoke.call_op)
      stencil_invoke.call_op.parent.erase_op(stencil_invoke.call_op)
      # Now erase all data loading for the origional call op
      for op in stencil_invoke.call_op.args:
        if isa(op.owner, fir.Convert):
          # If owner is a convert then it's an array, therefore delete the loading as we use the pointer
          # instead, otherwise it is passing a scalar (is a load) and we need to keep that
          InferGPUDataTransfer.erase_unused_data_ops(op.owner)

    stencils_module.regions[0].block.add_ops(alloc_funcs)
    driver_module.regions[0].block.add_ops(external_alloc_func_defs)

    v=FindFirstStencilBridgedFunction()
    v.traverse(driver_module)
    assert v.first_bridged_fn is not None
    v.first_bridged_fn.parent.insert_ops_before(call_ops, v.first_bridged_fn)

'''
    copy_back_fn, call_ops, external_def=GenerateDataCopyBack.generate_copy_back(named_gpu_llvm_ptr, handled_symbol_types, driver_module, ["su", "sw", "sv"])

    stencils_module.regions[0].block.add_op(copy_back_fn)
    driver_module.regions[0].block.add_op(external_def)

    v=FindLastTimerStopFunction()
    v.traverse(driver_module)
    assert v.last_timer_stop_fn is not None
    v.last_timer_stop_fn.parent.insert_ops_before(call_ops, v.last_timer_stop_fn)
'''
