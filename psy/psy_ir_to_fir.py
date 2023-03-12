from __future__ import annotations
from xdsl.dialects.builtin import (StringAttr, ModuleOp, IntegerAttr, IntegerType, ArrayAttr, i32, i64, f32, f64, IndexType, DictionaryAttr, IntAttr,
      Float16Type, Float32Type, Float64Type, FloatAttr, UnitAttr, DenseIntOrFPElementsAttr, VectorType, SymbolRefAttr, AnyFloat)
from xdsl.dialects import func, arith, cf, mpi #, gpu
from xdsl.dialects.experimental import math
from xdsl.ir import Operation, Attribute, ParametrizedAttribute, Region, Block, SSAValue, MLContext, BlockArgument
from psy.dialects import psy_ir, hstencil #, hpc_gpu
#from xdsl.dialects.experimental import stencil
from xdsl.dialects.llvm import LLVMPointerType
from util.list_ops import flatten
import uuid
from ftn.dialects import fir
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

binary_arith_op_matching={"ADD": [arith.Addi, arith.Addf], "SUB":[arith.Subi, arith.Subf], "MUL": [arith.Muli, arith.Mulf], "DIV": [arith.DivSI, arith.Divf], "REM": [arith.RemSI, None],
"MIN" : [arith.MinSI, arith.Minf], "MAX" : [arith.MaxSI, arith.Maxf], "POW" : [math.IPowIOp, None, None]}

binary_arith_psy_to_arith_comparison_op={"EQ": "eq", "NE": "ne", "GT": "sgt", "LT": "slt", "GE": "sge", "LE": "sle"}

gpu_module=None

@dataclass
class SSAValueCtx:
    """
    Context that relates identifiers from the AST to SSA values used in the flat representation.
    """
    dictionary: Dict[str, SSAValue] = field(default_factory=dict)
    parent_scope: Optional[SSAValueCtx] = None

    def __getitem__(self, identifier: str) -> Optional[SSAValue]:
        """Check if the given identifier is in the current scope, or a parent scope"""
        ssa_value = self.dictionary.get(identifier, None)
        if ssa_value:
            return ssa_value
        elif self.parent_scope:
            return self.parent_scope[identifier]
        else:
            return None

    def __setitem__(self, identifier: str, ssa_value: SSAValue):
        """Relate the given identifier and SSA value in the current scope"""
        if identifier in self.dictionary:
            raise Exception()
        else:
            self.dictionary[identifier] = ssa_value

class UserDefinedFunction:
  def __init__(self, full_name, args):
    self.full_name=full_name
    self.args=args

user_defined_functions={}

class ProgramState:
  def __init__(self):
    self.module_name=None
    self.routine_name=None
    self.return_block=None
    self.imports={}
    self.globals=[]
    self.global_fn_names=[]
    self.num_gpu_fns=0;
    self.requires_mpi=False

  def setRequiresMPI(self, requires_mpi):
    self.requires_mpi=requires_mpi

  def getRequiresMPI(self):
    return self.requires_mpi

  def setRoutineName(self, routine_name):
    self.routine_name=routine_name

  def unsetRoutineName(self):
    self.routine_name=None

  def getNumGPUFns(self):
    return self.num_gpu_fns

  def incrementNumGPUFns(self):
    self.num_gpu_fns+=1

  def getRoutineName(self):
    return self.routine_name

  def isInRoutine(self):
    return self.routine_name is not None

  def setModuleName(self, module_name):
    self.module_name=module_name

  def unsetModuleName(self):
    self.module_name=None

  def getModuleName(self):
    return self.module_name

  def isInModule(self):
    return self.module_name is not None

  def setReturnBlock(self, rb):
    self.return_block=rb

  def hasReturnBlock(self):
    return self.return_block is not NoneAttr

  def getReturnBlock(self):
    return self.return_block

  def clearReturnBlock(self):
    self.return_block=None

  def addImport(self, container_name, routine_name):
    self.imports[routine_name]=container_name

  def hasImport(self, routine_name):
    return routine_name in self.imports

  def getImportModule(self, routine_name):
    return self.imports[routine_name]

  def clearImports(self):
    self.imports.clear()

  def appendToGlobal(self, glob, fn_name=None):
    self.globals.append(glob)
    if fn_name is not None:
      self.global_fn_names.append(fn_name)

  def getGlobals(self):
    return self.globals

  def hasGlobalFnName(self, fn_name):
    return fn_name in self.global_fn_names

def psy_ir_to_fir(ctx: MLContext, input_module: ModuleOp):
    res_module = translate_program(input_module)
    # Now we need to gather the containers and inform the top level routine about these so it can generate the use
    #applyModuleUseToFloatingRegions(res_module, collectContainerNames(res_module))
    res_module.regions[0].move_blocks(input_module.regions[0])
    # Create program entry point
    #check_program_entry_point(input_module)

def check_program_entry_point(module: ModuleOp):
  for op in module.ops:
    if isinstance(op, func.FuncOp):
      if op.sym_name.data=="_QQmain": return
  assert False, "No program entry point"

def translate_program(input_module: ModuleOp) -> ModuleOp:
    # create an empty global context
    global_ctx = SSAValueCtx()
    globals_list=[]
    containers: List[func.FuncOp] = []
    body = Region()
    block = Block()
    for top_level_entry in input_module.ops:
      if isinstance(top_level_entry, psy_ir.FileContainer):
        for container in top_level_entry.children.blocks[0].ops:
          translate_container(global_ctx, container, block, globals_list)

      elif isinstance(top_level_entry, psy_ir.Container):
        print("NOT IMPLEMENTED")
        #containers.append(translate_container(global_ctx, top_level_entry))
      elif isinstance(top_level_entry, psy_ir.Routine):
        program_state = ProgramState()
        fn_ops=translate_fun_def(global_ctx, top_level_entry, program_state)
        if program_state.getRequiresMPI():
          fn_ops.regions[0].blocks[0].insert_op(mpi.Init.build(), 0)
          # Need to do this to pop finalize before the return at the end of the block
          fn_ops.regions[0].blocks[0].insert_op(mpi.Finalize.build(), len(fn_ops.regions[0].blocks[0].ops)-1)
        block.add_op(fn_ops)
        globals_list.extend(program_state.getGlobals())

    if len(globals_list) > 0:
      block.add_ops(globals_list)

    if gpu_module is not None:
      block.add_ops([gpu_module])

    body.add_block(block)
    return ModuleOp.from_region_or_ops(body)

def translate_container(ctx: SSAValueCtx, op: Operation, block, globals_lists) -> Operation:
  program_state = ProgramState()
  if isinstance(op, psy_ir.Container):
    program_state.setModuleName(op.attributes["container_name"].data)

    block.add_ops(translate_fun_def(ctx, routine, program_state) for routine in op.routines.blocks[0].ops)
  elif isinstance(op, psy_ir.Routine):
    block.add_op(translate_fun_def(ctx, op, program_state))

  globals_lists.extend(program_state.getGlobals())

def translate_fun_def(ctx: SSAValueCtx,
                      routine_def: psy_ir.Routine, program_state : ProgramState) -> Operation:
    routine_name = routine_def.attributes["routine_name"]

    def get_param(op: Operation) -> Tuple[str, Attribute]:
        assert isinstance(op, psy_ast.TypedVar)
        var_name = op.attributes.get('var_name')
        assert isinstance(var_name, StringAttr)
        name = var_name.data
        type_name = op.regions[0].blocks[0].ops[0]
        type = try_translate_type(type_name)
        assert type is not None
        return name, type

    #params = [get_param(op) for op in routine_def.params.blocks[0].ops]
    #param_names: List[str] = [p[0] for p in params]
   # param_types: List[Attribute] = [p[1] for p in params]
    #return_type = try_translate_type(fun_def.return_type.blocks[0].ops[0])
    #if return_type is None:
    #    return_type = choco_type.none_type

    body = Region()


    # Create a new nested scope and relate parameter identifiers with SSA values of block arguments
    # For now create this empty, will add in support for arguments later on!
    c = SSAValueCtx(dictionary=dict(), #zip(param_names, block.args)),
                    parent_scope=ctx)

    arg_types=[]
    arg_names=[]
    for arg in routine_def.args.data:
      translated_type=try_translate_type(arg.type)
      if isinstance(translated_type, fir.ArrayType) and translated_type.hasDeferredShape():
        # If this is an array with a deferred type then wrap it in a box as this will
        # contain parameter size information that is passed by the caller
        arg_type=fir.BoxType([translated_type])
      else:
        # Otherwise it's just a memory reference, nice and easy
        arg_type=fir.ReferenceType([translated_type])
      arg_names.append(arg.var_name.data)
      arg_types.append(arg_type)

    block = Block.from_arg_types(arg_types)

    for index, arg_name in enumerate(arg_names):
      c[arg_name] = block.args[index]

    # use the nested scope when translate the body of the function
    #block.add_ops(
    #    flatten([
    #        translate_def_or_stmt(c, op)
    #        for op in routine_def.local_var_declarations.blocks[0].ops
    #   ]))

    for import_statement in routine_def.imports.blocks[0].ops:
      assert isinstance(import_statement, psy_ir.Import)
      module_name=import_statement.import_name.data
      for fn in import_statement.specific_procedures.data:
        program_state.addImport(module_name, fn.data)

    # Add this function to program state imports
    program_state.addImport(program_state.getModuleName(), routine_name.data)

    to_add=[]
    program_state.setRoutineName(routine_name.data)
    for op in routine_def.local_var_declarations.blocks[0].ops:
      # Ignore dummys to/from procedure as these handled as block arguments
      if op.var.var_name.data not in arg_names:
        res=translate_def_or_stmt(c, op, program_state) # should be SSAValueCtx created above for routine
        if res is not None:
          to_add.append(res)

    is_function=not isinstance(routine_def.return_var, psy_ir.EmptyToken)
    if is_function:
      return_block=Block.from_arg_types([])
      load_return_var=fir.Load.create(operands=[c[routine_def.return_var.var_name.data]], result_types=[try_translate_type(routine_def.return_var.type)])
      return_block.add_ops([load_return_var, func.Return.create(operands=[load_return_var.results[0]])])
      program_state.setReturnBlock(return_block)

    for op in routine_def.routine_body.blocks[0].ops:
      res=translate_def_or_stmt(c, op, program_state) # should be SSAValueCtx created above for routine
      if res is not None:
        to_add.append(res)

    program_state.unsetRoutineName()
    program_state.clearImports()
    program_state.clearReturnBlock()

    block.add_ops(flatten(to_add))

    if not is_function:
       # A return is always needed at the end of the procedure
      block.add_op(func.Return.create())

    body.add_block(block)

    if is_function:
      # Need to return the variable at the end of the routine
      body.add_block(return_block)

    if routine_def.is_program.data:
      full_name="_QQmain"
    else:
      full_name=generateProcedureSymName(program_state, routine_name.data)

    user_defined_functions[routine_name.data]=UserDefinedFunction(full_name, arg_types)

    function_fir=func.FuncOp.from_region(full_name, arg_types, [try_translate_type(routine_def.return_var.type)] if is_function else [], body)
    #TODO - need to correlate against public routines to mark private or public!
    if routine_def.is_program.data:
      function_fir.attributes["sym_visibility"]=StringAttr("public")

    if len(arg_names) > 0:
      arg_attrs={}
      for arg_name in arg_names:
        arg_attrs[StringAttr("fir.bindc_name")]=StringAttr(arg_name)
      function_fir.attributes["arg_attrs"]=DictionaryAttr.from_dict(arg_attrs)
    return function_fir

def generateProcedureSymName(program_state : ProgramState, routine_name:str):
  return generateProcedurePrefix(program_state, routine_name, "P")

def generateProcedurePrefix(program_state : ProgramState, routine_name:str, procedure_identifier:str):
  if program_state.isInModule():
    return generateProcedurePrefixWithModuleName(program_state.getModuleName(), routine_name, procedure_identifier)
  else:
    return generateProcedurePrefixWithModuleName(None, routine_name, procedure_identifier)

def generateProcedurePrefixWithModuleName(module_name:str, routine_name:str, procedure_identifier:str):
  if module_name is not None:
    return "_QM"+module_name.lower()+procedure_identifier+routine_name.lower()
  else:
    return "_QQ"+routine_name.lower()

def generateVariableUniqueName(program_state : ProgramState, var_name:str):
  return generateProcedurePrefix(program_state, program_state.getRoutineName(), "F")+"E"+var_name

def translate_def_or_stmt(ctx: SSAValueCtx, op: Operation, program_state : ProgramState) -> List[Operation]:
    """
    Translate an operation that can either be a definition or statement
    """
    # first try to translate op as a definition:
    #   if op is a definition this will return a list of translated Operations
    ops = try_translate_def(ctx, op, program_state)
    if ops is not None:
        return ops
    # op has not been a definition, try to translate op as a statement:
    #   if op is a statement this will return a list of translated Operations
    ops = try_translate_stmt(ctx, op, program_state)
    if ops is not None:
        return ops
    # operation must have been translated by now
    return None
    raise Exception(f"Could not translate `{op}' as a definition or statement")


def try_translate_def(ctx: SSAValueCtx,
                      op: Operation, program_state : ProgramState) -> Optional[List[Operation]]:
    """
    Tries to translate op as a definition.
    Returns a list of the translated Operations if op is a definition, returns None otherwise.
    """
    #if isinstance(op, psy_ast.Routine):
    #    return [translate_fun_def(ctx, op)]
    if isinstance(op, psy_ir.VarDef):
        return translate_var_def(ctx, op, program_state)
    else:
        return None

def define_scalar_var(ctx: SSAValueCtx,
                      var_def: psy_ast.VarDef, program_state : ProgramState) -> List[Operation]:
    var_name = var_def.var.var_name
    assert isinstance(var_name, StringAttr)
    type = try_translate_type(var_def.var.type)

    ref_type=fir.ReferenceType([type])

    # Operand segment sizes is wrong here, either hack it like trying (but doesn't match!) or understand why missing
    fir_var_def = fir.Alloca.build(attributes={"bindc_name": var_name, "uniq_name": StringAttr(generateVariableUniqueName(program_state, var_name.data)),
      "in_type":type}, operands=[[],[]], regions=[[]], result_types=[ref_type])

    # relate variable identifier and SSA value by adding it into the current context
    ctx[var_name.data] = fir_var_def.results[0]
    return [fir_var_def]

def define_array_var(ctx: SSAValueCtx,
                      var_def: psy_ast.VarDef, program_state : ProgramState) -> List[Operation]:
    var_name = var_def.var.var_name
    type = try_translate_type(var_def.var.type)
    num_deferred=count_array_type_contains_deferred(type)

    region_args=[]
    if num_deferred:
      heap_type=fir.HeapType([type])
      type=fir.BoxType([heap_type])
      zero_bits=fir.ZeroBits.create(result_types=[heap_type])
      zero_val=arith.Constant.create(attributes={"value": IntegerAttr.from_index_int_value(0)},
                                         result_types=[IndexType()])
      shape_ops=[]
      for i in range(num_deferred):
        shape_ops.append(zero_val.results[0])
      shape=fir.Shape.create(operands=shape_ops, result_types=[fir.ShapeType([IntAttr.from_int(num_deferred)])])
      embox=fir.Embox.build(operands=[zero_bits.results[0], shape.results[0], [], []], regions=[[]], result_types=[type])
      has_val=fir.HasValue.create(operands=[embox.results[0]])
      region_args=[zero_bits, zero_val, shape, embox, has_val]

      glob=fir.Global.create(attributes={"linkName": StringAttr("internal"), "sym_name": StringAttr("_QFE"+var_name.data), "symref": SymbolRefAttr.from_str("_QFE"+var_name.data), "type": type},
          regions=[Region.from_operation_list(region_args)])
      addr_lookup=fir.AddressOf.create(attributes={"symbol": SymbolRefAttr.from_str("_QFE"+var_name.data)}, result_types=[fir.ReferenceType([type])])
      program_state.appendToGlobal(glob)
      ctx[var_name.data] = addr_lookup.results[0]
      return [addr_lookup]
    else:
      fir_var_def = fir.Alloca.build(attributes={"bindc_name": var_name, "uniq_name": StringAttr(generateVariableUniqueName(program_state, var_name.data)),
        "in_type":type}, operands=[[],[]], regions=[[]], result_types=[fir.ReferenceType([type])])
      ctx[var_name.data] = fir_var_def.results[0]
      return [fir_var_def]

def count_array_type_contains_deferred(type):
  occurances=0
  for s in type.shape.data:
    if isinstance(s, fir.DeferredAttr):
      occurances+=1
  return occurances

def translate_var_def(ctx: SSAValueCtx,
                      var_def: psy_ir.VarDef, program_state : ProgramState) -> List[Operation]:
    if isinstance(var_def.var.type, psy_ir.NamedType):
      return define_scalar_var(ctx, var_def, program_state)
    elif isinstance(var_def.var.type, psy_ir.ArrayType):
      return define_array_var(ctx, var_def, program_state)

def try_translate_type(op: Operation) -> Optional[Attribute]:
    """Tries to translate op as a type, returns None otherwise."""
    if isinstance(op, psy_ir.NamedType):
      if op.type_name.data == "integer": return i32
      if op.type_name.data == "real":
        if (isinstance(op.precision, psy_ir.EmptyAttr)): return f64
        if op.precision.data == 4: return f32
        if op.precision.data == 8: return f64
        raise Exception("Not sure how to interpret real")

    elif isinstance(op, psy_ir.ArrayType):
      array_shape=op.get_shape()
      array_size=[]
      i=0
      while i<len(array_shape):
        if isinstance(array_shape[i], psy_ir.DeferredAttr) or isinstance(array_shape[i], psy_ir.AssumedSizeAttr):
          array_size.append(fir.DeferredAttr())
        else:
          if isinstance(array_shape[i+1], int) and isinstance(array_shape[i], int):
            array_size.append((array_shape[i+1]-array_shape[i]) + 1)
            # Increment i now (will go up by two based on this and next loop round
            # as have done low to high size)
          i+=1
        i+=1

      arrayType=fir.ArrayType.from_type_and_list(try_translate_type(op.element_type), array_size)
      return arrayType

    return None

def try_translate_stmt(ctx: SSAValueCtx,
                       op: Operation, program_state : ProgramState) -> Optional[List[Operation]]:
    """
    Tries to translate op as a statement.
    If op is an expression, returns a list of the translated Operations.
    Returns None otherwise.
    """
    if isinstance(op, psy_ir.CallExpr):
      return translate_call_expr_stmt(ctx, op, program_state)
    if isinstance(op, psy_ir.Assign):
      return translate_assign(ctx, op, program_state)
    if isinstance(op, psy_ir.IfBlock):
      return translate_if(ctx, op, program_state)
    if isinstance(op, psy_ir.Loop):
      return translate_loop(ctx, op, program_state)
    if isinstance(op, psy_ir.Return):
      return translate_return(ctx, op, program_state)
    #if isinstance(op, hpc_gpu.GPULoop):
    #  return translate_gpu_loop(ctx, op, program_state)
    if isinstance(op, hstencil.HStencil_Stencil):
      return translate_hstencil_stencil(ctx, op, program_state)
    if isinstance(op, hstencil.HStencil_Result):
      return translate_hstencil_result(ctx, op, program_state)

    res = None #try_translate_expr(ctx, op)
    if res is None:
        return None
    else:
        return res[0]

def translate_stmt(ctx: SSAValueCtx, op: Operation, program_state : ProgramState) -> List[Operation]:
    """
    Translates op as a statement.
    If op is an expression, returns a list of the translated Operations.
    Fails otherwise.
    """
    ops = try_translate_stmt(ctx, op, program_state)
    if ops is None:
        raise Exception(f"Could not translate `{op}' as a statement")
    else:
        return ops

def translate_hstencil_access(ctx: SSAValueCtx, stencil_access: Operation, program_state : ProgramState) -> List[Operation]:
  assert isinstance(stencil_access.var.type, psy_ir.ArrayType)
  el_type=try_translate_type(stencil_access.var.type.element_type)
  assert el_type is not None
  access_op=stencil.Access.build(attributes={"offset": stencil_access.stencil_ops}, operands=[ctx[stencil_access.var.var_name.data]], result_types=[el_type])
  return [access_op], access_op.results[0]

def translate_hstencil_result(ctx: SSAValueCtx, stencil_result: Operation, program_state : ProgramState) -> List[Operation]:
  ops: List[Operation] = []
  for op in stencil_result.stencil_accesses.blocks[0].ops:
    stmt_ops, ssa = translate_expr(ctx, op, program_state)
    ops += stmt_ops

  assert isinstance(stencil_result.var.type, psy_ir.ArrayType)
  el_type=try_translate_type(stencil_result.var.type.element_type)
  assert el_type is not None
  rt=stencil.ResultType([el_type])

  store_result_op=stencil.StoreResult.create(operands=[ops[-1].results[0]], result_types=[rt])
  return_op=stencil.Return.create(operands=[store_result_op.results[0]])
  ops+=[store_result_op, return_op]

  block=Block()
  block.add_ops(ops)
  body=Region()
  body.add_block(block)

  array_sizes=get_array_sizes(stencil_result.var.type)

  apply_op=stencil.Apply.create(operands=[ctx[stencil_result.var.var_name.data]], regions=[body], result_types=[stencil.TempType.from_shape(array_sizes)])

  return [apply_op]

def get_array_sizes(array_type):
  shape_size=len(array_type.shape.data)
  assert shape_size % 2 == 0
  sizes=[]
  for i in range(0, shape_size, 2):
    sizes.append((array_type.shape.data[i+1].value.data-array_type.shape.data[i].value.data)+1)
  return sizes

def translate_hstencil_stencil(ctx: SSAValueCtx, stencil_stmt: Operation, program_state : ProgramState) -> List[Operation]:
  ops: List[Operation] = []
  new_ctx=SSAValueCtx()
  for field in stencil_stmt.input_fields.data:
    assert isinstance(field.type, psy_ir.ArrayType)
    array_sizes=get_array_sizes(field.type)
    external_load_op=stencil.External_Load.build(operands=[ctx[field.var_name.data]], result_types=[stencil.FieldType.from_shape(array_sizes)])
    load_op=stencil.Load.build(operands=[external_load_op.results[0]], result_types=[stencil.TempType.from_shape(array_sizes)])
    ops+=[external_load_op, load_op]
    new_ctx[field.var_name.data]=load_op.results[0]

  for op in stencil_stmt.body.blocks[0].ops:
    stmt_ops = translate_stmt(new_ctx, op, program_state)
    ops += stmt_ops

  out_var=stencil_stmt.output_fields.data[0].var_name.data
  store_op=stencil.Store.create(operands=[ops[-1].results[0], external_load_op.results[0]])
  external_store_op=stencil.External_Store.create(operands=[external_load_op.results[0], ctx[out_var]])
  ops+=[store_op, external_store_op]

  return ops

def translate_gpu_loop(ctx: SSAValueCtx, gpu_stmt: Operation, program_state : ProgramState) -> List[Operation]:
  global gpu_module
  ops: List[Operation] = []
  for op in gpu_stmt.loop.blocks[0].ops:
    stmt_ops = translate_stmt(ctx, op, program_state)
    ops += stmt_ops

  ops.append(gpu.ReturnOp.create())

  # For now empty block arguments, will be values in and out
  body = Region.from_operation_list(ops)
  gpu_fn=gpu.GPUFuncOp.from_region("gpu_fn_"+str(program_state.getNumGPUFns()), [], [], body)
  end_op=gpu.ModuleEndOp.create()
  if gpu_module is None:
    gpu_module=gpu.GPUModuleOp.from_region(Region.from_operation_list([gpu_fn, end_op]), "gpu_functions")
  else:
    pass # Need to add in ability to append GPU function here

  # Hacking in the "@" character on the GPU function name here
  launch_fn=gpu.LaunchFuncOp.create(attributes={"kernel":SymbolRefAttr.from_str("gpu_fns.@gpu_fn_"+str(program_state.getNumGPUFns()))})
  program_state.incrementNumGPUFns()
  return [launch_fn]

def translate_return(ctx: SSAValueCtx, return_stmt: psy_ir.Return, program_state : ProgramState) -> List[Operation]:
  return [cf.Branch.get(program_state.getReturnBlock())]

def translate_loop(ctx: SSAValueCtx,
                  for_stmt: psy_ir.Loop, program_state : ProgramState) -> List[Operation]:
    start, start_name = translate_expr(ctx, for_stmt.start.blocks[0].ops[0], program_state)
    conv_start=fir.Convert.create(operands=[start_name], result_types=[IndexType()])
    stop, stop_name = translate_expr(ctx, for_stmt.stop.blocks[0].ops[0], program_state)
    conv_stop=fir.Convert.create(operands=[stop_name], result_types=[IndexType()])
    step, step_name = translate_expr(ctx, for_stmt.step.blocks[0].ops[0], program_state)
    conv_step=fir.Convert.create(operands=[step_name], result_types=[IndexType()])

    ops: List[Operation] = []
    for op in for_stmt.body.blocks[0].ops:
        stmt_ops = translate_stmt(ctx, op, program_state)
        ops += stmt_ops
    #body = Region.from_operation_list(ops)

    iterator = ctx[for_stmt.variable.var_name.data]

    block = Block.from_arg_types([IndexType(), i32])
    store=fir.Store.create(operands=[block.args[1], iterator])

    add_iteration_count=arith.Addi.get(block.args[0], conv_step)
    load_iterator_var=fir.Load.create(operands=[iterator], result_types=[try_translate_type(for_stmt.variable.type)])
    convert_step_for_it=fir.Convert.create(operands=[conv_step.results[0]], result_types=[i32])
    add_to_iterator=arith.Addi.get(load_iterator_var.results[0], convert_step_for_it.results[0])
    block_result=fir.Result.create(operands=[add_iteration_count.results[0], add_to_iterator.results[0]])

    block.add_ops([store]+ops+[add_iteration_count, load_iterator_var, convert_step_for_it, add_to_iterator, block_result])
    body=Region()
    body.add_block(block)

    do_loop=fir.DoLoop.create(attributes={"finalValue": UnitAttr()},
    operands=[conv_start.results[0], conv_stop.results[0], conv_step.results[0], start_name], result_types=[IndexType(), i32], regions=[body])
    final_iterator_store=fir.Store.create(operands=[do_loop.results[1], iterator])
    return start+[conv_start]+stop+[conv_stop]+step+[conv_step, do_loop, final_iterator_store]


def translate_if(ctx: SSAValueCtx, if_stmt: psy_ir.If, program_state : ProgramState) -> List[Operation]:
    cond, cond_name = translate_expr(ctx, if_stmt.cond.blocks[0].ops[0], program_state)

    ops: List[Operation] = []
    for op in if_stmt.then.blocks[0].ops:
        stmt_ops = translate_stmt(ctx, op, program_state)
        ops += stmt_ops
    ops.append(fir.Result.create())
    then = Region.from_operation_list(ops)

    ops: List[Operation] = []
    for op in if_stmt.orelse.blocks[0].ops:
        stmt_ops = translate_stmt(ctx, op, program_state)
        ops += stmt_ops
    ops.append(fir.Result.create())
    orelse = Region.from_operation_list(ops)

    new_op = fir.If.create(operands=[cond_name], regions=[then, orelse])
    return cond + [new_op]

def get_store_conversion_if_needed(target_type, expr_type):
  if isinstance(target_type, fir.ReferenceType): return get_store_conversion_if_needed(target_type.type, expr_type)
  if isinstance(target_type, IntegerType):
    if isinstance(expr_type, IntegerType):
      if (target_type.width.data != expr_type.width.data):
        return target_type
      else:
        return None
    else:
      return target_type
  if isinstance(target_type, Float16Type) and not isinstance(expr_type, Float16Type):
    return target_type
  if isinstance(target_type, Float32Type) and not isinstance(expr_type, Float32Type):
    return target_type
  if isinstance(target_type, Float64Type) and not isinstance(expr_type, Float64Type):
    return target_type

  return None

def translate_assign(ctx: SSAValueCtx,
                     assign: psy_ast.Assign, program_state : ProgramState) -> List[Operation]:
    value_fir, value_var = translate_expr(ctx, assign.rhs.op, program_state)
    lhs_fir=[]
    # The targets of the assignment are references and not expressions, so grab from the ctx
    if isinstance(assign.lhs.op, psy_ir.ArrayReference):
      lhs_fir, lhs_var=translate_expr(ctx, assign.lhs.op, program_state)
      translated_target=lhs_var
    else:
      translated_target = ctx[assign.lhs.op.id.data]
    target_conversion_type=get_store_conversion_if_needed(translated_target.typ, value_var.typ)
    if target_conversion_type is not None:
      if isinstance(value_var.typ, fir.ReferenceType):
        # This is a reference, therefore load it
        converter=fir.Load.create(operands=[value_var], result_types=[target_conversion_type])
      else:
        # Otherwise data type conversion
        converter=fir.Convert.create(operands=[value_var], result_types=[target_conversion_type])
      value_fir.append(converter)
      return value_fir + lhs_fir + [fir.Store.create(operands=[converter.results[0], translated_target])]
    else:
      return value_fir + lhs_fir +[fir.Store.create(operands=[value_var, translated_target])]

def translate_call_expr_stmt(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:

    if call_expr.attributes["intrinsic"].data:
      return translate_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    else:
      return translate_user_call_expr(ctx, call_expr, program_state, is_expr)

def translate_intrinsic_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
    intrinsic_name=call_expr.attributes["func"].data
    if intrinsic_name == "allocate":
      return translate_allocate_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    if intrinsic_name == "deallocate":
      return translate_deallocate_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    if intrinsic_name.lower() == "print":
      return translate_print_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    if intrinsic_name.lower() == "mpi_commrank":
      return translate_mpi_commrank_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    if intrinsic_name.lower() == "mpi_commsize":
      return translate_mpi_commsize_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    if intrinsic_name.lower() == "mpi_send":
      return translate_mpi_send_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    if intrinsic_name.lower() == "mpi_recv":
      return translate_mpi_recv_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)

def translate_mpi_send_intrinsic_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
    program_state.setRequiresMPI(True)
    assert len(call_expr.args.blocks[0].ops) == 4
    assert isinstance(call_expr.args.blocks[0].ops[0], psy_ir.ExprName)

    ptr_type=try_translate_type(call_expr.args.blocks[0].ops[0].var.type)
    # Pointer type needs to be base type which might be wrapped in an array
    if isinstance(ptr_type, fir.ArrayType): ptr_type=ptr_type.type

    buffer_op, buffer_arg = translate_expr(ctx, call_expr.args.blocks[0].ops[0], program_state)
    convert_buffer=fir.Convert.create(operands=[buffer_arg],
                    result_types=[fir.LLVMPointerType([ptr_type])])
    count_op, count_arg = translate_expr(ctx, call_expr.args.blocks[0].ops[1], program_state)
    target_op, target_arg = translate_expr(ctx, call_expr.args.blocks[0].ops[2], program_state)
    tag_op, tag_arg = translate_expr(ctx, call_expr.args.blocks[0].ops[3], program_state)
    get_mpi_dtype_op=mpi.GetDtypeOp.get(ptr_type)
    mpi_send_op=mpi.Send.get(convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], target_arg, tag_arg)

    return buffer_op + count_op + target_op + tag_op + [convert_buffer, get_mpi_dtype_op, mpi_send_op]

def translate_mpi_recv_intrinsic_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
    program_state.setRequiresMPI(True)
    assert len(call_expr.args.blocks[0].ops) == 4
    assert isinstance(call_expr.args.blocks[0].ops[0], psy_ir.ExprName)

    ptr_type=try_translate_type(call_expr.args.blocks[0].ops[0].var.type)
    # Pointer type needs to be base type which might be wrapped in an array
    if isinstance(ptr_type, fir.ArrayType): ptr_type=ptr_type.type

    buffer_op, buffer_arg = translate_expr(ctx, call_expr.args.blocks[0].ops[0], program_state)
    convert_buffer=fir.Convert.create(operands=[buffer_arg],
                    result_types=[fir.LLVMPointerType([ptr_type])])
    count_op, count_arg = translate_expr(ctx, call_expr.args.blocks[0].ops[1], program_state)
    source_op, source_arg = translate_expr(ctx, call_expr.args.blocks[0].ops[2], program_state)
    tag_op, tag_arg = translate_expr(ctx, call_expr.args.blocks[0].ops[3], program_state)
    get_mpi_dtype_op=mpi.GetDtypeOp.get(ptr_type)
    mpi_recv_op=mpi.Recv.get(convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], source_arg, tag_arg)

    return buffer_op + count_op + source_op + tag_op + [convert_buffer, get_mpi_dtype_op, mpi_recv_op]

def type_to_mpi_datatype(typ):
  if typ==i32:
    return mpi.MPI_INT
  raise Exception(f"Could not translate type`{typ}' to MPI datatype as this unknown")

def translate_mpi_commrank_intrinsic_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
    program_state.setRequiresMPI(True)
    mpi_call=mpi.CommRank.get()
    return [mpi_call]

def translate_mpi_commsize_intrinsic_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
    program_state.setRequiresMPI(True)
    mpi_call=mpi.CommSize.get()
    return [mpi_call]

def translate_print_intrinsic_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
    arg_operands=[]

    # Start the IO session
    filename_str_op=generate_string_literal("./dummy.F90", program_state)
    arg1=arith.Constant.create(attributes={"value": IntegerAttr.from_int_and_width(-1, 32)}, result_types=[i32])
    arg2=fir.Convert.create(operands=[filename_str_op.results[0]], result_types=[fir.ReferenceType([IntegerType.from_width(8)])])
    arg3=arith.Constant.create(attributes={"value": IntegerAttr.from_int_and_width(3, 32)}, result_types=[i32])

    call1=fir.Call.create(attributes={"callee": SymbolRefAttr.from_str("_FortranAioBeginExternalListOutput")}, operands=[arg1.results[0],
      arg2.results[0], arg3.results[0]], result_types=[fir.ReferenceType([IntegerType.from_width(8)])])
    arg_operands.extend([filename_str_op, arg1, arg2, arg3, call1])

    insertExternalFunctionToGlobalState(program_state, "_FortranAioBeginExternalListOutput", [i32, fir.ReferenceType([IntegerType.from_width(8)]),
      i32], fir.ReferenceType([IntegerType.from_width(8)]))

    # Ignore first argument as it will be a star
    for argument in call_expr.args.blocks[0].ops[1:]:
      # For each argument need to issue a different print
      op, arg = translate_expr(ctx, argument, program_state)
      # Now do the actual print
      if isinstance(arg.typ, IntegerType) or isinstance(arg.typ, AnyFloat):
        arg_operands.extend(generatePrintForIntegerOrFloat(program_state, op[0], arg, call1.results[0]))


      if isinstance(arg.typ, fir.ReferenceType):
        if isinstance(arg.typ.type, fir.IntegerType) or isinstance(arg.typ.type, AnyFloat):
          load_op=fir.Load.create(operands=[arg], result_types=[arg.typ.type])
          print_ops=generatePrintForIntegerOrFloat(program_state, load_op, load_op.results[0], call1.results[0])
          arg_operands.append(ops[0])
          arg_operands.extend(print_ops)
        if isinstance(arg.typ.type, fir.CharType):
          # This is a reference to a string
          arg_operands.extend(generatePrintForString(program_state, op[0], arg, call1.results[0]))

    # Close out the IO
    call3=fir.Call.create(attributes={"callee": SymbolRefAttr.from_str("_FortranAioEndIoStatement")}, operands=[call1.results[0]],
      result_types=[IntegerType.from_width(32)])
    arg_operands.extend([call3])

    insertExternalFunctionToGlobalState(program_state, "_FortranAioEndIoStatement", [fir.ReferenceType([IntegerType.from_width(8)])], i32)

    return arg_operands

def generatePrintForIntegerOrFloat(program_state, op, arg, init_call_ssa):
    if isinstance(arg.typ, IntegerType):
      if arg.typ.width.data == 64:
        fn_name="_FortranAioOutputInteger64"
      elif arg.typ.width.data == 32:
        fn_name="_FortranAioOutputInteger32"
      else:
        raise Exception(f"Could not translate integer width`{arg.typ.width.data}' as only 32 or 64 bit supported")
    elif isinstance(arg.typ, AnyFloat):
      if isinstance(arg.typ, Float32Type):
        fn_name="_FortranAioOutputReal32"
      elif isinstance(arg.typ, Float64Type):
        fn_name="_FortranAioOutputReal64"
      else:
        raise Exception(f"Could not translate float type`{arg.typ}' as only 32 or 64 bit supported")
    else:
      raise Exception(f"Could not translate type`{arg.typ}' for printing")
    print_call=fir.Call.create(attributes={"callee": SymbolRefAttr.from_str(fn_name)}, operands=[init_call_ssa,
          arg], result_types=[IntegerType.from_width(1)])
    insertExternalFunctionToGlobalState(program_state, fn_name, [init_call_ssa.typ, arg.typ], IntegerType.from_width(1))
    return [op, print_call]

def generatePrintForString(program_state, op, arg, init_call_ssa):
    from_num=arg.typ.type.from_index.data
    to_num=arg.typ.type.to_index.data
    string_length=((to_num-from_num)+1)
    str_len=arith.Constant.create(attributes={"value": IntegerAttr.from_index_int_value(string_length)}, result_types=[IndexType()])
    arg2_2=fir.Convert.create(operands=[arg], result_types=[fir.ReferenceType([IntegerType.from_width(8)])])
    arg3_2=fir.Convert.create(operands=[str_len.results[0]], result_types=[i64])
    print_call=fir.Call.create(attributes={"callee": SymbolRefAttr.from_str("_FortranAioOutputAscii")}, operands=[init_call_ssa,
          arg2_2.results[0], arg3_2.results[0]], result_types=[IntegerType.from_width(1)])
    insertExternalFunctionToGlobalState(program_state, "_FortranAioOutputAscii", [init_call_ssa.typ,
        fir.ReferenceType([IntegerType.from_width(8)]), i64], IntegerType.from_width(1))
    return [op, str_len, arg2_2, arg3_2, print_call]

def insertExternalFunctionToGlobalState(program_state, function_name, args, result_type):
    if not program_state.hasGlobalFnName(function_name):
      fn=func.FuncOp.external(function_name, args, [result_type])
      program_state.appendToGlobal(fn, function_name)

def translate_deallocate_intrinsic_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:

    if len(call_expr.args.blocks[0].ops) != 1:
      raise Exception(f"For deallocate expected 1 argument but {len(call_expr.args.blocks[0].ops)} are present")

    op, arg = translate_expr(ctx, call_expr.args.blocks[0].ops[0], program_state)
    # The translate expression unboxes this for us, so we need to look into the operation that does that
    # which is a load, and then grab the origional SSA reference from that argument
    # We use the load initially here to load in the box and unbox it
    target_ssa=op[0].operands[0]

    box_type=get_nested_type(target_ssa.typ, fir.BoxType)
    heap_type=get_nested_type(target_ssa.typ, fir.HeapType)
    array_type=get_nested_type(target_ssa.typ, fir.ArrayType)
    num_deferred=count_array_type_contains_deferred(array_type)

    load_op=fir.Load.create(operands=[target_ssa], result_types=[box_type])
    box_addr_op=fir.BoxAddr.create(operands=[load_op.results[0]], result_types=[heap_type])


    freemem_op=fir.Freemem.create(operands=[arg])
    zero_bits_op=fir.ZeroBits.create(result_types=[heap_type])
    zero_val_op=arith.Constant.create(attributes={"value": IntegerAttr.from_index_int_value(0)},
                                         result_types=[IndexType()])
    shape_operands=[]
    for i in range(num_deferred):
      shape_operands.append(zero_val_op.results[0])
    shape_op=fir.Shape.create(operands=shape_operands, result_types=[fir.ShapeType([IntAttr.from_int(num_deferred)])])
    embox_op=fir.Embox.build(operands=[zero_bits_op.results[0], shape_op.results[0], [], []], regions=[[]], result_types=[box_type])
    store_op=fir.Store.create(operands=[embox_op.results[0], target_ssa])

    return op+[freemem_op, zero_bits_op, zero_val_op, shape_op, embox_op, store_op]


def translate_allocate_intrinsic_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
    ops: List[Operation] = []
    args: List[SSAValue] = []

    for index, arg in enumerate(call_expr.args.blocks[0].ops):
      op, ssa_arg = translate_expr(ctx, arg, program_state)
      if index==0:
        target_var=op
        var_name="_QFE"+arg.var.var_name.data
        # The translate expression already unboxes this for us, so we need to look into the operation that does that
        # which is a load and we don't care about here, and then grab the origional SSA reference from that argument
        target_ssa=op[0].operands[0]
      else:
        if op is not None: ops += op
        convert_op=fir.Convert.create(operands=[ssa_arg], result_types=[IndexType()])
        ops.append(convert_op)
        args.append(convert_op.results[0])
    heap_type=get_nested_type(target_ssa.typ, fir.HeapType)
    array_type=get_nested_type(target_ssa.typ, fir.ArrayType)

    allocmem_op=fir.Allocmem.build(attributes={"in_type":array_type, "uniq_name": StringAttr(var_name+".alloc")}, operands=[[], args], regions=[[]], result_types=[heap_type])
    shape_op=fir.Shape.create(operands=args, result_types=[fir.ShapeType([IntAttr.from_int(len(args))])])
    embox_op=fir.Embox.build(operands=[allocmem_op.results[0], shape_op.results[0], [], []], regions=[[]], result_types=[fir.BoxType([heap_type])])
    store_op=fir.Store.create(operands=[embox_op.results[0], target_ssa])
    ops+=[allocmem_op, shape_op, embox_op, store_op]
    return ops

def get_nested_type(in_type, search_type):
  if isinstance(in_type, search_type): return in_type
  return get_nested_type(in_type.type, search_type)

def has_nested_type(in_type, search_type):
  if isinstance(in_type, search_type): return True
  if getattr(in_type, "type", None) is None: return False
  return has_nested_type(in_type.type, search_type)

def translate_user_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
    ops: List[Operation] = []
    args: List[SSAValue] = []

    name=call_expr.attributes["func"]
    fn_info=user_defined_functions[name.data]

    for index, arg in enumerate(call_expr.args.blocks[0].ops):
        op, arg = translate_expr(ctx, arg, program_state)
        if op is not None: ops += op
        type_to_reference=arg.typ
        if not isinstance(type_to_reference, fir.ReferenceType) and not isinstance(type_to_reference, fir.ArrayType):
          if isinstance(type_to_reference, fir.HeapType):
            type_to_reference=type_to_reference.type
            convert_op=fir.Convert.create(operands=[arg], result_types=[fn_info.args[index]])
            ops+=[convert_op]
            args.append(convert_op.results[0])
          else:
            reference_creation=fir.Alloca.build(attributes={"in_type":arg.typ, "valuebyref": UnitAttr()}, operands=[[],[]],
                              regions=[[]], result_types=[fir.ReferenceType([type_to_reference])])
            store_op=fir.Store.create(operands=[arg, reference_creation.results[0]])
            ops+=[reference_creation, store_op]
            args.append(reference_creation.results[0])
        else:
          if isinstance(fn_info.args[index], fir.BoxType) and isinstance(type_to_reference, fir.ReferenceType):
            # We have a local array that is a reference array, but the calling function accepts a box as it
            # uses assumed sizes for the size of the array
            array_type=get_nested_type(type_to_reference, fir.ArrayType)
            val=array_type.shape.data[0].value.data
            constant_op=arith.Constant.create(attributes={"value": IntegerAttr.from_index_int_value(val)}, result_types=[IndexType()])
            shape_op=fir.Shape.create(operands=[constant_op.results[0]], result_types=[fir.ShapeType([IntAttr.from_int(1)])])
            embox_op=fir.Embox.build(operands=[arg, shape_op.results[0], [], []], regions=[[]], result_types=[fir.BoxType([array_type])])
            convert_op=fir.Convert.create(operands=[embox_op.results[0]], result_types=[fn_info.args[index]])
            ops+=[constant_op, shape_op, embox_op, convert_op]
            args.append(convert_op.results[0])
          else:
            args.append(arg)

    #assert program_state.hasImport(name.data)
    #full_name=generateProcedurePrefixWithModuleName(program_state.getImportModule(name.data), name.data, "P")

    # Need return type here for expression
    if is_expr:
      result_type=try_translate_type(call_expr.type)
      call = fir.Call.create(attributes={"callee": SymbolRefAttr.from_str(fn_info.full_name)}, operands=args, result_types=[result_type])
    else:
      call = fir.Call.create(attributes={"callee": SymbolRefAttr.from_str(fn_info.full_name)}, operands=args, result_types=[])
    ops.append(call)
    return ops

def translate_expr(ctx: SSAValueCtx,
                   op: Operation, program_state : ProgramState) -> Tuple[List[Operation], SSAValue]:
    """
    Translates op as an expression.
    If op is an expression, returns a list of the translated Operations
    and the ssa value representing the translated expression.
    Fails otherwise.
    """
    res = try_translate_expr(ctx, op, program_state)
    if res is None:
        raise Exception(f"Could not translate `{op}' as an expression")
    else:
        ops, ssa_value = res
        return ops, ssa_value

def try_translate_expr(
        ctx: SSAValueCtx,
        op: Operation, program_state : ProgramState) -> Optional[Tuple[List[Operation], SSAValue]]:
    """
    Tries to translate op as an expression.
    If op is an expression, returns a list of the translated Operations
    and the ssa value representing the translated expression.
    Returns None otherwise.
    """
    if isinstance(op, psy_ir.Literal):
      op = translate_literal(op, program_state)
      return [op], op.results[0]
    if isinstance(op, psy_ir.ExprName):
      ssa_value = ctx[op.id.data]
      assert isinstance(ssa_value, SSAValue)
      # We are limited here with type handling, need other floats - maybe a better way of doing this?
      if isinstance(ssa_value.typ, fir.ArrayType):
        return None, ssa_value
      elif isinstance(ssa_value.typ.type, IntegerType):
        result_type=i32
      elif isinstance(ssa_value.typ.type, Float32Type):
        result_type=f32
      elif isinstance(ssa_value.typ.type, Float64Type):
        result_type=f64
      elif isinstance(ssa_value.typ.type, fir.ArrayType):
        # Already have created the addressof reference so just return this
        return None, ssa_value
      elif isinstance(ssa_value.typ.type, fir.BoxType):
        # If it is a box type then unbox it
        load_op=fir.Load.create(operands=[ssa_value], result_types=[ssa_value.typ.type])
        boxaddr_op=fir.BoxAddr.create(operands=[load_op.results[0]], result_types=[ssa_value.typ.type.type])
        return [load_op, boxaddr_op], boxaddr_op.results[0]

      # This is a bit wierd, we have looked into the types of the reference and handled
      # these above, now issue the load (if indeed we want to do this)
      op=fir.Load.create(operands=[ssa_value], result_types=[result_type])
      return [op], op.results[0]

    if isinstance(op, psy_ir.BinaryOperation):
      return translate_binary_expr(ctx, op, program_state)
    if isinstance(op, psy_ir.UnaryOperation):
      return translate_unary_expr(ctx, op, program_state)
    if isinstance(op, psy_ir.CallExpr):
      call_expr= translate_call_expr_stmt(ctx, op, program_state, True)
      return call_expr, call_expr[-1].results[0]
    if isinstance(op, psy_ir.ArrayReference):
      return translate_array_reference_expr(ctx, op, program_state)
    if isinstance(op, hstencil.HStencil_Access):
      return translate_hstencil_access(ctx, op, program_state)

    assert False, "Unknown Expression"

def translate_array_reference_expr(ctx: SSAValueCtx, op: psy_ir.ArrayReference, program_state : ProgramState):
  expressions=[]
  ssa_list=[]

  boxdims_subtractor=None

  base_type=None

  if (has_nested_type(ctx[op.var.var_name.data].typ, fir.BoxType) and has_nested_type(ctx[op.var.var_name.data].typ, fir.HeapType)):
    # We need to debox this
    box_type=get_nested_type(ctx[op.var.var_name.data].typ, fir.BoxType)
    heap_type=get_nested_type(ctx[op.var.var_name.data].typ, fir.HeapType)

    load_op=fir.Load.create(operands=[ctx[op.var.var_name.data]], result_types=[box_type])
    zero_op=arith.Constant.create(attributes={"value": IntegerAttr.from_index_int_value(0)}, result_types=[IndexType()])
    boxdims_op=fir.BoxDims.create(operands=[load_op.results[0], zero_op.results[0]], result_types=[IndexType(), IndexType(), IndexType()])
    boxaddr_op=fir.BoxAddr.create(operands=[load_op.results[0]], result_types=[heap_type])

    expressions+=[load_op, zero_op, boxdims_op, boxaddr_op]
    ssa_list.append(boxaddr_op.results[0])
    base_type=boxaddr_op.results[0].typ
    # Below is needed for subtraction from each argument
    boxdims_subtractor=boxdims_op.results[0]
  else:
    # This is a reference to an array, nice and easy just use it directly
    ssa_list.append(ctx[op.var.var_name.data])
    base_type=ctx[op.var.var_name.data].typ

  for accessor in op.accessors.blocks[0].ops:
    # A lot of this is doing the subtraction to zero-base each index (default is starting at 1 in Fortran)
    # TODO - currently we assume always starts at 1 but in Fortran can set this so will need to keep track of that in the declaration and apply here
    expr, ssa=try_translate_expr(ctx, accessor, program_state)
    expressions.extend(expr)
    if boxdims_subtractor is None:
      subtraction_index=arith.Constant.create(attributes={"value": IntegerAttr.from_int_and_width(1, 64)},
                                         result_types=[i64])
    else:
      subtraction_index=fir.Convert.create(operands=[boxdims_subtractor], result_types=[i64])
    expressions.append(subtraction_index)

    lhs_conv_type, rhs_conv_type=get_expression_conversion_type(ssa.typ, subtraction_index.results[0].typ)

    lhs_conv=perform_data_conversion_if_needed(ssa, lhs_conv_type)
    if lhs_conv is not None:
      expressions.append(lhs_conv)
      lhs_ssa=lhs_conv.results[0]
    else:
      lhs_ssa=ssa

    rhs_conv=perform_data_conversion_if_needed(subtraction_index.results[0], rhs_conv_type)
    if rhs_conv is not None:
      expressions.append(rhs_conv)
      rhs_ssa=rhs_conv.results[0]
    else:
      rhs_ssa=subtraction_index.results[0]

    substract_expr=arith.Subi.get(lhs_ssa, rhs_ssa)
    expressions.append(substract_expr)
    ssa_list.append(substract_expr.results[0])

  fir_type=try_translate_type(op.var.type)

  coordinate_of=fir.CoordinateOf.create(attributes={"baseType": base_type}, operands=ssa_list, result_types=[fir.ReferenceType([fir_type.type])])
  expressions.append(coordinate_of)
  return expressions, coordinate_of.results[0]

def translate_unary_expr(ctx: SSAValueCtx,
        unary_expr: psy_ir.UnaryOperation, program_state : ProgramState) -> Tuple[List[Operation], SSAValue]:

  expr, expr_ssa_value = translate_expr(ctx, unary_expr.expr.blocks[0].ops[0], program_state)

  attr = unary_expr.op
  assert isinstance(attr, Attribute)

  if (attr.data == "NOT"):
    constant_true=arith.Constant.create(attributes={"value": IntegerAttr.from_int_and_width(1, 1)},
                                         result_types=[IntegerType.from_width(1)])
    xori=arith.XOrI.get(expr_ssa_value, constant_true.results[0])

    return expr + [constant_true, xori], xori.results[0]

  if (attr.data == "SQRT"):
    sqrt_op=math.SqrtOp.create(operands=[expr_ssa_value], result_types=[expr_ssa_value.typ])
    return expr + [sqrt_op], sqrt_op.results[0]

def get_expression_conversion_type(lhs_type, rhs_type):
  if isinstance(lhs_type, IntegerType):
    if isinstance(rhs_type, IntegerType):
      if lhs_type.width.data > rhs_type.width.data: return None, lhs_type
      if lhs_type.width.data < rhs_type.width.data: return rhs_type, None
      return None, None
    return rhs_type, None # assuming it is float, so we will convert lhs to this
  if isinstance(lhs_type, Float16Type):
    if isintance(rhs_type, Float32Type) or isintance(rhs_type, Float64Type): return rhs_type, None
  if isinstance(lhs_type, Float32Type):
    if isinstance(rhs_type, Float16Type): return None, lhs_type
    if isinstance(rhs_type, Float64Type): return rhs_type, None
  if isinstance(lhs_type, Float64Type):
    if isinstance(rhs_type, Float16Type) or isinstance(rhs_type, Float32Type): return None, lhs_type,
  return None, None

def perform_data_conversion_if_needed(expr_ssa, conv_type):
  if conv_type is not None:
    expr_conversion=fir.Convert.create(operands=[expr_ssa], result_types=[conv_type])
    return expr_conversion
  return None

def translate_binary_expr(
        ctx: SSAValueCtx,
        binary_expr: psy_ir.BinaryOperation, program_state : ProgramState) -> Tuple[List[Operation], SSAValue]:
    lhs, lhs_ssa_value = translate_expr(ctx, binary_expr.lhs.blocks[0].ops[0], program_state)
    rhs, rhs_ssa_value = translate_expr(ctx, binary_expr.rhs.blocks[0].ops[0], program_state)
    result_type = lhs_ssa_value.typ

    if isinstance(lhs_ssa_value.typ, fir.ReferenceType):
      load_op=fir.Load.create(operands=[lhs_ssa_value], result_types=[lhs_ssa_value.typ.type])
      lhs.append(load_op)
      lhs_ssa_value=load_op.results[0]

    if isinstance(rhs_ssa_value.typ, fir.ReferenceType):
      load_op=fir.Load.create(operands=[rhs_ssa_value], result_types=[rhs_ssa_value.typ.type])
      rhs.append(load_op)
      rhs_ssa_value=load_op.results[0]

    lhs_conv_type, rhs_conv_type=get_expression_conversion_type(lhs_ssa_value.typ, rhs_ssa_value.typ)
    lhs_conv=perform_data_conversion_if_needed(lhs_ssa_value, lhs_conv_type)
    if lhs_conv is not None:
      lhs.append(lhs_conv)
      lhs_ssa_value=lhs_conv.results[0]

    rhs_conv=perform_data_conversion_if_needed(rhs_ssa_value, rhs_conv_type)
    if rhs_conv is not None:
      rhs.append(rhs_conv)
      rhs_ssa_value=rhs_conv.results[0]

    #assert (lhs_ssa_value.typ == rhs_ssa_value.typ) or isinstance(lhs_ssa_value.typ, fir.ReferenceType) or isinstance(rhs_ssa_value.typ, fir.ReferenceType)

    attr = binary_expr.op
    assert isinstance(attr, Attribute)

    fir_binary_expr=get_arith_instance(binary_expr.op.data, lhs_ssa_value, rhs_ssa_value, program_state)

    return lhs + rhs + [fir_binary_expr], fir_binary_expr.results[0]

def get_arith_instance(operation:str, lhs, rhs, program_state : ProgramState):
  operand_type = lhs.typ
  if operation in binary_arith_op_matching:
    if isinstance(operand_type, IntegerType): index=0
    if isinstance(operand_type, Float16Type) or isinstance(operand_type, Float32Type) or isinstance(operand_type, Float64Type): index=1
    op_instance=binary_arith_op_matching[operation][index]
    if op_instance is None and operation == "POW":
      if (lhs.typ == f64 or lhs.typ == f32) and (rhs.typ == f64 or rhs.typ == f32):
        # Use math.powf
        return math.PowFOp.get(lhs, rhs)
      elif (lhs.typ == f64 or lhs.typ == f32) and (rhs.typ == i32):
        # Will call math.ipowi for integer, otherwise call into LLVM function directly
        call_name="llvm.powi."+str(lhs.typ)+".i32"
        fn_call_op=fir.Call.create(attributes={"callee": SymbolRefAttr.from_str(call_name)},
            operands=[lhs, rhs], result_types=[lhs.typ])
        insertExternalFunctionToGlobalState(program_state, call_name, [lhs.typ, rhs.typ], lhs.typ)
        return fn_call_op
      else:
        raise Exception(f"Could not translate `{lhs.typ}' and '{rhs.typ}' for POW operation")
    assert op_instance is not None, "Operation "+operation+" not implemented for type"
    return op_instance.get(lhs, rhs)

  if (operation == "AND"):
    assert isinstance(operand_type, IntegerType), "Integer type only supported for 'and'"
    return arith.AndI.get(lhs, rhs)
  if (operation == "OR"):
    assert isinstance(operand_type, IntegerType), "Integer type only supported for 'or'"
    return arith.OrI.get(lhs, rhs)

  if operation in binary_arith_psy_to_arith_comparison_op:
    return arith.Cmpi.from_mnemonic(lhs, rhs, binary_arith_psy_to_arith_comparison_op[operation])

  return None

def translate_literal(op: psy_ir.Literal, program_state : ProgramState) -> Operation:
    value = op.attributes["value"]

    if isinstance(value, IntegerAttr):
        return arith.Constant.create(attributes={"value": value},
                                         result_types=[value.typ])

    if isinstance(value, psy_ir.FloatAttr):
        return arith.Constant.create(attributes={"value": value},
                                         result_types=[value.type])

    if isinstance(value, StringAttr):
        return generate_string_literal(value.data, program_state)

    raise Exception(f"Could not translate `{op}' as a literal")

def generate_string_literal(string, program_state : ProgramState) -> Operation:
    string_literal=string.replace("\"", "")
    typ=fir.CharType([fir.IntAttr.from_int(1), fir.IntAttr.from_int(len(string_literal))])
    string_lit_op=fir.StringLit.create(attributes={"size": IntegerAttr.from_int_and_width(len(string_literal), 64), "value": StringAttr(string_literal)}, result_types=[typ])
    has_val_op=fir.HasValue.create(operands=[string_lit_op.results[0]])
    str_uuid=uuid.uuid4().hex.upper()
    glob=fir.Global.create(attributes={"linkName": StringAttr("linkonce"), "sym_name": StringAttr("_QQcl."+str_uuid), "symref": SymbolRefAttr.from_str("_QQcl."+str_uuid), "type": typ},
      regions=[Region.from_operation_list([string_lit_op, has_val_op])])
    program_state.appendToGlobal(glob)
    ref_type=fir.ReferenceType([typ])
    addr_lookup=fir.AddressOf.create(attributes={"symbol": SymbolRefAttr.from_str("_QQcl."+str_uuid)}, result_types=[ref_type])
    return addr_lookup
