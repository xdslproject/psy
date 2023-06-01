from __future__ import annotations
from xdsl.dialects.builtin import (StringAttr, ModuleOp, IntegerAttr, IntegerType, ArrayAttr, i32, i64, f32, f64, f16, IndexType, DictionaryAttr, IntAttr,
      Float16Type, Float32Type, Float64Type, FloatAttr, UnitAttr, DenseIntOrFPElementsAttr, SymbolRefAttr, AnyFloat, TupleType, UnrealizedConversionCastOp)
from xdsl.dialects import func, arith, cf, mpi #, gpu
from xdsl.dialects.experimental import math, fir
from xdsl.ir import Operation, Attribute, ParametrizedAttribute, Region, Block, SSAValue, MLContext, BlockArgument
from psy.dialects import psy_ir, psy_stencil #, hpc_gpu
from xdsl.dialects.experimental import stencil as experimental_stencil
from xdsl.dialects import stencil
from xdsl.dialects.llvm import LLVMPointerType
from xdsl.passes import ModulePass
from util.list_ops import flatten
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import copy

binary_arith_op_matching={"ADD": [arith.Addi, arith.Addf], "SUB":[arith.Subi, arith.Subf], "MUL": [arith.Muli, arith.Mulf], "DIV": [arith.DivSI, arith.Divf], "REM": [arith.RemSI, None],
"MIN" : [arith.MinSI, arith.Minf], "MAX" : [arith.MaxSI, arith.Maxf], "POW" : [math.IPowIOp, None, None], "SIGN": [None, math.CopySignOp]}

binary_arith_psy_to_arith_comparison_op={"EQ": "eq", "NE": "ne", "GT": "sgt", "LT": "slt", "GE": "sge", "LE": "sle"}

str_to_mpi_operation={"max": mpi.MpiOp.MPI_MAX, "min": mpi.MpiOp.MPI_MIN, "sum": mpi.MpiOp.MPI_SUM, "prod": mpi.MpiOp.MPI_PROD,
  "land": mpi.MpiOp.MPI_LAND, "band": mpi.MpiOp.MPI_BAND, "lor": mpi.MpiOp.MPI_LOR, "bor": mpi.MpiOp.MPI_BOR,
  "lxor": mpi.MpiOp.MPI_LXOR, "bxor": mpi.MpiOp.MPI_BXOR, "minloc": mpi.MpiOp.MPI_MINLOC,
  "maxloc": mpi.MpiOp.MPI_MAXLOC, "replace": mpi.MpiOp.MPI_REPLACE, "no_op": mpi.MpiOp.MPI_NO_OP}

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

@dataclass
class LowerPsyIR(ModulePass):

  name = 'lower-psy-ir'

  def apply(self, ctx: MLContext, input_module: ModuleOp):
    res_module = translate_program(input_module)
    res_module.regions[0].move_blocks(input_module.regions[0])

    # Create program entry point
    op=get_program_entry_point(input_module)
    if ("sym_visibility" in op.attributes and op.attributes["sym_visibility"].data == "private"):
      # Remove private from function visibility
      del op.attributes["sym_visibility"]
    apply_environment_to_module(input_module)

def get_program_entry_point(module: ModuleOp):
  for op in module.regions[0].block.ops:
    if isinstance(op, func.FuncOp):
      if op.sym_name.data=="_QQmain": return op
  assert False, "No program entry point, need a procedure called main"

def apply_environment_to_module(module: ModuleOp):
  if not check_has_environment(module):
    # No environment, therefore we need to insert this
    array_type=fir.SequenceType(fir.ReferenceType([IntegerType(8)]), type2=fir.ReferenceType([IntegerType(8)]))
    tuple_type=TupleType([i32, fir.ReferenceType([array_type])])
    typ=fir.ReferenceType([tuple_type])
    zero_bits=fir.ZeroBits.create(result_types=[typ])
    has_value=fir.HasValue.create(operands=[zero_bits.results[0]])

    region_args=[zero_bits, has_value]

    glob=fir.Global.create(attributes={"sym_name": StringAttr("_QQEnvironmentDefaults"),
                                      "symref": SymbolRefAttr("_QQEnvironmentDefaults"),
                                      "type": typ, "constant": UnitAttr()},
            regions=[Region([Block(region_args)])])

    module.regions[0].blocks[0].add_op(glob)

def check_has_environment(module: ModuleOp):
  for op in module.ops:
    if isinstance(op, fir.Global) and op.sym_name.data == "_QQEnvironmentDefaults": return True
  return False

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
          fn_ops.regions[0].blocks[0].insert_op_before(mpi.Init.build(), fn_ops.regions[0].blocks[0].first_op)
          # Need to do this to pop finalize before the return at the end of the block, hence insert
          # before the last operation
          fn_ops.regions[0].blocks[0].insert_op_before(mpi.Finalize.build(), fn_ops.regions[0].blocks[0].last_op)
        block.add_op(fn_ops)
        globals_list.extend(program_state.getGlobals())

    if len(globals_list) > 0:
      block.add_ops(globals_list)

    if gpu_module is not None:
      block.add_ops([gpu_module])

    body.add_block(block)
    return ModuleOp(body)

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
        type_name = op.regions[0].blocks[0].ops.first
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
      if isinstance(translated_type, fir.SequenceType) and translated_type.hasDeferredShape():
        # If this is an array with a deferred type then wrap it in a box as this will
        # contain parameter size information that is passed by the caller
        arg_type=fir.BoxType([translated_type])
      else:
        # Otherwise it's just a memory reference, nice and easy
        arg_type=fir.ReferenceType([translated_type])
      arg_names.append(arg.var_name.data)
      arg_types.append(arg_type)

    block = Block(arg_types=arg_types)

    for index, arg_name in enumerate(arg_names):
      c[arg_name] = block.args[index]

    # use the nested scope when translate the body of the function
    #block.add_ops(
    #    flatten([
    #        translate_def_or_stmt(c, op)
    #        for op in routine_def.local_var_declarations.blocks[0].ops
    #   ]))

    if len(routine_def.imports.blocks) > 0:
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
      return_block=Block(arg_types=[])
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
        arg_attrs["fir.bindc_name"]=StringAttr(arg_name)
      function_fir.attributes["arg_attrs"]=DictionaryAttr(arg_attrs)
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

def define_derived_var(ctx: SSAValueCtx,
                      var_def: psy_ir.VarDef, program_state : ProgramState) -> List[Operation]:
    var_name = var_def.var.var_name
    assert isinstance(var_name, StringAttr)
    assert isinstance(var_def.var.type, psy_ir.DerivedType)
    type_name=var_def.var.type.type.data
    if type_name == "mpi_request":
      constant=arith.Constant.create(attributes={"value": IntegerAttr.from_int_and_width(1, 32)}, result_types=[i32])
      mpi_request_alloc=mpi.AllocateTypeOp.get(mpi.RequestType, constant.results[0], var_name)
      ctx[var_name.data] = mpi_request_alloc.results[0]
      return [constant, mpi_request_alloc]
    else:
      raise Exception(f"Could not translate derived type `{type_name}' as this is unknown")

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
    if isinstance(var_def.var.type.element_type, psy_ir.DerivedType):
      type_name=var_def.var.type.element_type.type.data
      if type_name == "mpi_request":
        sizes=get_array_sizes(var_def.var.type)
        constant=arith.Constant.create(attributes={"value": IntegerAttr.from_int_and_width(sizes[0], 32)}, result_types=[i32])
        mpi_request_alloc=mpi.AllocateTypeOp.get(mpi.RequestType, constant.results[0], var_name)
        ctx[var_name.data] = mpi_request_alloc.results[0]
        return [constant, mpi_request_alloc]
      else:
        raise Exception(f"Unknown how to handle user derived type `{type_name}' in array definition")
    else:
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
        shape=fir.Shape.create(operands=shape_ops, result_types=[fir.ShapeType([IntAttr(num_deferred)])])
        embox=fir.Embox.build(operands=[zero_bits.results[0], shape.results[0], [], [], []], regions=[[]], result_types=[type])
        has_val=fir.HasValue.create(operands=[embox.results[0]])
        region_args=[zero_bits, zero_val, shape, embox, has_val]

        glob=fir.Global.create(attributes={"linkName": StringAttr("internal"), "sym_name": StringAttr("_QFE"+var_name.data), "symref": SymbolRefAttr("_QFE"+var_name.data), "type": type},
            regions=[Region([Block(region_args)])])
        addr_lookup=fir.AddressOf.create(attributes={"symbol": SymbolRefAttr("_QFE"+var_name.data)}, result_types=[fir.ReferenceType([type])])
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
    elif isinstance(var_def.var.type, psy_ir.DerivedType):
      return define_derived_var(ctx, var_def, program_state)

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

      arrayType=fir.SequenceType(try_translate_type(op.element_type), array_size)
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
    if isinstance(op, psy_stencil.PsyStencil_Stencil):
      return translate_psy_stencil_stencil(ctx, op, program_state)
    if isinstance(op, psy_stencil.PsyStencil_Result):
      return translate_psy_stencil_result(ctx, op, program_state)

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

def translate_psy_stencil_access(ctx: SSAValueCtx, stencil_access: Operation, program_state : ProgramState) -> List[Operation]:
  assert isinstance(stencil_access.var.type, psy_ir.ArrayType)
  el_type=try_translate_type(stencil_access.var.type.element_type)
  assert el_type is not None
  offsets=([value.data for value in stencil_access.stencil_ops.data])
  access_op=experimental_stencil.AccessOp.get(ctx[stencil_access.var.var_name.data], offsets)
  return [access_op], access_op.results[0]

def get_array_sizes(array_type):
  return get_size_from_lb_ub_array(array_type.shape.data)

def get_size_from_lb_ub_array(shape_array):
  shape_size=len(shape_array)
  assert shape_size % 2 == 0
  sizes=[]
  for i in range(0, shape_size, 2):
    sizes.append((shape_array[i+1].value.data-shape_array[i].value.data)+1)
  return sizes

def get_stencil_data_range(bounds_array, relative_offset, further_offset=0):
  range_vals=[]
  for rel_offset, d in zip(relative_offset, bounds_array):
    range_vals.append(d.data + rel_offset.data + further_offset)
  return experimental_stencil.IndexAttr.get(*range_vals)

def find_ops_with_type(ops, op_type):
  located_ops=[]
  for op in ops:
    if isinstance(op, op_type): located_ops.append(op)
  return located_ops

def locate_psy_stencil_deferred_array_info(var_name, ops):
  deferred_array_info=find_ops_with_type(ops, psy_stencil.PsyStencil_DeferredArrayInfo)
  for info in deferred_array_info:
    if info.var.var_name.data == var_name: return info
  return None

def interogate_stencil_field_inference_sizes(var_name, ops):
  deferred_array_info=locate_psy_stencil_deferred_array_info(var_name, ops)
  assert deferred_array_info is not None
  return get_size_from_lb_ub_array(deferred_array_info.shape.data)

def rebuild_deferred_fir_array_with_bounds(in_type, array_sizes):
  array_type=get_nested_type(in_type, fir.SequenceType)
  new_shape=[IntegerAttr(size, 64) for size in array_sizes]
  new_array=fir.SequenceType(array_type.type, new_shape)

  to_add=new_array
  parent_type=fir.SequenceType
  while True:
    parent=get_nested_parent_type(in_type, parent_type)
    if parent is None: break
    parent_type=type(parent)
    to_add=parent_type([to_add])

  return to_add

def translate_psy_stencil_result(ctx: SSAValueCtx, stencil_result: Operation, program_state : ProgramState) -> List[Operation]:

  ops: List[Operation] = []
  for op in stencil_result.stencil_accesses.blocks[0].ops:
    stmt_ops, ssa = translate_expr(ctx, op, program_state)
    ops += stmt_ops

  assert isinstance(stencil_result.out_field.type, psy_ir.ArrayType)
  el_type=try_translate_type(stencil_result.out_field.type.element_type)
  assert el_type is not None
  rt=experimental_stencil.ResultType(el_type)

  if el_type != ops[-1].results[0].typ:
    data_conv_op=perform_data_conversion_if_needed(ops[-1].results[0], el_type)
    #return_op=experimental_stencil.ReturnOp.get([data_conv_op.results[0]])
    ops+=[data_conv_op]
    return ops, data_conv_op.results[0]
  else:
    #return_op=experimental_stencil.ReturnOp.get([ops[-1].results[0]])
    #ops+=[return_op]
    return ops, ops[-1].results[0]



  num_deferred=stencil_result.out_field.type.get_num_deferred_dim()
  # Ensure either no dimensions are deferred or they all are
  assert num_deferred == 0 or num_deferred == stencil_result.out_field.type.get_num_dims()
  if num_deferred == 0:
    array_sizes=get_array_sizes(stencil_result.out_field.type)
  else:
    array_sizes=interogate_stencil_field_inference_sizes(stencil_result.out_field.var_name.data, stencil_result.parent.ops)

  assert len(array_sizes) == len(stencil_op.from_bounds.data)
  assert len(array_sizes) == len(stencil_op.to_bounds.data)

  lb_ints=[]
  for el in stencil_op.from_bounds.data:
    # We minus one as going from Fortran indexing to C style
    lb_ints.append(el.data-1)

  ub_ints=[]
  for el in stencil_op.to_bounds.data:
    ub_ints.append(el.data)

  lb=experimental_stencil.IndexAttr.get(*lb_ints)
  ub=experimental_stencil.IndexAttr.get(*ub_ints)
  stencil_temptype=experimental_stencil.TempType([-1] * len(array_sizes), el_type)
  apply_op=experimental_stencil.ApplyOp.get(block_ops, block, [stencil_temptype], lb, ub)

  return [apply_op]

def translate_psy_stencil_stencil(ctx: SSAValueCtx, stencil_stmt: Operation, program_state : ProgramState) -> List[Operation]:
  ops: List[Operation] = []
  new_ctx=SSAValueCtx()
  input_cast_ops={}

  # Build the lower and upper bounds up - note how we are picking off the first stencil
  # result here, we assume currently only one per stencil - need to enhance when we
  # support multiple ones
  #stencil_results_located=find_ops_with_type(stencil_stmt.body.blocks[0].ops, psy_stencil.PsyStencil_Result)
  #assert len(stencil_results_located) == 1
  #stencil_result_op=stencil_results_located[0]
  # -1 further offset for lb as Fortran array is 1 indexed, whereas C is 0 - so need to apply that here too
  #lb=get_stencil_data_range(stencil_result_op.from_bounds.data, stencil_result_op.min_relative_offset.data, -1)
  # No need for a further offset as loop in C is less than, rather than less than equals in Fortran so all good!
  #ub=get_stencil_data_range(stencil_result_op.to_bounds.data, stencil_result_op.max_relative_offset.data)

  for field in stencil_stmt.input_fields.data:
    if isinstance(field.type, psy_ir.ArrayType):
      num_deferred=field.type.get_num_deferred_dim()
      # Ensure either no dimensions are deferred or they all are
      assert num_deferred == 0 or num_deferred == field.type.get_num_dims()
      if num_deferred == 0:
        array_sizes=get_array_sizes(field.type)
      else:
        array_sizes=interogate_stencil_field_inference_sizes(field.var_name.data, stencil_stmt.body.blocks[0].ops)

      # For now hack these in, as need to ensure memref cast that is generated is of correct size of
      # input array
      lb=experimental_stencil.IndexAttr.get(*([-1]*len(array_sizes)))
      ub=experimental_stencil.IndexAttr.get(*[v-1 for v in array_sizes])
      el_type=try_translate_type(field.type.element_type)

      field_bounds=[]
      for dim in array_sizes:
        field_bounds.append((-1, dim-1))

      if num_deferred > 0:
        # Use an unrealized conversion to pop in the array size information here
        in_type=ctx[field.var_name.data].typ
        explicit_size_type=rebuild_deferred_fir_array_with_bounds(in_type, array_sizes)
        unreconciled_conv_op=UnrealizedConversionCastOp.create(operands=[ctx[field.var_name.data]], result_types=[explicit_size_type])
        external_load_op=experimental_stencil.ExternalLoadOp.get(unreconciled_conv_op.results[0], experimental_stencil.FieldType(field_bounds, el_type))
        ops+=[unreconciled_conv_op, external_load_op]
      else:
        external_load_op=experimental_stencil.ExternalLoadOp.get(ctx[field.var_name.data], experimental_stencil.FieldType(field_bounds, el_type))
        ops+=[external_load_op]
      cast_op=stencil.CastOp.get(external_load_op.results[0], experimental_stencil.StencilBoundsAttr(field_bounds), external_load_op.results[0].typ)
      input_cast_ops[field.var_name.data]=cast_op
      load_op=experimental_stencil.LoadOp.get(cast_op.results[0], lb, ub)
      ops+=[cast_op, load_op]
      new_ctx[field.var_name.data]=load_op.results[0]
    elif isinstance(field.type, psy_ir.NamedType):
      # This is a scalar and simply set outside the stencil loop and used in here
      # therefore include the context of this but don't need to do anything else
      scalar_var=ctx[field.var_name.data]
      if isinstance(scalar_var.typ, fir.ReferenceType):
        load_op=fir.Load.create(operands=[scalar_var], result_types=[scalar_var.typ.type])
        ops+=[load_op]
        new_ctx[field.var_name.data]=load_op.results[0]
      else:
        new_ctx[field.var_name.data]=scalar_var

  output_field_cast_ops=[]

  for field in stencil_stmt.output_fields.data:
    assert isinstance(field.type, psy_ir.ArrayType)
    num_deferred=field.type.get_num_deferred_dim()
    # Ensure either no dimensions are deferred or they all are
    assert num_deferred == 0 or num_deferred == field.type.get_num_dims()
    if num_deferred == 0:
      array_sizes=get_array_sizes(field.type)
    else:
      array_sizes=interogate_stencil_field_inference_sizes(field.var_name.data, stencil_stmt.body.blocks[0].ops)

    lb=experimental_stencil.IndexAttr.get(*([0]*len(array_sizes)))
    ub=experimental_stencil.IndexAttr.get(*[v for v in array_sizes])
    el_type=try_translate_type(field.type.element_type)
    if num_deferred > 0:
      # Use an unrealized conversion to pop in the array size information here
      in_type=ctx[field.var_name.data].typ
      explicit_size_type=rebuild_deferred_fir_array_with_bounds(in_type, array_sizes)
      unreconciled_conv_op=UnrealizedConversionCastOp.create(operands=[ctx[field.var_name.data]], result_types=[explicit_size_type])
      external_load_op=experimental_stencil.ExternalLoadOp.get(unreconciled_conv_op.results[0], experimental_stencil.FieldType(field_bounds, el_type))
      ops+=[unreconciled_conv_op, external_load_op]
    else:
      external_load_op=experimental_stencil.ExternalLoadOp.get(ctx[field.var_name.data], experimental_stencil.FieldType(field_bounds, el_type))
      ops+=[external_load_op]
    output_field_cast_op=stencil.CastOp.get(external_load_op.results[0], experimental_stencil.StencilBoundsAttr(field_bounds), external_load_op.results[0].typ)
    output_field_cast_ops.append((output_field_cast_op, external_load_op))
    ops+=[output_field_cast_op]
    #new_ctx[field.var_name.data]=output_field_cast_op.results[0]

  block_fields={}
  for var in stencil_stmt.input_fields.data:
    # Use the new ctx as this will point to the temporary
    block_fields[var.var_name.data]=new_ctx[var.var_name.data].typ

  block_types=[]
  block_idx={}
  block_ops=[]
  for index, (key, value) in enumerate(block_fields.items()):
    if isinstance(value, fir.ReferenceType):
      block_types.append(value.type)
      load_op=fir.Load.create(operands=[new_ctx[key]], result_types=[value.type])
      ops.append(load_op)
      block_ops.append(load_op)
    else:
      block_types.append(value)
      block_ops.append(new_ctx[key])
    block_idx[key]=index

  c = SSAValueCtx(dictionary=dict())
  block = Block(arg_types=block_types)

  for key, index in block_idx.items():
    c[key]=block.args[index]

  stencil_contents=[]
  result_ssa_vals=[]
  for op in stencil_stmt.body.blocks[0].ops:
    if not isinstance(op, psy_stencil.PsyStencil_DeferredArrayInfo):
      # Ignore the deferred array info statements
      assert isinstance(op, psy_stencil.PsyStencil_Result)
      result_ops, ssa_result=translate_psy_stencil_result(c, op, program_state)
      stencil_contents+=result_ops
      result_ssa_vals.append(ssa_result)

  return_op=experimental_stencil.ReturnOp.get(result_ssa_vals)
  stencil_contents.append(return_op)

  block.add_ops(stencil_contents)

  # With the from bounds we minus one to go into zero indexing regime
  from_bounds=[el.data-1 for el in stencil_stmt.from_bounds]
  lb=experimental_stencil.IndexAttr.get(*from_bounds)
  ub=experimental_stencil.IndexAttr.get(*stencil_stmt.to_bounds)

  stencil_temptypes=[]
  for result_ssa_val in result_ssa_vals:
    stencil_temptypes.append(experimental_stencil.TempType(field_bounds, result_ssa_val.typ))
  apply_op=experimental_stencil.ApplyOp.get(block_ops, block, stencil_temptypes, lb, ub)

  ops.append(apply_op)

  for index, out_tuple in enumerate(output_field_cast_ops):
    out_var_name=stencil_stmt.output_fields.data[index].var_name.data
    lb_indexes=experimental_stencil.IndexAttr.get(*([0] * len(array_sizes)))
    ub_indexes=experimental_stencil.IndexAttr.get(*array_sizes)
    store_op=experimental_stencil.StoreOp.get(apply_op.results[index], out_tuple[0].results[0], lb_indexes, ub_indexes)
    external_store_op=experimental_stencil.ExternalStoreOp.create(operands=[out_tuple[1].results[0], ctx[out_var_name]])
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
  body = Region([Block(ops)])
  gpu_fn=gpu.GPUFuncOp.from_region("gpu_fn_"+str(program_state.getNumGPUFns()), [], [], body)
  end_op=gpu.ModuleEndOp.create()
  if gpu_module is None:
    gpu_module=gpu.GPUModuleOp.from_region(Region([Block([gpu_fn, end_op])]), "gpu_functions")
  else:
    pass # Need to add in ability to append GPU function here

  # Hacking in the "@" character on the GPU function name here
  launch_fn=gpu.LaunchFuncOp.create(attributes={"kernel":SymbolRefAttr("gpu_fns.@gpu_fn_"+str(program_state.getNumGPUFns()))})
  program_state.incrementNumGPUFns()
  return [launch_fn]

def translate_return(ctx: SSAValueCtx, return_stmt: psy_ir.Return, program_state : ProgramState) -> List[Operation]:
  return [cf.Branch.get(program_state.getReturnBlock())]

def translate_loop(ctx: SSAValueCtx,
                  for_stmt: psy_ir.Loop, program_state : ProgramState) -> List[Operation]:
    start, start_name = translate_expr(ctx, for_stmt.start.blocks[0].ops.first, program_state)
    conv_start=fir.Convert.create(operands=[start_name], result_types=[IndexType()])
    stop, stop_name = translate_expr(ctx, for_stmt.stop.blocks[0].ops.first, program_state)
    conv_stop=fir.Convert.create(operands=[stop_name], result_types=[IndexType()])
    step, step_name = translate_expr(ctx, for_stmt.step.blocks[0].ops.first, program_state)
    conv_step=fir.Convert.create(operands=[step_name], result_types=[IndexType()])

    ops: List[Operation] = []
    for op in for_stmt.body.blocks[0].ops:
        stmt_ops = translate_stmt(ctx, op, program_state)
        ops += stmt_ops
    #body = Region.from_operation_list(ops)

    iterator = ctx[for_stmt.variable.var_name.data]

    block = Block(arg_types=[IndexType(), i32])
    store=fir.Store.create(operands=[block.args[1], iterator])

    add_iteration_count=arith.Addi(block.args[0], conv_step)
    load_iterator_var=fir.Load.create(operands=[iterator], result_types=[try_translate_type(for_stmt.variable.type)])
    convert_step_for_it=fir.Convert.create(operands=[conv_step.results[0]], result_types=[i32])
    add_to_iterator=arith.Addi(load_iterator_var.results[0], convert_step_for_it.results[0])
    block_result=fir.Result.create(operands=[add_iteration_count.results[0], add_to_iterator.results[0]])

    block.add_ops([store]+ops+[add_iteration_count, load_iterator_var, convert_step_for_it, add_to_iterator, block_result])
    body=Region()
    body.add_block(block)

    do_loop=fir.DoLoop.create(attributes={"finalValue": UnitAttr()},
    operands=[conv_start.results[0], conv_stop.results[0], conv_step.results[0], start_name], result_types=[IndexType(), i32], regions=[body])
    final_iterator_store=fir.Store.create(operands=[do_loop.results[1], iterator])
    return start+[conv_start]+stop+[conv_stop]+step+[conv_step, do_loop, final_iterator_store]


def translate_if(ctx: SSAValueCtx, if_stmt: psy_ir.If, program_state : ProgramState) -> List[Operation]:
    cond, cond_name = translate_expr(ctx, if_stmt.cond.blocks[0].ops.first, program_state)

    ops: List[Operation] = []
    for op in if_stmt.then.blocks[0].ops:
        stmt_ops = translate_stmt(ctx, op, program_state)
        ops += stmt_ops
    ops.append(fir.Result.create())
    then = Region([Block(ops)])

    ops: List[Operation] = []
    for op in if_stmt.orelse.blocks[0].ops:
        stmt_ops = translate_stmt(ctx, op, program_state)
        ops += stmt_ops
    ops.append(fir.Result.create())
    orelse = Region([Block(ops)])

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
    elif intrinsic_name == "deallocate":
      return translate_deallocate_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    elif intrinsic_name.lower() == "print":
      return translate_print_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    elif intrinsic_name.lower() == "mpi_commrank":
      return translate_mpi_commrank_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    elif intrinsic_name.lower() == "mpi_commsize":
      return translate_mpi_commsize_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    elif intrinsic_name.lower() == "mpi_send":
      return translate_mpi_send_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    elif intrinsic_name.lower() == "mpi_isend":
      return translate_mpi_send_intrinsic_call_expr(ctx, call_expr, program_state, is_expr, False)
    elif intrinsic_name.lower() == "mpi_recv":
      return translate_mpi_recv_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    elif intrinsic_name.lower() == "mpi_irecv":
      return translate_mpi_recv_intrinsic_call_expr(ctx, call_expr, program_state, is_expr, False)
    elif intrinsic_name.lower() == "mpi_wait":
      return translate_mpi_wait_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    elif intrinsic_name.lower() == "mpi_waitall":
      return translate_mpi_waitall_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    elif intrinsic_name.lower() == "mpi_reduce":
      return translate_mpi_reduce_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    elif intrinsic_name.lower() == "mpi_allreduce":
      return translate_mpi_allreduce_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    elif intrinsic_name.lower() == "mpi_bcast":
      return translate_mpi_bcast_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    elif intrinsic_name.lower() == "mpi_init" or intrinsic_name.lower() == "mpi_finalize":
      program_state.setRequiresMPI(True)
      return []
    elif intrinsic_name.lower() == "timer_init":
      return translate_timer_init_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    elif intrinsic_name.lower() == "timer_start":
      return translate_timer_start_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    elif intrinsic_name.lower() == "timer_stop":
      return translate_timer_stop_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    elif intrinsic_name.lower() == "timer_report":
      return translate_timer_report_intrinsic_call_expr(ctx, call_expr, program_state, is_expr)
    else:
      raise Exception(f"Could not translate intrinsic`{intrinsic_name}' as unknown")

def translate_mpi_bcast_intrinsic_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
    program_state.setRequiresMPI(True)
    assert len(call_expr.args.blocks[0].ops) == 3

    ops_list=flatten_ops_to_list(call_expr.args.blocks[0].ops)

    ptr_type, buffer_op, buffer_arg = translate_mpi_buffer(ctx, call_expr.args.blocks[0].ops.first, program_state)
    convert_buffer=fir.Convert.create(operands=[buffer_arg],
                    result_types=[fir.LLVMPointerType([ptr_type])])
    count_op, count_arg = translate_expr(ctx, ops_list, program_state)
    get_mpi_dtype_op=mpi.GetDtypeOp.get(ptr_type)
    root_op, root_arg = translate_expr(ctx, ops_list, program_state)

    result_ops=count_op + root_op + [convert_buffer, get_mpi_dtype_op]

    bcast_op=mpi.Bcast.get(convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], root_arg)
    result_ops.append(bcast_op)

    return result_ops


def translate_mpi_reduce_intrinsic_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
    program_state.setRequiresMPI(True)
    assert len(call_expr.args.blocks[0].ops) == 5

    ops_list=flatten_ops_to_list(call_expr.args.blocks[0].ops)
    assert isinstance(ops_list[3], psy_ir.Literal)

    send_ptr_type, send_buffer_op, send_buffer_arg = translate_mpi_buffer(ctx, ops_list[0], program_state)
    recv_ptr_type, recv_buffer_op, recv_buffer_arg = translate_mpi_buffer(ctx, ops_list[1], program_state)

    send_convert_buffer=fir.Convert.create(operands=[send_buffer_arg],
                    result_types=[fir.LLVMPointerType([send_ptr_type])])
    recv_convert_buffer=fir.Convert.create(operands=[recv_buffer_arg],
                    result_types=[fir.LLVMPointerType([recv_ptr_type])])
    count_op, count_arg = translate_expr(ctx, ops_list[2], program_state)
    get_mpi_dtype_op=mpi.GetDtypeOp.get(send_ptr_type) # Do this on the send buffer type

    mpi_op=str_to_mpi_operation[ops_list[3].value.data]
    root_op, root_arg = translate_expr(ctx, ops_list[4], program_state)

    result_ops=count_op + root_op + [send_convert_buffer, recv_convert_buffer, get_mpi_dtype_op]

    mpi_reduce_op=mpi.Reduce.get(send_convert_buffer.results[0], recv_convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], mpi_op, root_arg)
    result_ops.append(mpi_reduce_op)

    return result_ops

def translate_mpi_allreduce_intrinsic_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
    program_state.setRequiresMPI(True)
    assert len(call_expr.args.blocks[0].ops) == 4 or len(call_expr.args.blocks[0].ops) == 3

    has_send_buffer=len(call_expr.args.blocks[0].ops) == 4

    if has_send_buffer:
      send_buffer_idx=0
      recv_buffer_idx=1
      count_idx=2
      mpi_op_idx=3
    else:
      recv_buffer_idx=0
      count_idx=1
      mpi_op_idx=2

    ops_list=flatten_ops_to_list(call_expr.args.blocks[0].ops)

    assert isinstance(ops_list[mpi_op_idx], psy_ir.Literal)

    if has_send_buffer:
      send_ptr_type, send_buffer_op, send_buffer_arg = translate_mpi_buffer(ctx, ops_list[send_buffer_idx], program_state)
      send_convert_buffer=fir.Convert.create(operands=[send_buffer_arg],
                    result_types=[fir.LLVMPointerType([send_ptr_type])])

    recv_ptr_type, recv_buffer_op, recv_buffer_arg = translate_mpi_buffer(ctx, ops_list[recv_buffer_idx], program_state)
    recv_convert_buffer=fir.Convert.create(operands=[recv_buffer_arg],
                    result_types=[fir.LLVMPointerType([recv_ptr_type])])
    count_op, count_arg = translate_expr(ctx, ops_list[count_idx], program_state)
    get_mpi_dtype_op=mpi.GetDtypeOp.get(recv_ptr_type) # Do this on the recv buffer type

    mpi_op=str_to_mpi_operation[ops_list[mpi_op_idx].value.data]

    if has_send_buffer:
      result_ops=count_op + [send_convert_buffer, recv_convert_buffer, get_mpi_dtype_op]
      mpi_reduce_op=mpi.Allreduce.get(send_convert_buffer.results[0], recv_convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], mpi_op)
    else:
      result_ops=count_op + [recv_convert_buffer, get_mpi_dtype_op]
      mpi_reduce_op=mpi.Allreduce.get(None, recv_convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], mpi_op)

    result_ops.append(mpi_reduce_op)

    return result_ops

def translate_mpi_buffer(ctx: SSAValueCtx, ops: psy_ir.ExprName, program_state : ProgramState):
    assert isinstance(ops, psy_ir.ExprName)
    ptr_type=try_translate_type(ops.var.type)
    # Pointer type needs to be base type which might be wrapped in an array
    if isinstance(ptr_type, fir.SequenceType): ptr_type=ptr_type.type

    buffer_op, buffer_arg = translate_expr(ctx, ops, program_state)
    if not isinstance(buffer_arg.typ, fir.ReferenceType):
      if isinstance(buffer_op, list) and len(buffer_op) == 1 and isinstance(buffer_op[0], fir.Load):
        # We do this as translate expression assumes we want to use the value rather than the reference,
        # so it loads the value from the fir.referencetype, hence we go in and grab the reference type.
        # This is needed if a scalar is passed to the call as the buffer argument
        buffer_arg=buffer_op[0].memref
      else:
        raise Exception(f"Unable to process MPI argument`{buffer_arg}' as it is not a reference and can not be translated")
    return ptr_type, buffer_op, buffer_arg


def translate_mpi_wait_intrinsic_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
    program_state.setRequiresMPI(True)
    assert len(call_expr.args.blocks[0].ops) == 1

    to_return=[]
    request_op, request_arg = translate_expr(ctx, call_expr.args.blocks[0].ops.first, program_state)
    if request_op is not None: to_return+=request_op

    if isinstance(request_arg.typ, mpi.VectorType):
      element_index=arith.Constant.create(attributes={"value": IntegerAttr.from_int_and_width(0, 32)},
                                         result_types=[i32])
      load_op=mpi.VectorGetOp.get(request_arg, element_index.results[0])
      wait_op=mpi.Wait.get(load_op.results[0])
      to_return+=[element_index, load_op, wait_op]
    else:
      wait_op=mpi.Wait.get(request_arg)
      to_return.append(wait_op)

    return to_return

def flatten_ops_to_list(ops):
    ops_list=[]
    for op in ops:
      ops_list.append(op)
    return ops_list

def translate_mpi_waitall_intrinsic_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
    program_state.setRequiresMPI(True)
    assert len(call_expr.args.blocks[0].ops) == 2

    ops_list=flatten_ops_to_list(call_expr.args.blocks[0].ops)

    request_ops, request_args = translate_expr(ctx, ops_list[0], program_state)
    count_op, count_arg = translate_expr(ctx, ops_list[1], program_state)
    wait_op=mpi.Waitall.get(request_args, count_arg)
    return count_op + [wait_op]


def translate_mpi_send_intrinsic_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False, blocking=True) -> List[Operation]:
    program_state.setRequiresMPI(True)
    if blocking:
      assert len(call_expr.args.blocks[0].ops) == 4
    else:
      assert len(call_expr.args.blocks[0].ops) == 5

    ops_list=flatten_ops_to_list(call_expr.args.blocks[0].ops)

    assert isinstance(ops_list[0], psy_ir.ExprName)

    ptr_type, buffer_op, buffer_arg = translate_mpi_buffer(ctx, ops_list[0], program_state)

    convert_buffer=fir.Convert.create(operands=[buffer_arg],
                    result_types=[fir.LLVMPointerType([ptr_type])])
    count_op, count_arg = translate_expr(ctx, ops_list[1], program_state)
    target_op, target_arg = translate_expr(ctx, ops_list[2], program_state)
    tag_op, tag_arg = translate_expr(ctx, ops_list[3], program_state)
    get_mpi_dtype_op=mpi.GetDtypeOp.get(ptr_type)

    result_ops=count_op + target_op + tag_op + [convert_buffer, get_mpi_dtype_op]

    if blocking:
      mpi_send_op=mpi.Send.get(convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], target_arg, tag_arg)
      result_ops.append(mpi_send_op)
    else:
      request_op, request_arg = translate_expr(ctx, ops_list[4], program_state)
      if request_op is not None: result_ops+=request_op

      if isinstance(request_arg.typ, mpi.VectorType):
        element_index=arith.Constant.create(attributes={"value": IntegerAttr.from_int_and_width(0, 32)},
                                         result_types=[i32])
        load_op=mpi.VectorGetOp.get(request_arg, element_index.results[0])
        mpi_send_op=mpi.Isend.get(convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], target_arg, tag_arg, load_op.results[0])
        result_ops+=[element_index, load_op, mpi_send_op]
      else:
        mpi_send_op=mpi.Isend.get(convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], target_arg, tag_arg, request_arg)
        result_ops.append(mpi_send_op)

    return result_ops

def translate_mpi_recv_intrinsic_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False, blocking=True) -> List[Operation]:
    program_state.setRequiresMPI(True)
    if blocking:
      assert len(call_expr.args.blocks[0].ops) == 4
    else:
      assert len(call_expr.args.blocks[0].ops) == 5

    ops_list=flatten_ops_to_list(call_expr.args.blocks[0].ops)

    assert isinstance(ops_list[0], psy_ir.ExprName)

    ptr_type, buffer_op, buffer_arg = translate_mpi_buffer(ctx, ops_list[0], program_state)

    convert_buffer=fir.Convert.create(operands=[buffer_arg],
                    result_types=[fir.LLVMPointerType([ptr_type])])
    count_op, count_arg = translate_expr(ctx, ops_list[1], program_state)
    source_op, source_arg = translate_expr(ctx, ops_list[2], program_state)
    tag_op, tag_arg = translate_expr(ctx, ops_list[3], program_state)
    get_mpi_dtype_op=mpi.GetDtypeOp.get(ptr_type)

    result_ops=count_op + source_op + tag_op + [convert_buffer, get_mpi_dtype_op]

    if blocking:
      mpi_recv_op=mpi.Recv.get(convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], source_arg, tag_arg)
      result_ops.append(mpi_recv_op)
    else:
      request_op, request_arg = translate_expr(ctx, ops_list[4], program_state)
      if request_op is not None: result_ops+=request_op

      if isinstance(request_arg.typ, mpi.VectorType):
        element_index=arith.Constant.create(attributes={"value": IntegerAttr.from_int_and_width(0, 32)},
                                         result_types=[i32])
        load_op=mpi.VectorGetOp.get(request_arg, element_index.results[0])
        mpi_recv_op=mpi.Irecv.get(convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], source_arg, tag_arg, load_op.results[0])
        result_ops+=[element_index, load_op, mpi_recv_op]
      else:
        mpi_recv_op=mpi.Irecv.get(convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], source_arg, tag_arg, request_arg)
        result_ops.append(mpi_recv_op)

    return result_ops

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

def translate_timer_report_intrinsic_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
    insertExternalFunctionToGlobalState(program_state, "_QMdl_timerPtimer_report", [], None)

    init_call=fir.Call.create(attributes={"callee": SymbolRefAttr("_QMdl_timerPtimer_report")}, operands=[], result_types=[])

    return [init_call]

def translate_timer_init_intrinsic_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
    insertExternalFunctionToGlobalState(program_state, "_QMdl_timerPtimer_init", [], None)

    init_call=fir.Call.create(attributes={"callee": SymbolRefAttr("_QMdl_timerPtimer_init")}, operands=[], result_types=[])

    return [init_call]

def translate_timer_start_intrinsic_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
    assert len(call_expr.args.blocks[0].ops) == 2

    ops_list=flatten_ops_to_list(call_expr.args.blocks[0].ops)

    op_ctrl, arg_ctrl = translate_expr(ctx, ops_list[0], program_state)
    op_desc, arg_desc = translate_expr(ctx, ops_list[1], program_state)

    assert isinstance(arg_ctrl.owner, fir.Load)
    assert arg_ctrl.owner.memref.typ == fir.ReferenceType([i32])

    deferred_char_type=fir.ReferenceType([fir.CharacterType([fir.IntAttr(1), fir.DeferredAttr()])])
    convert_op=fir.Convert.create(operands=[arg_desc], result_types=[deferred_char_type])


    embox_to_found=arith.Constant.create(attributes={"value": IntegerAttr.from_index_int_value(arg_desc.typ.type.to_index.data)}, result_types=[IndexType()])

    embox_op=emboxchar_op=fir.Emboxchar.create(operands=[convert_op.results[0], embox_to_found.results[0]], result_types=[fir.BoxCharType([fir.IntAttr(1)])])

    absent_op=fir.Absent.create(operands=[], result_types=[fir.ReferenceType([i64])])

    start_call=fir.Call.create(attributes={"callee": SymbolRefAttr("_QMdl_timerPtimer_start")},
      operands=[arg_ctrl.owner.memref, embox_op.results[0], absent_op.results[0]], result_types=[])

    insertExternalFunctionToGlobalState(program_state, "_QMdl_timerPtimer_start", [start_call.operands[0].typ,
      start_call.operands[1].typ, start_call.operands[2].typ], None)

    return op_ctrl+op_desc+[convert_op, embox_to_found, embox_op, absent_op, start_call]

def translate_timer_stop_intrinsic_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
    assert len(call_expr.args.blocks[0].ops) == 1
    op_ctrl, arg_ctrl = translate_expr(ctx, call_expr.args.blocks[0].ops.first, program_state)

    assert isinstance(arg_ctrl.owner, fir.Load)
    assert arg_ctrl.owner.memref.typ == fir.ReferenceType([i32])

    stop_call=fir.Call.create(attributes={"callee": SymbolRefAttr("_QMdl_timerPtimer_stop")},
      operands=[arg_ctrl.owner.memref], result_types=[])

    insertExternalFunctionToGlobalState(program_state, "_QMdl_timerPtimer_stop", [stop_call.operands[0].typ], None)

    return op_ctrl+[stop_call]

def translate_print_intrinsic_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
    arg_operands=[]

    # Start the IO session
    filename_str_op=generate_string_literal("./dummy.F90", program_state)
    arg1=arith.Constant.create(attributes={"value": IntegerAttr.from_int_and_width(-1, 32)}, result_types=[i32])
    arg2=fir.Convert.create(operands=[filename_str_op.results[0]], result_types=[fir.ReferenceType([IntegerType(8)])])
    arg3=arith.Constant.create(attributes={"value": IntegerAttr.from_int_and_width(3, 32)}, result_types=[i32])

    call1=fir.Call.create(attributes={"callee": SymbolRefAttr("_FortranAioBeginExternalListOutput")}, operands=[arg1.results[0],
      arg2.results[0], arg3.results[0]], result_types=[fir.ReferenceType([IntegerType(8)])])
    arg_operands.extend([filename_str_op, arg1, arg2, arg3, call1])

    insertExternalFunctionToGlobalState(program_state, "_FortranAioBeginExternalListOutput", [i32, fir.ReferenceType([IntegerType(8)]),
      i32], fir.ReferenceType([IntegerType(8)]))

    # Ignore first argument as it will be a star
    for index, argument in enumerate(call_expr.args.blocks[0].ops):
      if index == 0: continue
      # For each argument need to issue a different print
      op, arg = translate_expr(ctx, argument, program_state)
      # Now do the actual print
      if isinstance(arg.typ, IntegerType) or isinstance(arg.typ, AnyFloat):
        arg_operands.extend(generatePrintForIntegerOrFloat(program_state, op[0], arg, call1.results[0]))


      if isinstance(arg.typ, fir.ReferenceType):
        if isinstance(arg.typ.type, fir.IntegerType) or isinstance(arg.typ.type, AnyFloat):
          load_op=fir.Load.create(operands=[arg], result_types=[arg.typ.type])
          print_ops=generatePrintForIntegerOrFloat(program_state, load_op, load_op.results[0], call1.results[0])
          arg_operands.append(ops.first)
          arg_operands.extend(print_ops)
        if isinstance(arg.typ.type, fir.CharacterType):
          # This is a reference to a string
          arg_operands.extend(generatePrintForString(program_state, op[0], arg, call1.results[0]))

    # Close out the IO
    call3=fir.Call.create(attributes={"callee": SymbolRefAttr("_FortranAioEndIoStatement")}, operands=[call1.results[0]],
      result_types=[IntegerType(32)])
    arg_operands.extend([call3])

    insertExternalFunctionToGlobalState(program_state, "_FortranAioEndIoStatement", [fir.ReferenceType([IntegerType(8)])], i32)

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
    print_call=fir.Call.create(attributes={"callee": SymbolRefAttr(fn_name)}, operands=[init_call_ssa,
          arg], result_types=[IntegerType(1)])
    insertExternalFunctionToGlobalState(program_state, fn_name, [init_call_ssa.typ, arg.typ], IntegerType(1))
    return [op, print_call]

def generatePrintForString(program_state, op, arg, init_call_ssa):
    from_num=arg.typ.type.from_index.data
    to_num=arg.typ.type.to_index.data
    string_length=((to_num-from_num)+1)
    str_len=arith.Constant.create(attributes={"value": IntegerAttr.from_index_int_value(string_length)}, result_types=[IndexType()])
    arg2_2=fir.Convert.create(operands=[arg], result_types=[fir.ReferenceType([IntegerType(8)])])
    arg3_2=fir.Convert.create(operands=[str_len.results[0]], result_types=[i64])
    print_call=fir.Call.create(attributes={"callee": SymbolRefAttr("_FortranAioOutputAscii")}, operands=[init_call_ssa,
          arg2_2.results[0], arg3_2.results[0]], result_types=[IntegerType(1)])
    insertExternalFunctionToGlobalState(program_state, "_FortranAioOutputAscii", [init_call_ssa.typ,
        fir.ReferenceType([IntegerType(8)]), i64], IntegerType(1))
    return [op, str_len, arg2_2, arg3_2, print_call]

def insertExternalFunctionToGlobalState(program_state, function_name, args, result_type):
    if not program_state.hasGlobalFnName(function_name):
      if result_type is not None:
        fn=func.FuncOp.external(function_name, args, [result_type])
      else:
        fn=func.FuncOp.external(function_name, args, [])
      program_state.appendToGlobal(fn, function_name)

def translate_deallocate_intrinsic_call_expr(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:

    if len(call_expr.args.blocks[0].ops) != 1:
      raise Exception(f"For deallocate expected 1 argument but {len(call_expr.args.blocks[0].ops)} are present")

    op, arg = translate_expr(ctx, call_expr.args.blocks[0].ops.first, program_state)
    # The translate expression unboxes this for us, so we need to look into the operation that does that
    # which is a load, and then grab the origional SSA reference from that argument
    # We use the load initially here to load in the box and unbox it
    target_ssa=op[0].operands[0]

    box_type=get_nested_type(target_ssa.typ, fir.BoxType)
    heap_type=get_nested_type(target_ssa.typ, fir.HeapType)
    array_type=get_nested_type(target_ssa.typ, fir.SequenceType)
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
    shape_op=fir.Shape.create(operands=shape_operands, result_types=[fir.ShapeType([IntAttr(num_deferred)])])
    embox_op=fir.Embox.build(operands=[zero_bits_op.results[0], shape_op.results[0], [], [], []], regions=[[]], result_types=[box_type])
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
    array_type=get_nested_type(target_ssa.typ, fir.SequenceType)

    allocmem_op=fir.Allocmem.build(attributes={"in_type":array_type, "uniq_name": StringAttr(var_name+".alloc")}, operands=[[], args], regions=[[]], result_types=[heap_type])
    shape_op=fir.Shape.create(operands=args, result_types=[fir.ShapeType([IntAttr(len(args))])])
    embox_op=fir.Embox.build(operands=[allocmem_op.results[0], shape_op.results[0], [], [], []], regions=[[]], result_types=[fir.BoxType([heap_type])])
    store_op=fir.Store.create(operands=[embox_op.results[0], target_ssa])
    ops+=[allocmem_op, shape_op, embox_op, store_op]
    return ops

def get_nested_type(in_type, search_type):
  if isinstance(in_type, search_type): return in_type
  return get_nested_type(in_type.type, search_type)

def get_nested_parent_type(in_type, search_type):
  if not hasattr(in_type, "type"): return None
  if isinstance(in_type.type, search_type): return in_type
  return get_nested_parent_type(in_type.type, search_type)

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
        if not isinstance(type_to_reference, fir.ReferenceType) and not isinstance(type_to_reference, fir.SequenceType):
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
            array_type=get_nested_type(type_to_reference, fir.SequenceType)
            val=array_type.shape.data[0].value.data
            constant_op=arith.Constant.create(attributes={"value": IntegerAttr.from_index_int_value(val)}, result_types=[IndexType()])
            shape_op=fir.Shape.create(operands=[constant_op.results[0]], result_types=[fir.ShapeType([IntAttr(1)])])
            embox_op=fir.Embox.build(operands=[arg, shape_op.results[0], [], [], []], regions=[[]], result_types=[fir.BoxType([array_type])])
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
      call = fir.Call.create(attributes={"callee": SymbolRefAttr(fn_info.full_name)}, operands=args, result_types=[result_type])
    else:
      call = fir.Call.create(attributes={"callee": SymbolRefAttr(fn_info.full_name)}, operands=args, result_types=[])
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
      if isinstance(ssa_value, BlockArgument):
        # This is a block argument, therefore already loaded when was passed in
        return [], ssa_value
      has_nested_type = hasattr(ssa_value.typ, "type")
      assert isinstance(ssa_value, SSAValue)
      # We are limited here with type handling, need other floats - maybe a better way of doing this?
      if isinstance(ssa_value.typ, fir.SequenceType):
        return None, ssa_value
      elif isinstance(ssa_value.typ, mpi.VectorType):
        return None, ssa_value
      elif isinstance(ssa_value.typ, IntegerType) or (has_nested_type and isinstance(ssa_value.typ.type, IntegerType)):
        result_type=i32
      elif isinstance(ssa_value.typ, Float32Type) or (has_nested_type and isinstance(ssa_value.typ.type, Float32Type)):
        result_type=f32
      elif isinstance(ssa_value.typ, Float64Type) or (has_nested_type and isinstance(ssa_value.typ.type, Float64Type)):
        result_type=f64
      elif isinstance(ssa_value.typ, fir.SequenceType) or (has_nested_type and isinstance(ssa_value.typ.type, fir.SequenceType)):
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

    if isinstance(op, psy_stencil.PsyStencil_DimIndex):
      stencil_index_op=experimental_stencil.IndexOp.build(attributes={"dim": IntegerAttr(op.index, 32), "offset": experimental_stencil.IndexAttr.get(0,0,0)}, result_types=[IndexType()])
      target_type=try_translate_type(op.original_type)
      assert target_type is not None
      data_conv_op=perform_data_conversion_if_needed(stencil_index_op.results[0], target_type)
      assert data_conv_op is not None
      return [stencil_index_op, data_conv_op], data_conv_op.results[0]

    if isinstance(op, psy_ir.BinaryOperation):
      return translate_binary_expr(ctx, op, program_state)
    if isinstance(op, psy_ir.UnaryOperation):
      return translate_unary_expr(ctx, op, program_state)
    if isinstance(op, psy_ir.NaryOperation):
      return translate_nary_expr(ctx, op, program_state)
    if isinstance(op, psy_ir.CallExpr):
      call_expr= translate_call_expr_stmt(ctx, op, program_state, True)
      return call_expr, call_expr[-1].results[0]
    if isinstance(op, psy_ir.ArrayReference):
      return translate_array_reference_expr(ctx, op, program_state)
    if isinstance(op, psy_stencil.PsyStencil_Access):
      return translate_psy_stencil_access(ctx, op, program_state)

    assert False, "Unknown Expression"

def unpack_mpi_array(ctx: SSAValueCtx, op: psy_ir.ArrayReference, program_state : ProgramState):
    assert isinstance(ctx[op.var.var_name.data].typ, mpi.VectorType)
    # For now just support one dimensional access here to keep simple
    assert len(op.accessors.blocks[0].ops) == 1
    expr, ssa=try_translate_expr(ctx, op.accessors.blocks[0].ops.first, program_state)

    access_op=mpi.VectorGetOp.get(ctx[op.var.var_name.data], ssa)
    return expr+[access_op], access_op.results[0]

def translate_array_reference_expr(ctx: SSAValueCtx, op: psy_ir.ArrayReference, program_state : ProgramState):
  expressions=[]
  ssa_list=[]

  base_type=None

  if isinstance(ctx[op.var.var_name.data].typ, mpi.VectorType):
    # If this is an mpi array then we handle it differently
    return unpack_mpi_array(ctx, op, program_state)

  if (has_nested_type(ctx[op.var.var_name.data].typ, fir.BoxType) and has_nested_type(ctx[op.var.var_name.data].typ, fir.HeapType)):
    # We need to debox this
    box_type=get_nested_type(ctx[op.var.var_name.data].typ, fir.BoxType)
    heap_type=get_nested_type(ctx[op.var.var_name.data].typ, fir.HeapType)
    array_type=get_nested_type(ctx[op.var.var_name.data].typ, fir.SequenceType)

    load_op=fir.Load.create(operands=[ctx[op.var.var_name.data]], result_types=[box_type])

    boxdims_ops=[]
    ops_to_add=[]
    for i in range(array_type.getNumberDims()):
      dim_constant_op=arith.Constant.create(attributes={"value": IntegerAttr.from_index_int_value(i)}, result_types=[IndexType()])
      box_addr_op=fir.BoxDims.create(operands=[load_op.results[0], dim_constant_op.results[0]], result_types=[IndexType(), IndexType(), IndexType()])
      boxdims_ops.append(box_addr_op)
      ops_to_add+=[dim_constant_op, box_addr_op]

    boxaddr_op=fir.BoxAddr.create(operands=[load_op.results[0]], result_types=[heap_type])
    fir_type=fir.SequenceType(array_type.type, [fir.DeferredAttr()])

    convert_to_1d_op=fir.Convert.create(operands=[boxaddr_op.results[0]], result_types=[fir.ReferenceType([fir_type])])

    expressions+=[load_op] + ops_to_add + [boxaddr_op, convert_to_1d_op]
    base_type=convert_to_1d_op.results[0].typ

    ssa_list.append(convert_to_1d_op.results[0])

    arg_ssa_list=[]
    for idx, accessor in enumerate(op.accessors.blocks[0].ops):
      expr, ssa=try_translate_expr(ctx, accessor, program_state)
      expressions.extend(expr)

      index_conv=perform_data_conversion_if_needed(ssa, boxdims_ops[idx].results[0].typ)
      if index_conv is not None:
        expressions.append(index_conv)
        ssa=index_conv.results[0]

      substract_expr=arith.Subi(ssa, boxdims_ops[idx].results[0])
      expressions.append(substract_expr)

      prev_dim=substract_expr.results[0]
      for i in range(0, idx):
        multiply_op=arith.Muli(boxdims_ops[i].results[1], prev_dim)
        expressions.append(multiply_op)
        prev_dim=multiply_op.results[0]

      arg_ssa_list.append(prev_dim)

    top_level=arg_ssa_list[0]
    for i in range(1, len(arg_ssa_list)):
      add_op=arith.Addi(top_level, arg_ssa_list[i])
      expressions.append(add_op)
      top_level=add_op.results[0]

    ssa_list.append(top_level)

  else:
    # This is a reference to an array, nice and easy just use it directly
    ssa_list.append(ctx[op.var.var_name.data])
    base_type=ctx[op.var.var_name.data].typ

    for idx, accessor in enumerate(op.accessors.blocks[0].ops):
      # A lot of this is doing the subtraction to zero-base each index (default is starting at 1 in Fortran)
      # TODO - currently we assume always starts at 1 but in Fortran can set this so will need to keep track of that in the declaration and apply here
      expr, ssa=try_translate_expr(ctx, accessor, program_state)
      expressions.extend(expr)

      subtraction_index=arith.Constant.create(attributes={"value": IntegerAttr.from_int_and_width(1, 64)},
                                          result_types=[i64])
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

      substract_expr=arith.Subi(lhs_ssa, rhs_ssa)
      expressions.append(substract_expr)
      ssa_list.append(substract_expr.results[0])

    fir_type=try_translate_type(op.var.type)

  coordinate_of=fir.CoordinateOf.create(attributes={"baseType": base_type}, operands=ssa_list, result_types=[fir.ReferenceType([fir_type.type])])
  expressions.append(coordinate_of)
  return expressions, coordinate_of.results[0]

def translate_nary_expr(ctx: SSAValueCtx,
        unary_expr: psy_ir.UnaryOperation, program_state : ProgramState) -> Tuple[List[Operation], SSAValue]:
  expr_ssa=[]
  ssa_type=None
  ops_to_add=[]
  for op in unary_expr.expr.blocks[0].ops:
    expr, expr_ssa_value = translate_expr(ctx, op, program_state)
    if ssa_type is None:
      ssa_type=expr_ssa_value.typ
    if ssa_type != expr_ssa_value.typ:
      raise Exception(f"All operand types must be same type for nary operation")
    expr_ssa.append(expr_ssa_value)
    ops_to_add+=expr

  attr = unary_expr.op
  if attr.data == "MIN" or attr.data == "MAX":
    if isinstance(ssa_type, IntegerType):
      comparison_op_str="slt" if attr.data == "MIN" else "sgt"
    else:
      comparison_op_str="olt" if attr.data == "MIN" else "ogt"
    prev_min_ssa=expr_ssa[0]
    comparison_op=arith.Cmpi if isinstance(ssa_type, IntegerType) else arith.Cmpf
    for idx in range(1, len(expr_ssa)):
      compare_op=comparison_op.get(prev_min_ssa, expr_ssa[idx], comparison_op_str)
      select_op=arith.Select.get(compare_op.results[0], prev_min_ssa, expr_ssa[idx])
      prev_min_ssa=select_op.results[0]
      ops_to_add+=[compare_op, select_op]
    return ops_to_add, prev_min_ssa
  else:
    raise Exception(f"Nary operation '{attr.data}' not supported")


def translate_unary_expr(ctx: SSAValueCtx,
        unary_expr: psy_ir.UnaryOperation, program_state : ProgramState) -> Tuple[List[Operation], SSAValue]:

  expr, expr_ssa_value = translate_expr(ctx, unary_expr.expr.blocks[0].ops.first, program_state)

  attr = unary_expr.op
  assert isinstance(attr, Attribute)

  if (attr.data == "NOT"):
    constant_true=arith.Constant.create(attributes={"value": IntegerAttr.from_int_and_width(1, 1)},
                                         result_types=[IntegerType(1)])
    xori=arith.XOrI.get(expr_ssa_value, constant_true.results[0])

    return expr + [constant_true, xori], xori.results[0]

  if (attr.data == "SQRT"):
    sqrt_op=math.SqrtOp.get(expr_ssa_value)
    return expr + [sqrt_op], sqrt_op.results[0]

  if (attr.data == "ABS"):
    if isinstance(expr_ssa_value.typ, AnyFloat):
      abs_op=math.AbsFOp.get(expr_ssa_value)
      return expr + [abs_op], abs_op.results[0]
    elif isinstance(expr_ssa_value.typ, IntegerType):
      abs_op=math.AbsIOp.get(expr_ssa_value)
      return expr + [abs_op], abs_op.results[0]
    else:
      raise Exception(f"Can only issue abs on int or float, but issued on {expr_ssa_value.typ}")

  if (attr.data == "MINUS"):
    if isinstance(expr_ssa_value.typ, AnyFloat):
      negf_op=arith.Negf.get(expr_ssa_value)
      return expr + [negf_op], negf_op.results[0]
    elif isinstance(expr_ssa_value.typ, IntegerType):
      constant_op=arith.Constant.create(attributes={"value": IntegerAttr(0, expr_ssa_value.typ)}, result_types=[expr_ssa_value.typ])
      sub_op=arith.Subi.get(constant_op, expr_ssa_value)
      return expr + [constant_op, sub_op], sub_op.results[0]
    else:
      raise Exception(f"Can only issue minus on int or float, but issued on {expr_ssa_value.typ}")

  raise Exception(f"Unable to handle unary expression `{attr.data}`")

def get_expression_conversion_type(lhs_type, rhs_type):
  if isinstance(lhs_type, IntegerType):
    if isinstance(rhs_type, IntegerType):
      if lhs_type.width.data > rhs_type.width.data: return None, lhs_type
      if lhs_type.width.data < rhs_type.width.data: return rhs_type, None
      return None, None
    elif isinstance(rhs_type, Float16Type) or isinstance(rhs_type, Float32Type) or isinstance(rhs_type, Float64Type):
      return rhs_type, None
    else:
      return None, None
  if isinstance(lhs_type, Float16Type):
    if isintance(rhs_type, Float32Type) or isintance(rhs_type, Float64Type): return rhs_type, None
  if isinstance(lhs_type, Float32Type):
    if isinstance(rhs_type, Float16Type) or isinstance(rhs_type, IntegerType): return None, lhs_type
    if isinstance(rhs_type, Float64Type): return rhs_type, None
  if isinstance(lhs_type, Float64Type):
    if isinstance(rhs_type, Float16Type) or isinstance(rhs_type, Float32Type) or isinstance(rhs_type, IntegerType): return None, lhs_type,
  return None, None

def perform_data_conversion_if_needed(expr_ssa, conv_type):
  if conv_type is not None:
    expr_conversion=fir.Convert.create(operands=[expr_ssa], result_types=[conv_type])
    return expr_conversion
  return None

def translate_binary_expr(
        ctx: SSAValueCtx,
        binary_expr: psy_ir.BinaryOperation, program_state : ProgramState) -> Tuple[List[Operation], SSAValue]:
    lhs, lhs_ssa_value = translate_expr(ctx, binary_expr.lhs.blocks[0].ops.first, program_state)
    rhs, rhs_ssa_value = translate_expr(ctx, binary_expr.rhs.blocks[0].ops.first, program_state)
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
        fn_call_op=fir.Call.create(attributes={"callee": SymbolRefAttr(call_name)},
            operands=[lhs, rhs], result_types=[lhs.typ])
        insertExternalFunctionToGlobalState(program_state, call_name, [lhs.typ, rhs.typ], lhs.typ)
        return fn_call_op
      else:
        raise Exception(f"Could not translate `{lhs.typ}' and '{rhs.typ}' for POW operation")
    assert op_instance is not None, "Operation "+operation+" not implemented for type"

    attributes=dir(op_instance)
    if "get" in attributes:
      return op_instance.get(lhs, rhs)
    else:
      return op_instance(lhs, rhs)

  if (operation == "AND"):
    assert isinstance(operand_type, IntegerType), "Integer type only supported for 'and'"
    return arith.AndI(lhs, rhs)
  if (operation == "OR"):
    assert isinstance(operand_type, IntegerType), "Integer type only supported for 'or'"
    return arith.OrI(lhs, rhs)

  if operation in binary_arith_psy_to_arith_comparison_op:
    return arith.Cmpi.get(lhs, rhs, binary_arith_psy_to_arith_comparison_op[operation])

  raise Exception(f"Unable to handle arithmetic instance `{operation}`")

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
    typ=fir.CharacterType([fir.IntAttr(1), fir.IntAttr(len(string_literal))])
    string_lit_op=fir.StringLit.create(attributes={"size": IntegerAttr.from_int_and_width(len(string_literal), 64), "value": StringAttr(string_literal)}, result_types=[typ])
    has_val_op=fir.HasValue.create(operands=[string_lit_op.results[0]])
    str_uuid=uuid.uuid4().hex.upper()
    glob=fir.Global.create(attributes={"linkName": StringAttr("linkonce"), "sym_name": StringAttr("_QQcl."+str_uuid), "symref": SymbolRefAttr("_QQcl."+str_uuid), "type": typ},
      regions=[Region([Block([string_lit_op, has_val_op])])])
    program_state.appendToGlobal(glob)
    ref_type=fir.ReferenceType([typ])
    addr_lookup=fir.AddressOf.create(attributes={"symbol": SymbolRefAttr("_QQcl."+str_uuid)}, result_types=[ref_type])
    return addr_lookup
