from __future__ import annotations
from xdsl.dialects.builtin import (StringAttr, ModuleOp, IntegerAttr, IntegerType, ArrayAttr, i32, f32, IndexType,
      Float16Type, Float32Type, Float64Type, FlatSymbolRefAttr, FloatAttr, UnitAttr)
from xdsl.dialects import func, arith, cf
from xdsl.ir import Operation, Attribute, ParametrizedAttribute, Region, Block, SSAValue, MLContext
from psy.dialects import psy_ir

from util.list_ops import flatten
from ftn.dialects import fir
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

binary_arith_op_matching={"ADD": [arith.Addi, arith.Addf], "SUB":[arith.Subi, arith.Subf], "MUL": [arith.Muli, arith.Mulf], "DIV": [arith.DivSI, arith.Divf], "REM": [arith.RemSI, None], 
"MIN" : [arith.MinSI, arith.Minf], "MAX" : [arith.MaxSI, arith.Maxf]}

binary_arith_psy_to_arith_comparison_op={"EQ": "eq", "NE": "ne", "GT": "sgt", "LT": "slt", "GE": "sge", "LE": "sle"}

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
            
class ProgramState:
  def __init__(self):
    self.module_name=None
    self.routine_name=None
    self.return_block=None
    self.imports={}
  
  def setRoutineName(self, routine_name):
    self.routine_name=routine_name
    
  def unsetRoutineName(self):
    self.routine_name=None
  
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
    

def psy_ir_to_fir(ctx: MLContext, input_module: ModuleOp):
    res_module = translate_program(input_module)        
    # Now we need to gather the containers and inform the top level routine about these so it can generate the use
    #applyModuleUseToFloatingRegions(res_module, collectContainerNames(res_module))
    res_module.regions[0].move_blocks(input_module.regions[0])
    # Create program entry point
    check_program_entry_point(input_module)
    
def check_program_entry_point(module: ModuleOp):
  for op in module.ops:
    if isinstance(op, func.FuncOp):
      if op.sym_name.data=="_QQmain": return
  assert False, "No program entry point"

def translate_program(input_module: ModuleOp) -> ModuleOp:
    # create an empty global context
    global_ctx = SSAValueCtx()
    containers: List[func.FuncOp] = []
    body = Region()
    block = Block()
    for top_level_entry in input_module.ops:
      if isinstance(top_level_entry, psy_ir.FileContainer):                   
        for container in top_level_entry.children.blocks[0].ops:
          translate_container(global_ctx, container, block)         
        
      elif isinstance(top_level_entry, psy_ir.Container):        
        print("NOT IMPLEMENTED")
        #containers.append(translate_container(global_ctx, top_level_entry))      
          
    body.add_block(block)
    return ModuleOp.from_region_or_ops(body)
    
def translate_container(ctx: SSAValueCtx, op: Operation, block) -> Operation:  
  if isinstance(op, psy_ir.Container):        
    program_state = ProgramState()
    program_state.setModuleName(op.attributes["container_name"].data)
    
    block.add_ops(translate_fun_def(ctx, routine, program_state) for routine in op.routines.blocks[0].ops)    
  elif isinstance(op, psy_ir.Routine):
    program_state = ProgramState()  
   
    block.add_op(translate_fun_def(ctx, op, program_state))
    
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
      arg_type=fir.ReferenceType([try_translate_type(arg.type)])
      arg_names.append(arg.var_name.data)
      c[arg.var_name.data] = arg_type
      arg_types.append(arg_type)
      
    block = Block.from_arg_types(arg_types)
                    
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

    function_fir=func.FuncOp.from_region(full_name, arg_types, [try_translate_type(routine_def.return_var.type)] if is_function else [], body)
    #TODO - need to correlate against public routines to mark private or public!
    if routine_def.is_program.data:
      function_fir.attributes["sym_visibility"]=StringAttr("public")
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
        
def translate_var_def(ctx: SSAValueCtx,
                      var_def: psy_ast.VarDef, program_state : ProgramState) -> List[Operation]:
   
    var_name = var_def.var.var_name
    assert isinstance(var_name, StringAttr)
    type = try_translate_type(var_def.var.type)
    
    ref_type=fir.ReferenceType([type])

    # Operand segment sizes is wrong here, either hack it like trying (but doesn't match!) or understand why missing
    fir_var_def = fir.Alloca.create(attributes={"bindc_name": var_name, "uniq_name": StringAttr(generateVariableUniqueName(program_state, var_name.data)), 
      "in_type":type}, operands=[], result_types=[ref_type])

    # relate variable identifier and SSA value by adding it into the current context
    ctx[var_name.data] = fir_var_def.results[0]
    return [fir_var_def]   
    
def try_translate_type(op: Operation) -> Optional[Attribute]:
    """Tries to translate op as a type, returns None otherwise."""    
    if isinstance(op, psy_ir.NamedType):
      if op.type_name.data == "integer": return i32
      if op.type_name.data == "real": return f32

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
    
    # The targets of the assignment are references and not expressions, so grab from the ctx
    translated_target = ctx[assign.lhs.op.id.data]
    target_conversion_type=get_store_conversion_if_needed(translated_target.typ, value_var.typ)
    if target_conversion_type is not None:
      converter=fir.Convert.create(operands=[value_var], result_types=[target_conversion_type])
      value_fir.append(converter)
      return value_fir + [fir.Store.create(operands=[converter.results[0], translated_target])]
    else:
      return value_fir + [fir.Store.create(operands=[value_var, translated_target])]
        
def translate_call_expr_stmt(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
    ops: List[Operation] = []
    args: List[SSAValue] = []

    #for arg in call_expr.args.blocks[0].ops:
    #    op, arg = translate_expr(ctx, arg)
    #    ops += op
    #    args.append(arg)      

    name = call_expr.attributes["func"]
    assert program_state.hasImport(name.data)
    full_name=generateProcedurePrefixWithModuleName(program_state.getImportModule(name.data), name.data, "P") 
    # Need return type here for expression
    if is_expr:
      result_type=try_translate_type(call_expr.type)
      call = fir.Call.create(attributes={"callee": FlatSymbolRefAttr.from_str(full_name)}, result_types=[result_type])
    else:
      call = fir.Call.create(attributes={"callee": FlatSymbolRefAttr.from_str(full_name)}, result_types=[])
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
      op = translate_literal(op)
      return [op], op.results[0]
    if isinstance(op, psy_ir.ExprName):
      ssa_value = ctx[op.id.data]
      assert isinstance(ssa_value, SSAValue)
      # We are limited here with type handling, need other floats - maybe a better way of doing this?
      if isinstance(ssa_value.typ.type, IntegerType):
        result_type=i32
      elif isinstance(ssa_value.typ.type, Float32Type):
        result_type=f32      
      
      op=fir.Load.create(operands=[ssa_value], result_types=[result_type])
      
      return [op], op.results[0]
    if isinstance(op, psy_ir.BinaryOperation):
      return translate_binary_expr(ctx, op, program_state)
    if isinstance(op, psy_ir.UnaryOperation):
      return translate_unary_expr(ctx, op, program_state)
    if isinstance(op, psy_ir.CallExpr):      
      call_expr= translate_call_expr_stmt(ctx, op, program_state, True)      
      return call_expr, call_expr[0].results[0]
        
    assert False, "Unknown Expression"
    
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
    if isintance(rhs_type, Float16Type): return None, lhs_type
    if isintance(rhs_type, Float64Type): return rhs_type, None
  if isinstance(lhs_type, Float64Type):
    if isintance(rhs_type, Float16Type) or isintance(rhs_type, Float32Type): return None, lhs_type, 
  return None
    
def translate_binary_expr(
        ctx: SSAValueCtx,
        binary_expr: psy_ir.BinaryOperation, program_state : ProgramState) -> Tuple[List[Operation], SSAValue]:
    lhs, lhs_ssa_value = translate_expr(ctx, binary_expr.lhs.blocks[0].ops[0], program_state)
    rhs, rhs_ssa_value = translate_expr(ctx, binary_expr.rhs.blocks[0].ops[0], program_state)
    result_type = lhs_ssa_value.typ
    
    
    lhs_conv_type, rhs_conv_type=get_expression_conversion_type(lhs_ssa_value.typ, rhs_ssa_value.typ)
    if lhs_conv_type is not None:
      lhs_conversion=fir.Convert.create(operands=[lhs_ssa_value], result_types=[lhs_conv_type])
      lhs.append(lhs_conversion)
      lhs_ssa_value=lhs_conversion.results[0]
      
    if rhs_conv_type is not None:
      rhs_conversion=fir.Convert.create(operands=[rhs_ssa_value], result_types=[rhs_conv_type])
      rhs.append(rhs_conversion)
      rhs_ssa_value=rhs_conversion.results[0]
        
    #assert (lhs_ssa_value.typ == rhs_ssa_value.typ) or isinstance(lhs_ssa_value.typ, fir.ReferenceType) or isinstance(rhs_ssa_value.typ, fir.ReferenceType)
    
    attr = binary_expr.op
    assert isinstance(attr, Attribute)    
    
    fir_binary_expr=get_arith_instance(binary_expr.op.data, lhs_ssa_value, rhs_ssa_value)    
    
    return lhs + rhs + [fir_binary_expr], fir_binary_expr.results[0] 
    
def get_arith_instance(operation:str, lhs, rhs):
  operand_type = lhs.typ
  if operation in binary_arith_op_matching:    
    if isinstance(operand_type, IntegerType): index=0
    if isinstance(operand_type, Float16Type) or isinstance(operand_type, Float32Type) or isinstance(operand_type, Float64Type): index=1
    op_instance=binary_arith_op_matching[operation][index]
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
    
def translate_literal(op: psy_ir.Literal) -> Operation:
    value = op.attributes["value"]      

    if isinstance(value, IntegerAttr):
        return arith.Constant.create(attributes={"value": value},
                                         result_types=[value.typ])  
        
    if isinstance(value, psy_ir.FloatAttr):        
        return arith.Constant.create(attributes={"value": value},
                                         result_types=[value.type])  

    if isinstance(value, StringAttr):
        return arith.Constant.create(attributes={"value": value},
                                         result_types=[i32]) # Need to replace with string type!
    
    raise Exception(f"Could not translate `{op}' as a literal")    
