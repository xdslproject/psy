from __future__ import annotations
from xdsl.dialects.builtin import (StringAttr, ModuleOp, IntegerAttr, IntegerType, ArrayAttr, i32, f32, 
      Float16Type, Float32Type, Float64Type, FlatSymbolRefAttr, FloatAttr)
from xdsl.dialects import func, arith
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
    wrap_top_levelcall_from_main(ctx, input_module)
    
def wrap_top_levelcall_from_main(ctx: MLContext, module: ModuleOp):
  found_routine=find_floating_region(module)
  assert found_routine is not None
  
  body = Region()
  block = Block()
  
  callexpr = fir.Call.create(attributes={"callee": FlatSymbolRefAttr.from_str(found_routine.sym_name.data)}, result_types=[])  
  # A return is always needed at the end of the procedure
  block.add_ops([callexpr, func.Return.create()])
  body.add_block(block)
  main = func.FuncOp.from_region("_QQmain", [], [], body)
  # Need to set sym_visibility here so the main function appears in Flang generated output
  main.attributes["sym_visibility"]=StringAttr("public")
  module.regions[0].blocks[0].add_ops([main])
  
def find_floating_region(module: ModuleOp):
  for op in module.ops:
    if isinstance(op, func.FuncOp):
      if op.sym_name.data.startswith("_QQ"): return op      
  return None  

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
    block = Block()
    # create a new nested scope and
    # relate parameter identifiers with SSA values of block arguments
    #c = SSAValueCtx(dictionary=dict(zip(param_names, block.args)),
    #                parent_scope=ctx)
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
      res=translate_def_or_stmt(ctx, op, program_state) # should be SSAValueCtx created above for routine
      if res is not None:
        to_add.append(res)
        
    for op in routine_def.routine_body.blocks[0].ops:
      res=translate_def_or_stmt(ctx, op, program_state) # should be SSAValueCtx created above for routine
      if res is not None:        
        to_add.append(res)
        
    # A return is always needed at the end of the procedure
    to_add.append([func.Return.create()])
        
    program_state.unsetRoutineName()
    program_state.clearImports()
    block.add_ops(flatten(to_add))
    body.add_block(block)
    
    full_name=generateProcedureSymName(program_state, routine_name.data)

    return func.FuncOp.from_region(full_name, [], [], body)
    
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
        
def split_multi_assign(
        assign: psy_ast.Assign) -> Tuple[List[Operation], Operation]:
    """Get the list of targets of a multi assign, as well as the expression value."""
    if isinstance(assign.rhs.op, psy_ir.Assign):
        targets, value = split_multi_assign(assign.rhs.op)
        return [assign.target.op] + targets, value
    return [assign.lhs.op], assign.rhs.op        
        
def translate_assign(ctx: SSAValueCtx,
                     assign: psy_ast.Assign, program_state : ProgramState) -> List[Operation]:
    targets, value = split_multi_assign(assign)
    value_fir, value_var = translate_expr(ctx, value)        

    translated_targets = [translate_expr(ctx, target) for target in targets]
    targets_fir = [
        target_op for target in translated_targets for target_op in target[0]
    ]    
    targets_var = [target[1] for target in translated_targets]
    
    assigns: List[Operation] = [
        fir.Store.create(operands=[value_var, target_var])
        for target_var in targets_var
    ]      
    
    return value_fir + targets_fir + assigns        
        
def translate_call_expr_stmt(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, program_state : ProgramState) -> List[Operation]:
    ops: List[Operation] = []
    args: List[SSAValue] = []

    #for arg in call_expr.args.blocks[0].ops:
    #    op, arg = translate_expr(ctx, arg)
    #    ops += op
    #    args.append(arg)      

    name = call_expr.attributes["func"]
    assert program_state.hasImport(name.data)
    full_name=generateProcedurePrefixWithModuleName(program_state.getImportModule(name.data), name.data, "P") 
    call = fir.Call.create(attributes={"callee": FlatSymbolRefAttr.from_str(full_name)}, result_types=[])
    ops.append(call)
    return ops
    
def translate_expr(ctx: SSAValueCtx,
                   op: Operation) -> Tuple[List[Operation], SSAValue]:
    """
    Translates op as an expression.
    If op is an expression, returns a list of the translated Operations
    and the ssa value representing the translated expression.
    Fails otherwise.
    """
    res = try_translate_expr(ctx, op)
    if res is None:
        raise Exception(f"Could not translate `{op}' as an expression")
    else:
        ops, ssa_value = res
        return ops, ssa_value
  
def try_translate_expr(
        ctx: SSAValueCtx,
        op: Operation) -> Optional[Tuple[List[Operation], SSAValue]]:
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
      return [], ssa_value    
    if isinstance(op, psy_ir.BinaryOperation):
      return translate_binary_expr(ctx, op)
    #    return translate_binary_expr(ctx, op)
    #if isinstance(op, psy_ast.CallExpr):
    #    print("No call expression here!")
    #    return translate_call_expr(ctx, op)
    
    assert False, "Unknown Expression"
    
def translate_binary_expr(
        ctx: SSAValueCtx,
        binary_expr: psy_ast.BinaryExpr) -> Tuple[List[Operation], SSAValue]:
    lhs, lhs_ssa_value = translate_expr(ctx, binary_expr.lhs.blocks[0].ops[0])
    rhs, rhs_ssa_value = translate_expr(ctx, binary_expr.rhs.blocks[0].ops[0])
    result_type = rhs_ssa_value.typ
    if binary_expr.op.data != "is":
        assert lhs_ssa_value.typ == rhs_ssa_value.typ

    if binary_expr.op.data in ['!=', '==', '<', '<=', '>', '>=', 'is']:
        result_type = psy_type.bool_type

    attr = binary_expr.op
    assert isinstance(attr, Attribute)

    # Need to consider special case when the binary operation has a different execution order for and & or?
    
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
                                         result_types=[i32])  
        
    if isinstance(value, psy_ir.FloatAttr):        
        return arith.Constant.create(attributes={"value": FloatAttr.from_value(value.data)},
                                         result_types=[f32])  

    if isinstance(value, StringAttr):
        return arith.Constant.create(attributes={"value": value},
                                         result_types=[i32]) # Need to replace with string type!
    
    raise Exception(f"Could not translate `{op}' as a literal")    
