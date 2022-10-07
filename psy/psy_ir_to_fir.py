from __future__ import annotations
from xdsl.dialects.builtin import StringAttr, ModuleOp, IntegerAttr, ArrayAttr, i32, f32
from xdsl.dialects import func
from xdsl.ir import Operation, Attribute, ParametrizedAttribute, Region, Block, SSAValue, MLContext
from psy.dialects import psy_ir

from util.list_ops import flatten
from ftn.dialects import fir
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


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


def psy_ir_to_fir(ctx: MLContext, input_module: ModuleOp):
    res_module = translate_program(input_module)    
    # Now we need to gather the containers and inform the top level routine about these so it can generate the use
    #applyModuleUseToFloatingRegions(res_module, collectContainerNames(res_module))
    res_module.regions[0].move_blocks(input_module.regions[0])
    # Create program entry point
    #wrap_top_levelcall_from_main(ctx, input_module)

def translate_program(input_module: ModuleOp) -> ModuleOp:
    # create an empty global context
    global_ctx = SSAValueCtx()
    containers: List[func.FuncOp] = []
    for top_level_entry in input_module.ops:
      if isinstance(top_level_entry, psy_ir.FileContainer):        
       pass #containers.extend([translate_container(global_ctx, container) for container in top_level_entry.children.blocks[0].ops])
        
        
      elif isinstance(top_level_entry, psy_ir.Container):        
        print("NOT IMPLEMENTED")
        #containers.append(translate_container(global_ctx, top_level_entry))      
                
    return ModuleOp.from_region_or_ops(translate_container(global_ctx, top_level_entry.children.blocks[0].ops[0]))
    
def translate_container(ctx: SSAValueCtx, op: Operation) -> Operation:  
  if isinstance(op, psy_ir.Container):    
    body = Region()
    block = Block()
    block.add_ops(translate_fun_def(ctx, routine) for routine in op.routines.blocks[0].ops)

    body.add_block(block)    
    return body
  elif isinstance(op, psy_ir.Routine):    
    return translate_fun_def(ctx, op)    
    
def translate_fun_def(ctx: SSAValueCtx,
                      routine_def: psy_ir.Routine) -> Operation:
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
    
    to_add=[]
    for op in routine_def.local_var_declarations.blocks[0].ops:
      res=translate_def_or_stmt(ctx, op) # should be SSAValueCtx created above for routine
      if res is not None:
        to_add.append(res)
        
    for op in routine_def.routine_body.blocks[0].ops:
      res=translate_def_or_stmt(ctx, op) # should be SSAValueCtx created above for routine
      if res is not None:        
        to_add.append(res)
        
    block.add_ops(flatten(to_add))
    body.add_block(block)

    return func.FuncOp.from_region(routine_name, [], [], body)   
    
def translate_def_or_stmt(ctx: SSAValueCtx, op: Operation) -> List[Operation]:
    """
    Translate an operation that can either be a definition or statement
    """
    # first try to translate op as a definition:
    #   if op is a definition this will return a list of translated Operations
    ops = try_translate_def(ctx, op)
    if ops is not None:
        return ops
    # op has not been a definition, try to translate op as a statement:
    #   if op is a statement this will return a list of translated Operations
    #ops = try_translate_stmt(ctx, op)
    #if ops is not None:
    #    return ops
    # operation must have been translated by now
    return None
    raise Exception(f"Could not translate `{op}' as a definition or statement")


def try_translate_def(ctx: SSAValueCtx,
                      op: Operation) -> Optional[List[Operation]]:
    """
    Tries to translate op as a definition.
    Returns a list of the translated Operations if op is a definition, returns None otherwise.
    """
    #if isinstance(op, psy_ast.Routine):
    #    return [translate_fun_def(ctx, op)]    
    if isinstance(op, psy_ir.VarDef):
        return translate_var_def(ctx, op)
    else:
        return None
        
def translate_var_def(ctx: SSAValueCtx,
                      var_def: psy_ast.VarDef) -> List[Operation]:
   
    var_name = var_def.var.var_name
    assert isinstance(var_name, StringAttr)
    type = try_translate_type(var_def.var.type)
    
    ref_type=fir.ReferenceType([type])

    fir_var_def = fir.Alloca.create(attributes={"bindc_name": var_name, "uniq_name": StringAttr("hello"), "in_type":type}, operands=[], result_types=[ref_type])

    # relate variable identifier and SSA value by adding it into the current context
    ctx[var_name.data] = fir_var_def.results[0]    
    return [fir_var_def]     
    
def try_translate_type(op: Operation) -> Optional[Attribute]:
    """Tries to translate op as a type, returns None otherwise."""    
    if isinstance(op, psy_ir.NamedType):
      if op.type_name.data == "integer": return i32
      if op.type_name.data == "real": return f32

    return None
  
