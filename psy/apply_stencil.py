from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue, Region, Block
from xdsl.dialects.builtin import IntegerAttr, StringAttr, ArrayAttr, IntAttr
from xdsl.pattern_rewriter import (GreedyRewritePatternApplier,
                                   PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, op_type_rewrite_pattern)

from psy.dialects import psy_ir, hstencil
from util.visitor import Visitor
from enum import Enum

class AccessMode(Enum):
    WRITE = 1
    READ = 2

class CollectLoopsToReplace(Visitor):
  def __init__(self, indicies):
    self.applicable_indicies = {index: None for index in indicies}
    self.top_loop=None
    self.bottom_loop=None

  def traverse_loop(self, dl: psy_ir.Loop):
    if (dl.variable.var_name.data in self.applicable_indicies):
      self.applicable_indicies[dl.variable.var_name.data]=dl
      if self.top_loop is None: self.top_loop=dl
      self.bottom_loop=dl
    for op in dl.body.blocks[0].ops:
      self.traverse(op)

class CollectArrayVariableIndexes(Visitor):
  def __init__(self):
    self.array_indexes=[]

  def traverse_binary_expr(self, binary_expr: psy_ir.BinaryOperation):
    for op in binary_expr.lhs.blocks[0].ops:
      self.traverse(op)
    for op in binary_expr.rhs.blocks[0].ops:
      self.traverse(op)

  def traverse_expr_name(self, id_expr: psy_ir.ExprName):
    self.array_indexes.append(id_expr.var.var_name.data)

class CollectApplicabeVariables(Visitor):
  def __init__(self):
    self.written_variables={}
    self.read_variables={}
    self.currentMode=AccessMode.WRITE

  def traverse_binary_expr(self, binary_expr: psy_ir.BinaryOperation):
    for op in binary_expr.lhs.blocks[0].ops:
      self.traverse(op)
    for op in binary_expr.rhs.blocks[0].ops:
      self.traverse(op)

  def traverse_array_reference(self, member_access_expr: psy_ir.ArrayReference):
    if self.currentMode == AccessMode.WRITE:
      self.written_variables[member_access_expr.var.var_name.data]=member_access_expr
    else:
      self.read_variables[member_access_expr.var.var_name.data]=member_access_expr

  def traverse_expr_name(self, id_expr: psy_ir.ExprName):
    if self.currentMode == AccessMode.WRITE:
      self.written_variables[id_expr.var.var_name.data]=id_expr
    else:
      self.read_variables[id_expr.var.var_name.data]=id_expr

  def traverse_loop(self, dl: psy_ir.Loop):
    for op in dl.body.blocks[0].ops:
      self.traverse(op)

  def traverse_assign(self, assign:psy_ir.Assign):
    self.currentMode=AccessMode.WRITE
    for op in assign.lhs.blocks[0].ops:
      self.traverse(op)
    self.currentMode=AccessMode.READ
    for op in assign.rhs.blocks[0].ops:
      self.traverse(op)

class CollectArrayRelativeOffsets(Visitor):
  def __init__(self):
    self.offset_val=None

  def traverse_binary_operation(self, binary_expr: psy_ir.BinaryOperation):    
    for op in binary_expr.lhs.blocks[0].ops:
      self.traverse(op)
    for op in binary_expr.rhs.blocks[0].ops:
      self.traverse(op)

    if binary_expr.op.data == "SUB":       
      self.offset_val=-self.offset_val

  def traverse_literal(self, literal:psy_ir.Literal):    
    self.offset_val=literal.value.value.data

class ReplaceAbsoluteArrayIndexWithStencil(RewritePattern):
  @op_type_rewrite_pattern
  def match_and_rewrite(
            self, array_reference: psy_ir.ArrayReference, rewriter: PatternRewriter):

    stencil_relative_offsets=[]
    for accessor in array_reference.accessors.blocks[0].ops:
      visitor=CollectArrayRelativeOffsets()
      visitor.traverse(accessor)
      if visitor.offset_val is None:
        stencil_relative_offsets.append(IntAttr.from_int(0))
      else:
        stencil_relative_offsets.append(IntAttr.from_int(visitor.offset_val))
        
    parent=array_reference.parent
    idx = parent.ops.index(array_reference)
    array_reference.detach()
    access_op=hstencil.HStencil_Access.build(attributes={"var": array_reference.var, "stencil_ops": ArrayAttr(stencil_relative_offsets)})
    rewriter.insert_op_at_pos(access_op, parent, idx)

class ApplyStencilRewriter(RewritePattern):
    def __init__(self):
      self.called_procedures=[]

    def allIndexesFilled(self, index_dict):
      for v in index_dict.values():
        if v is None: return False
      return True

    @op_type_rewrite_pattern
    def match_and_rewrite(
            self, for_loop: psy_ir.Loop, rewriter: PatternRewriter):

        visitor = CollectApplicabeVariables()
        visitor.traverse(for_loop)

        read_vars=[]
        access_variables=[]
        for read_var_k, read_var_v in visitor.read_variables.items():
          read_vars.append(read_var_v.var)
          for index in read_var_v.accessors.blocks[0].ops:
            v2=CollectArrayVariableIndexes()
            v2.traverse(index)
            access_variables.extend(v2.array_indexes)

        write_vars=[]
        for write_var_v in visitor.written_variables.values():
          write_vars.append(write_var_v.var)

        v3=CollectLoopsToReplace(access_variables)
        v3.traverse(for_loop)

        if self.allIndexesFilled(v3.applicable_indicies):
          parent=v3.top_loop.parent
          idx = parent.ops.index(v3.top_loop)
          v3.top_loop.detach()
          loop_body=v3.bottom_loop.body.blocks[0].ops[0]
          v3.bottom_loop.body.blocks[0].ops[0].detach()

          # For now assume the loop body is an assignment, can extend later on to be more flexible
          rhs=loop_body.rhs.blocks[0].ops[0]
          rhs.detach()

          replaceArrayIndexWithStencil=ReplaceAbsoluteArrayIndexWithStencil()
          walker = PatternRewriteWalker(GreedyRewritePatternApplier([replaceArrayIndexWithStencil]), apply_recursively=False)
          walker.rewrite_module(rhs)

          stencil_result=hstencil.HStencil_Result.build(attributes={"var": loop_body.lhs.blocks[0].ops[0].var, "stencil_ops": ArrayAttr([])}, regions=[[rhs]])

          stencil_op=hstencil.HStencil_Stencil.build(attributes={"input_fields": ArrayAttr(read_vars), "output_fields":ArrayAttr(write_vars)}, regions=[[stencil_result]])
          rewriter.insert_op_at_pos(stencil_op, parent, idx)


def apply_stencil_analysis(ctx: psy_ir.MLContext, module: ModuleOp) -> ModuleOp:
    applyStencilRewriter=ApplyStencilRewriter()
    walker = PatternRewriteWalker(GreedyRewritePatternApplier([applyStencilRewriter]), apply_recursively=False)
    walker.rewrite_module(module)

    #print(1 + "J")
    return module
