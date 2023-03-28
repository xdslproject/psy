from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue, Region, Block
from xdsl.dialects.builtin import IntegerAttr, StringAttr, ArrayAttr, IntAttr
from xdsl.pattern_rewriter import (GreedyRewritePatternApplier,
                                   PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, op_type_rewrite_pattern)

from psy.dialects import psy_ir, psy_stencil
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

class CollectApplicableVariables(Visitor):
  def __init__(self):
    self.written_variables={}
    self.read_variables={}
    self.written_to_read={}
    self.current_written_variable=None
    self.current_read_variables=[]
    self.currentMode=AccessMode.WRITE

  def traverse_binary_expr(self, binary_expr: psy_ir.BinaryOperation):
    for op in binary_expr.lhs.blocks[0].ops:
      self.traverse(op)
    for op in binary_expr.rhs.blocks[0].ops:
      self.traverse(op)

  def traverse_array_reference(self, member_access_expr: psy_ir.ArrayReference):
    if self.currentMode == AccessMode.WRITE:
      self.written_variables[member_access_expr.var.var_name.data]=member_access_expr
      self.current_written_variable=member_access_expr.var.var_name.data
    else:
      self.read_variables[member_access_expr.var.var_name.data]=member_access_expr
      self.current_read_variables.append(member_access_expr.var.var_name.data)

  def traverse_expr_name(self, id_expr: psy_ir.ExprName):
    if self.currentMode == AccessMode.WRITE:
      self.written_variables[id_expr.var.var_name.data]=id_expr
      self.current_written_variable=id_expr.var.var_name.data
    else:
      self.read_variables[id_expr.var.var_name.data]=id_expr
      self.current_read_variables.append(id_expr.var.var_name.data)

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
    assert self.current_written_variable is not None
    self.written_to_read[self.current_written_variable]=self.current_read_variables
    self.current_written_variable=None
    self.current_read_variables=[]

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

class RemoveEmptyLoops(RewritePattern):
  @op_type_rewrite_pattern
  def match_and_rewrite(
            self, for_loop: psy_ir.Loop, rewriter: PatternRewriter):
      if len(for_loop.body.blocks[0].ops) == 0:
        for_loop.detach()

class ReplaceAbsoluteArrayIndexWithStencil(RewritePattern):
  @op_type_rewrite_pattern
  def match_and_rewrite(
            self, array_reference: psy_ir.ArrayReference, rewriter: PatternRewriter):

    stencil_relative_offsets=[]
    for accessor in array_reference.accessors.blocks[0].ops:
      visitor=CollectArrayRelativeOffsets()
      visitor.traverse(accessor)
      if visitor.offset_val is None:
        stencil_relative_offsets.append(IntAttr(0))
      else:
        stencil_relative_offsets.append(IntAttr(visitor.offset_val))

    parent=array_reference.parent
    idx = parent.ops.index(array_reference)
    array_reference.detach()
    access_op=psy_stencil.PsyStencil_Access.build(attributes={"var": array_reference.var, "stencil_ops": ArrayAttr(stencil_relative_offsets)})
    rewriter.insert_op_at_pos(access_op, parent, idx)

class ApplyStencilRewriter(RewritePattern):
    def __init__(self):
      self.called_procedures=[]

    def allIndexesFilled(self, index_dict):
      for v in index_dict.values():
        if v is None: return False
      return True

    def locate_assignment(self, ops, target_name):
      for op in ops:
        if isinstance(op, psy_ir.Assign):
          if op.lhs.blocks[0].ops[0].var.var_name.data == target_name:
            return op
      return None

    def handle_stencil_for_target(self, visitor, target_var_name, for_loop: psy_ir.Loop, rewriter: PatternRewriter):
        read_vars=[]
        access_variables=[]
        for read_var_name in visitor.written_to_read[target_var_name]:
          read_var_v=visitor.read_variables[read_var_name]
          read_vars.append(read_var_v.var)
          if isinstance(read_var_v, psy_ir.ArrayReference):
            for index in read_var_v.accessors.blocks[0].ops:
              v2=CollectArrayVariableIndexes()
              v2.traverse(index)
              access_variables.extend(v2.array_indexes)

        if len(access_variables) == 0: return None, None, None

        v3=CollectLoopsToReplace(access_variables)
        v3.traverse(for_loop)

        if self.allIndexesFilled(v3.applicable_indicies):
          loop_body=v3.bottom_loop.body.blocks[0].ops
          assign_op=self.locate_assignment(loop_body, target_var_name)
          assert assign_op is not None

          rhs=assign_op.rhs.blocks[0].ops[0]
          rhs.detach()

          replaceArrayIndexWithStencil=ReplaceAbsoluteArrayIndexWithStencil()
          walker = PatternRewriteWalker(GreedyRewritePatternApplier([replaceArrayIndexWithStencil]), apply_recursively=False)
          walker.rewrite_module(rhs)

          write_var=visitor.written_variables[target_var_name].var

          stencil_result=psy_stencil.PsyStencil_Result.build(attributes={"var": assign_op.lhs.blocks[0].ops[0].var, "stencil_ops": ArrayAttr([])}, regions=[[rhs]])
          stencil_op=psy_stencil.PsyStencil_Stencil.build(attributes={"input_fields": ArrayAttr(read_vars), "output_fields":ArrayAttr([write_var])}, regions=[[stencil_result]])
          return v3.top_loop, assign_op, stencil_op
        else:
          return None, None, None


    @op_type_rewrite_pattern
    def match_and_rewrite(
            self, for_loop: psy_ir.Loop, rewriter: PatternRewriter):

        visitor = CollectApplicableVariables()
        visitor.traverse(for_loop)

        for written_var in visitor.written_to_read.keys():
          top_loop, assignment_op, stencil_op=self.handle_stencil_for_target(visitor, written_var, for_loop, rewriter)
          if top_loop is not None and assignment_op is not None and stencil_op is not None:
            # Detach assignment op and then jam stencil into parent of top loop
            assignment_op.detach()
            top_loop.parent.add_op(stencil_op)
          else:
            # If one is none, ensure all are
            assert top_loop is None and assignment_op is None and stencil_op is None

        # Now go through and remove any subloops that are empty
        walker = PatternRewriteWalker(GreedyRewritePatternApplier([RemoveEmptyLoops()]), walk_regions_first=True)
        walker.rewrite_module(for_loop)


def apply_stencil_analysis(ctx: psy_ir.MLContext, module: ModuleOp) -> ModuleOp:
    applyStencilRewriter=ApplyStencilRewriter()
    walker = PatternRewriteWalker(GreedyRewritePatternApplier([applyStencilRewriter]), apply_recursively=False)
    walker.rewrite_module(module)

    #print(1 + "J")
    return module
