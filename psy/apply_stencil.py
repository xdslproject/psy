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

class GetVariableValue(Visitor):
  def __init__(self, var_name, dag_top_level):
    self.var_name=var_name
    self.dag_top_level=dag_top_level
    self.var_value=None

  def traverse_assign(self, assign:psy_ir.Assign):
    if isinstance(assign.lhs.blocks[0].ops[0], psy_ir.ExprName):
      if (assign.lhs.blocks[0].ops[0].var.var_name.data == self.var_name):
        cv=GetConstantValue(self.dag_top_level)
        cv.traverse(assign.rhs.blocks[0].ops[0])
        self.var_value=cv.literal

class GetConstantValue(Visitor):
  def __init__(self, dag_top_level):
    self.literal=None
    self.store=None
    self.active_op=0
    self.dag_top_level=dag_top_level

  def traverse_literal(self, literal: psy_ir.Literal):
    if self.active_op > 0:
      self.store=literal.value.value.data
    else:
      self.literal=literal.value.value.data

  def traverse_binary_operation(self, bin_op: psy_ir.BinaryOperation):
    self.active_op+=1
    self.traverse(bin_op.lhs.blocks[0].ops[0])
    lhs_v=self.store
    self.store=None
    self.traverse(bin_op.rhs.blocks[0].ops[0])
    if bin_op.op.data == "SUB":
      self.store=lhs_v - self.store
    elif bin_op.op.data == "ADD":
      self.store=lhs_v - self.store
    elif bin_op.op.data == "MUL":
      self.store=lhs_v * self.store
    elif bin_op.op.data == "DIV":
      self.store=lhs_v / self.store
    else:
      raise Exception(f"Unable to handle binary operation '{bin_op.op.data}' in range extraction")
    self.active_op-=1
    if self.active_op == 0: self.literal=self.store

  def traverse_expr_name(self, id_expr: psy_ir.ExprName):
    find_var_val=GetVariableValue(id_expr.var.var_name.data, self.dag_top_level)
    find_var_val.traverse(self.dag_top_level)
    if find_var_val.var_value is None:
      raise Exception(f"Can not find value for variable {id_expr.var.var_name.data}")
    else:
      self.store=find_var_val.var_value
      if self.active_op == 0: self.literal=self.store

  def traverse_assign(self, assign:psy_ir.Assign):
    self.active_op+=1
    for op in assign.rhs.blocks[0].ops:
      self.traverse(op)
    self.active_op-=1
    if self.active_op == 0: self.literal=self.store

class LocateAssignment(Visitor):
  def __init__(self, name):
    self.name=name
    self.assign=[]

  def traverse_assign(self, assign:psy_ir.Assign):
    if (assign.lhs.blocks[0].ops[0].var.var_name.data == self.name):
      self.assign.append(assign)

class CollectLoopsWithVariableName(Visitor):
  def __init__(self, indicies):
    self.applicable_indicies = {index: None for index in indicies}
    self.located_loops={}

  def traverse_loop(self, dl: psy_ir.Loop):
    if (dl.variable.var_name.data in self.applicable_indicies):
      self.located_loops[dl.variable.var_name.data]=dl
    for op in dl.body.blocks[0].ops:
      self.traverse(op)

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
    self.ordered_writes={}
    self.write_index=0
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
    self.written_to_read[self.write_index]=set(self.current_read_variables)
    self.ordered_writes[self.write_index]=self.current_written_variable
    self.write_index+=1
    self.current_written_variable=None
    self.current_read_variables=[]

class DetermineMinMaxRelativeOffsetsAcrossStencilAccesses(Visitor):
  def __init__(self, index):
    self.min=None
    self.max=None
    self.index=index

  def traverse_psy_stencil__access(self, stencil_access: psy_stencil.PsyStencil_Access):
    if len(stencil_access.stencil_ops.data) > self.index:
      val=stencil_access.stencil_ops.data[self.index].data
      if self.min is None or self.min > val: self.min=val
      if self.max is None or self.max < val: self.max=val

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

    def build_bounds(index_variable_names, constants_dir, index):
      bounds=[]
      for var_name in index_variable_names:
        if constants_dir[var_name][index] is not None:
          bounds.append(IntAttr(constants_dir[var_name][index]))
      return bounds

    def get_dag_top_level(node):
      if node.parent == None: return node
      return ApplyStencilRewriter.get_dag_top_level(node.parent)

    def handle_stencil_for_target(self, visitor, index, target_var_name, for_loop: psy_ir.Loop, rewriter: PatternRewriter, unique_var_idx:int):
        read_vars=[]
        access_variables=[]
        for read_var_name in visitor.written_to_read[index]:
          read_var_v=visitor.read_variables[read_var_name]
          read_vars.append(read_var_v.var)
          if isinstance(read_var_v, psy_ir.ArrayReference):
            for idx in read_var_v.accessors.blocks[0].ops:
              v2=CollectArrayVariableIndexes()
              v2.traverse(idx)
              access_variables.extend(v2.array_indexes)

        if len(access_variables) == 0: return None, None, None

        written_var=visitor.written_variables[target_var_name]
        # Needs to be an array that we are writing into
        assert isinstance(written_var, psy_ir.ArrayReference)
        index_variable_names=[]
        for idx in written_var.accessors.blocks[0].ops:
          v2=CollectArrayVariableIndexes()
          v2.traverse(idx)
          index_variable_names.extend(v2.array_indexes)

        # Now we collect the literal range (lb, ub) that each written to index is operating over
        loop_var_name=CollectLoopsWithVariableName(index_variable_names)
        loop_var_name.traverse(for_loop)
        loop_numeric_bounds={}
        for key, value in loop_var_name.located_loops.items():
            top_level=ApplyStencilRewriter.get_dag_top_level(value.start.blocks[0].ops[0])
            from_bound=GetConstantValue(top_level)
            from_bound.traverse(value.start.blocks[0].ops[0])

            to_bound=GetConstantValue(top_level)
            to_bound.traverse(value.stop.blocks[0].ops[0])

            loop_numeric_bounds[key]=[from_bound.literal, to_bound.literal]

        v3=CollectLoopsToReplace(access_variables)
        v3.traverse(for_loop)

        if self.allIndexesFilled(v3.applicable_indicies):
          loop_body=v3.bottom_loop.body.blocks[0].ops

          # This is needed for multiple assignments with the same name, thats why we have unique_var_idx
          # here, which is tracked from the caller. This is the index per variable name, so we can jump
          # to the corresponding assignment and use it
          v=LocateAssignment(target_var_name)
          v.traverse(for_loop)
          assert len(v.assign) != 0
          if len(v.assign) == 1:
            assign_op=v.assign[0]
          else:
            assign_op=v.assign[unique_var_idx]
          assert assign_op is not None

          rhs=assign_op.rhs.blocks[0].ops[0]
          rhs.detach()

          replaceArrayIndexWithStencil=ReplaceAbsoluteArrayIndexWithStencil()
          walker = PatternRewriteWalker(GreedyRewritePatternApplier([replaceArrayIndexWithStencil]), apply_recursively=False)
          walker.rewrite_module(rhs)

          min_indicies=[]
          max_indicies=[]

          for i in range(len(index_variable_names)):
            offset_indexes=DetermineMinMaxRelativeOffsetsAcrossStencilAccesses(i)
            offset_indexes.traverse(rhs)
            if offset_indexes.min is not None:
              min_indicies.append(IntAttr(offset_indexes.min))
            if offset_indexes.max is not None:
              max_indicies.append(IntAttr(offset_indexes.max))

          # Grab the lower and upper bounds for the stencil application
          lb=ApplyStencilRewriter.build_bounds(index_variable_names, loop_numeric_bounds, 0)
          ub=ApplyStencilRewriter.build_bounds(index_variable_names, loop_numeric_bounds, 1)

          assert len(min_indicies) == len(lb)
          assert len(max_indicies) == len(ub)
          assert len(lb) == len(ub)
          assert len(min_indicies) == len(max_indicies)

          write_var=visitor.written_variables[target_var_name].var

          # For now assume only one result per stencil, hence use same stencil read_vars as input_fields to stencil result
          stencil_result=psy_stencil.PsyStencil_Result.build(attributes={"out_field": assign_op.lhs.blocks[0].ops[0].var,
              "input_fields": ArrayAttr(read_vars), "stencil_ops": ArrayAttr([]),
              "from_bounds": ArrayAttr(lb), "to_bounds": ArrayAttr(ub),
              "min_relative_offset": ArrayAttr(min_indicies), "max_relative_offset": ArrayAttr(max_indicies)}, regions=[[rhs]])
          stencil_op=psy_stencil.PsyStencil_Stencil.build(attributes={"input_fields": ArrayAttr(read_vars), "output_fields":ArrayAttr([write_var])}, regions=[[stencil_result]])
          return v3.top_loop, assign_op, stencil_op
        else:
          return None, None, None


    @op_type_rewrite_pattern
    def match_and_rewrite(
            self, for_loop: psy_ir.Loop, rewriter: PatternRewriter):

        visitor = CollectApplicableVariables()
        visitor.traverse(for_loop)

        unique_written_vars={}

        for index, written_var in visitor.ordered_writes.items():
          unique_var_idx=unique_written_vars.get(written_var, 0)
          top_loop, assignment_op, stencil_op=self.handle_stencil_for_target(visitor, index, written_var, for_loop, rewriter, unique_var_idx)
          unique_written_vars[written_var]=unique_var_idx+1
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
