from abc import ABC
from typing import TypeVar, cast
from dataclasses import dataclass
from ftn.dialects import fir
from xdsl.utils.hints import isa
from xdsl.dialects.memref import MemRefType
from xdsl.ir import Operation, SSAValue, OpResult, Attribute, MLContext, Block, Region

from xdsl.pattern_rewriter import (RewritePattern, PatternRewriter,
                                   op_type_rewrite_pattern,
                                   PatternRewriteWalker,
                                   GreedyRewritePatternApplier)
from xdsl.dialects import builtin, func, llvm, arith

class RewriteFirConvert(RewritePattern):
  """
  Rewrites FIR convert operation into standard dialect equivalent
  """
  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: fir.Convert, rewriter: PatternRewriter, /):  
    in_type=op.value.typ
    out_type=op.results[0].typ
    if isinstance(in_type, builtin.Float32Type) and isinstance(out_type, 
      builtin.Float64Type) or isinstance(in_type, builtin.Float16Type) and isinstance(out_type, 
      builtin.Float64Type) or isinstance(in_type, builtin.Float16Type) and isinstance(out_type, 
      builtin.Float32Type) :
      new_conv=arith.ExtFOp.get(op.value, out_type)
      rewriter.replace_matched_op(new_conv)
      
    if isinstance(in_type, builtin.Float64Type) and isinstance(out_type, 
      builtin.Float32Type) or isinstance(in_type, builtin.Float64Type) and isinstance(out_type, 
      builtin.Float16Type) or isinstance(in_type, builtin.Float32Type) and isinstance(out_type, 
      builtin.Float16Type) :
      new_conv=arith.TruncFOp.get(op.value, out_type)
      rewriter.replace_matched_op(new_conv)

def rewrite_fir_to_standard(ctx: MLContext, module: builtin.ModuleOp):
    walker = PatternRewriteWalker(GreedyRewritePatternApplier([
              RewriteFirConvert(),
    ]),
                                   apply_recursively=True)
    walker.rewrite_module(module.regions[0].blocks[0].ops[0])
