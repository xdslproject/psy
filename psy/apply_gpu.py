from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue, Region, Block
from xdsl.dialects.builtin import IntegerAttr, StringAttr, ArrayAttr, IntAttr
from xdsl.pattern_rewriter import (GreedyRewritePatternApplier,
                                   PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, op_type_rewrite_pattern)

from psy.dialects import psy_ir, hpc_gpu
from util.visitor import Visitor

class ApplyGPURewriter(RewritePattern):
    def __init__(self):
      self.called_procedures=[]
      
    def checkIfProgramProcedure(self, node):
      if isinstance(node, psy_ir.Routine):
        return node.is_program.data
      if node is None: return False
      return self.checkIfProgramProcedure(node.parent)
      
    def hasGPULoopInParents(self, node):      
      if isinstance(node, hpc_gpu.GPULoop): return True      
      if node is None: return False
      return self.hasGPULoopInParents(node.parent)

    @op_type_rewrite_pattern
    def match_and_rewrite(  # type: ignore reportIncompatibleMethodOverride
            self, for_loop: psy_ir.Loop, rewriter: PatternRewriter):

        block = for_loop.parent
        assert isinstance(block, Block)
        
        if (self.hasGPULoopInParents(block.parent)): return
        # This is hacked in currently, but sufficient for our initial needs, only consider loops not in
        # the program entry subroutine
        if self.checkIfProgramProcedure(block.parent): return

        idx = block.ops.index(for_loop)

        for_loop.detach()

        #visitor = CollectWrittenVariables()
        #visitor.traverse(for_loop)

        copy_in_vars=[]
        copy_out_vars=[]
        create_vars=[]
        private_vars=[]

        gpu_loop = hpc_gpu.GPULoop.get([for_loop])
        rewriter.insert_op_at_pos(gpu_loop, block, idx)        


def apply_gpu_analysis(ctx: psy_ir.MLContext, module: ModuleOp) -> ModuleOp:
    applyGPURewriter=ApplyGPURewriter()
    walker = PatternRewriteWalker(GreedyRewritePatternApplier([applyGPURewriter]), apply_recursively=False)
    walker.rewrite_module(module)

    #print(1 + "J")
    return module
