from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Type, Union
from xdsl.dialects.builtin import IntegerAttr, StringAttr, IntegerType, Float32Type, i32, f32, ArrayAttr, IntAttr
from xdsl.ir import Data, MLContext, Operation, ParametrizedAttribute
from xdsl.irdl import (AnyOf, AttributeDef, SingleBlockRegionDef, builder, AnyAttr, ResultDef, OperandDef,
                       irdl_attr_definition, irdl_op_definition, ParameterDef)  

@irdl_op_definition
class GPULoop(Operation):
  name = "hpc.gpu.gpuloop"
    
  loop = SingleBlockRegionDef()
    
  #vector_length = AttributeDef(IntAttr)
  #num_workers = AttributeDef(IntAttr)
  #num_inner_loops_to_collapse = AttributeDef(IntAttr)
    
  #copy_in_vars= SingleBlockRegionDef()
  #copy_out_vars= SingleBlockRegionDef()
  #create_vars= SingleBlockRegionDef()
  #private_vars= SingleBlockRegionDef()
    
  @staticmethod
  def get(loop: List[Operation],          
          verify_op: bool = True) -> ParallelLoop:
    res = GPULoop.build(regions=[loop])
    if verify_op:
        res.verify(verify_nested_ops=False)
    return res

  def verify_(self) -> None:
    pass
    
