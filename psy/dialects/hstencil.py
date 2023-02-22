from __future__ import annotations
from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr

from xdsl.ir import MLContext, Operation
from xdsl.irdl import irdl_op_definition, SingleBlockRegion, AnyAttr, OpAttr


@dataclass
class HStencil:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(HStencil_Stencil)
        self.ctx.register_op(HStencil_Access)
        self.ctx.register_op(HStencil_Result)

# Operations
@irdl_op_definition
class HStencil_Stencil(Operation):
    name: str = "hstencil.stencil"

    input_fields: OpAttr[ArrayAttr]
    output_fields: OpAttr[ArrayAttr]
    body: SingleBlockRegion

@irdl_op_definition
class HStencil_Access(Operation):
    name: str = "hstencil.access"

    var: OpAttr[AnyAttr()]
    stencil_ops: OpAttr[ArrayAttr]

@irdl_op_definition
class HStencil_Result(Operation):
    name: str = "hstencil.result"

    var: OpAttr[AnyAttr()]
    stencil_ops: OpAttr[ArrayAttr]
    stencil_accesses: SingleBlockRegion
