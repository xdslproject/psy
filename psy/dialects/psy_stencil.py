from __future__ import annotations
from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr

from xdsl.ir import MLContext, Operation
from xdsl.irdl import irdl_op_definition, SingleBlockRegion, AnyAttr, OpAttr


@dataclass
class Psy_Stencil:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(PsyStencil_Stencil)
        self.ctx.register_op(PsyStencil_Access)
        self.ctx.register_op(PsyStencil_Result)

# Operations
@irdl_op_definition
class PsyStencil_Stencil(Operation):
    name: str = "psy.stencil.stencil"

    input_fields: OpAttr[ArrayAttr]
    output_fields: OpAttr[ArrayAttr]
    body: SingleBlockRegion

@irdl_op_definition
class PsyStencil_Access(Operation):
    name: str = "psy.stencil.access"

    var: OpAttr[AnyAttr()]
    stencil_ops: OpAttr[ArrayAttr]

@irdl_op_definition
class PsyStencil_Result(Operation):
    name: str = "psy.stencil.result"

    var: OpAttr[AnyAttr()]
    stencil_ops: OpAttr[ArrayAttr]
    stencil_accesses: SingleBlockRegion
