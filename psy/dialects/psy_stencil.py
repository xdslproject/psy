from __future__ import annotations
from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr, IntAttr
from psy.dialects.psy_ir import NamedType, DerivedType, ArrayType
from xdsl.ir import MLContext, Dialect
from xdsl.irdl import irdl_op_definition, SingleBlockRegion, AnyAttr, OpAttr, IRDLOperation, AnyOf

@irdl_op_definition
class PsyStencil_Stencil(IRDLOperation):
    name: str = "psy.stencil.stencil"

    input_fields: OpAttr[ArrayAttr]
    output_fields: OpAttr[ArrayAttr]
    body: SingleBlockRegion

@irdl_op_definition
class PsyStencil_Access(IRDLOperation):
    name: str = "psy.stencil.access"

    var: OpAttr[AnyAttr()]
    stencil_ops: OpAttr[ArrayAttr]

@irdl_op_definition
class PsyStencil_DimIndex(IRDLOperation):
    name: str = "psy.dimension.index"

    index: OpAttr[IntAttr]
    original_type : OpAttr[AnyOf([NamedType, DerivedType, ArrayType])]

@irdl_op_definition
class PsyStencil_DeferredArrayInfo(IRDLOperation):
    name: str = "psy.stencil.deferredarrayinfo"

    var: OpAttr[AnyAttr()]
    shape: OpAttr[ArrayAttr]

@irdl_op_definition
class PsyStencil_Result(IRDLOperation):
    name: str = "psy.stencil.result"

    from_bounds: OpAttr[ArrayAttr]
    to_bounds: OpAttr[ArrayAttr]
    min_relative_offset: OpAttr[ArrayAttr]
    max_relative_offset: OpAttr[ArrayAttr]
    out_field: OpAttr[AnyAttr()]
    input_fields: OpAttr[ArrayAttr]
    stencil_ops: OpAttr[ArrayAttr]
    stencil_accesses: SingleBlockRegion

psyStencil = Dialect([
  PsyStencil_Stencil,
  PsyStencil_Access,
  PsyStencil_DeferredArrayInfo,
  PsyStencil_Result,
], [])
