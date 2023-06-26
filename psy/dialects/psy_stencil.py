from __future__ import annotations
from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr, IntAttr
from psy.dialects.psy_ir import NamedType, DerivedType, ArrayType
from xdsl.traits import NoTerminator, IsTerminator
from xdsl.ir import MLContext, Dialect
from xdsl.irdl import irdl_op_definition, SingleBlockRegion, AnyAttr, attr_def, IRDLOperation, AnyOf, region_def, opt_attr_def

@irdl_op_definition
class PsyStencil_Stencil(IRDLOperation):
    name: str = "psy.stencil.stencil"

    input_fields= attr_def(ArrayAttr)
    output_fields= attr_def(ArrayAttr)
    from_bounds= attr_def(ArrayAttr)
    to_bounds= attr_def(ArrayAttr)
    min_relative_offset= attr_def(ArrayAttr)
    max_relative_offset= attr_def(ArrayAttr)
    body: SingleBlockRegion = region_def("single_block")

    traits = frozenset([NoTerminator()])

@irdl_op_definition
class PsyStencil_Access(IRDLOperation):
    name: str = "psy.stencil.access"

    var= attr_def(AnyAttr())
    stencil_ops= attr_def(ArrayAttr)
    op_mapping=opt_attr_def(ArrayAttr)

@irdl_op_definition
class PsyStencil_DimIndex(IRDLOperation):
    name: str = "psy.dimension.index"

    index= attr_def(IntAttr)
    original_type = attr_def(AnyOf([NamedType, DerivedType, ArrayType]))

@irdl_op_definition
class PsyStencil_DeferredArrayInfo(IRDLOperation):
    name: str = "psy.stencil.deferredarrayinfo"

    var= attr_def(AnyAttr())
    shape= attr_def(ArrayAttr)

@irdl_op_definition
class PsyStencil_Result(IRDLOperation):
    name: str = "psy.stencil.result"

    out_field= attr_def(AnyAttr())
    input_fields= attr_def(ArrayAttr)
    stencil_ops= attr_def(ArrayAttr)
    stencil_accesses: SingleBlockRegion = region_def("single_block")

    traits = frozenset([NoTerminator()])

@irdl_op_definition
class PsyStencil_IntermediateResult(IRDLOperation):
    name: str = "psy.stencil.intermediate_result"

    out_field= attr_def(AnyAttr())
    input_fields= attr_def(ArrayAttr)
    stencil_ops= attr_def(ArrayAttr)
    stencil_accesses: SingleBlockRegion = region_def("single_block")

    traits = frozenset([NoTerminator()])

psyStencil = Dialect([
  PsyStencil_Stencil,
  PsyStencil_Access,
  PsyStencil_DeferredArrayInfo,
  PsyStencil_Result,
], [])
