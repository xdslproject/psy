from __future__ import annotations
from dataclasses import dataclass
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *
from xdsl.dialects.builtin import *

from typing import TypeAlias, List, cast, Type, Sequence, Optional

from xdsl.ir import (MLContext, TYPE_CHECKING, Data, ParametrizedAttribute,
                     Operation)
from xdsl.irdl import (irdl_attr_definition, attr_constr_coercion,
                       irdl_to_attr_constraint, irdl_op_definition, builder,
                       ParameterDef, SingleBlockRegionDef, TypeVar, Generic,
                       GenericData, AttrConstraint, Any, Attribute, Region,
                       VerifyException, AnyAttr)


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
    
    input_fields=AttributeDef(ArrayAttr)
    output_fields=AttributeDef(ArrayAttr)
    body=SingleBlockRegionDef()

@irdl_op_definition
class HStencil_Access(Operation):
    name: str = "hstencil.access"

    var = AttributeDef(AnyAttr())
    stencil_ops=AttributeDef(ArrayAttr)

@irdl_op_definition
class HStencil_Result(Operation):
    name: str = "hstencil.result"

    var = AttributeDef(AnyAttr())
    stencil_ops=AttributeDef(ArrayAttr)
    stencil_accesses=SingleBlockRegionDef()
