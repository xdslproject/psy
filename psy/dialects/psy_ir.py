from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Type, Union

from xdsl.dialects.builtin import IntegerAttr, StringAttr, ArrayAttr, ArrayOfConstraint, AnyAttr, IntAttr, FloatAttr
from xdsl.ir import Data, MLContext, Operation, ParametrizedAttribute
from xdsl.irdl import (AnyOf, ParameterDef, irdl_attr_definition, irdl_op_definition
                       , OpAttr, SingleBlockRegion)
from xdsl.parser import Parser
from xdsl.printer import Printer

@irdl_attr_definition
class BoolAttr(Data[bool]):
    name = "bool"
    data: bool

    @staticmethod
    def parse_parameter(parser: Parser) -> BoolAttr:
        data = parser.parse_str_literal()
        if data == "True": return True
        if data == "False": return False
        raise Exception(f"bool parsing resulted in {data}")
        return None

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(f'"{self.data}"')

    @staticmethod
    def from_bool(data: bool) -> BoolAttr:
        return BoolAttr(data)

@irdl_attr_definition
class DerivedType(ParametrizedAttribute):
    name = "derivedtype"

    type : ParameterDef[StringAttr]

    @staticmethod
    def from_str(type: str) -> DerivedType:
        return DerivedType([StringAttr.from_str(type)])

    @staticmethod
    def from_string_attr(data: StringAttr) -> DerivedType:
        return DerivedType([data])

@irdl_attr_definition
class EmptyAttr(ParametrizedAttribute):
    name="empty"

@irdl_attr_definition
class NamedType(ParametrizedAttribute):
    name = "psy.ir.named_type"

    type_name : ParameterDef[StringAttr]
    kind : ParameterDef[AnyOf([StringAttr, EmptyAttr])]
    precision : ParameterDef[AnyOf([IntAttr, EmptyAttr])]

    def set_kind(self, kind):
      self.parameters[1]=kind

    def set_precision(self, precision):
      self.parameters[2]=precision

@irdl_attr_definition
class AnonymousAttr(ParametrizedAttribute):
    name = "anonymous"

@irdl_attr_definition
class DeferredAttr(ParametrizedAttribute):
    name = "deferred"

@irdl_attr_definition
class AssumedSizeAttr(ParametrizedAttribute):
    name = "assumed_size"

# Ideally would use vector, but need unknown dimension types (and ranges too!)
@irdl_attr_definition
class ArrayType(ParametrizedAttribute):
    name = "arraytype"

    shape : ParameterDef[ArrayAttr]
    element_type : ParameterDef[AnyOf([NamedType, DerivedType])]

    def get_num_dims(self) -> int:
        return len(self.parameters[0].data)

    def get_shape(self) -> List[int]:
        shape=[]
        for i in self.shape.data:
          if isinstance(i, DeferredAttr) or len(i.parameters) == 0:
            shape.append(DeferredAttr())
          else:
            shape.append(i.parameters[0].data)

        return shape

    @staticmethod
    def from_type_and_list(
            referenced_type: Attribute,
            shape: List[Union[int, IntegerAttr, AnonymousAttr]] = None) -> ArrayType:
        if shape is None:
            shape = [1]
        if (isinstance(referenced_type, str) and referenced_type in VarDef.TYPE_MAP_TO_PSY.keys()):
          type=VarDef.TYPE_MAP_TO_PSY[referenced_type]
        else:
          type=referenced_type
        return ArrayType([
            ArrayAttr.from_list([(IntegerAttr.build(d) if isinstance(d, int) else d) for d in shape]),
            type]
        )

    @staticmethod
    def from_params(
        referenced_type: Attribute,
        shape: ArrayAttr) -> ArrayType:
        return ArrayType([shape, referenced_type])

@irdl_op_definition
class FileContainer(Operation):
    name = "psy.ir.filecontainer"

    file_name: OpAttr[StringAttr]
    children: SingleBlockRegion

    @staticmethod
    def get(file_name: str,
            containers: List[Operation],
            verify_op: bool = True) -> FileContainer:
      res = FileContainer.build(attributes={"file_name": file_name}, regions=[containers])
      if verify_op:
        res.verify(verify_nested_ops=False)
      return res

    def verify_(self) -> None:
      pass

@irdl_op_definition
class Container(Operation):
    name = "psy.ir.container"

    container_name: OpAttr[StringAttr]
    imports: SingleBlockRegion
    routines: SingleBlockRegion
    default_visibility: OpAttr[StringAttr]
    is_program: OpAttr[BoolAttr]
    public_routines: OpAttr[ArrayAttr]
    private_routines: OpAttr[ArrayAttr]

    @staticmethod
    def get(container_name: str,
            default_visibility: str,
            public_routines: List[str],
            private_routines: List[str],
            imports: List[Operation],
            routines: List[Operation],
            verify_op: bool = True) -> Container:
      res = Container.build(attributes={"container_name": container_name, "default_visibility": default_visibility,
                                        "is_program": False, "public_routines": ArrayAttr.from_list(public_routines),
                                        "private_routines": ArrayAttr.from_list(private_routines)},
                                        regions=[imports, routines])
      if verify_op:
        res.verify(verify_nested_ops=False)
      return res

    def verify_(self) -> None:
      pass

@irdl_op_definition
class Import(Operation):
    name = "psy.ir.import"

    import_name: OpAttr[StringAttr]
    specific_procedures: OpAttr[ArrayAttr]

    @staticmethod
    def get(import_name: str,
            specific_procedures: List[str],
            verify_op: bool = True) -> Container:
      res = Import.build(attributes={"import_name": import_name, "specific_procedures": ArrayAttr.from_list(specific_procedures)})
      if verify_op:
        res.verify(verify_nested_ops=False)
      return res

    def verify_(self) -> None:
      pass

@irdl_op_definition
class Routine(Operation):
    name = "psy.ir.routine"

    routine_name: OpAttr[StringAttr]
    imports: SingleBlockRegion
    args: OpAttr[ArrayAttr]
    return_var: OpAttr[AnyAttr()]
    local_var_declarations: SingleBlockRegion
    routine_body: SingleBlockRegion
    is_program: OpAttr[BoolAttr]

    @staticmethod
    def get(routine_name: Union[str, StringAttr],
            return_var: Union[str, StringAttr],
            imports: List[Operation],
            args: List[Operation],
            local_var_declarations: List[Operation],
            routine_body: List[Operation],
            is_program=False,
            verify_op: bool = True) -> Routine:
        if return_var is None:
          return_var=EmptyToken()
        if isinstance(routine_name, str):
          routine_name=StringAttr(routine_name)

        res = Routine.build(attributes={"routine_name": routine_name, "return_var": return_var,
                                        "args": ArrayAttr.from_list(args), "is_program": BoolAttr(is_program)},
                                        regions=[imports, local_var_declarations, routine_body])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def isFunction(self):
      return not isinstance(self.attributes["return_var"], EmptyToken)

    def verify_(self) -> None:
      pass

@irdl_op_definition
class ArrayReference(Operation):
    name="psy.ir.array_reference"

    var: OpAttr[AnyAttr()]
    accessors: SingleBlockRegion

    @staticmethod
    def get(var,
            accessors: List[Operation],
            verify_op: bool = True) -> ExprName:
        res = ArrayReference.build(attributes={"var": var}, regions=[accessors])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res


@irdl_op_definition
class ExprName(Operation):
    name = "psy.ir.id_expr"

    id: OpAttr[StringAttr]
    var: OpAttr[AnyAttr()]

    @staticmethod
    def get(name: Union[str, StringAttr], v, verify_op: bool = True) -> ExprName:
        res = ExprName.build(attributes={"id": StringAttr(name), "var": v})
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

@irdl_attr_definition
class StructureMember(ParametrizedAttribute):
  name = "psy.ir.structure_member"

  member_name : ParameterDef[StringAttr]
  children : ParameterDef[AnyOf([AnyAttr(), EmptyAttr])]

@irdl_op_definition
class StructureReference(Operation):
    name = "psy.ir.structure_reference"

    var: OpAttr[AnyAttr()]
    member: OpAttr[AnyAttr()]

    @staticmethod
    def get(var, member: Union[str, StringAttr], verify_op: bool = True) -> ExprName:
        res = StructureReference.build(attributes={"var": var, "member": member})
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

@irdl_attr_definition
class Token(ParametrizedAttribute):
    name = "psy.ir.token"

    var_name : ParameterDef[StringAttr]
    type : ParameterDef[AnyOf([NamedType, DerivedType, ArrayType])]

@irdl_attr_definition
class EmptyToken(EmptyAttr):
    name = "psy.ir.emptytoken"

@irdl_op_definition
class VarDef(Operation):
    name = "psy.ir.var_def"

    var: OpAttr[AnyAttr()]
    is_proc_argument: OpAttr[BoolAttr]
    is_constant: OpAttr[BoolAttr]
    intent: OpAttr[StringAttr]

    @staticmethod
    def get(var,
            is_proc_argument: bool = False,
            is_constant: bool = False,
            intent: str = "",
            verify_op: bool = True) -> VarDef:
        #TODO: This is a bit nasty how we feed in both string and IR nodes, with arrays will be hard to fix though?
        res = VarDef.build(attributes={"var": var, "is_proc_argument": BoolAttr.from_bool(is_proc_argument),
          "is_constant": BoolAttr.from_bool(is_constant), "intent": StringAttr(intent)})
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass

@irdl_op_definition
class Assign(Operation):
    name = "psy.ir.assign"

    lhs: SingleBlockRegion
    rhs: SingleBlockRegion

    @staticmethod
    def get(lhs: Operation,
            rhs: Operation,
            verify_op: bool = True) -> Assign:
        res = Assign.build(regions=[[lhs], [rhs]])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass

@irdl_op_definition
class Literal(Operation):
    name = "psy.ir.literal"

    value: OpAttr[AnyOf([StringAttr, IntegerAttr, FloatAttr])]

    @staticmethod
    def get(value: Union[None, bool, int, str, float], width=None,
            verify_op: bool = True) -> Literal:
        if type(value) is int:
            attr = IntegerAttr.from_int_and_width(value, width)
        elif type(value) is float:
            attr = FloatAttr.from_float_and_width(value, width)
        elif type(value) is str:
            attr = StringAttr.from_str(value)
        else:
            raise Exception(f"Unknown literal of type {type(value)}")
        res = Literal.create(attributes={"value": attr})
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

@irdl_op_definition
class IfBlock(Operation):
    name = "psy.ir.ifblock"

    cond: SingleBlockRegion
    then: SingleBlockRegion
    orelse: SingleBlockRegion

    @staticmethod
    def get(cond: Operation,
            then: List[Operation],
            orelse: List[Operation],
            verify_op: bool = True) -> If:
        res = IfBlock.build(regions=[[cond], then, orelse])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass

@irdl_op_definition
class Loop(Operation):
    name = "psy.ir.loop"

    variable: OpAttr[AnyAttr()]
    start: SingleBlockRegion
    stop: SingleBlockRegion
    step: SingleBlockRegion
    body: SingleBlockRegion

    @staticmethod
    def get(variable,
            start: Operation,
            stop: Operation,
            step: Operation,
            body: List[Operation],
            verify_op: bool = True) -> If:
        res = Loop.build(attributes={"variable": variable}, regions=[[start], [stop], [step], body])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass

@irdl_op_definition
class Return(Operation):
    name = "psy.ir.return"

@irdl_op_definition
class BinaryOperation(Operation):
    name = "psy.ir.binaryoperation"

    op: OpAttr[StringAttr]
    lhs: SingleBlockRegion
    rhs: SingleBlockRegion

    @staticmethod
    def get_valid_ops() -> List[str]:
        return [
        # Arithmetic Operators. ('REM' is remainder AKA 'MOD' in Fortran.)
        'ADD', 'SUB', 'MUL', 'DIV', 'REM', 'POW', 'SUM',
        # Relational Operators
        'EQ', 'NE', 'GT', 'LT', 'GE', 'LE',
        # Logical Operators
        'AND', 'OR',
        # Other Maths Operators
        'SIGN', 'MIN', 'MAX',
        # Casting operators
        'REAL', 'INT', 'CAST',
        # Array Query Operators
        'SIZE', 'LBOUND', 'UBOUND',
        # Matrix and Vector Operators
        'MATMUL', 'DOT_PRODUCT']

    @staticmethod
    def get(op: str,
            lhs: Operation,
            rhs: Operation,
            verify_op: bool = True) -> BinaryExpr:
        res = BinaryOperation.build(attributes={"op": StringAttr(op)}, regions=[[lhs], [rhs]])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass

@irdl_op_definition
class UnaryOperation(Operation):
    name = "psy.ir.unaryoperation"

    op: OpAttr[StringAttr]
    expr: SingleBlockRegion

    @staticmethod
    def get_valid_ops() -> List[str]:
        return [
        # Arithmetic Operators
        'MINUS', 'PLUS', 'SQRT', 'EXP', 'LOG', 'LOG10', 'SUM',
        # Logical Operators
        'NOT',
        # Trigonometric Operators
        'COS', 'SIN', 'TAN', 'ACOS', 'ASIN', 'ATAN',
        # Other Maths Operators
        'ABS', 'CEIL',
        # Casting Operators
        'REAL', 'INT', 'NINT']

    @staticmethod
    def get(op: str,
            expr: Operation,
            verify_op: bool = True) -> BinaryExpr:
        res = UnaryOperation.build(attributes={"op": StringAttr(op)}, regions=[[expr]])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass

@irdl_op_definition
class NaryOperation(Operation):
    name = "psy.ir.naryoperation"

    op: OpAttr[StringAttr]
    expr: SingleBlockRegion

    @staticmethod
    def get_valid_ops() -> List[str]:
        return ['MIN', 'MAX', 'SUM']

    @staticmethod
    def get(op: str,
            args: List[Operation],
            verify_op: bool = True) -> BinaryExpr:
        res = UnaryOperation.build(attributes={"op": StringAttr(op)}, regions=[args])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass

@irdl_op_definition
class Range(Operation):
    name = "psy.ir.range"

    start: SingleBlockRegion
    stop: SingleBlockRegion
    step: SingleBlockRegion

    @staticmethod
    def get(start: List[Operation],
            stop: List[Operation],
            step: List[Operation],
            verify_op: bool = True) -> CallExpr:
        res = Range.build(regions=[start, stop, step])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass

@irdl_op_definition
class CallExpr(Operation):
    name = "psy.ir.call_expr"

    func: OpAttr[StringAttr]
    intrinsic: OpAttr[BoolAttr]
    type: OpAttr[AnyOf([NamedType, DerivedType, ArrayType, EmptyAttr])]
    args: SingleBlockRegion

    @staticmethod
    def get(func: str,
            args: List[Operation],
            type:EmptyAttr =EmptyAttr(),
            intrinsic: bool=False,
            verify_op: bool = True) -> CallExpr:
        res = CallExpr.build(regions=[args], attributes={"func": StringAttr(func), "type": type, "intrinsic": BoolAttr(intrinsic)})
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass

@dataclass
class psyIR:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_attr(BoolAttr)
        self.ctx.register_attr(AnonymousAttr)
        self.ctx.register_attr(DeferredAttr)
        self.ctx.register_attr(EmptyAttr)
        self.ctx.register_attr(AssumedSizeAttr)
        self.ctx.register_attr(DerivedType)
        self.ctx.register_attr(NamedType)
        self.ctx.register_attr(ArrayType)
        self.ctx.register_attr(Token)
        self.ctx.register_attr(EmptyToken)
        self.ctx.register_attr(StructureMember)

        self.ctx.register_op(FileContainer)
        self.ctx.register_op(Container)
        self.ctx.register_op(Routine)
        self.ctx.register_op(Import)
        self.ctx.register_op(Return)
        self.ctx.register_op(VarDef)
        self.ctx.register_op(Assign)
        self.ctx.register_op(IfBlock)
        self.ctx.register_op(Loop)
        self.ctx.register_op(Literal)
        self.ctx.register_op(ExprName)
        self.ctx.register_op(ArrayReference)
        self.ctx.register_op(StructureReference)
        self.ctx.register_op(BinaryOperation)
        self.ctx.register_op(UnaryOperation)
        self.ctx.register_op(NaryOperation)
        self.ctx.register_op(Range)
        self.ctx.register_op(CallExpr)

    @staticmethod
    def get_type(annotation: str) -> Operation:
        return TypeName.get(annotation)

    @staticmethod
    def get_statement_op_types() -> List[Type[Operation]]:
        statements: List[Type[Operation]] = [
            Assign, If, Do
        ]
        return statements + psyIR.get_expression_op_types()

    @staticmethod
    def get_expression_op_types() -> List[Type[Operation]]:
        return [
            BinaryExpr, CallExpr, Literal, ExprName, MemberAccess, ArrayAccess
        ]

    @staticmethod
    def get_type_op_types() -> List[Type[Operation]]:
        return []
