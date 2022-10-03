from psy.dialects import psy_ir
from xdsl.dialects.builtin import StringAttr, IntegerAttr
import sys

BINARY_OP_TO_SYMBOL={
  "ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/", "EQ": ".eq.", "NE": ".ne.", "GT": ".gt.", 
  "LT": ".lt.", "GE": ".ge.", "LE": ".le.", "AND": ".and.", "OR": ".or."
}

BINARY_OP_TO_INTRINSIC={
  "REM": "MOD", "POW": "POW", "SUM": "SUM", "SIGN": "SIGN", "MIN": "MIN", "MAX": "MAX",
  "REAL": "REAL", "INT": "INT", "CAST": "CAST", "SIZE": "SIZE", "LBOUND": "LBOUND", 
  "UBOUND": "UBOUND", "MATMUL": "MATMUL", "DOT_PRODUCT": "DOT_PRODUCT"
}

UNARY_OP_TO_SYMBOL={
  "MINUS": "-", "PLUS": "+", "NOT": ".not."
}

UNARY_OP_TO_INTRINSIC={
  "SQRT": "SQRT", "EXP": "EXP", "LOG": "LOG", "LOG10": "LOG10", "SUM": "SUM",
  "COS": "COS", "SIN": "SIN", "TAN": "TAN", "ACOS": "ACOS", "ASIN": "ASIN", "ATAN": "ATAN",
  "ABS": "ABS", "CEIL": "CEIL", "REAL": "REAL", "INT": "INT", "NINT": "NINT"
}

class FortranPrinter():
  def __init__(self):
    self.incr=0
    self.inside_expr=0;
    
  def print_op(self, op, stream=sys.stdout):    
    if isinstance(op, psy_ir.FileContainer):
      for child in op.children.blocks[0].ops:   
        self.print_op(child)
    elif isinstance(op, psy_ir.Container):
      self.print_container(op)
    elif isinstance(op, psy_ir.Routine):      
      self.print_routine(op)
    elif isinstance(op, psy_ir.VarDef):
      self.print_vardef(op)
    elif isinstance(op, psy_ir.CallExpr):
      self.print_callexpr(op)
    elif isinstance(op, psy_ir.Literal):
      self.print_literal(op)
    elif isinstance(op, psy_ir.BinaryOperation):   
      self.print_binaryoperation(op)
    elif isinstance(op, psy_ir.UnaryOperation):
      self.print_unaryoperation(op)
    elif isinstance(op, psy_ir.Assign):
      self.print_assign(op)
    elif isinstance(op, psy_ir.ExprName):          
      print(op.var.var_name.data, end="")          
    elif isinstance(op, psy_ir.IfBlock):
      self.print_if(op)
    elif isinstance(op, psy_ir.Loop):
      self.print_do(op)
    elif isinstance(op, psy_ir.Import):
      self.print_import(op)
    elif isinstance(op, psy_ir.MemberAccess):
      self.print_memberaccess(op)
    elif isinstance(op, psy_ir.ArrayAccess):
      self.print_arrayaccess(op)
    elif isinstance(op, psy_ir.Return):
      self.print_indent()
      print("return")
    else:
        raise Exception(f"Trying to print unknown operation '{op.name}'")
        
  def print_binaryoperation(self, op):
    if op.attributes["op"].data in BINARY_OP_TO_SYMBOL:
      self.print_op(op.lhs.blocks[0].ops[0])
      print(" "+BINARY_OP_TO_SYMBOL[op.attributes["op"].data]+" ", end="")
      self.print_op(op.rhs.blocks[0].ops[0])
    elif op.attributes["op"].data in BINARY_OP_TO_INTRINSIC:
      print(BINARY_OP_TO_INTRINSIC[op.attributes["op"].data]+"(", end="")
      self.print_op(op.lhs.blocks[0].ops[0])
      print(", ", end="")
      self.print_op(op.rhs.blocks[0].ops[0])
      print(")", end="")
    else:
      raise Exception(f"Unknown binary operation {op.attributes['op'].data}")
        
  def print_unaryoperation(self, op):
    if op.attributes["op"].data in UNARY_OP_TO_SYMBOL:
      print(UNARY_OP_TO_SYMBOL[op.attributes["op"].data]+" ", end="")
      self.print_op(op.expr.blocks[0].ops[0])          
    elif op.attributes["op"].data in UNARY_OP_TO_INTRINSIC:
      print(UNARY_OP_TO_INTRINSIC[op.attributes["op"].data]+"(", end="")
      self.print_op(op.expr.blocks[0].ops[0])
      print(")", end="")
    else:
      raise Exception(f"Unknown unary operation {op.attributes['op'].data}")
        
  def print_arrayaccess(self, op):
    self.print_op(op.var.blocks[0].ops[0])
    print("(", end="")
    needs_comma=False
    for member in op.accessors.blocks[0].ops:
      if (needs_comma): print(", ", end="")
      needs_comma=True
      self.inside_expr+=1
      self.print_op(member)
      self.inside_expr-=1
    print(")", end="")        
      
  def print_callexpr(self, op):
    if self.inside_expr==0: self.print_indent()
    print(f"call {op.func.data}(", end='')    
    for index, arg in enumerate(op.args.blocks[0].ops):
      if (index > 0): print(", ", end="")
      self.inside_expr+=1
      self.print_op(arg)
      self.inside_expr-=1
    print(")", end="")
    if self.inside_expr==0: print("") 
        
  def print_memberaccess(self, op):
    print(f"{op.var.var_name.data}", end="")
    for member in op.fields.data:
      print(f"%{member.data}", end="")
        
  def print_import(self, op):
    print(f"use {op.import_name.data}", end="")
    if len(op.specific_procedures.data) > 1:
      print(", only : ", end="")      
      for index, proc in enumerate(op.specific_procedures.data):
        if (index > 0): print(", ", end="")        
        print(proc.data, end="")      
    print("") # force a newline
        
  def print_vardef(self, op):
    self.print_indent()
    type_str=self.generate_typestring(op.var.type)
    type_str+=self.generate_vardeclaration_extra_type_info(op)
    print(f"{type_str} :: {op.var.var_name.data}")
    
  def generate_vardeclaration_extra_type_info(self, var_def):
    extra_info=""
    if var_def.is_proc_argument.data:    
      extra_info+=f", intent({var_def.intent.data})"
    return extra_info
    
  def generate_typestring(self, type):  
    if isinstance(type, psy_ir.DerivedType):
      type_str=f"type({type.parameters[0].data})"
    elif isinstance(type, psy_ir.ArrayType):
      type_str=self.generate_typestring(type.element_type)
      type_str+=", dimension("      
      for index, dim_size in enumerate(type.shape.data):
        if isinstance(dim_size, psy_ir.AnonymousAttr):
          if (index > 0): type_str+=(",")          
          type_str+=":"
      type_str+=")"
    else:      
      type_str=f"{type.parameters[0].data}"      
      if (not isinstance(type.parameters[1], psy_ir.EmptyAttr)) and (len(type.parameters[1].data) > 0):
        type_str+=f"(kind={type.parameters[1].data})"
      elif (not isinstance(type.parameters[2], psy_ir.EmptyAttr)) and (type_str=="float" or type_str=="integer") and type.parameters[2].data != 4:
        type_str+=f"({type.parameters[2].data})"
    return type_str  
        
  def print_container(self, op):    
    self.print_indent()
    print(f"module {op.container_name.data}")
    self.incr+=2
    for import_stmt in op.imports.blocks[0].ops:
      self.print_indent()
      self.print_op(import_stmt)
    print("")
    self.print_indent()
    print("implicit none")
    self.print_indent()
    print(f"{op.default_visibility.data}\n")
    self.print_container_level_routine_visibility("public", op.public_routines.data)
    self.print_container_level_routine_visibility("private", op.private_routines.data)  
    print("contains")      
    for block in op.routines.blocks[0].ops:
      self.print_op(block)        
    self.incr-=2
    self.print_indent()
    print(f"end module {op.container_name.data}\n")
    
  def print_container_level_routine_visibility(self, visibility, vis_list):
    if len(vis_list) > 0:
      self.print_indent()
      print(f"{visibility} :: ", end="")      
      for index, member in enumerate(vis_list):
        if (index > 0): print(", ", end="")        
        print(member.data, end="")
      print("\n")
        
  def print_literal(self, op):    
    literal_val=op.attributes["value"]
    if isinstance(literal_val, IntegerAttr):
      print(op.attributes["value"].parameters[0].data, end="")
    elif isinstance(literal_val, psy_ir.FloatAttr):
      print(op.attributes["value"].data, end="")      
        
  def print_assign(self, op):    
    self.print_indent()
    self.inside_expr+=1
    self.print_op(op.lhs.blocks[0].ops[0])
    self.inside_expr-=1
    print("=", end="")
    self.inside_expr+=1
    self.print_op(op.rhs.blocks[0].ops[0])
    self.inside_expr-=1
    print("") # For a new line      
        
  def print_if(self, op):    
    self.print_indent()
    print("if (", end="")
    self.inside_expr+=1
    self.print_op(op.cond.blocks[0].ops[0])
    self.inside_expr-=1
    print(") then")      
    self.incr+=2
    self.print_op(op.then.blocks[0].ops[0])
    self.incr-=2      
    if len(op.orelse.blocks[0].ops) > 0:
      self.print_indent()
      print("else")
      self.incr+=2
      self.print_op(op.orelse.blocks[0].ops[0])
      self.incr-=2
    self.print_indent()
    print("end if")
  
  def print_do(self, op):    
    self.print_indent()
    
    print("do ", end="")
    if not isinstance(op.variable, psy_ir.Token):
      raise Exception(f"Loop variable must be a token")
    self.print_token_identifier(op.variable)
    print("=", end="")
    self.inside_expr+=1
    self.print_op(op.start.blocks[0].ops[0])
    self.inside_expr-=1
    print(", ", end="")
    self.inside_expr+=1
    self.print_op(op.stop.blocks[0].ops[0])
    self.inside_expr-=1
    print(", ", end="")
    self.inside_expr+=1
    self.print_op(op.step.blocks[0].ops[0])
    self.inside_expr-=1
    print("")
    self.incr+=2
    self.print_op(op.body.blocks[0].ops[0])
    self.incr-=2
    self.print_indent()
    print("end do")
      
  def print_routine(self, op):
    print("")
    self.print_out_routine(op)
    
  def print_out_routine(self, op):
    self.print_indent()
    if op.is_program.data:
      print(f"program {op.routine_name.data}")    
    else:
      if op.isFunction():
        print("function", end="")
      else:
        print("subroutine", end="")
      print(f" {op.routine_name.data}(", end="")      
      for index, arg in enumerate(op.args.data):
        if (index > 0): print(", ", end="")
        print(arg.var_name.data, end="")
      print(")", end="")
      if op.isFunction():
        print(" result(", end="")
        self.print_token_identifier(op.return_var)        
        print(")", end="")
      print("") # Force newline
    self.incr+=2
    for import_stmt in op.imports.blocks[0].ops:
      self.print_indent()
      self.print_op(import_stmt)
    for block in op.local_var_declarations.blocks[0].ops:
      self.print_op(block)
    if len(op.local_var_declarations.blocks[0].ops) > 0: print("")
    for block in op.routine_body.blocks[0].ops:
      self.print_op(block)
    self.incr-=2
    self.print_indent()
    if op.is_program.data:
      print(f"end program {op.routine_name.data}")
    elif op.isFunction():
      print(f"end function {op.routine_name.data}")
    else:
      print(f"end subroutine {op.routine_name.data}")
      
  def print_indent(self):
    print(" "*self.incr, end="")
    
  def print_token_identifier(self, token):
    print(token.var_name.data, end="")

def print_fortran(instructions, stream=sys.stdout):
  fortran_printer=FortranPrinter()
  for op in instructions:
    fortran_printer.print_op(op, stream=stream)
    
