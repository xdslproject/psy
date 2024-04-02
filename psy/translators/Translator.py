from typing import List
from xdsl.ir import Operation
from psy.support import SSAValueCtx, ProgramState
from psy.dialects import psy_ir

class Translator:
  
    def __init__(self):
        self.__function_names = None
        self.__constant_names = None
              
    def setFnNames(self,fn_names):
        self.__function_names = fn_names
        
    def setConstNames(self,const_names):
        self.__constant_names = const_names       
         
    def exports(self,obj='functions'):
        match obj:
            case 'functions':
                return self.__function_names
            case 'constants':
                return self.__constant_names
            case _:
                return None
    
    def setName(self, name):
        self.__name = name
        
    def getName(self):
        return self.__name
        
    def translate(self, function_name : str, ctx: SSAValueCtx, call_expr: psy_ir.CallExpr, 
        program_state : ProgramState, is_expr=False) -> List[Operation]:
        pass
   
    @staticmethod
    def flatten_ops_to_list(ops):
        ops_list=[]
        for op in ops:
          ops_list.append(op)
        return ops_list