from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
from xdsl.ir import SSAValue
from xdsl.dialects import func
from xdsl.dialects.experimental import fir


class ProgramState:
  def __init__(self):
    self.module_name=None
    self.module_translators = {}
    self.routine_name=None
    self.return_block=None
    self.imports={}
    self.globals=[]
    self.global_fn_names=[]
    self.num_gpu_fns=0;
    self.requires_mpi=False
    self.stencil_intermediate_results={}
    
  def setRequiresMPI(self, requires_mpi):
    self.requires_mpi=requires_mpi

  def getRequiresMPI(self):
    return self.requires_mpi

  def setRoutineName(self, routine_name):
    self.routine_name=routine_name

  def unsetRoutineName(self):
    self.routine_name=None

  def getNumGPUFns(self):
    return self.num_gpu_fns

  def incrementNumGPUFns(self):
    self.num_gpu_fns+=1

  def getRoutineName(self):
    return self.routine_name

  def isInRoutine(self):
    return self.routine_name is not None

  def setModuleName(self, module_name):
    self.module_name=module_name

  def unsetModuleName(self):
    self.module_name=None

  def getModuleName(self):
    return self.module_name

  def isInModule(self):
    return self.module_name is not None

  def addModuleTranslator(self, module_translator: Translator):
    self.module_translators[module_translator.getName()] = module_translator

  def removeModuleTranslator(self, module_translator: Translator):
    self.module_translator[module_translator.getName()]=None

  def getModuleTranslator(self, module_name) -> Translator:
    return self.module_translators[module_name] 

  def setReturnBlock(self, rb):
    self.return_block=rb

  def hasReturnBlock(self):
    return self.return_block is not NoneAttr

  def getReturnBlock(self):
    return self.return_block

  def clearReturnBlock(self):
    self.return_block=None

  def addImport(self, container_name, routine_name):
    self.imports[routine_name]=container_name

  def hasImport(self, routine_name):
    return routine_name in self.imports

  def getImportModule(self, routine_name):
    return self.imports[routine_name]

  def clearImports(self):
    self.imports.clear()

  def appendToGlobal(self, glob, fn_name=None):
    self.globals.append(glob)
    if fn_name is not None:
      self.global_fn_names.append(fn_name)

  def getGlobals(self):
    return self.globals

  def hasGlobalFnName(self, fn_name):
    return fn_name in self.global_fn_names

  def clearStencilIntermediateResults(self):
    self.stencil_intermediate_results.clear()

  def addStencilIntermediateResult(self, name, ssa_result):
    self.stencil_intermediate_results[name]=ssa_result

  def getStencilIntermediateResult(self, name):
    return self.stencil_intermediate_results[name]
    
  def insertExternalFunctionToGlobalState(self, function_name, args, result_type):
      if not self.hasGlobalFnName(function_name):
        if result_type is not None:
          fn=func.FuncOp.external(function_name, args, [result_type])
        else:
          fn=func.FuncOp.external(function_name, args, [])
        self.appendToGlobal(fn, function_name)

@dataclass
class SSAValueCtx:
    """
    Context that relates identifiers from the AST to SSA values used in the flat representation.
    """
    dictionary: Dict[str, SSAValue] = field(default_factory=dict)
    parent_scope: Optional[SSAValueCtx] = None

    def __getitem__(self, identifier: str) -> Optional[SSAValue]:
        """Check if the given identifier is in the current scope, or a parent scope"""
        ssa_value = self.dictionary.get(identifier, None)
        if ssa_value:
            return ssa_value
        elif self.parent_scope:
            return self.parent_scope[identifier]
        else:
            return None

    def __setitem__(self, identifier: str, ssa_value: SSAValue):
        """Relate the given identifier and SSA value in the current scope"""
        if identifier in self.dictionary:
            raise Exception()
        else:
            self.dictionary[identifier] = ssa_value




