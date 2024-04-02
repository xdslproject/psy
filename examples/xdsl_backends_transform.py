# -----------------------------------------------------------------------------
# BSD 3-Clause License
#
# Copyright (c) 2018-2022, Science and Technology Facilities Council.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------

from __future__ import print_function
import sys
from psyclone.psyir.backend.fortran import FortranWriter
from psyclone.psyir.backend.xdsl import xDSLWriter
from xdsl.printer import Printer
from xdsl.dialects.builtin import ModuleOp

def trans(psy):
    #fvisitor = FortranWriter()
    writer = xDSLWriter()
    printer = Printer(stream=sys.stdout)    
    routine_list=[]
    for invoke in psy.invokes.invoke_list:
        sched = invoke.schedule        
        routine_list.append(writer(sched))

    top_level=ModuleOp(routine_list)
    printer.print_op(top_level)

    f = open("psy_output.mlir", "w")
    p2=Printer(stream=f)
    p2.print_op(top_level)
    f.close()

    print("")
    print("")
    print("")

    '''
    print(dir(psy.container))
    print((psy.container.root))

    tranformed=writer(psy.container)

    top_level=ModuleOp.from_region_or_ops([tranformed])
    printer.print_op(top_level)


    schedule = psy.invokes.get('tra_adv').schedule
    # This seems strange, but need to do this as each kernel is separate and we want in the same module
    container_to_AST_mappings={}
    for idx, kern in enumerate(schedule.kernels()):
      module_name=kern.get_kernel_schedule().parent.name
      if module_name in container_to_AST_mappings.keys():
        kernel_subroutine_ast=writer(kern.get_kernel_schedule())
        container_to_AST_mappings[module_name].routines.blocks[0].ops.append(kernel_subroutine_ast)
      else:
        container_to_AST_mappings[module_name]=writer(kern.get_kernel_schedule().parent)

    containers = []

    for entry in container_to_AST_mappings.values():
      containers.append(entry)
    containers.append(psy_layer)
    top_level=ModuleOp.from_region_or_ops(containers)
    printer.print_op(top_level)
    '''
