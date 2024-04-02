from typing import List
from xdsl.ir import Operation
from xdsl.dialects import mpi
from psy.support import SSAValueCtx, ProgramState
from psy.translators.Translator import Translator
from psy.dialects import psy_ir


class MpiTranslator(Translator):

    def __init__(self):
        super().__init__()
        self.setName('mpi')
        self.setFnNames([ 'mpi_commrank', 'mpi_commsize', 'mpi_send', 'mpi_isend', 'mpi_recv', 'mpi_irecv', 'mpi_wait',
        'mpi_waitall', 'mpi_reduce', 'mpi_allreduce', 'mpi_bcast', 'mpi_init','mpi_finalize'])


    def translate(self, function_name: str, ctx: SSAValueCtx,
                                 call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
        if function_name.lower() == "mpi_commrank":
          return self.translate_mpi_commrank_function_call_expr(ctx, call_expr, program_state, is_expr)
        elif function_name.lower() == "mpi_commsize":
          return self.translate_mpi_commsize_function_call_expr(ctx, call_expr, program_state, is_expr)
        elif function_name.lower() == "mpi_send":
          return self.translate_mpi_send_function_call_expr(ctx, call_expr, program_state, is_expr)
        elif function_name.lower() == "mpi_isend":
          return self.translate_mpi_send_function_call_expr(ctx, call_expr, program_state, is_expr, False)
        elif function_name.lower() == "mpi_recv":
          return self.translate_mpi_recv_function_call_expr(ctx, call_expr, program_state, is_expr)
        elif function_name.lower() == "mpi_irecv":
          return self.translate_mpi_recv_function_call_expr(ctx, call_expr, program_state, is_expr, False)
        elif function_name.lower() == "mpi_wait":
          return self.translate_mpi_wait_function_call_expr(ctx, call_expr, program_state, is_expr)
        elif function_name.lower() == "mpi_waitall":
          return self.translate_mpi_waitall_function_call_expr(ctx, call_expr, program_state, is_expr)
        elif function_name.lower() == "mpi_reduce":
          return self.translate_mpi_reduce_function_call_expr(ctx, call_expr, program_state, is_expr)
        elif function_name.lower() == "mpi_allreduce":
          return self.translate_mpi_allreduce_function_call_expr(ctx, call_expr, program_state, is_expr)
        elif function_name.lower() == "mpi_bcast":
          return self.translate_mpi_bcast_function_call_expr(ctx, call_expr, program_state, is_expr)
        elif function_name.lower() == "mpi_init" or function_name.lower() == "mpi_finalize":
          program_state.setRequiresMPI(True)
          return []

        # The function we were looking for isn't in the MPI module
        return None;


    def translate_mpi_bcast_function_call_expr(self, ctx: SSAValueCtx,
                                 call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
        program_state.setRequiresMPI(True)
        assert len(call_expr.args.blocks[0].ops) == 3

        ops_list=Translator.flatten_ops_to_list(call_expr.args.blocks[0].ops)

        ptr_type, buffer_op, buffer_arg = translate_mpi_buffer(ctx, call_expr.args.blocks[0].ops.first, program_state)
        convert_buffer=fir.Convert.create(operands=[buffer_arg],
                        result_types=[fir.LLVMPointerType([ptr_type])])
        count_op, count_arg = translate_expr(ctx, ops_list, program_state)
        get_mpi_dtype_op=mpi.GetDtypeOp(ptr_type)
        root_op, root_arg = translate_expr(ctx, ops_list, program_state)

        result_ops=count_op + root_op + [convert_buffer, get_mpi_dtype_op]

        bcast_op=mpi.Bcast(convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], root_arg)
        result_ops.append(bcast_op)

        return result_ops


    def translate_mpi_reduce_function_call_expr(self, ctx: SSAValueCtx,
                                 call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
        program_state.setRequiresMPI(True)
        assert len(call_expr.args.blocks[0].ops) == 5

        ops_list=Translator.flatten_ops_to_list(call_expr.args.blocks[0].ops)
        assert isinstance(ops_list[3], psy_ir.Literal)

        send_ptr_type, send_buffer_op, send_buffer_arg = translate_mpi_buffer(ctx, ops_list[0], program_state)
        recv_ptr_type, recv_buffer_op, recv_buffer_arg = translate_mpi_buffer(ctx, ops_list[1], program_state)

        send_convert_buffer=fir.Convert.create(operands=[send_buffer_arg],
                        result_types=[fir.LLVMPointerType([send_ptr_type])])
        recv_convert_buffer=fir.Convert.create(operands=[recv_buffer_arg],
                        result_types=[fir.LLVMPointerType([recv_ptr_type])])
        count_op, count_arg = translate_expr(ctx, ops_list[2], program_state)
        get_mpi_dtype_op=mpi.GetDtypeOp(send_ptr_type) # Do this on the send buffer type

        mpi_op=str_to_mpi_operation[ops_list[3].value.data]
        root_op, root_arg = translate_expr(ctx, ops_list[4], program_state)

        result_ops=count_op + root_op + [send_convert_buffer, recv_convert_buffer, get_mpi_dtype_op]

        mpi_reduce_op=mpi.Reduce(send_convert_buffer.results[0], recv_convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], mpi_op, root_arg)
        result_ops.append(mpi_reduce_op)

        return result_ops

    def translate_mpi_allreduce_function_call_expr(self, ctx: SSAValueCtx,
                                 call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
        program_state.setRequiresMPI(True)
        assert len(call_expr.args.blocks[0].ops) == 4 or len(call_expr.args.blocks[0].ops) == 3

        has_send_buffer=len(call_expr.args.blocks[0].ops) == 4

        if has_send_buffer:
          send_buffer_idx=0
          recv_buffer_idx=1
          count_idx=2
          mpi_op_idx=3
        else:
          recv_buffer_idx=0
          count_idx=1
          mpi_op_idx=2

        ops_list=Translator.flatten_ops_to_list(call_expr.args.blocks[0].ops)

        assert isinstance(ops_list[mpi_op_idx], psy_ir.Literal)

        if has_send_buffer:
          send_ptr_type, send_buffer_op, send_buffer_arg = translate_mpi_buffer(ctx, ops_list[send_buffer_idx], program_state)
          send_convert_buffer=fir.Convert.create(operands=[send_buffer_arg],
                        result_types=[fir.LLVMPointerType([send_ptr_type])])

        recv_ptr_type, recv_buffer_op, recv_buffer_arg = translate_mpi_buffer(ctx, ops_list[recv_buffer_idx], program_state)
        recv_convert_buffer=fir.Convert.create(operands=[recv_buffer_arg],
                        result_types=[fir.LLVMPointerType([recv_ptr_type])])
        count_op, count_arg = translate_expr(ctx, ops_list[count_idx], program_state)
        get_mpi_dtype_op=mpi.GetDtypeOp(recv_ptr_type) # Do this on the recv buffer type

        mpi_op=str_to_mpi_operation[ops_list[mpi_op_idx].value.data]

        if has_send_buffer:
          result_ops=count_op + [send_convert_buffer, recv_convert_buffer, get_mpi_dtype_op]
          mpi_reduce_op=mpi.Allreduce(send_convert_buffer.results[0], recv_convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], mpi_op)
        else:
          result_ops=count_op + [recv_convert_buffer, get_mpi_dtype_op]
          mpi_reduce_op=mpi.Allreduce(None, recv_convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], mpi_op)

        result_ops.append(mpi_reduce_op)

        return result_ops

    def translate_mpi_buffer(self, ctx: SSAValueCtx, ops: psy_ir.ExprName, program_state : ProgramState):
        assert isinstance(ops, psy_ir.ExprName)
        ptr_type=try_translate_type(ops.var.type)
        # Pointer type needs to be base type which might be wrapped in an array
        if isinstance(ptr_type, fir.SequenceType): ptr_type=ptr_type.type

        buffer_op, buffer_arg = translate_expr(ctx, ops, program_state)
        if not isinstance(buffer_arg.type, fir.ReferenceType):
          if isinstance(buffer_op, list) and len(buffer_op) == 1 and isinstance(buffer_op[0], fir.Load):
            # We do this as translate expression assumes we want to use the value rather than the reference,
            # so it loads the value from the fir.referencetype, hence we go in and grab the reference type.
            # This is needed if a scalar is passed to the call as the buffer argument
            buffer_arg=buffer_op[0].memref
          else:
            raise Exception(f"Unable to process MPI argument`{buffer_arg}' as it is not a reference and can not be translated")
        return ptr_type, buffer_op, buffer_arg


    def translate_mpi_wait_function_call_expr(self, ctx: SSAValueCtx,
                                 call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
        program_state.setRequiresMPI(True)
        assert len(call_expr.args.blocks[0].ops) == 1

        to_return=[]
        request_op, request_arg = translate_expr(ctx, call_expr.args.blocks[0].ops.first, program_state)
        if request_op is not None: to_return+=request_op

        if isinstance(request_arg.type, mpi.VectorType):
          element_index=arith.Constant.create(properties={"value": IntegerAttr.from_int_and_width(0, 32)},
                                             result_types=[i32])
          load_op=mpi.VectorGetOp(request_arg, element_index.results[0])
          wait_op=mpi.Wait(load_op.results[0])
          to_return+=[element_index, load_op, wait_op]
        else:
          wait_op=mpi.Wait(request_arg)
          to_return.append(wait_op)

        return to_return


    def translate_mpi_waitall_function_call_expr(self, ctx: SSAValueCtx,
                                 call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
        program_state.setRequiresMPI(True)
        assert len(call_expr.args.blocks[0].ops) == 2

        ops_list=Translator.flatten_ops_to_list(call_expr.args.blocks[0].ops)

        request_ops, request_args = translate_expr(ctx, ops_list[0], program_state)
        count_op, count_arg = translate_expr(ctx, ops_list[1], program_state)
        wait_op=mpi.Waitall(request_args, count_arg)
        return count_op + [wait_op]


    def translate_mpi_send_function_call_expr(self, ctx: SSAValueCtx,
                                 call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False, blocking=True) -> List[Operation]:
        program_state.setRequiresMPI(True)
        if blocking:
          assert len(call_expr.args.blocks[0].ops) == 4
        else:
          assert len(call_expr.args.blocks[0].ops) == 5

        ops_list=Translator.flatten_ops_to_list(call_expr.args.blocks[0].ops)

        assert isinstance(ops_list[0], psy_ir.ExprName)

        ptr_type, buffer_op, buffer_arg = translate_mpi_buffer(ctx, ops_list[0], program_state)

        convert_buffer=fir.Convert.create(operands=[buffer_arg],
                        result_types=[fir.LLVMPointerType([ptr_type])])
        count_op, count_arg = translate_expr(ctx, ops_list[1], program_state)
        target_op, target_arg = translate_expr(ctx, ops_list[2], program_state)
        tag_op, tag_arg = translate_expr(ctx, ops_list[3], program_state)
        get_mpi_dtype_op=mpi.GetDtypeOp(ptr_type)

        result_ops=count_op + target_op + tag_op + [convert_buffer, get_mpi_dtype_op]

        if blocking:
          mpi_send_op=mpi.Send(convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], target_arg, tag_arg)
          result_ops.append(mpi_send_op)
        else:
          request_op, request_arg = translate_expr(ctx, ops_list[4], program_state)
          if request_op is not None: result_ops+=request_op

          if isinstance(request_arg.type, mpi.VectorType):
            element_index=arith.Constant.create(properties={"value": IntegerAttr.from_int_and_width(0, 32)},
                                             result_types=[i32])
            load_op=mpi.VectorGetOp(request_arg, element_index.results[0])
            mpi_send_op=mpi.Isend(convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], target_arg, tag_arg, load_op.results[0])
            result_ops+=[element_index, load_op, mpi_send_op]
          else:
            mpi_send_op=mpi.Isend(convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], target_arg, tag_arg, request_arg)
            result_ops.append(mpi_send_op)

        return result_ops

    def translate_mpi_recv_function_call_expr(self, ctx: SSAValueCtx,
                                 call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False, blocking=True) -> List[Operation]:
        program_state.setRequiresMPI(True)
        if blocking:
          assert len(call_expr.args.blocks[0].ops) == 4
        else:
          assert len(call_expr.args.blocks[0].ops) == 5

        ops_list=Translator.flatten_ops_to_list(call_expr.args.blocks[0].ops)

        assert isinstance(ops_list[0], psy_ir.ExprName)

        ptr_type, buffer_op, buffer_arg = translate_mpi_buffer(ctx, ops_list[0], program_state)

        convert_buffer=fir.Convert.create(operands=[buffer_arg],
                        result_types=[fir.LLVMPointerType([ptr_type])])
        count_op, count_arg = translate_expr(ctx, ops_list[1], program_state)
        source_op, source_arg = translate_expr(ctx, ops_list[2], program_state)
        tag_op, tag_arg = translate_expr(ctx, ops_list[3], program_state)
        get_mpi_dtype_op=mpi.GetDtypeOp(ptr_type)

        result_ops=count_op + source_op + tag_op + [convert_buffer, get_mpi_dtype_op]

        if blocking:
          mpi_recv_op=mpi.Recv(convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], source_arg, tag_arg)
          result_ops.append(mpi_recv_op)
        else:
          request_op, request_arg = translate_expr(ctx, ops_list[4], program_state)
          if request_op is not None: result_ops+=request_op

          if isinstance(request_arg.type, mpi.VectorType):
            element_index=arith.Constant.create(properties={"value": IntegerAttr.from_int_and_width(0, 32)},
                                             result_types=[i32])
            load_op=mpi.VectorGetOp(request_arg, element_index.results[0])
            mpi_recv_op=mpi.Irecv(convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], source_arg, tag_arg, load_op.results[0])
            result_ops+=[element_index, load_op, mpi_recv_op]
          else:
            mpi_recv_op=mpi.Irecv(convert_buffer.results[0], count_arg, get_mpi_dtype_op.results[0], source_arg, tag_arg, request_arg)
            result_ops.append(mpi_recv_op)

        return result_ops

    def type_to_mpi_datatype(typ):
      if typ==i32:
        return mpi.MPI_INT
      raise Exception(f"Could not translate type`{typ}' to MPI datatype as this unknown")

    def translate_mpi_commrank_function_call_expr(self, ctx: SSAValueCtx,
                                 call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
        program_state.setRequiresMPI(True)
        mpi_call=mpi.CommRank()
        return [mpi_call]

    def translate_mpi_commsize_function_call_expr(self, ctx: SSAValueCtx,
                                 call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
        program_state.setRequiresMPI(True)
        mpi_call=mpi.CommSize()
        return [mpi_call]


