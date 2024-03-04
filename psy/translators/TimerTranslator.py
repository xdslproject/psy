from typing import List
from xdsl.ir import Operation
from xdsl.dialects import arith
from xdsl.dialects.experimental import fir
from xdsl.dialects.builtin import SymbolRefAttr, IntegerAttr, i32, i64, IndexType
from psy.support import SSAValueCtx, ProgramState
from psy.translators.Translator import Translator
from psy.dialects import psy_ir
from psy.psy_ir_to_fir import translate_expr


class TimerTranslator(Translator):
  
    def __init__(self):
        super().__init__()
        self.setName('timer')
        self.setFnNames([ 'timer_init', 'timer_start', 'timer_stop', 'timer_report' ])
 

    def translate(self, function_name: str, ctx: SSAValueCtx,
                                 call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
        if function_name.lower() == "timer_init":
          return self.translate_timer_init_function_call_expr(ctx, call_expr, program_state, is_expr)
        elif function_name.lower() == "timer_start":
          return self.translate_timer_start_function_call_expr(ctx, call_expr, program_state, is_expr)
        elif function_name.lower() == "timer_stop":
          return self.translate_timer_stop_function_call_expr(ctx, call_expr, program_state, is_expr)
        elif function_name.lower() == "timer_report":
          return self.translate_timer_report_function_call_expr(ctx, call_expr, program_state, is_expr)
        
        # The function we were looking for isn't in the Timer module
        return None;

        
    def translate_timer_report_function_call_expr(self, ctx: SSAValueCtx,
                                 call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
        program_state.insertExternalFunctionToGlobalState("_QMdl_timerPtimer_report", [], None)

        init_call=fir.Call.create(attributes={"callee": SymbolRefAttr("_QMdl_timerPtimer_report")}, operands=[], result_types=[])

        return [init_call]
        

    def translate_timer_init_function_call_expr(self, ctx: SSAValueCtx,
                                 call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
        program_state.insertExternalFunctionToGlobalState("_QMdl_timerPtimer_init", [], None)

        init_call=fir.Call.create(attributes={"callee": SymbolRefAttr("_QMdl_timerPtimer_init")}, operands=[], result_types=[])

        return [init_call]
        

    def translate_timer_start_function_call_expr(self, ctx: SSAValueCtx,
                                 call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
        assert len(call_expr.args.blocks[0].ops) == 2

        ops_list=Translator.flatten_ops_to_list(call_expr.args.blocks[0].ops)

        op_ctrl, arg_ctrl = translate_expr(ctx, ops_list[0], program_state)
        op_desc, arg_desc = translate_expr(ctx, ops_list[1], program_state)

        assert isinstance(arg_ctrl.owner, fir.Load)
        assert arg_ctrl.owner.memref.type == fir.ReferenceType([i32])

        deferred_char_type=fir.ReferenceType([fir.CharacterType([fir.IntAttr(1), fir.DeferredAttr()])])
        convert_op=fir.Convert.create(operands=[arg_desc], result_types=[deferred_char_type])


        embox_to_found=arith.Constant.create(attributes={"value": IntegerAttr.from_index_int_value(arg_desc.type.type.to_index.data)}, result_types=[IndexType()])

        embox_op=emboxchar_op=fir.Emboxchar.create(operands=[convert_op.results[0], embox_to_found.results[0]], result_types=[fir.BoxCharType([fir.IntAttr(1)])])

        absent_op=fir.Absent.create(operands=[], result_types=[fir.ReferenceType([i64])])

        start_call=fir.Call.create(attributes={"callee": SymbolRefAttr("_QMdl_timerPtimer_start")},
          operands=[arg_ctrl.owner.memref, embox_op.results[0], absent_op.results[0]], result_types=[])

        program_state.insertExternalFunctionToGlobalState("_QMdl_timerPtimer_start", [start_call.operands[0].type,
          start_call.operands[1].type, start_call.operands[2].type], None)

        return op_ctrl+op_desc+[convert_op, embox_to_found, embox_op, absent_op, start_call]
        

    def translate_timer_stop_function_call_expr(self, ctx: SSAValueCtx,
                                 call_expr: psy_ir.CallExpr, program_state : ProgramState, is_expr=False) -> List[Operation]:
        assert len(call_expr.args.blocks[0].ops) == 1
        op_ctrl, arg_ctrl = translate_expr(ctx, call_expr.args.blocks[0].ops.first, program_state)

        assert isinstance(arg_ctrl.owner, fir.Load)
        assert arg_ctrl.owner.memref.type == fir.ReferenceType([i32])

        stop_call=fir.Call.create(attributes={"callee": SymbolRefAttr("_QMdl_timerPtimer_stop")},
          operands=[arg_ctrl.owner.memref], result_types=[])

        program_state.insertExternalFunctionToGlobalState("_QMdl_timerPtimer_stop", [stop_call.operands[0].type], None)

        return op_ctrl+[stop_call]