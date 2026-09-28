#include "codegen_riscv.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef TILELANG_ENABLE_RISCV_MLIR
#define TILELANG_ENABLE_RISCV_MLIR 0
#endif

#if TILELANG_ENABLE_RISCV_MLIR
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#endif

#include <tvm/arith/analyzer.h>
#include <tvm/ir/attrs.h>
#include <tvm/node/structural_equal.h>
#include <tvm/runtime/device_api.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../op/builtin.h"
#include "../op/utils.h"

namespace tvm {
namespace codegen {

namespace {

using FunctionEntry = std::pair<std::string, tir::PrimFunc>;

std::string BuildPlaceholderModule(const std::vector<FunctionEntry>& functions) {
  std::ostringstream os;
  os << "module {\n";
  os << "  // Placeholder MLIR module for the riscv backend.\n";
  os << "  // Rebuild TileLang with TILELANG_RISCV_MLIR_MODE=ON after the vendored\n";
  os << "  // LLVM/MLIR toolchain is installed to enable the real C++ MLIR builder.\n";
  for (const auto& [name, func] : functions) {
    (void)func;
    os << "  // pending lowering for @" << name << "\n";
  }
  os << "}\n";
  return os.str();
}

#if TILELANG_ENABLE_RISCV_MLIR
bool IsStructuredLoopKind(tir::ForKind kind) {
  return kind == tir::ForKind::kSerial || kind == tir::ForKind::kUnrolled ||
         kind == tir::ForKind::kParallel || kind == tir::ForKind::kVectorized;
}

bool IsSupportedGeneralLoopKind(tir::ForKind kind) {
  return kind == tir::ForKind::kSerial || kind == tir::ForKind::kUnrolled ||
         kind == tir::ForKind::kParallel || kind == tir::ForKind::kVectorized;
}

bool IsIntegerLikeType(DataType dtype) {
  return dtype.is_int() || dtype.is_uint() || dtype.is_bool();
}

bool IsFloatLikeType(DataType dtype) {
  DataType element_dtype = dtype.element_of();
  return element_dtype.is_float() || element_dtype.is_float16() ||
         element_dtype.is_bfloat16() || element_dtype.is_float8() ||
         element_dtype.is_float4();
}

bool IsShiftLikeType(DataType dtype) { return dtype.is_int() || dtype.is_uint(); }

bool IsLoopBreakCall(const tir::CallNode* op) {
  return op != nullptr &&
         (op->op.same_as(tvm::tl::loop_break()) || op->op.same_as(tir::builtin::break_loop()));
}

bool ContainsDirectLoopBreak(const tir::Stmt& stmt) {
  if (!stmt.defined()) {
    return false;
  }
  if (stmt.as<tir::ForNode>() != nullptr) {
    return false;
  }
  if (const auto* seq = stmt.as<tir::SeqStmtNode>()) {
    for (const tir::Stmt& child : seq->seq) {
      if (ContainsDirectLoopBreak(child)) {
        return true;
      }
    }
    return false;
  }
  if (const auto* eval = stmt.as<tir::EvaluateNode>()) {
    if (const auto* call = eval->value.as<tir::CallNode>()) {
      return IsLoopBreakCall(call);
    }
    return false;
  }
  if (const auto* if_node = stmt.as<tir::IfThenElseNode>()) {
    if (ContainsDirectLoopBreak(if_node->then_case)) {
      return true;
    }
    return if_node->else_case.defined() && ContainsDirectLoopBreak(if_node->else_case.value());
  }
  if (const auto* let = stmt.as<tir::LetStmtNode>()) {
    return ContainsDirectLoopBreak(let->body);
  }
  if (const auto* attr = stmt.as<tir::AttrStmtNode>()) {
    return ContainsDirectLoopBreak(attr->body);
  }
  if (const auto* alloc = stmt.as<tir::AllocateNode>()) {
    return ContainsDirectLoopBreak(alloc->body);
  }
  if (const auto* realize = stmt.as<tir::BufferRealizeNode>()) {
    return ContainsDirectLoopBreak(realize->body);
  }
  if (const auto* assert_stmt = stmt.as<tir::AssertStmtNode>()) {
    return ContainsDirectLoopBreak(assert_stmt->body);
  }
  if (const auto* block_realize = stmt.as<tir::BlockRealizeNode>()) {
    return ContainsDirectLoopBreak(block_realize->block);
  }
  if (const auto* block = stmt.as<tir::BlockNode>()) {
    return ContainsDirectLoopBreak(block->body);
  }
  return false;
}

bool IsLowerableScalarType(DataType dtype) {
  if (dtype.lanes() < 1) {
    return false;
  }
  DataType element_dtype = dtype.element_of();
  if (element_dtype.is_bool() || element_dtype.is_int() || element_dtype.is_uint()) {
    return true;
  }
  if (element_dtype.is_float16() || element_dtype.is_bfloat16()) {
    return true;
  }
  if (element_dtype.is_float8() || element_dtype.is_float4()) {
    return true;
  }
  return element_dtype.is_float() && (element_dtype.bits() == 32 || element_dtype.bits() == 64);
}

bool IsSupportedUnaryMathCall(const tir::CallNode* op) {
  const auto* op_node = op->op.as<OpNode>();
  if (op_node == nullptr || op->args.size() != 1) {
    return false;
  }
  if (op_node->name == "tir.abs" || op_node->name == "tir.fabs" ||
      llvm::StringRef(op_node->name).ends_with("fabs")) {
    return IsFloatLikeType(op->dtype) || op->dtype.is_int() || op->dtype.is_uint();
  }
  if (!IsFloatLikeType(op->dtype)) {
    return false;
  }
  return op_node->name == "tir.sqrt" || op_node->name == "tir.rsqrt" ||
         op_node->name == "tir.exp2" || op_node->name == "tir.log2" ||
         op_node->name == "tir.exp" || op_node->name == "tir.log" ||
         op_node->name == "tir.log1p" || op_node->name == "tir.sigmoid" ||
         op_node->name == "tir.tanh" || op_node->name == "tir.ceil" ||
         op_node->name == "tir.floor" || op_node->name == "tir.trunc" ||
         op_node->name == "tir.sin" || op_node->name == "tir.cos" ||
         op_node->name == "tir.erf" || op_node->name == "tir.nearbyint";
}

bool IsSupportedUnaryIntrinsicCall(const tir::CallNode* op) {
  const auto* op_node = op->op.as<OpNode>();
  if (op_node == nullptr || op->args.size() != 1) {
    return false;
  }
  if (op_node->name == "tir.isnan" || op_node->name == "tir.isinf") {
    return IsFloatLikeType(op->args[0].dtype()) && op->dtype.is_bool();
  }
  if (op_node->name == "tir.isfinite") {
    return IsFloatLikeType(op->args[0].dtype()) && op->dtype.is_bool();
  }
  if (op_node->name == "tir.popcount") {
    return IsIntegerLikeType(op->args[0].dtype()) && IsIntegerLikeType(op->dtype);
  }
  if (op_node->name == "tir.bitwise_not") {
    return IsIntegerLikeType(op->args[0].dtype()) && IsIntegerLikeType(op->dtype);
  }
  return false;
}

bool IsSupportedPackedX2IntrinsicCall(const tir::CallNode* op) {
  const auto* op_node = op->op.as<OpNode>();
  if (op_node == nullptr || !IsLowerableScalarType(op->dtype) || op->dtype.lanes() != 2) {
    return false;
  }
  if (!IsFloatLikeType(op->dtype)) {
    return false;
  }
  if (op_node->name == "tl.abs2") {
    return op->args.size() == 1 && op->args[0].dtype() == op->dtype;
  }
  if (op_node->name == "tl.max2" || op_node->name == "tl.min2" || op_node->name == "tl.add2" ||
      op_node->name == "tl.sub2" || op_node->name == "tl.mul2") {
    return op->args.size() == 2 && op->args[0].dtype() == op->dtype &&
           op->args[1].dtype() == op->dtype;
  }
  if (op_node->name == "tl.fma2") {
    return op->args.size() == 3 && op->args[0].dtype() == op->dtype &&
           op->args[1].dtype() == op->dtype && op->args[2].dtype() == op->dtype;
  }
  return false;
}

bool IsSupportedBitcastCall(const tir::CallNode* op) {
  if (!op->op.same_as(tir::builtin::reinterpret()) || op->args.size() != 1) {
    return false;
  }
  DataType source_dtype = op->args[0].dtype();
  DataType target_dtype = op->dtype;
  if (!IsLowerableScalarType(source_dtype) || !IsLowerableScalarType(target_dtype)) {
    return false;
  }
  if (source_dtype.is_handle() || target_dtype.is_handle()) {
    return false;
  }
  return source_dtype.bits() == target_dtype.bits();
}

bool IsReinterpretCall(const tir::CallNode* op) {
  return op != nullptr && op->op.same_as(tir::builtin::reinterpret());
}

void RejectUnsupportedReinterpretCall(const tir::CallNode* op) {
  std::string source_dtype = op != nullptr && !op->args.empty()
                                 ? runtime::DLDataTypeToString(op->args[0].dtype())
                                 : "<unknown>";
  std::string target_dtype =
      op != nullptr ? runtime::DLDataTypeToString(op->dtype) : "<unknown>";
  LOG(FATAL) << "Unsupported tir.reinterpret in riscv lowering from "
             << source_dtype << " to " << target_dtype
             << ". The backend only supports scalar/vector bitcasts between non-handle "
                "types with identical bit width; pointer/handle reinterpret paths need "
                "an explicit RISC-V runtime and memory model.";
}

bool IsSupportedBinaryIntrinsicCall(const tir::CallNode* op) {
  if (op->args.size() != 2) {
    return false;
  }
  if (op->op.same_as(tir::builtin::bitwise_and()) ||
      op->op.same_as(tir::builtin::bitwise_or())) {
    return IsIntegerLikeType(op->args[0].dtype()) && IsIntegerLikeType(op->args[1].dtype()) &&
           IsIntegerLikeType(op->dtype);
  }
  if (op->op.same_as(tir::builtin::shift_left()) ||
      op->op.same_as(tir::builtin::shift_right())) {
    return IsShiftLikeType(op->args[0].dtype()) && IsShiftLikeType(op->args[1].dtype()) &&
           IsShiftLikeType(op->dtype);
  }
  const auto* op_node = op->op.as<OpNode>();
  if (op_node == nullptr) {
    return false;
  }
  if (op_node->name == "tir.bitwise_xor") {
    return IsIntegerLikeType(op->args[0].dtype()) && IsIntegerLikeType(op->args[1].dtype()) &&
           IsIntegerLikeType(op->dtype);
  }
  if (op_node->name == "tir.copysign") {
    return IsFloatLikeType(op->args[0].dtype()) &&
           IsFloatLikeType(op->args[1].dtype()) && IsFloatLikeType(op->dtype);
  }
  if (op_node->name == "tir.pow") {
    if (IsFloatLikeType(op->args[0].dtype()) && IsFloatLikeType(op->args[1].dtype()) &&
        IsFloatLikeType(op->dtype)) {
      return true;
    }
    if (IsFloatLikeType(op->args[0].dtype()) && IsIntegerLikeType(op->args[1].dtype()) &&
        IsFloatLikeType(op->dtype)) {
      return true;
    }
    return IsIntegerLikeType(op->args[0].dtype()) && IsIntegerLikeType(op->args[1].dtype()) &&
           IsIntegerLikeType(op->dtype);
  }
  return false;
}

bool IsIfThenElseCall(const tir::CallNode* op) {
  const auto* op_node = op->op.as<OpNode>();
  return op_node != nullptr && op_node->name == "tir.if_then_else" && op->args.size() == 3;
}

bool IsInfinityCall(const tir::CallNode* op) {
  const auto* op_node = op->op.as<OpNode>();
  bool valid_args =
      op->args.empty() ||
      (op->args.size() == 1 && op->args[0].as<tir::StringImmNode>() != nullptr);
  return op_node != nullptr && (op_node->name == "tl.infinity" ||
                                op_node->name == "tir.infinity") &&
         valid_args && IsFloatLikeType(op->dtype);
}

bool IsThreadLaunchIterVar(const tir::IterVarNode* iter_var) {
  return iter_var != nullptr &&
         llvm::StringRef(iter_var->thread_tag).starts_with("threadIdx");
}

bool IsBlockLaunchIterVar(const tir::IterVarNode* iter_var) {
  return iter_var != nullptr &&
         llvm::StringRef(iter_var->thread_tag).starts_with("blockIdx");
}

bool IsThreadReturnCall(const tir::CallNode* op) {
  return op != nullptr && op->op.same_as(tir::builtin::thread_return());
}

bool IsCooperativeThreadIntrinsicName(const ffi::String& op_name) {
  llvm::StringRef name(op_name.c_str());
  if (name == "tl.sync_warp" || name == "tl.sync_grid" ||
      name == "tl.tl_shuffle_elect" || name == "tl.any_sync" ||
      name == "tl.all_sync" || name == "tl.ballot_sync" ||
      name == "tl.ballot" || name == "tl.activemask" ||
      name == "tl.syncthreads_count" || name == "tl.syncthreads_and" ||
      name == "tl.syncthreads_or" || name == "tl.match_any_sync" ||
      name == "tl.match_all_sync") {
    return true;
  }
  if (name.starts_with("tl.shfl") || name.starts_with("tl.warp_reduce_")) {
    return true;
  }
  if (name.starts_with("tir.tvm_warp_shuffle") || name == "tir.tvm_warp_activemask") {
    return true;
  }
  return false;
}

bool IsAtomicIntrinsicName(const ffi::String& op_name) {
  llvm::StringRef name(op_name.c_str());
  return name.starts_with("tl.atomic_") || name.starts_with("tl.tileop.atomic");
}

bool IsUnsupportedTileReductionIntrinsicName(const ffi::String& op_name) {
  llvm::StringRef name(op_name.c_str());
  return name == "tl.tileop.reduce" || name == "tl.tileop.finalize_reducer";
}

bool IsUnsupportedTileScanIntrinsicName(const ffi::String& op_name) {
  return llvm::StringRef(op_name.c_str()) == "tl.tileop.cumsum";
}

bool IsLowerableTileScanIntrinsicName(const ffi::String& op_name) {
  return llvm::StringRef(op_name.c_str()) == "tl.tileop.cumsum";
}

bool IsLowerableTileReductionIntrinsicName(const ffi::String& op_name) {
  return llvm::StringRef(op_name.c_str()) == "tl.tileop.reduce";
}

bool IsFinalizeReducerIntrinsicName(const ffi::String& op_name) {
  return llvm::StringRef(op_name.c_str()) == "tl.tileop.finalize_reducer";
}

enum class TileReductionKind {
  kSum,
  kAbsSum,
  kMax,
  kAbsMax,
  kMin,
  kBitAnd,
  kBitOr,
  kBitXor,
};

std::optional<TileReductionKind> ParseTileReductionKind(llvm::StringRef kind) {
  if (kind == "sum") {
    return TileReductionKind::kSum;
  }
  if (kind == "abssum") {
    return TileReductionKind::kAbsSum;
  }
  if (kind == "max") {
    return TileReductionKind::kMax;
  }
  if (kind == "absmax") {
    return TileReductionKind::kAbsMax;
  }
  if (kind == "min") {
    return TileReductionKind::kMin;
  }
  if (kind == "bitand") {
    return TileReductionKind::kBitAnd;
  }
  if (kind == "bitor") {
    return TileReductionKind::kBitOr;
  }
  if (kind == "bitxor") {
    return TileReductionKind::kBitXor;
  }
  return std::nullopt;
}

bool TileReductionNeedsAbs(TileReductionKind kind) {
  return kind == TileReductionKind::kAbsSum || kind == TileReductionKind::kAbsMax;
}

bool TileReductionIsBitwise(TileReductionKind kind) {
  return kind == TileReductionKind::kBitAnd || kind == TileReductionKind::kBitOr ||
         kind == TileReductionKind::kBitXor;
}

bool TileReductionCanUseZeroIdentity(TileReductionKind kind) {
  return kind == TileReductionKind::kSum || kind == TileReductionKind::kAbsSum ||
         kind == TileReductionKind::kAbsMax || kind == TileReductionKind::kBitOr ||
         kind == TileReductionKind::kBitXor;
}

enum class ScalarAtomicRMWKind {
  kAdd,
  kMax,
  kMin,
};

std::optional<ScalarAtomicRMWKind> GetScalarAtomicRMWKind(const tir::CallNode* op) {
  if (op == nullptr) {
    return std::nullopt;
  }
  if (op->op.same_as(tl::atomic_add_elem_op()) ||
      op->op.same_as(tl::atomic_add_ret_elem_op())) {
    return ScalarAtomicRMWKind::kAdd;
  }
  if (op->op.same_as(tl::atomic_max_elem_op()) ||
      op->op.same_as(tl::atomic_max_ret_elem_op())) {
    return ScalarAtomicRMWKind::kMax;
  }
  if (op->op.same_as(tl::atomic_min_elem_op()) ||
      op->op.same_as(tl::atomic_min_ret_elem_op())) {
    return ScalarAtomicRMWKind::kMin;
  }
  return std::nullopt;
}

bool IsScalarAtomicRMWCall(const tir::CallNode* op) {
  return GetScalarAtomicRMWKind(op).has_value();
}

bool IsScalarAtomicLoadCall(const tir::CallNode* op) {
  return op != nullptr && op->op.same_as(tl::atomic_load_elem_op());
}

bool IsScalarAtomicStoreCall(const tir::CallNode* op) {
  return op != nullptr && op->op.same_as(tl::atomic_store_elem_op());
}

bool IsVectorAtomicAddCall(const tir::CallNode* op) {
  return op != nullptr &&
         (op->op.same_as(tl::atomic_addx2_elem_op()) ||
          op->op.same_as(tl::atomic_addx4_elem_op()));
}

int VectorAtomicAddLaneCount(const tir::CallNode* op) {
  ICHECK(IsVectorAtomicAddCall(op)) << "Vector atomic add expected atomic_addx2/addx4";
  if (op->op.same_as(tl::atomic_addx2_elem_op())) {
    return 2;
  }
  return 4;
}

const char* VectorAtomicAddContext(const tir::CallNode* op) {
  ICHECK(IsVectorAtomicAddCall(op)) << "Vector atomic add expected atomic_addx2/addx4";
  if (op->op.same_as(tl::atomic_addx2_elem_op())) {
    return "atomic_addx2";
  }
  return "atomic_addx4";
}

bool ScalarAtomicRMWReturnsValue(const tir::CallNode* op) {
  if (op == nullptr) {
    return false;
  }
  if (op->op.same_as(tl::atomic_add_ret_elem_op()) ||
      op->op.same_as(tl::atomic_max_ret_elem_op()) ||
      op->op.same_as(tl::atomic_min_ret_elem_op())) {
    return true;
  }
  return !op->dtype.is_handle();
}

const char* ScalarAtomicRMWContext(ScalarAtomicRMWKind kind) {
  switch (kind) {
    case ScalarAtomicRMWKind::kAdd:
      return "atomic_add";
    case ScalarAtomicRMWKind::kMax:
      return "atomic_max";
    case ScalarAtomicRMWKind::kMin:
      return "atomic_min";
  }
  TVM_FFI_UNREACHABLE();
}

bool IsCudaPipelineOrTargetSyncIntrinsicName(const ffi::String& op_name) {
  llvm::StringRef name(op_name.c_str());
  return name.starts_with("tir.ptx_") || name.starts_with("tl.ptx_") ||
         name.starts_with("tl.tma_") || name.starts_with("tl.tileop.tma") ||
         name.starts_with("tl.create_tma") || name.starts_with("tl.mbarrier") ||
         name.starts_with("tir.ptx_mma") || name.starts_with("tl.tcgen05_") ||
         name.starts_with("tl.tileop.tcgen05") || name.starts_with("tl.cluster_") ||
         name == "tl.block_rank_in_cluster" || name.starts_with("tl.clc_") ||
         name.starts_with("tl.warpgroup_") || name == "tl.wait_wgmma" ||
         name == "tl.set_max_nreg" || name == "tl.no_set_max_nreg" ||
         name == "tl.deallocate_tmem" || name == "tl.get_lane_idx" ||
         name == "tl.get_warp_idx_sync" || name == "tl.get_warp_idx" ||
         name == "tl.get_warp_group_idx";
}

bool IsThreadIndexHelperIntrinsicName(const ffi::String& op_name) {
  llvm::StringRef name(op_name.c_str());
  return name == "tl.get_lane_idx" || name == "tl.get_warp_idx_sync" ||
         name == "tl.get_warp_idx" || name == "tl.get_warp_group_idx";
}

bool IsSerializedNoOpTargetSyncCall(const tir::CallNode* op) {
  return op != nullptr &&
         (op->op.same_as(tir::builtin::ptx_arrive_barrier()) ||
          op->op.same_as(tvm::tl::ptx_cp_async_barrier_noinc()) ||
          op->op.same_as(tvm::tl::mbarrier_expect_tx()) ||
          op->op.same_as(tvm::tl::mbarrier_wait_parity()) ||
          op->op.same_as(tvm::tl::ptx_fence_barrier_init()) ||
          op->op.same_as(tvm::tl::fence_proxy_async()) ||
          op->op.same_as(tvm::tl::wait_wgmma()) ||
          op->op.same_as(tvm::tl::set_max_nreg()) ||
          op->op.same_as(tvm::tl::no_set_max_nreg()) ||
          op->op.same_as(tvm::tl::warpgroup_fence_operand()));
}

std::string GetCallExternName(const tir::CallNode* op) {
  if (op == nullptr || !op->op.same_as(tir::builtin::call_extern()) || op->args.empty()) {
    return "<unknown>";
  }
  const auto* name = op->args[0].as<tir::StringImmNode>();
  if (name == nullptr) {
    return "<non-constant>";
  }
  return name->value;
}

bool IsFloat2HalfRZCallExtern(const tir::CallNode* op) {
  return op != nullptr && op->op.same_as(tir::builtin::call_extern()) &&
         GetCallExternName(op) == "__float2half_rz";
}

bool IsAtomicAddOffsetCallExtern(const tir::CallNode* op) {
  return op != nullptr && op->op.same_as(tir::builtin::call_extern()) &&
         GetCallExternName(op) == "tl_atomic_add_offset";
}

bool IsSerializedNoOpCallExternName(const std::string& extern_name) {
  return extern_name == "tl::fp8_transpose_v_128x224_fa3_src_ldsm_stsm_barrier_each_iter" ||
         extern_name == "tl::fp8_pv_ptx_unit_accumulate_fa3_raw_64x128x224";
}

bool IsSerializedFp8TmaLoadCallExternName(const std::string& extern_name) {
  return extern_name == "tl::fp8_tma_load_4d_ptx";
}

bool IsMatchSyncCallExtern(const tir::CallNode* op) {
  if (op == nullptr || !op->op.same_as(tir::builtin::call_extern())) {
    return false;
  }
  std::string extern_name = GetCallExternName(op);
  return extern_name == "__match_any_sync" || extern_name == "__match_all_sync";
}

struct AddressOfBufferLoadAccess {
  tir::Buffer buffer;
  Array<PrimExpr> indices;
};

std::optional<AddressOfBufferLoadAccess> MatchAddressOfBufferLoad(const PrimExpr& expr) {
  const auto* call = expr.as<tir::CallNode>();
  if (call == nullptr || !call->op.same_as(tir::builtin::address_of()) || call->args.size() != 1U) {
    return std::nullopt;
  }
  const auto* load = call->args[0].as<tir::BufferLoadNode>();
  if (load == nullptr) {
    return std::nullopt;
  }
  return AddressOfBufferLoadAccess{load->buffer, load->indices};
}

void RejectUnsupportedCallExtern(const tir::CallNode* op) {
  std::string extern_name = GetCallExternName(op);
  if (extern_name == "__match_any_sync" || extern_name == "__match_all_sync") {
    LOG(FATAL) << "tl."
               << (extern_name == "__match_any_sync" ? "match_any_sync" : "match_all_sync")
               << " expression is not supported yet in riscv lowering because it "
                  "requires cooperative thread semantics across serialized lanes.";
  }
  LOG(FATAL) << "Unsupported call_extern in riscv lowering: "
             << extern_name
             << ". Opaque or CUDA-specific extern calls need an explicit RISC-V MLIR lowering "
                "or a documented rejection entry.";
}

void RejectCudaPipelineOrTargetSyncIntrinsic(const ffi::String& op_name) {
  LOG(FATAL) << "Unsupported CUDA/target-specific intrinsic in riscv lowering: "
             << op_name
             << ". TMA, mbarrier, tensorcore, cluster, launch-control, warpgroup, TMEM, "
                "register-hint, and lane/warp-index "
                "intrinsics require a RISC-V execution model before lowering.";
}

void RejectUnsupportedTileReductionIntrinsic(const ffi::String& op_name) {
  if (llvm::StringRef(op_name.c_str()) == "tl.tileop.finalize_reducer") {
    LOG(FATAL) << "Unsupported reducer finalization intrinsic in riscv lowering: "
               << op_name
               << ". T.finalize_reducer may lower to a cross-thread AllReduce when the "
                  "reducer is replicated; this requires an explicit RISC-V cooperative "
                  "execution model before it can be compiled.";
  }
  LOG(FATAL) << "Unsupported tile reduction intrinsic in riscv lowering: "
             << op_name
             << ". Tile-level reductions such as T.reduce_sum, T.reduce_max, and "
                "T.reduce_absmax need an explicit RISC-V MLIR lowering or a documented "
                "execution model before they can be compiled.";
}

void RejectUnsupportedTileScanIntrinsic(const ffi::String& op_name) {
  LOG(FATAL) << "Unsupported tile scan intrinsic in riscv lowering: "
             << op_name
             << ". Tile-level prefix scans such as T.cumsum need an explicit RISC-V "
                "cooperative execution model before they can be compiled.";
}

void RejectUnsupportedCooperativeThreadExpression(const ffi::String& op_name) {
  LOG(FATAL) << op_name
             << " expression is not supported yet in riscv lowering because it "
             << "requires cooperative thread semantics across serialized lanes.";
}

void RejectUnsupportedCooperativeThreadStatement(const ffi::String& op_name) {
  LOG(FATAL) << op_name
             << " inside non-unit thread launch is not supported yet in riscv "
             << "lowering because it requires cooperative thread semantics.";
}

mlir::Value CreateFloatConstant(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::Type type, double value) {
  mlir::FloatType float_type = mlir::cast<mlir::FloatType>(type);
  return builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(float_type, value));
}

mlir::Value CreateInfinityConstant(mlir::OpBuilder& builder, mlir::Location loc,
                                   mlir::Type type) {
  mlir::FloatType float_type = mlir::cast<mlir::FloatType>(type);
  llvm::APFloat inf = llvm::APFloat::getInf(float_type.getFloatSemantics(), false);
  return builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(float_type, inf));
}

mlir::Value CreateIntegerConstant(mlir::OpBuilder& builder, mlir::Location loc,
                                  mlir::Type type, int64_t value) {
  return builder.create<mlir::arith::ConstantIntOp>(loc, type, value);
}

mlir::Value LowerSupportedUnaryMathCall(mlir::OpBuilder& builder, mlir::Location loc,
                                        const tir::CallNode* op, mlir::Value arg) {
  ICHECK(IsSupportedUnaryMathCall(op))
      << "Unsupported TIR call in riscv lowering: " << op->op;
  const auto* op_node = op->op.as<OpNode>();
  ICHECK(op_node != nullptr);
  if (op_node->name == "tir.sqrt") {
    return builder.create<mlir::math::SqrtOp>(loc, arg);
  }
  if (op_node->name == "tir.rsqrt") {
    return builder.create<mlir::math::RsqrtOp>(loc, arg);
  }
  if (op_node->name == "tir.exp2") {
    return builder.create<mlir::math::Exp2Op>(loc, arg);
  }
  if (op_node->name == "tir.log2") {
    return builder.create<mlir::math::Log2Op>(loc, arg);
  }
  if (op_node->name == "tir.exp") {
    return builder.create<mlir::math::ExpOp>(loc, arg);
  }
  if (op_node->name == "tir.log") {
    return builder.create<mlir::math::LogOp>(loc, arg);
  }
  if (op_node->name == "tir.log1p") {
    return builder.create<mlir::math::Log1pOp>(loc, arg);
  }
  if (op_node->name == "tir.tanh") {
    return builder.create<mlir::math::TanhOp>(loc, arg);
  }
  if (op_node->name == "tir.ceil") {
    return builder.create<mlir::math::CeilOp>(loc, arg);
  }
  if (op_node->name == "tir.floor") {
    return builder.create<mlir::math::FloorOp>(loc, arg);
  }
  if (op_node->name == "tir.trunc") {
    return builder.create<mlir::math::TruncOp>(loc, arg);
  }
  if (op_node->name == "tir.sin") {
    return builder.create<mlir::math::SinOp>(loc, arg);
  }
  if (op_node->name == "tir.cos") {
    return builder.create<mlir::math::CosOp>(loc, arg);
  }
  if (op_node->name == "tir.erf") {
    return builder.create<mlir::math::ErfOp>(loc, arg);
  }
  if (op_node->name == "tir.nearbyint") {
    // RISC-V MLIR lowering currently fixes nearbyint to round-to-nearest-even.
    return builder.create<mlir::math::RoundEvenOp>(loc, arg);
  }
  if (op_node->name == "tir.sigmoid") {
    mlir::Value zero = CreateFloatConstant(builder, loc, arg.getType(), 0.0);
    mlir::Value one = CreateFloatConstant(builder, loc, arg.getType(), 1.0);
    mlir::Value neg = builder.create<mlir::arith::SubFOp>(loc, zero, arg);
    mlir::Value exp = builder.create<mlir::math::ExpOp>(loc, neg);
    mlir::Value denom = builder.create<mlir::arith::AddFOp>(loc, one, exp);
    return builder.create<mlir::arith::DivFOp>(loc, one, denom);
  }
  if (op_node->name == "tir.abs" || op_node->name == "tir.fabs" ||
      llvm::StringRef(op_node->name).ends_with("fabs")) {
    if (IsFloatLikeType(op->dtype)) {
      return builder.create<mlir::math::AbsFOp>(loc, arg);
    }
    if (op->dtype.is_uint()) {
      return arg;
    }
    mlir::Value zero = CreateIntegerConstant(builder, loc, arg.getType(), 0);
    mlir::Value neg = builder.create<mlir::arith::SubIOp>(loc, zero, arg);
    mlir::Value cond =
        builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, arg, zero);
    return builder.create<mlir::arith::SelectOp>(loc, cond, neg, arg);
  }
  LOG(FATAL) << "Unsupported TIR math call in riscv lowering: " << op_node->name;
  TVM_FFI_UNREACHABLE();
}

mlir::Value LowerSupportedUnaryIntrinsicCall(mlir::OpBuilder& builder, mlir::Location loc,
                                             const tir::CallNode* op, mlir::Value arg) {
  ICHECK(IsSupportedUnaryIntrinsicCall(op))
      << "Unsupported TIR intrinsic in riscv lowering: " << op->op;
  const auto* op_node = op->op.as<OpNode>();
  ICHECK(op_node != nullptr);
  if (op_node->name == "tir.isnan") {
    return builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::UNO, arg, arg);
  }
  if (op_node->name == "tir.isinf") {
    mlir::Value abs = builder.create<mlir::math::AbsFOp>(loc, arg);
    mlir::Value inf = CreateInfinityConstant(builder, loc, arg.getType());
    return builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, abs, inf);
  }
  if (op_node->name == "tir.isfinite") {
    mlir::Value abs = builder.create<mlir::math::AbsFOp>(loc, arg);
    mlir::Value inf = CreateInfinityConstant(builder, loc, arg.getType());
    return builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::ONE, abs, inf);
  }
  if (op_node->name == "tir.popcount") {
    return builder.create<mlir::math::CtPopOp>(loc, arg);
  }
  if (op_node->name == "tir.bitwise_not") {
    mlir::Value all_ones = CreateIntegerConstant(builder, loc, arg.getType(), -1);
    return builder.create<mlir::arith::XOrIOp>(loc, arg, all_ones);
  }
  LOG(FATAL) << "Unsupported TIR intrinsic in riscv lowering: " << op_node->name;
  TVM_FFI_UNREACHABLE();
}

mlir::Value LowerSupportedPackedX2IntrinsicCall(mlir::OpBuilder& builder, mlir::Location loc,
                                                const tir::CallNode* op, mlir::Value lhs,
                                                mlir::Value rhs = mlir::Value(),
                                                mlir::Value extra = mlir::Value()) {
  ICHECK(IsSupportedPackedX2IntrinsicCall(op))
      << "Unsupported packed x2 intrinsic in riscv lowering: " << op->op;
  mlir::VectorType type = mlir::cast<mlir::VectorType>(lhs.getType());
  llvm::SmallVector<mlir::Value, 2> lanes;
  lanes.reserve(2);

  const auto* op_node = op->op.as<OpNode>();
  ICHECK(op_node != nullptr);
  for (int64_t lane = 0; lane < 2; ++lane) {
    mlir::Value lhs_lane = builder.create<mlir::vector::ExtractOp>(loc, lhs, lane);
    if (op_node->name == "tl.abs2") {
      lanes.push_back(builder.create<mlir::math::AbsFOp>(loc, lhs_lane));
      continue;
    }
    mlir::Value rhs_lane = builder.create<mlir::vector::ExtractOp>(loc, rhs, lane);
    if (op_node->name == "tl.add2") {
      lanes.push_back(builder.create<mlir::arith::AddFOp>(loc, lhs_lane, rhs_lane));
      continue;
    }
    if (op_node->name == "tl.sub2") {
      lanes.push_back(builder.create<mlir::arith::SubFOp>(loc, lhs_lane, rhs_lane));
      continue;
    }
    if (op_node->name == "tl.mul2") {
      lanes.push_back(builder.create<mlir::arith::MulFOp>(loc, lhs_lane, rhs_lane));
      continue;
    }
    if (op_node->name == "tl.fma2") {
      mlir::Value extra_lane = builder.create<mlir::vector::ExtractOp>(loc, extra, lane);
      mlir::Value mul = builder.create<mlir::arith::MulFOp>(loc, lhs_lane, rhs_lane);
      lanes.push_back(builder.create<mlir::arith::AddFOp>(loc, mul, extra_lane));
      continue;
    }
    mlir::arith::CmpFPredicate predicate = op_node->name == "tl.min2"
                                               ? mlir::arith::CmpFPredicate::OLT
                                               : mlir::arith::CmpFPredicate::OGT;
    mlir::Value cond = builder.create<mlir::arith::CmpFOp>(loc, predicate, lhs_lane, rhs_lane);
    lanes.push_back(builder.create<mlir::arith::SelectOp>(loc, cond, lhs_lane, rhs_lane));
  }
  return builder.create<mlir::vector::FromElementsOp>(loc, type, lanes);
}

mlir::Value LowerSupportedBinaryIntrinsicCall(mlir::OpBuilder& builder, mlir::Location loc,
                                              const tir::CallNode* op, mlir::Value lhs,
                                              mlir::Value rhs) {
  ICHECK(IsSupportedBinaryIntrinsicCall(op))
      << "Unsupported TIR intrinsic in riscv lowering: " << op->op;
  if (op->op.same_as(tir::builtin::bitwise_and())) {
    return builder.create<mlir::arith::AndIOp>(loc, lhs, rhs);
  }
  if (op->op.same_as(tir::builtin::bitwise_or())) {
    return builder.create<mlir::arith::OrIOp>(loc, lhs, rhs);
  }
  if (op->op.same_as(tir::builtin::shift_left())) {
    if (auto integer_type = mlir::dyn_cast<mlir::IntegerType>(lhs.getType());
        integer_type && integer_type.getWidth() < 32) {
      mlir::Type promoted_type = builder.getIntegerType(32);
      bool is_unsigned = op->args[0].dtype().is_uint() || op->args[0].dtype().is_bool();
      mlir::Value promoted_lhs =
          is_unsigned ? builder.create<mlir::arith::ExtUIOp>(loc, promoted_type, lhs).getResult()
                      : builder.create<mlir::arith::ExtSIOp>(loc, promoted_type, lhs).getResult();
      mlir::Value promoted_rhs = builder.create<mlir::arith::ExtUIOp>(loc, promoted_type, rhs);
      mlir::Value shifted = builder.create<mlir::arith::ShLIOp>(loc, promoted_lhs, promoted_rhs);
      return builder.create<mlir::arith::TruncIOp>(loc, lhs.getType(), shifted);
    }
    return builder.create<mlir::arith::ShLIOp>(loc, lhs, rhs);
  }
  if (op->op.same_as(tir::builtin::shift_right())) {
    if (auto integer_type = mlir::dyn_cast<mlir::IntegerType>(lhs.getType());
        integer_type && integer_type.getWidth() < 32) {
      mlir::Type promoted_type = builder.getIntegerType(32);
      bool is_unsigned = op->args[0].dtype().is_uint() || op->args[0].dtype().is_bool();
      mlir::Value promoted_lhs =
          is_unsigned ? builder.create<mlir::arith::ExtUIOp>(loc, promoted_type, lhs).getResult()
                      : builder.create<mlir::arith::ExtSIOp>(loc, promoted_type, lhs).getResult();
      mlir::Value promoted_rhs = builder.create<mlir::arith::ExtUIOp>(loc, promoted_type, rhs);
      mlir::Value shifted =
          is_unsigned ? builder.create<mlir::arith::ShRUIOp>(loc, promoted_lhs, promoted_rhs).getResult()
                      : builder.create<mlir::arith::ShRSIOp>(loc, promoted_lhs, promoted_rhs).getResult();
      return builder.create<mlir::arith::TruncIOp>(loc, lhs.getType(), shifted);
    }
    if (op->dtype.is_uint()) {
      return builder.create<mlir::arith::ShRUIOp>(loc, lhs, rhs);
    }
    return builder.create<mlir::arith::ShRSIOp>(loc, lhs, rhs);
  }
  const auto* op_node = op->op.as<OpNode>();
  ICHECK(op_node != nullptr);
  if (op_node->name == "tir.bitwise_xor") {
    return builder.create<mlir::arith::XOrIOp>(loc, lhs, rhs);
  }
  if (op_node->name == "tir.copysign") {
    return builder.create<mlir::math::CopySignOp>(loc, lhs, rhs);
  }
  if (op_node->name == "tir.pow") {
    if (IsFloatLikeType(op->args[0].dtype()) && IsFloatLikeType(op->args[1].dtype())) {
      return builder.create<mlir::math::PowFOp>(loc, lhs, rhs);
    }
    if (IsFloatLikeType(op->args[0].dtype()) && IsIntegerLikeType(op->args[1].dtype())) {
      return builder.create<mlir::math::FPowIOp>(loc, lhs, rhs);
    }
    return builder.create<mlir::math::IPowIOp>(loc, lhs, rhs);
  }
  LOG(FATAL) << "Unsupported TIR intrinsic in riscv lowering: " << op_node->name;
  TVM_FFI_UNREACHABLE();
}

class TIRToMLIRLowerer final : private tir::StmtFunctor<void(const tir::Stmt&)>,
                               private tir::ExprFunctor<mlir::Value(const PrimExpr&)> {
public:
  TIRToMLIRLowerer()
      : context_(),
        builder_(&context_),
        loc_(builder_.getUnknownLoc()),
        module_(mlir::ModuleOp::create(loc_)) {
    registry_.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                     mlir::linalg::LinalgDialect, mlir::math::MathDialect,
                     mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                     mlir::vector::VectorDialect>();
    context_.appendDialectRegistry(registry_);
    context_.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                         mlir::linalg::LinalgDialect, mlir::math::MathDialect,
                         mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                         mlir::vector::VectorDialect>();
    builder_.setInsertionPointToStart(module_.getBody());
  }

  std::string Lower(const std::vector<FunctionEntry>& functions) {
    for (const auto& [name, func] : functions) {
      LowerFunction(name, func);
    }

    std::string mlir_text;
    llvm::raw_string_ostream os(mlir_text);
    module_.print(os);
    os.flush();
    return mlir_text;
  }

private:
  using tir::ExprFunctor<mlir::Value(const PrimExpr&)>::VisitExpr;
  using tir::StmtFunctor<void(const tir::Stmt&)>::VisitStmt;

  struct SavedBinding {
    bool had_value{false};
    mlir::Value value;
  };

  struct PackedScalarViewBinding {
    tir::Buffer source_buffer;
    DataType source_dtype;
    DataType view_dtype;
  };

  struct ElementwiseLoopNestMatch {
    llvm::SmallVector<const tir::ForNode*, 4> loops;
    const tir::BlockRealizeNode* block_realize{nullptr};
    const tir::BlockNode* block{nullptr};
    const tir::BufferStoreNode* store{nullptr};
    llvm::SmallVector<tir::Var, 4> block_vars;
    std::vector<tir::Buffer> input_buffers;
  };

  struct DeferredLoopBindings {
    llvm::SmallVector<tir::Buffer, 4> alloc_buffers;
    llvm::SmallVector<tir::MatchBufferRegion, 4> match_buffers;
  };

  struct ScopedBufferBindings {
    std::vector<std::pair<const Object*, SavedBinding>> buffer_bindings;
    std::vector<std::pair<const Object*, std::optional<tir::Buffer>>> buffer_owners;
    std::vector<std::pair<const Object*, std::optional<tir::Buffer>>> packed_owners;
    std::vector<std::pair<const Object*, std::optional<mlir::Value>>> thread_local_backings;
    std::vector<const Object*> thread_local_keys;
  };

  struct ThreadLaunchFrame {
    const tir::IterVarNode* iter_var{nullptr};
    std::optional<int64_t> extent;
    bool body_uses_iter_var{false};
  };

  struct ThreadLocalBlockAllocBinding {
    tir::Buffer buffer;
    mlir::Value backing;
  };

  struct BreakLoopFrame {
    mlir::Value flag;
  };

  struct SerializedWarpReplayBufferElementKey {
    const Object* buffer_data{nullptr};
    int64_t linear_index{0};

    bool operator==(const SerializedWarpReplayBufferElementKey& other) const {
      return buffer_data == other.buffer_data && linear_index == other.linear_index;
    }
  };

  struct SerializedWarpReplayBufferElementKeyHash {
    size_t operator()(const SerializedWarpReplayBufferElementKey& key) const {
      size_t h1 = std::hash<const Object*>{}(key.buffer_data);
      size_t h2 = std::hash<int64_t>{}(key.linear_index);
      return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
  };

  enum class SerializedWarpIndexClass {
    kInvariant,
    kLaneLinear,
    kUnsupported,
  };

  enum class SerializedWarpShuffleKind {
    kSync,
    kXor,
    kDown,
    kUp,
  };

  enum class SerializedWarpMatchKind {
    kAny,
    kAll,
  };

  enum class SerializedWarpVoteKind {
    kAny,
    kAll,
    kBallot,
    kActiveMask,
  };

  struct SerializedWarpMatchReplayPattern {
    tir::Buffer load_buffer;
    PrimExpr index_expr;
    DataType value_dtype;
    std::optional<PrimExpr> guard_upper_bound;
    std::optional<PrimExpr> fallback_expr;
  };

  enum class LaunchTrackingKind { kNone, kBlock, kThread };

  struct AccessPtrBinding {
    tir::Buffer buffer;
    mlir::Value memref;
    llvm::SmallVector<mlir::Value, 4> indices;
  };

  struct ContiguousAccessPtrBinding {
    tir::Buffer buffer;
    mlir::Value memref;
    llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4> lane_indices;
  };

  struct StructuredIndexPattern {
    llvm::SmallVector<unsigned, 4> dims;

    bool operator==(const StructuredIndexPattern& other) const { return dims == other.dims; }
  };

  struct ReductionLoopNestMatch {
    enum class Kind {
      kAdd,
      kMin,
      kMax,
      kBitAnd,
      kBitOr,
      kBitXor,
    };

    llvm::SmallVector<const tir::ForNode*, 4> loops;
    const tir::BlockRealizeNode* block_realize{nullptr};
    const tir::BlockNode* block{nullptr};
    const tir::IfThenElseNode* init_if{nullptr};
    const tir::BufferStoreNode* update_store{nullptr};
    llvm::SmallVector<tir::Var, 4> block_vars;
    llvm::SmallVector<tir::Var, 4> output_vars;
    tir::Buffer output_buffer;
    PrimExpr init_value;
    PrimExpr reduction_expr;
    llvm::SmallVector<size_t, 4> reduction_dims;
    Kind kind{Kind::kAdd};
  };

  class StructuredExprAnalyzer final : private tir::ExprFunctor<bool(const PrimExpr&)> {
  public:
    explicit StructuredExprAnalyzer(llvm::ArrayRef<tir::Var> block_vars)
        : block_vars_(block_vars.begin(), block_vars.end()) {}

    bool Analyze(const PrimExpr& expr) { return VisitExpr(expr); }

    const std::vector<tir::Buffer>& input_buffers() const { return input_buffers_; }

    StructuredIndexPattern input_pattern(const tir::Buffer& buffer) const {
      auto it = input_patterns_.find(buffer.get());
      ICHECK(it != input_patterns_.end())
          << "Missing structured input access for buffer: " << buffer->name;
      return it->second;
    }

  private:
    using tir::ExprFunctor<bool(const PrimExpr&)>::VisitExpr;

    std::optional<StructuredIndexPattern> ClassifyIndices(const Array<PrimExpr>& indices) const {
      StructuredIndexPattern pattern;
      pattern.dims.reserve(indices.size());
      size_t next_block_dim = 0;
      for (const PrimExpr& index : indices) {
        const auto* var = index.as<tir::VarNode>();
        if (var == nullptr) {
          return std::nullopt;
        }
        bool matched = false;
        while (next_block_dim < block_vars_.size()) {
          if (block_vars_[next_block_dim].get() == var) {
            pattern.dims.push_back(static_cast<unsigned>(next_block_dim));
            ++next_block_dim;
            matched = true;
            break;
          }
          ++next_block_dim;
        }
        if (!matched) {
          return std::nullopt;
        }
      }
      return pattern;
    }

    template <typename T>
    bool VisitBinary(const T* op) {
      return VisitExpr(op->a) && VisitExpr(op->b);
    }

    bool VisitExpr_(const tir::VarNode* op) final {
      for (const tir::Var& block_var : block_vars_) {
        if (block_var.get() == op) {
          return false;
        }
      }
      return true;
    }

    bool VisitExpr_(const tir::BufferLoadNode* op) final {
      if (op->predicate.defined() && !tir::is_one(op->predicate.value())) {
        return false;
      }
      std::optional<StructuredIndexPattern> pattern = ClassifyIndices(op->indices);
      if (!pattern.has_value()) {
        return false;
      }
      auto it = input_patterns_.find(op->buffer.get());
      if (it != input_patterns_.end() && !(it->second == pattern.value())) {
        return false;
      }
      input_patterns_[op->buffer.get()] = pattern.value();
      if (std::find_if(input_buffers_.begin(), input_buffers_.end(),
                       [&](const tir::Buffer& buffer) { return buffer.get() == op->buffer.get(); }) ==
          input_buffers_.end()) {
        input_buffers_.push_back(op->buffer);
      }
      return true;
    }

    bool VisitExpr_(const tir::AddNode* op) final { return VisitBinary(op); }
    bool VisitExpr_(const tir::SubNode* op) final { return VisitBinary(op); }
    bool VisitExpr_(const tir::MulNode* op) final { return VisitBinary(op); }
    bool VisitExpr_(const tir::DivNode* op) final { return VisitBinary(op); }
    bool VisitExpr_(const tir::ModNode* op) final { return VisitBinary(op); }
    bool VisitExpr_(const tir::FloorDivNode* op) final { return VisitBinary(op); }
    bool VisitExpr_(const tir::FloorModNode* op) final { return VisitBinary(op); }
    bool VisitExpr_(const tir::MinNode* op) final { return VisitBinary(op); }
    bool VisitExpr_(const tir::MaxNode* op) final { return VisitBinary(op); }
    bool VisitExpr_(const tir::EQNode* op) final { return VisitBinary(op); }
    bool VisitExpr_(const tir::NENode* op) final { return VisitBinary(op); }
    bool VisitExpr_(const tir::LTNode* op) final { return VisitBinary(op); }
    bool VisitExpr_(const tir::LENode* op) final { return VisitBinary(op); }
    bool VisitExpr_(const tir::GTNode* op) final { return VisitBinary(op); }
    bool VisitExpr_(const tir::GENode* op) final { return VisitBinary(op); }
    bool VisitExpr_(const tir::AndNode* op) final { return VisitBinary(op); }
    bool VisitExpr_(const tir::OrNode* op) final { return VisitBinary(op); }
    bool VisitExpr_(const tir::NotNode* op) final { return VisitExpr(op->a); }

    bool VisitExpr_(const tir::SelectNode* op) final {
      return VisitExpr(op->condition) && VisitExpr(op->true_value) && VisitExpr(op->false_value);
    }

    bool VisitExpr_(const tir::CastNode* op) final { return VisitExpr(op->value); }
    bool VisitExpr_(const tir::BroadcastNode* op) final { return VisitExpr(op->value); }
    bool VisitExpr_(const tir::RampNode* op) final {
      return VisitExpr(op->base) && VisitExpr(op->stride);
    }
    bool VisitExpr_(const IntImmNode* op) final {
      (void)op;
      return true;
    }
    bool VisitExpr_(const FloatImmNode* op) final {
      (void)op;
      return true;
    }
    bool VisitExpr_(const tir::CallNode* op) final {
      if (IsInfinityCall(op)) {
        return true;
      }
      if (IsFloat2HalfRZCallExtern(op)) {
        ICHECK_EQ(op->args.size(), 2U)
            << "__float2half_rz call_extern expects one value argument";
        return VisitExpr(op->args[1]);
      }
      if (IsAtomicAddOffsetCallExtern(op)) {
        ICHECK_EQ(op->args.size(), 4U)
            << "tl_atomic_add_offset expects <name, base_ptr, offset, value>";
        return MatchAddressOfBufferLoad(op->args[1]).has_value() && VisitExpr(op->args[2]) &&
               VisitExpr(op->args[3]);
      }
      if (IsMatchSyncCallExtern(op)) {
        ICHECK_EQ(op->args.size(), 3U)
            << GetCallExternName(op) << " call_extern expects <name, mask, value>";
        return VisitExpr(op->args[2]);
      }
      if (IsSupportedBitcastCall(op)) {
        return VisitExpr(op->args[0]);
      }
      if (IsReinterpretCall(op)) {
        return false;
      }
      if (IsSupportedPackedX2IntrinsicCall(op)) {
        if (op->args.size() == 1) {
          return VisitExpr(op->args[0]);
        }
        if (op->args.size() == 2) {
          return VisitExpr(op->args[0]) && VisitExpr(op->args[1]);
        }
        return VisitExpr(op->args[0]) && VisitExpr(op->args[1]) && VisitExpr(op->args[2]);
      }
      if (IsSupportedUnaryMathCall(op) || IsSupportedUnaryIntrinsicCall(op)) {
        return VisitExpr(op->args[0]);
      }
      if (IsSupportedBinaryIntrinsicCall(op)) {
        return VisitExpr(op->args[0]) && VisitExpr(op->args[1]);
      }
      return false;
    }
    bool VisitExprDefault_(const Object* op) final {
      (void)op;
      return false;
    }

    std::vector<tir::Var> block_vars_;
    std::vector<tir::Buffer> input_buffers_;
    std::unordered_map<const Object*, StructuredIndexPattern> input_patterns_;
  };

  class StructuredRegionExprLowerer final
      : private tir::ExprFunctor<mlir::Value(const PrimExpr&)> {
  public:
    StructuredRegionExprLowerer(TIRToMLIRLowerer* outer, llvm::ArrayRef<tir::Var> block_vars,
                                llvm::ArrayRef<tir::Buffer> input_buffers,
                                llvm::ArrayRef<mlir::Value> input_values,
                                llvm::ArrayRef<StructuredIndexPattern> input_patterns)
        : outer_(outer), block_vars_(block_vars.begin(), block_vars.end()) {
      ICHECK_EQ(input_buffers.size(), input_values.size());
      ICHECK_EQ(input_buffers.size(), input_patterns.size());
      for (size_t i = 0; i < input_buffers.size(); ++i) {
        input_values_[input_buffers[i].get()] = {input_values[i], input_patterns[i]};
      }
    }

    mlir::Value Lower(const PrimExpr& expr) { return VisitExpr(expr); }

  private:
    using tir::ExprFunctor<mlir::Value(const PrimExpr&)>::VisitExpr;

    bool IndicesMatch(const Array<PrimExpr>& indices, const StructuredIndexPattern& pattern) const {
      if (indices.size() != pattern.dims.size()) {
        return false;
      }
      for (size_t i = 0; i < indices.size(); ++i) {
        const auto* var = indices[i].as<tir::VarNode>();
        if (var == nullptr || pattern.dims[i] >= block_vars_.size() ||
            var != block_vars_[pattern.dims[i]].get()) {
          return false;
        }
      }
      return true;
    }

    mlir::Value VisitExpr_(const tir::VarNode* op) final {
      for (const tir::Var& block_var : block_vars_) {
        ICHECK(block_var.get() != op)
            << "Direct loop-index use is not supported in structured linalg region lowering";
      }
      return outer_->LookupVarValue(tvm::ffi::GetRef<tir::Var>(op));
    }

    mlir::Value VisitExpr_(const tir::BufferLoadNode* op) final {
      ICHECK(!op->predicate.defined() || tir::is_one(op->predicate.value()))
          << "Predicated buffer loads are not supported in linalg.generic region lowering";
      auto it = input_values_.find(op->buffer.get());
      ICHECK(it != input_values_.end())
          << "Unbound structured input buffer in linalg region lowering: " << op->buffer->name;
      ICHECK(IndicesMatch(op->indices, it->second.pattern))
          << "Unsupported buffer indexing pattern in structured linalg region lowering";
      return outer_->CastValue(it->second.value, op->buffer->dtype, op->dtype);
    }

    mlir::Value VisitExpr_(const tir::AddNode* op) final {
      mlir::Value lhs = outer_->CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
      mlir::Value rhs = outer_->CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
      if (IsFloatLikeType(op->dtype)) {
        return outer_->builder_.create<mlir::arith::AddFOp>(outer_->loc_, lhs, rhs);
      }
      return outer_->builder_.create<mlir::arith::AddIOp>(outer_->loc_, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::SubNode* op) final {
      mlir::Value lhs = outer_->CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
      mlir::Value rhs = outer_->CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
      if (IsFloatLikeType(op->dtype)) {
        return outer_->builder_.create<mlir::arith::SubFOp>(outer_->loc_, lhs, rhs);
      }
      return outer_->builder_.create<mlir::arith::SubIOp>(outer_->loc_, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::MulNode* op) final {
      mlir::Value lhs = outer_->CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
      mlir::Value rhs = outer_->CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
      if (IsFloatLikeType(op->dtype)) {
        return outer_->builder_.create<mlir::arith::MulFOp>(outer_->loc_, lhs, rhs);
      }
      return outer_->builder_.create<mlir::arith::MulIOp>(outer_->loc_, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::DivNode* op) final {
      mlir::Value lhs = outer_->CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
      mlir::Value rhs = outer_->CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
      if (IsFloatLikeType(op->dtype)) {
        return outer_->builder_.create<mlir::arith::DivFOp>(outer_->loc_, lhs, rhs);
      }
      if (op->dtype.is_uint() || op->dtype.is_bool()) {
        return outer_->builder_.create<mlir::arith::DivUIOp>(outer_->loc_, lhs, rhs);
      }
      return outer_->builder_.create<mlir::arith::DivSIOp>(outer_->loc_, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::ModNode* op) final {
      mlir::Value lhs = outer_->CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
      mlir::Value rhs = outer_->CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
      if (IsFloatLikeType(op->dtype)) {
        return outer_->builder_.create<mlir::arith::RemFOp>(outer_->loc_, lhs, rhs);
      }
      if (op->dtype.is_uint() || op->dtype.is_bool()) {
        return outer_->builder_.create<mlir::arith::RemUIOp>(outer_->loc_, lhs, rhs);
      }
      return outer_->builder_.create<mlir::arith::RemSIOp>(outer_->loc_, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::FloorDivNode* op) final {
      mlir::Value lhs = outer_->CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
      mlir::Value rhs = outer_->CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
      ICHECK(!IsFloatLikeType(op->dtype))
          << "tir.FloorDiv on floating-point dtype is not supported yet";
      if (op->dtype.is_uint() || op->dtype.is_bool()) {
        return outer_->builder_.create<mlir::arith::DivUIOp>(outer_->loc_, lhs, rhs);
      }
      return outer_->builder_.create<mlir::arith::FloorDivSIOp>(outer_->loc_, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::FloorModNode* op) final {
      mlir::Value lhs = outer_->CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
      mlir::Value rhs = outer_->CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
      ICHECK(!IsFloatLikeType(op->dtype))
          << "tir.FloorMod on floating-point dtype is not supported yet";
      if (op->dtype.is_uint() || op->dtype.is_bool()) {
        return outer_->builder_.create<mlir::arith::RemUIOp>(outer_->loc_, lhs, rhs);
      }
      mlir::Value quotient = outer_->builder_.create<mlir::arith::FloorDivSIOp>(outer_->loc_, lhs, rhs);
      mlir::Value product = outer_->builder_.create<mlir::arith::MulIOp>(outer_->loc_, quotient, rhs);
      return outer_->builder_.create<mlir::arith::SubIOp>(outer_->loc_, lhs, product);
    }

    mlir::Value VisitExpr_(const tir::MinNode* op) final {
      DataType compare_dtype = op->dtype;
      mlir::Value lhs = outer_->CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
      mlir::Value rhs = outer_->CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
      mlir::Value cond;
      if (IsFloatLikeType(compare_dtype)) {
        cond = outer_->builder_.create<mlir::arith::CmpFOp>(
            outer_->loc_, mlir::arith::CmpFPredicate::OLT, lhs, rhs);
      } else if (compare_dtype.is_uint() || compare_dtype.is_bool()) {
        cond = outer_->builder_.create<mlir::arith::CmpIOp>(
            outer_->loc_, mlir::arith::CmpIPredicate::ult, lhs, rhs);
      } else {
        cond = outer_->builder_.create<mlir::arith::CmpIOp>(
            outer_->loc_, mlir::arith::CmpIPredicate::slt, lhs, rhs);
      }
      return outer_->builder_.create<mlir::arith::SelectOp>(outer_->loc_, cond, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::MaxNode* op) final {
      DataType compare_dtype = op->dtype;
      mlir::Value lhs = outer_->CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
      mlir::Value rhs = outer_->CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
      mlir::Value cond;
      if (IsFloatLikeType(compare_dtype)) {
        cond = outer_->builder_.create<mlir::arith::CmpFOp>(
            outer_->loc_, mlir::arith::CmpFPredicate::OGT, lhs, rhs);
      } else if (compare_dtype.is_uint() || compare_dtype.is_bool()) {
        cond = outer_->builder_.create<mlir::arith::CmpIOp>(
            outer_->loc_, mlir::arith::CmpIPredicate::ugt, lhs, rhs);
      } else {
        cond = outer_->builder_.create<mlir::arith::CmpIOp>(
            outer_->loc_, mlir::arith::CmpIPredicate::sgt, lhs, rhs);
      }
      return outer_->builder_.create<mlir::arith::SelectOp>(outer_->loc_, cond, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::CastNode* op) final {
      mlir::Value value = VisitExpr(op->value);
      return outer_->CastValue(value, op->value.dtype(), op->dtype);
    }

    mlir::Value VisitExpr_(const tir::ShuffleNode* op) final {
      return outer_->LowerShuffle(op);
    }

    mlir::Value VisitExpr_(const tir::CallNode* op) final {
      if (const auto* op_node = op->op.as<OpNode>()) {
        if (IsUnsupportedTileReductionIntrinsicName(op_node->name)) {
          RejectUnsupportedTileReductionIntrinsic(op_node->name);
        }
        if (IsUnsupportedTileScanIntrinsicName(op_node->name)) {
          RejectUnsupportedTileScanIntrinsic(op_node->name);
        }
      }
      if (IsInfinityCall(op)) {
        return outer_->CreateInfinityValue(op->dtype);
      }
      if (IsFloat2HalfRZCallExtern(op)) {
        ICHECK_EQ(op->args.size(), 2U)
            << "__float2half_rz call_extern expects one value argument";
        mlir::Value arg = VisitExpr(op->args[1]);
        return outer_->CastValue(arg, op->args[1].dtype(), op->dtype);
      }
      if (IsAtomicAddOffsetCallExtern(op)) {
        return outer_->LowerAtomicAddOffsetCallExtern(op);
      }
      if (IsMatchSyncCallExtern(op) && outer_->CanLowerCooperativeCallAsSingleThread(op)) {
        return outer_->LowerSingleThreadCooperativeCall(op);
      }
      if (outer_->CanLowerMatchSyncCallAsSerializedWarpReplay(op)) {
        return outer_->LowerSerializedWarpReplayMatchSyncCall(op);
      }
      if (outer_->CanLowerVoteCallAsSerializedWarpReplay(op)) {
        return outer_->LowerSerializedWarpReplayVoteCall(op);
      }
      if (outer_->CanLowerSyncthreadsOrCallAsSerializedThreadReplay(op)) {
        return outer_->LowerSerializedThreadReplaySyncthreadsOrCall(op);
      }
      if (IsSupportedBitcastCall(op)) {
        mlir::Value arg = outer_->CastValue(VisitExpr(op->args[0]), op->args[0].dtype(),
                                            op->args[0].dtype());
        return outer_->LowerBitcastCall(op, arg);
      }
      if (IsReinterpretCall(op)) {
        RejectUnsupportedReinterpretCall(op);
      }
      if (IsSupportedPackedX2IntrinsicCall(op)) {
        mlir::Value lhs =
            outer_->CastValue(VisitExpr(op->args[0]), op->args[0].dtype(), op->dtype);
        if (op->args.size() == 1) {
          return LowerSupportedPackedX2IntrinsicCall(outer_->builder_, outer_->loc_, op, lhs);
        }
        mlir::Value rhs =
            outer_->CastValue(VisitExpr(op->args[1]), op->args[1].dtype(), op->dtype);
        if (op->args.size() == 2) {
          return LowerSupportedPackedX2IntrinsicCall(outer_->builder_, outer_->loc_, op, lhs, rhs);
        }
        mlir::Value extra =
            outer_->CastValue(VisitExpr(op->args[2]), op->args[2].dtype(), op->dtype);
        return LowerSupportedPackedX2IntrinsicCall(outer_->builder_, outer_->loc_, op, lhs, rhs,
                                                   extra);
      }
      if (IsSupportedUnaryMathCall(op)) {
        mlir::Value arg =
            outer_->CastValue(VisitExpr(op->args[0]), op->args[0].dtype(), op->dtype);
        return LowerSupportedUnaryMathCall(outer_->builder_, outer_->loc_, op, arg);
      }
      if (IsSupportedUnaryIntrinsicCall(op)) {
        mlir::Value arg = outer_->CastValue(VisitExpr(op->args[0]), op->args[0].dtype(),
                                            op->args[0].dtype());
        return LowerSupportedUnaryIntrinsicCall(outer_->builder_, outer_->loc_, op, arg);
      }
      if (IsSupportedBinaryIntrinsicCall(op)) {
        mlir::Value lhs =
            outer_->CastValue(VisitExpr(op->args[0]), op->args[0].dtype(), op->dtype);
        mlir::Value rhs =
            outer_->CastValue(VisitExpr(op->args[1]), op->args[1].dtype(), op->dtype);
        return LowerSupportedBinaryIntrinsicCall(outer_->builder_, outer_->loc_, op, lhs, rhs);
      }
      if (outer_->CanLowerCooperativeCallAsSingleThread(op)) {
        return outer_->LowerSingleThreadCooperativeCall(op);
      }
      LOG(FATAL) << "Unsupported TIR call in elementwise linalg.generic lowering: " << op->op;
      TVM_FFI_UNREACHABLE();
    }

    mlir::Value VisitExpr_(const tir::EQNode* op) final {
      DataType compare_dtype = op->a.dtype();
      mlir::Value lhs = outer_->CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
      mlir::Value rhs = outer_->CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
      if (IsFloatLikeType(compare_dtype)) {
        return outer_->builder_.create<mlir::arith::CmpFOp>(
            outer_->loc_, mlir::arith::CmpFPredicate::OEQ, lhs, rhs);
      }
      return outer_->builder_.create<mlir::arith::CmpIOp>(outer_->loc_,
                                                          mlir::arith::CmpIPredicate::eq, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::NENode* op) final {
      DataType compare_dtype = op->a.dtype();
      mlir::Value lhs = outer_->CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
      mlir::Value rhs = outer_->CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
      if (IsFloatLikeType(compare_dtype)) {
        return outer_->builder_.create<mlir::arith::CmpFOp>(
            outer_->loc_, mlir::arith::CmpFPredicate::UNE, lhs, rhs);
      }
      return outer_->builder_.create<mlir::arith::CmpIOp>(outer_->loc_,
                                                          mlir::arith::CmpIPredicate::ne, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::LTNode* op) final {
      DataType compare_dtype = op->a.dtype();
      mlir::Value lhs = outer_->CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
      mlir::Value rhs = outer_->CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
      if (IsFloatLikeType(compare_dtype)) {
        return outer_->builder_.create<mlir::arith::CmpFOp>(
            outer_->loc_, mlir::arith::CmpFPredicate::OLT, lhs, rhs);
      }
      mlir::arith::CmpIPredicate predicate =
          compare_dtype.is_uint() || compare_dtype.is_bool() ? mlir::arith::CmpIPredicate::ult
                                                             : mlir::arith::CmpIPredicate::slt;
      return outer_->builder_.create<mlir::arith::CmpIOp>(outer_->loc_, predicate, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::LENode* op) final {
      DataType compare_dtype = op->a.dtype();
      mlir::Value lhs = outer_->CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
      mlir::Value rhs = outer_->CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
      if (IsFloatLikeType(compare_dtype)) {
        return outer_->builder_.create<mlir::arith::CmpFOp>(
            outer_->loc_, mlir::arith::CmpFPredicate::OLE, lhs, rhs);
      }
      mlir::arith::CmpIPredicate predicate =
          compare_dtype.is_uint() || compare_dtype.is_bool() ? mlir::arith::CmpIPredicate::ule
                                                             : mlir::arith::CmpIPredicate::sle;
      return outer_->builder_.create<mlir::arith::CmpIOp>(outer_->loc_, predicate, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::GTNode* op) final {
      DataType compare_dtype = op->a.dtype();
      mlir::Value lhs = outer_->CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
      mlir::Value rhs = outer_->CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
      if (IsFloatLikeType(compare_dtype)) {
        return outer_->builder_.create<mlir::arith::CmpFOp>(
            outer_->loc_, mlir::arith::CmpFPredicate::OGT, lhs, rhs);
      }
      mlir::arith::CmpIPredicate predicate =
          compare_dtype.is_uint() || compare_dtype.is_bool() ? mlir::arith::CmpIPredicate::ugt
                                                             : mlir::arith::CmpIPredicate::sgt;
      return outer_->builder_.create<mlir::arith::CmpIOp>(outer_->loc_, predicate, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::GENode* op) final {
      DataType compare_dtype = op->a.dtype();
      mlir::Value lhs = outer_->CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
      mlir::Value rhs = outer_->CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
      if (IsFloatLikeType(compare_dtype)) {
        return outer_->builder_.create<mlir::arith::CmpFOp>(
            outer_->loc_, mlir::arith::CmpFPredicate::OGE, lhs, rhs);
      }
      mlir::arith::CmpIPredicate predicate =
          compare_dtype.is_uint() || compare_dtype.is_bool() ? mlir::arith::CmpIPredicate::uge
                                                             : mlir::arith::CmpIPredicate::sge;
      return outer_->builder_.create<mlir::arith::CmpIOp>(outer_->loc_, predicate, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::AndNode* op) final {
      mlir::Value lhs = outer_->LowerLogicalValue(VisitExpr(op->a), op->a.dtype());
      mlir::Value rhs = outer_->LowerLogicalValue(VisitExpr(op->b), op->b.dtype());
      return outer_->builder_.create<mlir::arith::AndIOp>(outer_->loc_, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::OrNode* op) final {
      mlir::Value lhs = outer_->LowerLogicalValue(VisitExpr(op->a), op->a.dtype());
      mlir::Value rhs = outer_->LowerLogicalValue(VisitExpr(op->b), op->b.dtype());
      return outer_->builder_.create<mlir::arith::OrIOp>(outer_->loc_, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::NotNode* op) final {
      mlir::Value value = outer_->LowerLogicalValue(VisitExpr(op->a), op->a.dtype());
      mlir::Value one = outer_->CreateBooleanTrueLike(value.getType());
      return outer_->builder_.create<mlir::arith::XOrIOp>(outer_->loc_, value, one);
    }

    mlir::Value VisitExpr_(const tir::SelectNode* op) final {
      mlir::Value cond = outer_->LowerLogicalValue(VisitExpr(op->condition), op->condition.dtype());
      mlir::Value true_value =
          outer_->CastValue(VisitExpr(op->true_value), op->true_value.dtype(), op->dtype);
      mlir::Value false_value =
          outer_->CastValue(VisitExpr(op->false_value), op->false_value.dtype(), op->dtype);
      return outer_->builder_.create<mlir::arith::SelectOp>(outer_->loc_, cond, true_value,
                                                            false_value);
    }

    mlir::Value VisitExpr_(const tir::BroadcastNode* op) final {
      return outer_->LowerBroadcast(op);
    }

    mlir::Value VisitExpr_(const tir::RampNode* op) final { return outer_->LowerRamp(op); }

    mlir::Value VisitExpr_(const IntImmNode* op) final {
      mlir::Type type = outer_->LowerScalarType(op->dtype);
      return outer_->ConstantIntLike(op->value, type);
    }

    mlir::Value VisitExpr_(const FloatImmNode* op) final {
      mlir::FloatType type = mlir::cast<mlir::FloatType>(outer_->LowerScalarType(op->dtype));
      return outer_->builder_.create<mlir::arith::ConstantOp>(
          outer_->loc_, outer_->builder_.getFloatAttr(type, op->value));
    }

    mlir::Value VisitExprDefault_(const Object* op) final {
      LOG(FATAL) << "Unsupported TIR expr for linalg.generic region lowering: " << op->GetTypeKey();
      TVM_FFI_UNREACHABLE();
    }

    TIRToMLIRLowerer* outer_;
    std::vector<tir::Var> block_vars_;
    struct InputBinding {
      mlir::Value value;
      StructuredIndexPattern pattern;
    };
    std::unordered_map<const Object*, InputBinding> input_values_;
  };

  using ValueMap = std::unordered_map<const Object*, mlir::Value>;
  using PrimExprMap = std::unordered_map<const Object*, PrimExpr>;
  using SerializedWarpReplayBufferElementExprMap =
      std::unordered_map<SerializedWarpReplayBufferElementKey, PrimExpr,
                         SerializedWarpReplayBufferElementKeyHash>;
  using PackedScalarViewMap = std::unordered_map<const Object*, PackedScalarViewBinding>;
  using PackedDataOwnerMap = std::unordered_map<const Object*, tir::Buffer>;
  using BufferOwnerMap = std::unordered_map<const Object*, tir::Buffer>;

  SavedBinding SaveAndSet(ValueMap& map, const Object* key, mlir::Value value) {
    SavedBinding saved;
    auto it = map.find(key);
    if (it != map.end()) {
      saved.had_value = true;
      saved.value = it->second;
    }
    map[key] = value;
    return saved;
  }

  void RestoreBinding(ValueMap& map, const Object* key, const SavedBinding& saved) {
    if (saved.had_value) {
      map[key] = saved.value;
    } else {
      map.erase(key);
    }
  }

  std::optional<PrimExpr> SaveAndSetPrimExpr(PrimExprMap& map, const Object* key,
                                             const PrimExpr& value) {
    auto it = map.find(key);
    if (it == map.end()) {
      map[key] = value;
      return std::nullopt;
    }
    PrimExpr saved = it->second;
    it->second = value;
    return saved;
  }

  void RestorePrimExprBinding(PrimExprMap& map, const Object* key,
                              const std::optional<PrimExpr>& saved) {
    if (saved.has_value()) {
      map[key] = saved.value();
    } else {
      map.erase(key);
    }
  }

  void RestorePackedScalarViewBinding(PackedScalarViewMap& map, const Object* key,
                                      const std::optional<PackedScalarViewBinding>& saved) {
    if (saved.has_value()) {
      map[key] = saved.value();
    } else {
      map.erase(key);
    }
  }

  void RestorePackedDataOwnerBinding(PackedDataOwnerMap& map, const Object* key,
                                     const std::optional<tir::Buffer>& saved) {
    if (saved.has_value()) {
      map[key] = saved.value();
    } else {
      map.erase(key);
    }
  }

  void RestoreBufferOwnerBinding(BufferOwnerMap& map, const Object* key,
                                 const std::optional<tir::Buffer>& saved) {
    if (saved.has_value()) {
      map[key] = saved.value();
    } else {
      map.erase(key);
    }
  }

  mlir::Type LowerScalarType(DataType dtype) {
    if (dtype.lanes() > 1) {
      return mlir::VectorType::get({dtype.lanes()}, LowerScalarType(dtype.element_of()));
    }
    ICHECK_EQ(dtype.lanes(), 1) << "Unsupported vector lane count in riscv: " << dtype;
    if (dtype.is_bool()) {
      return builder_.getI1Type();
    }
    if (dtype.is_int() || dtype.is_uint()) {
      return builder_.getIntegerType(dtype.bits());
    }
    if (dtype.is_float16()) {
      return builder_.getF16Type();
    }
    if (dtype.is_bfloat16()) {
      return builder_.getBF16Type();
    }
    if (dtype.is_float8_e3m4()) {
      return builder_.getType<mlir::Float8E3M4Type>();
    }
    if (dtype.is_float8_e4m3()) {
      return builder_.getType<mlir::Float8E4M3Type>();
    }
    if (dtype.is_float8_e4m3b11fnuz()) {
      return builder_.getType<mlir::Float8E4M3B11FNUZType>();
    }
    if (dtype.is_float8_e4m3fn()) {
      return builder_.getType<mlir::Float8E4M3FNType>();
    }
    if (dtype.is_float8_e4m3fnuz()) {
      return builder_.getType<mlir::Float8E4M3FNUZType>();
    }
    if (dtype.is_float8_e5m2()) {
      return builder_.getType<mlir::Float8E5M2Type>();
    }
    if (dtype.is_float8_e5m2fnuz()) {
      return builder_.getType<mlir::Float8E5M2FNUZType>();
    }
    if (dtype.is_float8_e8m0fnu()) {
      return builder_.getType<mlir::Float8E8M0FNUType>();
    }
    if (dtype.is_float4_e2m1fn()) {
      return builder_.getType<mlir::Float4E2M1FNType>();
    }
    if (dtype.is_float() && dtype.bits() == 32) {
      return builder_.getF32Type();
    }
    if (dtype.is_float() && dtype.bits() == 64) {
      return builder_.getF64Type();
    }
    LOG(FATAL) << "Unsupported scalar dtype in riscv MLIR lowering: " << dtype;
    TVM_FFI_UNREACHABLE();
  }

  mlir::Value ConstantIntLike(int64_t value, mlir::Type type) {
    if (type.isIndex()) {
      return builder_.create<mlir::arith::ConstantIndexOp>(loc_, value);
    }
    return builder_.create<mlir::arith::ConstantIntOp>(loc_, type, value);
  }

  mlir::Value ZeroIndex() { return ConstantIntLike(0, builder_.getIndexType()); }

  mlir::MemRefType LowerMemRefType(DataType element_dtype, const Array<PrimExpr>& shape_exprs) {
    llvm::SmallVector<int64_t, 4> shape;
    shape.reserve(shape_exprs.size());
    for (const PrimExpr& dim : shape_exprs) {
      if (const auto* imm = dim.as<IntImmNode>()) {
        shape.push_back(imm->value);
      } else {
        shape.push_back(mlir::ShapedType::kDynamic);
      }
    }
    return mlir::MemRefType::get(shape, LowerScalarType(element_dtype));
  }

  llvm::SmallVector<int64_t, 4> LowerStaticRowMajorStrides(const Array<PrimExpr>& shape_exprs) {
    llvm::SmallVector<int64_t, 4> strides(shape_exprs.size(), mlir::ShapedType::kDynamic);
    int64_t stride = 1;
    for (int i = static_cast<int>(shape_exprs.size()) - 1; i >= 0; --i) {
      strides[i] = stride;
      const auto* imm = shape_exprs[i].as<IntImmNode>();
      if (imm == nullptr || stride == mlir::ShapedType::kDynamic) {
        stride = mlir::ShapedType::kDynamic;
      } else {
        stride *= imm->value;
      }
    }
    return strides;
  }

  llvm::SmallVector<int64_t, 4> LowerBufferStrides(const tir::Buffer& buffer) {
    if (buffer->strides.empty()) {
      return LowerStaticRowMajorStrides(buffer->shape);
    }
    llvm::SmallVector<int64_t, 4> strides;
    strides.reserve(buffer->strides.size());
    for (const PrimExpr& stride : buffer->strides) {
      std::optional<int64_t> static_stride = GetOptionalStaticInt(stride);
      if (!static_stride.has_value()) {
        strides.push_back(mlir::ShapedType::kDynamic);
        continue;
      }
      ICHECK_GT(static_stride.value(), 0)
          << "Static buffer strides must be positive in riscv: " << buffer->name;
      strides.push_back(static_stride.value());
    }
    return strides;
  }

  mlir::MemRefType LowerBufferMemRefType(const tir::Buffer& buffer) {
    mlir::MemRefType base_type = LowerMemRefType(buffer->dtype, buffer->shape);
    if (HasCompactRowMajorLayout(buffer) && tir::is_zero(buffer->elem_offset)) {
      return base_type;
    }

    std::optional<int64_t> static_offset = GetOptionalStaticInt(buffer->elem_offset);
    int64_t offset = mlir::ShapedType::kDynamic;
    if (static_offset.has_value()) {
      ICHECK_GE(static_offset.value(), 0)
          << "Negative elem_offset is not supported for riscv: " << buffer->name;
      offset = static_offset.value();
    }
    mlir::StridedLayoutAttr layout =
        mlir::StridedLayoutAttr::get(&context_, offset, LowerBufferStrides(buffer));
    return mlir::MemRefType::get(base_type.getShape(), base_type.getElementType(), layout);
  }

  llvm::SmallVector<int64_t, 4> LowerStaticShape(const Array<PrimExpr>& shape_exprs) {
    llvm::SmallVector<int64_t, 4> shape;
    shape.reserve(shape_exprs.size());
    for (const PrimExpr& dim : shape_exprs) {
      if (const auto* imm = dim.as<IntImmNode>()) {
        shape.push_back(imm->value);
      } else {
        shape.push_back(mlir::ShapedType::kDynamic);
      }
    }
    return shape;
  }

  llvm::SmallVector<mlir::Value, 4> LowerDynamicSizes(const Array<PrimExpr>& shape_exprs) {
    llvm::SmallVector<mlir::Value, 4> dynamic_sizes;
    for (const PrimExpr& dim : shape_exprs) {
      if (!dim.as<IntImmNode>()) {
        dynamic_sizes.push_back(AsIndex(VisitExpr(dim), dim.dtype()));
      }
    }
    return dynamic_sizes;
  }

  llvm::SmallVector<mlir::Value, 4> LowerDynamicLayoutSymbols(const tir::Buffer& buffer) {
    llvm::SmallVector<mlir::Value, 4> symbol_operands;
    if (HasDynamicElemOffset(buffer)) {
      symbol_operands.push_back(AsIndex(VisitExpr(buffer->elem_offset), buffer->elem_offset.dtype()));
    }
    for (const PrimExpr& stride : buffer->strides) {
      if (!stride.as<IntImmNode>()) {
        symbol_operands.push_back(AsIndex(VisitExpr(stride), stride.dtype()));
      }
    }
    return symbol_operands;
  }

  llvm::SmallVector<mlir::OpFoldResult, 4> LowerBufferShapeOpFoldResults(
      const tir::Buffer& buffer) {
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    sizes.reserve(buffer->shape.size());
    for (const PrimExpr& dim : buffer->shape) {
      sizes.push_back(LowerIndexOpFoldResult(dim));
    }
    return sizes;
  }

  llvm::SmallVector<mlir::OpFoldResult, 4> LowerBufferStrideOpFoldResults(
      const tir::Buffer& buffer) {
    llvm::SmallVector<mlir::OpFoldResult, 4> strides;
    strides.reserve(buffer->shape.size());
    if (!buffer->strides.empty()) {
      for (const PrimExpr& stride : buffer->strides) {
        strides.push_back(LowerIndexOpFoldResult(stride));
      }
      return strides;
    }

    arith::Analyzer analyzer;
    PrimExpr running_stride = Integer(1);
    std::vector<PrimExpr> row_major(buffer->shape.size());
    for (int i = static_cast<int>(buffer->shape.size()) - 1; i >= 0; --i) {
      row_major[i] = running_stride;
      running_stride = analyzer.Simplify(running_stride * buffer->shape[i]);
    }
    for (const PrimExpr& stride : row_major) {
      strides.push_back(LowerIndexOpFoldResult(stride));
    }
    return strides;
  }

  mlir::OpFoldResult LowerIndexOpFoldResult(const PrimExpr& expr) {
    if (const auto* imm = expr.as<IntImmNode>()) {
      return builder_.getIndexAttr(imm->value);
    }
    return AsIndex(VisitExpr(expr), expr.dtype());
  }

  llvm::SmallVector<mlir::OpFoldResult, 4> LowerRegionOffsets(const Array<Range>& region) {
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    offsets.reserve(region.size());
    for (const Range& range : region) {
      offsets.push_back(LowerIndexOpFoldResult(range->min));
    }
    return offsets;
  }

  llvm::SmallVector<mlir::OpFoldResult, 4> LowerRegionSizes(const Array<Range>& region) {
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    sizes.reserve(region.size());
    for (const Range& range : region) {
      sizes.push_back(LowerIndexOpFoldResult(range->extent));
    }
    return sizes;
  }

  llvm::SmallVector<mlir::OpFoldResult, 4> UnitStrides(size_t rank) {
    llvm::SmallVector<mlir::OpFoldResult, 4> strides;
    strides.reserve(rank);
    for (size_t i = 0; i < rank; ++i) {
      strides.push_back(builder_.getIndexAttr(1));
    }
    return strides;
  }

  bool IsStaticOne(const PrimExpr& expr) {
    if (const auto* imm = expr.as<IntImmNode>()) {
      return imm->value == 1;
    }
    return false;
  }

  bool AreStaticEqual(const PrimExpr& lhs, const PrimExpr& rhs) {
    const auto* lhs_imm = lhs.as<IntImmNode>();
    const auto* rhs_imm = rhs.as<IntImmNode>();
    return lhs_imm != nullptr && rhs_imm != nullptr && lhs_imm->value == rhs_imm->value;
  }

  mlir::Value LinearizeCompactBufferIndices(const tir::Buffer& buffer,
                                            llvm::ArrayRef<mlir::Value> indices) {
    ICHECK_EQ(indices.size(), buffer->shape.size())
        << "Packed scalar view index rank mismatch in riscv: " << buffer->name;
    ICHECK(!indices.empty())
        << "Packed scalar view currently expects at least one index in riscv";
    mlir::Value linear = indices[0];
    for (size_t dim = 1; dim < indices.size(); ++dim) {
      mlir::Value extent = AsIndex(VisitExpr(buffer->shape[dim]), buffer->shape[dim].dtype());
      linear = builder_.create<mlir::arith::MulIOp>(loc_, linear, extent);
      linear = builder_.create<mlir::arith::AddIOp>(loc_, linear, indices[dim]);
    }
    return linear;
  }

  bool IndicesMatchIdentityVars(const Array<PrimExpr>& indices, llvm::ArrayRef<tir::Var> vars) {
    if (indices.size() != vars.size()) {
      return false;
    }
    for (size_t i = 0; i < indices.size(); ++i) {
      const auto* var = indices[i].as<tir::VarNode>();
      if (var == nullptr || var != vars[i].get()) {
        return false;
      }
    }
    return true;
  }

  bool ExprMatchesVar(const PrimExpr& expr, const tir::Var& var) {
    const auto* expr_var = expr.as<tir::VarNode>();
    return expr_var != nullptr && expr_var == var.get();
  }

  bool IsZeroValue(const PrimExpr& expr) { return tir::is_zero(expr); }

  bool CollectVarsEqualZero(const PrimExpr& expr,
                            llvm::SmallVectorImpl<const tir::VarNode*>* vars) {
    if (const auto* and_node = expr.as<tir::AndNode>()) {
      return CollectVarsEqualZero(and_node->a, vars) && CollectVarsEqualZero(and_node->b, vars);
    }
    const auto* eq = expr.as<tir::EQNode>();
    if (eq == nullptr) {
      return false;
    }
    if (const auto* lhs = eq->a.as<tir::VarNode>(); lhs != nullptr && IsZeroValue(eq->b)) {
      vars->push_back(lhs);
      return true;
    }
    if (const auto* rhs = eq->b.as<tir::VarNode>(); rhs != nullptr && IsZeroValue(eq->a)) {
      vars->push_back(rhs);
      return true;
    }
    return false;
  }

  bool MatchesReductionInitCondition(const PrimExpr& expr,
                                     llvm::ArrayRef<tir::Var> reduction_vars) {
    llvm::SmallVector<const tir::VarNode*, 4> init_vars;
    if (!CollectVarsEqualZero(expr, &init_vars) || init_vars.size() != reduction_vars.size()) {
      return false;
    }
    for (const tir::Var& reduction_var : reduction_vars) {
      bool matched = false;
      for (const tir::VarNode* init_var : init_vars) {
        if (init_var == reduction_var.get()) {
          matched = true;
          break;
        }
      }
      if (!matched) {
        return false;
      }
    }
    return true;
  }

  bool ExprUsesVar(const PrimExpr& expr, const tir::Var& var) const {
    return tir::UsesVar(expr, [target = var.get()](const tir::VarNode* candidate) {
      return candidate == target;
    });
  }

  bool BufferUsesVar(const tir::Buffer& buffer, const tir::Var& var) const {
    for (const PrimExpr& dim : buffer->shape) {
      if (ExprUsesVar(dim, var)) {
        return true;
      }
    }
    for (const PrimExpr& stride : buffer->strides) {
      if (ExprUsesVar(stride, var)) {
        return true;
      }
    }
    return ExprUsesVar(buffer->elem_offset, var);
  }

  bool MatchBufferUsesVar(const tir::MatchBufferRegion& match_buffer, const tir::Var& var) const {
    if (BufferUsesVar(match_buffer->buffer, var)) {
      return true;
    }
    for (const Range& range : match_buffer->source->region) {
      if (ExprUsesVar(range->min, var) || ExprUsesVar(range->extent, var)) {
        return true;
      }
    }
    return false;
  }

  void CollectBuffersBackedByVar(const tir::Stmt& stmt, const tir::Var& var,
                                 std::vector<tir::Buffer>* buffers) const {
    ICHECK(buffers != nullptr);
    tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
      auto collect = [&](const tir::Buffer& buffer) {
        if (!buffer.defined() || buffer->data.get() != var.get()) {
          return;
        }
        auto it = std::find_if(buffers->begin(), buffers->end(), [&](const tir::Buffer& candidate) {
          return candidate.get() == buffer.get();
        });
        if (it == buffers->end()) {
          buffers->push_back(buffer);
        }
      };
      if (const auto* load = node.as<tir::BufferLoadNode>()) {
        collect(load->buffer);
        return;
      }
      if (const auto* store = node.as<tir::BufferStoreNode>()) {
        collect(store->buffer);
        return;
      }
      if (const auto* region = node.as<tir::BufferRegionNode>()) {
        collect(region->buffer);
        return;
      }
      if (const auto* decl = node.as<tir::DeclBufferNode>()) {
        collect(decl->buffer);
      }
    });
  }

  bool HaveEquivalentPointerBackedBufferViews(const tir::Buffer& lhs, const tir::Buffer& rhs) const {
    if (lhs->dtype != rhs->dtype || lhs.scope() != rhs.scope() ||
        lhs->shape.size() != rhs->shape.size() || lhs->strides.size() != rhs->strides.size()) {
      return false;
    }
    arith::Analyzer analyzer;
    for (size_t i = 0; i < lhs->shape.size(); ++i) {
      if (!analyzer.CanProveEqual(lhs->shape[i], rhs->shape[i])) {
        return false;
      }
    }
    for (size_t i = 0; i < lhs->strides.size(); ++i) {
      if (!analyzer.CanProveEqual(lhs->strides[i], rhs->strides[i])) {
        return false;
      }
    }
    return analyzer.CanProveEqual(lhs->elem_offset, rhs->elem_offset);
  }

  bool IsPointerBackedHandleReinterpretLet(const tir::LetStmtNode* let,
                                           const tir::CallNode** reinterpret_call) const {
    if (reinterpret_call != nullptr) {
      *reinterpret_call = nullptr;
    }
    if (let == nullptr || !let->var.defined() || !let->var->dtype.is_handle()) {
      return false;
    }
    if (let->var->type_annotation.as<PointerTypeNode>() == nullptr) {
      return false;
    }
    const auto* call = let->value.as<tir::CallNode>();
    if (call == nullptr || !IsReinterpretCall(call) || call->args.size() != 1U ||
        !call->dtype.is_handle() || !IsIntegerLikeType(call->args[0].dtype()) ||
        call->args[0].dtype().lanes() != 1) {
      return false;
    }
    if (reinterpret_call != nullptr) {
      *reinterpret_call = call;
    }
    return true;
  }

  bool BufferIsBound(const tir::Buffer& buffer) const {
    return buffer_values_.count(buffer.get()) != 0 ||
           buffer_values_.count(buffer->data.get()) != 0;
  }

  void AppendUniqueBuffer(const tir::Buffer& buffer, std::vector<tir::Buffer>* buffers) const {
    auto it = std::find_if(buffers->begin(), buffers->end(), [&](const tir::Buffer& candidate) {
      return candidate.get() == buffer.get() || candidate->data.get() == buffer->data.get();
    });
    if (it == buffers->end()) {
      buffers->push_back(buffer);
    }
  }

  bool AllBlockAllocBuffersAreShared(const ffi::Array<tir::Buffer>& alloc_buffers) const {
    for (const tir::Buffer& buffer : alloc_buffers) {
      if (!tl::IsSharedBuffer(buffer)) {
        return false;
      }
    }
    return true;
  }

  bool SameBuffer(const tir::Buffer& lhs, const tir::Buffer& rhs) const {
    return lhs.get() == rhs.get() || lhs->data.get() == rhs->data.get();
  }

  bool StmtUsesBuffer(const tir::Stmt& stmt, const tir::Buffer& buffer) const {
    bool found = false;
    tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
      if (found) {
        return;
      }
      if (const auto* load = node.as<tir::BufferLoadNode>()) {
        found = SameBuffer(load->buffer, buffer);
        return;
      }
      if (const auto* store = node.as<tir::BufferStoreNode>()) {
        found = SameBuffer(store->buffer, buffer);
        return;
      }
      if (const auto* region = node.as<tir::BufferRegionNode>()) {
        found = SameBuffer(region->buffer, buffer);
      }
    });
    return found;
  }

  void CollectSharedBufferStores(const tir::Stmt& stmt, std::vector<tir::Buffer>* buffers) const {
    ICHECK(buffers != nullptr);
    tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
      const auto* store = node.as<tir::BufferStoreNode>();
      if (store == nullptr || !tl::IsSharedBuffer(store->buffer)) {
        return;
      }
      AppendUniqueBuffer(store->buffer, buffers);
    });
  }

  bool StmtReadsAnySharedBuffer(const tir::Stmt& stmt,
                                llvm::ArrayRef<tir::Buffer> buffers) const {
    if (buffers.empty()) {
      return false;
    }
    bool found = false;
    tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
      if (found) {
        return;
      }
      const auto* load = node.as<tir::BufferLoadNode>();
      if (load == nullptr || !tl::IsSharedBuffer(load->buffer)) {
        return;
      }
      found = std::any_of(buffers.begin(), buffers.end(), [&](const tir::Buffer& buffer) {
        return SameBuffer(load->buffer, buffer);
      });
    });
    return found;
  }

  bool StmtContainsCooperativeThreadIntrinsic(const tir::Stmt& stmt) const {
    bool found = false;
    tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
      if (found) {
        return;
      }
      const auto* call = node.as<tir::CallNode>();
      if (call == nullptr) {
        return;
      }
      const auto* op_node = call->op.as<OpNode>();
      found = op_node != nullptr && IsCooperativeThreadIntrinsicName(op_node->name);
    });
    return found;
  }

  bool StmtUsesVar(const tir::Stmt& stmt, const tir::Var& var) const {
    bool found = false;
    tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
      if (found) {
        return;
      }
      const auto* candidate = node.as<tir::VarNode>();
      found = candidate != nullptr && candidate == var.get();
    });
    return found;
  }

  void CollectSharedBlockAllocBuffers(const tir::Stmt& stmt,
                                      std::vector<tir::Buffer>* buffers) const {
    if (const auto* seq = stmt.as<tir::SeqStmtNode>()) {
      for (const tir::Stmt& child : seq->seq) {
        CollectSharedBlockAllocBuffers(child, buffers);
      }
      return;
    }
    if (const auto* attr = stmt.as<tir::AttrStmtNode>()) {
      CollectSharedBlockAllocBuffers(attr->body, buffers);
      return;
    }
    if (const auto* realize = stmt.as<tir::BlockRealizeNode>()) {
      const tir::BlockNode* block = realize->block.as<tir::BlockNode>();
      ICHECK(block != nullptr);
      for (const tir::Buffer& buffer : block->alloc_buffers) {
        if (tl::IsSharedBuffer(buffer)) {
          AppendUniqueBuffer(buffer, buffers);
        }
      }
      CollectSharedBlockAllocBuffers(block->body, buffers);
      return;
    }
    if (const auto* block = stmt.as<tir::BlockNode>()) {
      for (const tir::Buffer& buffer : block->alloc_buffers) {
        if (tl::IsSharedBuffer(buffer)) {
          AppendUniqueBuffer(buffer, buffers);
        }
      }
      CollectSharedBlockAllocBuffers(block->body, buffers);
      return;
    }
    if (const auto* if_node = stmt.as<tir::IfThenElseNode>()) {
      CollectSharedBlockAllocBuffers(if_node->then_case, buffers);
      if (if_node->else_case.defined()) {
        CollectSharedBlockAllocBuffers(if_node->else_case.value(), buffers);
      }
      return;
    }
    if (const auto* for_node = stmt.as<tir::ForNode>()) {
      CollectSharedBlockAllocBuffers(for_node->body, buffers);
      return;
    }
    if (const auto* let = stmt.as<tir::LetStmtNode>()) {
      CollectSharedBlockAllocBuffers(let->body, buffers);
      return;
    }
    if (const auto* allocate = stmt.as<tir::AllocateNode>()) {
      CollectSharedBlockAllocBuffers(allocate->body, buffers);
      return;
    }
    if (const auto* realize = stmt.as<tir::BufferRealizeNode>()) {
      CollectSharedBlockAllocBuffers(realize->body, buffers);
      return;
    }
  }

  void CollectLocalVarBlockAllocBuffers(const tir::Stmt& stmt,
                                        std::vector<tir::Buffer>* buffers) const {
    if (const auto* seq = stmt.as<tir::SeqStmtNode>()) {
      for (const tir::Stmt& child : seq->seq) {
        CollectLocalVarBlockAllocBuffers(child, buffers);
      }
      return;
    }
    if (const auto* attr = stmt.as<tir::AttrStmtNode>()) {
      CollectLocalVarBlockAllocBuffers(attr->body, buffers);
      return;
    }
    if (const auto* realize = stmt.as<tir::BlockRealizeNode>()) {
      const tir::BlockNode* block = realize->block.as<tir::BlockNode>();
      ICHECK(block != nullptr);
      for (const tir::Buffer& buffer : block->alloc_buffers) {
        if (buffer.defined() && buffer.scope() == "local.var") {
          AppendUniqueBuffer(buffer, buffers);
        }
      }
      CollectLocalVarBlockAllocBuffers(block->body, buffers);
      return;
    }
    if (const auto* block = stmt.as<tir::BlockNode>()) {
      for (const tir::Buffer& buffer : block->alloc_buffers) {
        if (buffer.defined() && buffer.scope() == "local.var") {
          AppendUniqueBuffer(buffer, buffers);
        }
      }
      CollectLocalVarBlockAllocBuffers(block->body, buffers);
      return;
    }
    if (const auto* if_node = stmt.as<tir::IfThenElseNode>()) {
      CollectLocalVarBlockAllocBuffers(if_node->then_case, buffers);
      if (if_node->else_case.defined()) {
        CollectLocalVarBlockAllocBuffers(if_node->else_case.value(), buffers);
      }
      return;
    }
    if (const auto* for_node = stmt.as<tir::ForNode>()) {
      CollectLocalVarBlockAllocBuffers(for_node->body, buffers);
      return;
    }
    if (const auto* let = stmt.as<tir::LetStmtNode>()) {
      CollectLocalVarBlockAllocBuffers(let->body, buffers);
      return;
    }
    if (const auto* allocate = stmt.as<tir::AllocateNode>()) {
      CollectLocalVarBlockAllocBuffers(allocate->body, buffers);
      return;
    }
    if (const auto* realize = stmt.as<tir::BufferRealizeNode>()) {
      CollectLocalVarBlockAllocBuffers(realize->body, buffers);
      return;
    }
  }

  void CollectSerializedWarpReplayThreadLocalBlockAllocBuffers(
      const tir::Stmt& stmt, llvm::ArrayRef<ThreadLocalBlockAllocBinding> existing_bindings,
      std::vector<tir::Buffer>* buffers) const {
    std::unordered_set<const Object*> replay_operands =
        CollectCooperativeReplayOperandBuffers(stmt);
    if (replay_operands.empty()) {
      return;
    }
    std::unordered_set<const Object*> seen;
    tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
      const auto* block = node.as<tir::BlockNode>();
      if (block == nullptr) {
        return;
      }
      for (const tir::Buffer& buffer : block->alloc_buffers) {
        if (tl::IsSharedBuffer(buffer) ||
            ThreadLocalBindingsContainBuffer(existing_bindings, buffer) ||
            !CanMaterializeThreadLocalBlockAllocForSerializedWarpReplay(buffer)) {
          continue;
        }
        if (replay_operands.count(buffer.get()) == 0 &&
            replay_operands.count(buffer->data.get()) == 0) {
          continue;
        }
        if (seen.insert(buffer.get()).second) {
          buffers->push_back(buffer);
        }
      }
    });
  }

  bool IsPreboundThreadLocalBlockBuffer(const tir::Buffer& buffer) const {
    auto contains = [&](const Object* key) {
      auto it = prebound_thread_local_block_buffers_.find(key);
      return it != prebound_thread_local_block_buffers_.end() && it->second > 0;
    };
    return contains(buffer.get()) || contains(buffer->data.get());
  }

  bool IsScalarLocalVarBuffer(const tir::Buffer& buffer) const {
    if (!buffer.defined() || buffer.scope() != "local.var" || buffer->shape.size() != 1) {
      return false;
    }
    std::optional<int64_t> extent = GetOptionalStaticInt(buffer->shape[0]);
    return extent.has_value() && extent.value() == 1;
  }

  void BindSharedBlockAllocBuffersForThreadLaunch(llvm::ArrayRef<tir::Buffer> buffers,
                                                  const tir::IterVarNode* iter_var,
                                                  ScopedBufferBindings* saved) {
    for (const tir::Buffer& buffer : buffers) {
      ICHECK(!BufferUsesVar(buffer, iter_var->var))
          << "shared block allocation shape/layout cannot depend on the serialized thread index "
          << "in riscv lowering: " << buffer->name;
      ValidateContiguousBuffer(buffer);
      mlir::Value alloc = CreateAlloca(buffer->shape, buffer->dtype);
      BindBufferAliases(buffer, alloc, &saved->buffer_bindings, &saved->buffer_owners);
      if (buffer->dtype.lanes() > 1) {
        saved->packed_owners.emplace_back(
            buffer->data.get(), SaveAndSetPackedDataOwner(buffer->data.get(), buffer));
      }
    }
  }

  std::vector<ThreadLocalBlockAllocBinding> CreateThreadLocalBlockAllocBackings(
      llvm::ArrayRef<tir::Buffer> buffers, const tir::IterVarNode* iter_var,
      const PrimExpr& thread_extent) {
    std::vector<ThreadLocalBlockAllocBinding> bindings;
    bindings.reserve(buffers.size());
    for (const tir::Buffer& buffer : buffers) {
      ICHECK(!tl::IsSharedBuffer(buffer))
          << "thread-local block allocation backing received a shared buffer: "
          << buffer->name;
      ICHECK(!BufferUsesVar(buffer, iter_var->var))
          << "local block allocation shape/layout cannot depend on the serialized thread "
          << "index when it crosses a sync_threads phase in riscv lowering: "
          << buffer->name;
      ValidateContiguousBuffer(buffer);
      Array<PrimExpr> backing_shape;
      backing_shape.push_back(thread_extent);
      if (!IsScalarLocalVarBuffer(buffer)) {
        for (const PrimExpr& dim : buffer->shape) {
          backing_shape.push_back(dim);
        }
      }
      ThreadLocalBlockAllocBinding binding{buffer, CreateAlloca(backing_shape, buffer->dtype)};
      InitializeThreadLocalBackingFromExpr(
          binding.backing, buffer, thread_extent, LookupLocalVarInitForDataVar(buffer->data));
      bindings.push_back(binding);
    }
    return bindings;
  }

  mlir::Value CreateThreadLocalBlockAllocSubview(const ThreadLocalBlockAllocBinding& binding,
                                                mlir::Value thread_index) {
    const tir::Buffer& buffer = binding.buffer;
    mlir::MemRefType backing_type = mlir::cast<mlir::MemRefType>(binding.backing.getType());
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    llvm::SmallVector<mlir::OpFoldResult, 4> strides;
    offsets.reserve(buffer->shape.size() + 1);
    sizes.reserve(buffer->shape.size() + 1);
    strides.reserve(buffer->shape.size() + 1);

    offsets.push_back(AsIndex(thread_index, DataType::Int(32)));
    sizes.push_back(builder_.getIndexAttr(1));
    strides.push_back(builder_.getIndexAttr(1));
    if (!IsScalarLocalVarBuffer(buffer)) {
      for (const PrimExpr& dim : buffer->shape) {
        offsets.push_back(builder_.getIndexAttr(0));
        sizes.push_back(LowerIndexOpFoldResult(dim));
        strides.push_back(builder_.getIndexAttr(1));
      }
    }

    mlir::MemRefType result_type = mlir::memref::SubViewOp::inferRankReducedResultType(
        LowerStaticShape(buffer->shape), backing_type, offsets, sizes, strides);
    return builder_
        .create<mlir::memref::SubViewOp>(loc_, result_type, binding.backing, offsets, sizes,
                                         strides)
        .getResult();
  }

  void BindThreadLocalBlockAllocBuffersForPhase(
      llvm::ArrayRef<ThreadLocalBlockAllocBinding> bindings, mlir::Value thread_index,
      ScopedBufferBindings* saved) {
    for (const ThreadLocalBlockAllocBinding& binding : bindings) {
      mlir::Value subview = CreateThreadLocalBlockAllocSubview(binding, thread_index);
      BindBufferAliases(binding.buffer, subview, &saved->buffer_bindings,
                        &saved->buffer_owners);
      auto save_backing = [&](const Object* key) {
        auto it = active_thread_local_backings_.find(key);
        if (it == active_thread_local_backings_.end()) {
          saved->thread_local_backings.emplace_back(key, std::nullopt);
        } else {
          saved->thread_local_backings.emplace_back(key, it->second);
        }
        active_thread_local_backings_[key] = binding.backing;
      };
      save_backing(binding.buffer.get());
      save_backing(binding.buffer->data.get());
      saved->thread_local_keys.push_back(binding.buffer.get());
      saved->thread_local_keys.push_back(binding.buffer->data.get());
      ++prebound_thread_local_block_buffers_[binding.buffer.get()];
      ++prebound_thread_local_block_buffers_[binding.buffer->data.get()];
      if (binding.buffer->dtype.lanes() > 1) {
        saved->packed_owners.emplace_back(
            binding.buffer->data.get(),
            SaveAndSetPackedDataOwner(binding.buffer->data.get(), binding.buffer));
      }
    }
  }

  llvm::ArrayRef<ThreadLocalBlockAllocBinding> CurrentActiveThreadLocalBlockAllocBindings() const {
    if (active_thread_local_bindings_stack_.empty()) {
      return {};
    }
    return active_thread_local_bindings_stack_.back();
  }

  void RestoreScopedBufferBindings(ScopedBufferBindings* saved) {
    RestoreBindings(buffer_values_, saved->buffer_bindings);
    RestoreBufferOwnerBindings(saved->buffer_owners);
    RestorePackedDataOwnerBindings(saved->packed_owners);
    for (auto it = saved->thread_local_backings.rbegin();
         it != saved->thread_local_backings.rend(); ++it) {
      if (it->second.has_value()) {
        active_thread_local_backings_[it->first] = it->second.value();
      } else {
        active_thread_local_backings_.erase(it->first);
      }
    }
    for (const Object* key : saved->thread_local_keys) {
      auto it = prebound_thread_local_block_buffers_.find(key);
      ICHECK(it != prebound_thread_local_block_buffers_.end() && it->second > 0)
          << "missing prebound thread-local buffer tracking entry in riscv lowering";
      if (--it->second == 0) {
        prebound_thread_local_block_buffers_.erase(it);
      }
    }
  }

  bool InLogicalThreadRegion() const { return !thread_launch_stack_.empty(); }

  bool InBreakLoopRegion() const { return !break_loop_stack_.empty(); }

  mlir::Value CurrentBreakLoopFlag() const {
    ICHECK(InBreakLoopRegion()) << "loop_break requires an active enclosing loop";
    return break_loop_stack_.back().flag;
  }

  void StoreBreakLoopFlag(mlir::Value flag, bool value) {
    llvm::SmallVector<mlir::Value, 1> indices{ZeroIndex()};
    builder_.create<mlir::memref::StoreOp>(
        loc_, ConstantIntLike(value ? 1 : 0, builder_.getI1Type()), flag, indices);
  }

  mlir::Value LoadBreakLoopFlag(mlir::Value flag) {
    return builder_.create<mlir::memref::LoadOp>(loc_, flag,
                                                 llvm::SmallVector<mlir::Value, 1>{ZeroIndex()});
  }

  template <typename F>
  void EmitIfCurrentLoopNotBroken(F&& body_builder) {
    mlir::Value flag = LoadBreakLoopFlag(CurrentBreakLoopFlag());
    mlir::Value not_broken = builder_.create<mlir::arith::CmpIOp>(
        loc_, mlir::arith::CmpIPredicate::eq, flag,
        ConstantIntLike(0, builder_.getI1Type()));
    mlir::scf::IfOp if_op = builder_.create<mlir::scf::IfOp>(loc_, not_broken, false);
    mlir::OpBuilder::InsertionGuard guard(builder_);
    builder_.setInsertionPoint(if_op.thenYield());
    body_builder();
  }

  bool HasNonUnitLaunchExtent(const std::vector<ThreadLaunchFrame>& stack) const {
    for (const ThreadLaunchFrame& frame : stack) {
      if (!frame.extent.has_value() || frame.extent.value() != 1) {
        return true;
      }
    }
    return false;
  }

  bool InNonUnitLogicalThreadRegion() const {
    return HasNonUnitLaunchExtent(thread_launch_stack_);
  }

  bool InNonUnitLogicalThreadRegionThatUsesThreadVar() const {
    for (const ThreadLaunchFrame& frame : thread_launch_stack_) {
      bool non_unit = !frame.extent.has_value() || frame.extent.value() != 1;
      if (non_unit && frame.body_uses_iter_var) {
        return true;
      }
    }
    return false;
  }

  bool StmtContainsPhaseBoundarySyncDeep(const tir::Stmt& stmt) const {
    if (!stmt.defined()) {
      return false;
    }
    bool found = false;
    tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
      if (found) {
        return;
      }
      if (const auto* stmt_node = node.as<tir::StmtNode>()) {
        const tir::Stmt stmt_ref = tvm::ffi::GetRef<tir::Stmt>(stmt_node);
        if (IsPhaseBoundarySyncStmt(stmt_ref)) {
          found = true;
        }
      }
    });
    return found;
  }

  bool StmtContainsImplicitThreadSensitiveOp(const tir::Stmt& stmt) const {
    if (!stmt.defined()) {
      return false;
    }
    bool found = false;
    tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
      if (found) {
        return;
      }
      if (const auto* call = node.as<tir::CallNode>()) {
        if (IsThreadReturnCall(call)) {
          found = true;
          return;
        }
        const auto* op_node = call->op.as<OpNode>();
        if (op_node == nullptr) {
          return;
        }
        if (IsCooperativeThreadIntrinsicName(op_node->name) ||
            IsThreadIndexHelperIntrinsicName(op_node->name)) {
          found = true;
        }
        return;
      }
      if (const auto* attr = node.as<tir::AttrStmtNode>()) {
        if (attr->attr_key != tir::attr::thread_extent) {
          return;
        }
        const auto* iter_var = attr->node.as<tir::IterVarNode>();
        if (iter_var == nullptr || !IsThreadLaunchIterVar(iter_var)) {
          return;
        }
        std::optional<int64_t> static_extent = GetOptionalStaticInt(attr->value);
        if (!static_extent.has_value() || static_extent.value() != 1) {
          found = true;
        }
        return;
      }
      if (const auto* loop = node.as<tir::ForNode>()) {
        if (!loop->thread_binding.defined()) {
          return;
        }
        const auto* iter_var = loop->thread_binding.as<tir::IterVarNode>();
        if (iter_var == nullptr || !IsThreadLaunchIterVar(iter_var)) {
          return;
        }
        std::optional<int64_t> static_extent = GetOptionalStaticInt(loop->extent);
        if (!static_extent.has_value() || static_extent.value() != 1) {
          found = true;
        }
      }
    });
    return found;
  }

  bool ShouldCollapseThreadInvariantLaunchBody(const tir::Stmt& body,
                                              const tir::Var& thread_var) const {
    if (StmtUsesVar(body, thread_var)) {
      return false;
    }
    if (StmtContainsPhaseBoundarySyncDeep(body)) {
      return false;
    }
    if (StmtContainsImplicitThreadSensitiveOp(body)) {
      return false;
    }
    return true;
  }

  bool InNonUnitLaunchRegion() const {
    return HasNonUnitLaunchExtent(block_launch_stack_) ||
           HasNonUnitLaunchExtent(thread_launch_stack_);
  }

  mlir::Value CurrentThreadIdxXValue(llvm::StringRef context) const {
    std::string context_name = context.str();
    for (auto it = thread_launch_stack_.rbegin(); it != thread_launch_stack_.rend(); ++it) {
      if (it->iter_var == nullptr || it->iter_var->thread_tag != "threadIdx.x") {
        continue;
      }
      auto scalar_it = scalar_values_.find(it->iter_var->var.get());
      ICHECK(scalar_it != scalar_values_.end())
          << context_name << " found threadIdx.x without a bound serialized induction value";
      return scalar_it->second;
    }
    LOG(FATAL) << context_name
               << " requires an active threadIdx.x launch in riscv lowering";
    TVM_FFI_UNREACHABLE();
  }

  const ThreadLaunchFrame* CurrentThreadIdxXFrame() const {
    for (auto it = thread_launch_stack_.rbegin(); it != thread_launch_stack_.rend(); ++it) {
      if (it->iter_var != nullptr && it->iter_var->thread_tag == "threadIdx.x") {
        return &*it;
      }
    }
    return nullptr;
  }

  PrimExpr ResolveBoundPrimExpr(const PrimExpr& expr) const {
    class BoundExprResolver final : public tir::StmtExprMutator {
    public:
      explicit BoundExprResolver(const PrimExprMap& bound) : bound_(bound) {}

      PrimExpr Resolve(const PrimExpr& expr) { return VisitExpr(expr); }

    private:
      PrimExpr VisitExpr_(const tir::VarNode* op) final {
        auto it = bound_.find(op);
        if (it == bound_.end()) {
          return tvm::ffi::GetRef<PrimExpr>(op);
        }
        if (!active_.insert(op).second) {
          return tvm::ffi::GetRef<PrimExpr>(op);
        }
        PrimExpr resolved = VisitExpr(it->second);
        active_.erase(op);
        return resolved;
      }

      const PrimExprMap& bound_;
      std::unordered_set<const Object*> active_;
    };

    BoundExprResolver resolver(bound_prim_exprs_);
    return resolver.Resolve(expr);
  }

  bool BufferIsParamBackedForSerializedWarpReplay(const tir::Buffer& buffer) const {
    return function_param_buffers_.count(buffer.get()) != 0 ||
           function_param_buffer_data_.count(buffer->data.get()) != 0;
  }

  std::string SanitizeHelperSymbolToken(std::string token) const {
    for (char& ch : token) {
      if (!std::isalnum(static_cast<unsigned char>(ch))) {
        ch = '_';
      }
    }
    return token;
  }

  std::string GetPointerBackedBufferViewHelperSymbol(const tir::Buffer& buffer) const {
    std::string symbol =
        "tilelang_riscv_ptr_i64_to_memref_" +
        SanitizeHelperSymbolToken(runtime::DLDataTypeToString(buffer->dtype)) + "_r" +
        std::to_string(buffer->shape.size());
    for (const PrimExpr& dim : buffer->shape) {
      if (const auto* imm = dim.as<IntImmNode>()) {
        symbol += "_s" + std::to_string(imm->value);
      } else {
        symbol += "_d";
      }
    }
    return symbol;
  }

  std::string EnsurePointerBackedBufferViewHelper(const tir::Buffer& buffer) {
    std::string symbol = GetPointerBackedBufferViewHelperSymbol(buffer);
    if (!pointer_backed_buffer_view_helpers_.insert(symbol).second) {
      return symbol;
    }

    llvm::SmallVector<mlir::Type, 4> arg_types;
    arg_types.push_back(builder_.getI64Type());
    for (const PrimExpr& dim : buffer->shape) {
      if (!dim.as<IntImmNode>()) {
        arg_types.push_back(builder_.getIndexType());
      }
    }

    mlir::OpBuilder::InsertionGuard guard(builder_);
    builder_.setInsertionPointToStart(module_.getBody());
    mlir::func::FuncOp func_op = builder_.create<mlir::func::FuncOp>(
        loc_, symbol,
        builder_.getFunctionType(arg_types, llvm::ArrayRef<mlir::Type>{LowerBufferMemRefType(buffer)}));
    func_op.setPrivate();
    return symbol;
  }

  mlir::Value MaterializePointerBackedBufferView(mlir::Value raw_addr, const tir::Buffer& buffer) {
    ValidateContiguousBuffer(buffer);
    ICHECK(buffer.scope() == "global" || buffer.scope().empty())
        << "Pointer-backed buffers currently require global scope in riscv lowering: "
        << buffer->name;

    raw_addr = CastValueToType(raw_addr, builder_.getI64Type(), false);
    llvm::SmallVector<mlir::Value, 4> args;
    args.push_back(raw_addr);
    for (const PrimExpr& dim : buffer->shape) {
      if (!dim.as<IntImmNode>()) {
        args.push_back(AsIndex(VisitExpr(dim), dim.dtype()));
      }
    }

    std::string symbol = EnsurePointerBackedBufferViewHelper(buffer);
    mlir::Type result_type = LowerBufferMemRefType(buffer);
    mlir::func::CallOp call =
        builder_.create<mlir::func::CallOp>(loc_, symbol, llvm::ArrayRef<mlir::Type>{result_type},
                                            llvm::ArrayRef<mlir::Value>{args});
    return call.getResult(0);
  }

  template <typename BodyEmitter>
  bool TryEmitPointerBackedHandleLet(const tir::LetStmtNode* let, BodyEmitter&& body_emitter) {
    const tir::CallNode* reinterpret_call = nullptr;
    if (!IsPointerBackedHandleReinterpretLet(let, &reinterpret_call)) {
      return false;
    }

    tir::Var handle_var = tvm::ffi::GetRef<tir::Var>(let->var.get());
    std::vector<tir::Buffer> pointer_buffers;
    CollectBuffersBackedByVar(let->body, handle_var, &pointer_buffers);
    if (pointer_buffers.empty()) {
      return false;
    }

    for (size_t i = 1; i < pointer_buffers.size(); ++i) {
      ICHECK(HaveEquivalentPointerBackedBufferViews(pointer_buffers.front(), pointer_buffers[i]))
          << "Pointer-backed handle let '" << let->var->name_hint
          << "' materializes multiple incompatible buffer views in riscv lowering";
    }

    mlir::Value raw_addr = VisitExpr(reinterpret_call->args[0]);
    mlir::Value view = MaterializePointerBackedBufferView(raw_addr, pointer_buffers.front());
    std::vector<std::pair<const Object*, SavedBinding>> saved_bindings;
    std::vector<std::pair<const Object*, std::optional<tir::Buffer>>> saved_buffer_owners;
    saved_bindings.reserve(pointer_buffers.size() * 2);
    saved_buffer_owners.reserve(pointer_buffers.size() * 2);
    for (const tir::Buffer& buffer : pointer_buffers) {
      BindBufferAliases(buffer, view, &saved_bindings, &saved_buffer_owners);
    }
    std::optional<PrimExpr> saved_expr =
        SaveAndSetPrimExpr(bound_prim_exprs_, let->var.get(), let->value);
    body_emitter();
    RestorePrimExprBinding(bound_prim_exprs_, let->var.get(), saved_expr);
    RestoreBindings(buffer_values_, saved_bindings);
    RestoreBufferOwnerBindings(saved_buffer_owners);
    return true;
  }

  bool InSingleNonUnitThreadIdxXRegion() const {
    int non_unit_thread_dims = 0;
    for (const ThreadLaunchFrame& frame : thread_launch_stack_) {
      bool non_unit = !frame.extent.has_value() || frame.extent.value() != 1;
      if (!non_unit) {
        continue;
      }
      ++non_unit_thread_dims;
      if (frame.iter_var == nullptr || frame.iter_var->thread_tag != "threadIdx.x") {
        return false;
      }
    }
    return non_unit_thread_dims == 1;
  }

  bool CanTrackSerializedWarpReplayBufferElements(const tir::Buffer& buffer) const {
    if (!buffer.defined() || BufferIsParamBackedForSerializedWarpReplay(buffer) ||
        tl::IsSharedBuffer(buffer) || buffer->dtype.lanes() != 1) {
      return false;
    }
    int64_t element_count = 1;
    for (const PrimExpr& dim : buffer->shape) {
      std::optional<int64_t> static_extent = GetOptionalStaticInt(dim);
      if (!static_extent.has_value() || static_extent.value() <= 0) {
        return false;
      }
      if (element_count > 4096 / static_extent.value()) {
        return false;
      }
      element_count *= static_extent.value();
    }
    return true;
  }

  std::optional<int64_t> GetStaticSerializedWarpReplayBufferElementCount(
      const tir::Buffer& buffer) const {
    if (!CanTrackSerializedWarpReplayBufferElements(buffer)) {
      return std::nullopt;
    }
    int64_t element_count = 1;
    for (const PrimExpr& dim : buffer->shape) {
      const auto* imm = dim.as<IntImmNode>();
      if (imm == nullptr || imm->value <= 0) {
        return std::nullopt;
      }
      element_count *= imm->value;
    }
    return element_count;
  }

  bool CanMaterializeThreadLocalBlockAllocForSerializedWarpReplay(const tir::Buffer& buffer) const {
    std::optional<int64_t> element_count =
        GetStaticSerializedWarpReplayBufferElementCount(buffer);
    return element_count.has_value() && element_count.value() <= 16;
  }

  std::unordered_set<const Object*> CollectCooperativeReplayOperandBuffers(
      const tir::Stmt& stmt) const {
    std::unordered_set<const Object*> buffers;
    tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
      const auto* call = node.as<tir::CallNode>();
      if (call == nullptr) {
        return;
      }
      auto collect_load_buffer = [&](const PrimExpr& expr) {
        const auto* load = expr.as<tir::BufferLoadNode>();
        if (load != nullptr) {
          buffers.insert(load->buffer.get());
          buffers.insert(load->buffer->data.get());
        }
      };
      if (GetWarpReduceCombineKind(call).has_value() && call->args.size() == 1U) {
        collect_load_buffer(call->args[0]);
        return;
      }
      if (GetSerializedWarpShuffleKind(call).has_value() && call->args.size() == 4U) {
        collect_load_buffer(call->args[1]);
      }
    });
    return buffers;
  }

  std::optional<int64_t> GetSerializedWarpReplayStaticBufferLinearIndex(
      const tir::Buffer& buffer, const Array<PrimExpr>& indices) const {
    if (!CanTrackSerializedWarpReplayBufferElements(buffer) ||
        indices.size() != buffer->shape.size()) {
      return std::nullopt;
    }
    int64_t linear_index = 0;
    int64_t stride = 1;
    for (size_t i = indices.size(); i > 0; --i) {
      size_t dim = i - 1;
      std::optional<int64_t> static_extent = GetOptionalStaticInt(buffer->shape[dim]);
      ICHECK(static_extent.has_value() && static_extent.value() > 0);
      std::optional<int64_t> static_index =
          GetOptionalStaticInt(ResolveBoundPrimExpr(indices[dim]));
      if (!static_index.has_value() || static_index.value() < 0 ||
          static_index.value() >= static_extent.value()) {
        return std::nullopt;
      }
      linear_index += static_index.value() * stride;
      stride *= static_extent.value();
    }
    return linear_index;
  }

  bool IsSerializedWarpReplayTrackedScalarBuffer(const tir::Buffer& buffer) const {
    if (!CanTrackSerializedWarpReplayBufferElements(buffer) || buffer->shape.size() != 1) {
      return false;
    }
    std::optional<int64_t> static_extent = GetOptionalStaticInt(buffer->shape[0]);
    return static_extent.has_value() && static_extent.value() == 1;
  }

  PrimExpr CastSerializedWarpReplayTrackedExpr(const tir::Buffer& buffer,
                                               const PrimExpr& expr) const {
    if (!buffer.defined() || expr.dtype() == buffer->dtype) {
      return expr;
    }
    return tir::Cast(buffer->dtype, expr);
  }

  void SetSerializedWarpReplayTrackedBufferElementExpr(const tir::Buffer& buffer,
                                                       int64_t linear_index,
                                                       const PrimExpr& expr) {
    serialized_warp_replay_buffer_element_exprs_[{buffer->data.get(), linear_index}] =
        CastSerializedWarpReplayTrackedExpr(buffer, expr);
  }

  void ClearSerializedWarpReplayTrackedBufferElementExpr(const tir::Buffer& buffer,
                                                         int64_t linear_index) {
    serialized_warp_replay_buffer_element_exprs_.erase({buffer->data.get(), linear_index});
  }

  void ClearSerializedWarpReplayTrackedBufferElementExprs(const tir::Buffer& buffer) {
    if (!buffer.defined()) {
      return;
    }
    const Object* data = buffer->data.get();
    for (auto it = serialized_warp_replay_buffer_element_exprs_.begin();
         it != serialized_warp_replay_buffer_element_exprs_.end();) {
      if (it->first.buffer_data == data) {
        it = serialized_warp_replay_buffer_element_exprs_.erase(it);
      } else {
        ++it;
      }
    }
  }

  std::optional<PrimExpr> LookupSerializedWarpReplayTrackedBufferElementExpr(
      const tir::Buffer& buffer, int64_t linear_index) const {
    auto it = serialized_warp_replay_buffer_element_exprs_.find(
        SerializedWarpReplayBufferElementKey{buffer->data.get(), linear_index});
    if (it != serialized_warp_replay_buffer_element_exprs_.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  bool IsSerializedWarpReplayTrackedScalarLoad(const tir::BufferLoadNode* load) const {
    return load != nullptr && (!load->predicate.defined() || tir::is_one(load->predicate.value())) &&
           load->buffer->dtype.lanes() == 1 && load->indices.size() == 1 &&
           IsSerializedWarpReplayTrackedScalarBuffer(load->buffer) &&
           GetOptionalStaticInt(ResolveBoundPrimExpr(load->indices[0])) == std::optional<int64_t>(0);
  }

  void SetSerializedWarpReplayTrackedBufferExpr(const tir::Buffer& buffer, const PrimExpr& expr) {
    SetSerializedWarpReplayTrackedBufferElementExpr(buffer, 0, expr);
    serialized_warp_replay_buffer_exprs_[buffer.get()] = expr;
    serialized_warp_replay_buffer_exprs_[buffer->data.get()] = expr;
  }

  void ClearSerializedWarpReplayTrackedBufferExpr(const tir::Buffer& buffer) {
    ClearSerializedWarpReplayTrackedBufferElementExpr(buffer, 0);
    serialized_warp_replay_buffer_exprs_.erase(buffer.get());
    serialized_warp_replay_buffer_exprs_.erase(buffer->data.get());
  }

  std::optional<PrimExpr> LookupSerializedWarpReplayTrackedBufferExpr(
      const tir::Buffer& buffer) const {
    auto it = serialized_warp_replay_buffer_exprs_.find(buffer.get());
    if (it != serialized_warp_replay_buffer_exprs_.end()) {
      return it->second;
    }
    it = serialized_warp_replay_buffer_exprs_.find(buffer->data.get());
    if (it != serialized_warp_replay_buffer_exprs_.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  bool CanUseDirectThreadLocalLoadForSerializedWarpReplay(const PrimExpr& expr) const {
    PrimExpr resolved = ResolveBoundPrimExpr(expr);
    const auto* load = resolved.as<tir::BufferLoadNode>();
    return load != nullptr && (!load->predicate.defined() || tir::is_one(load->predicate.value())) &&
           load->buffer->dtype.lanes() == 1 && IsPreboundThreadLocalBlockBuffer(load->buffer) &&
           BufferIsBound(load->buffer);
  }

  PrimExpr GetSerializedWarpReplayCandidateExpr(const PrimExpr& expr) const {
    if (CanUseDirectThreadLocalLoadForSerializedWarpReplay(expr)) {
      return ResolveBoundPrimExpr(expr);
    }
    return ResolveSerializedWarpReplayExpr(expr);
  }

  std::optional<mlir::Value> TryLoadSerializedWarpReplayDirectThreadLocalValue(
      const PrimExpr& expr, llvm::ArrayRef<ThreadLocalBlockAllocBinding> bindings,
      mlir::Value thread_index, DataType result_dtype) {
    PrimExpr resolved = ResolveBoundPrimExpr(expr);
    const auto* load = resolved.as<tir::BufferLoadNode>();
    if (load == nullptr) {
      resolved = ResolveSerializedWarpReplayExpr(expr);
      load = resolved.as<tir::BufferLoadNode>();
    }
    if (load == nullptr || (load->predicate.defined() && !tir::is_one(load->predicate.value())) ||
        load->buffer->dtype.lanes() != 1) {
      return std::nullopt;
    }

    const ThreadLocalBlockAllocBinding* binding = nullptr;
    for (const ThreadLocalBlockAllocBinding& candidate : bindings) {
      if (candidate.buffer.get() == load->buffer.get() ||
          candidate.buffer->data.get() == load->buffer->data.get()) {
        binding = &candidate;
        break;
      }
    }
    if (binding == nullptr) {
      auto backing_it = active_thread_local_backings_.find(load->buffer.get());
      if (backing_it == active_thread_local_backings_.end()) {
        backing_it = active_thread_local_backings_.find(load->buffer->data.get());
      }
      if (backing_it == active_thread_local_backings_.end()) {
        return std::nullopt;
      }
      ThreadLocalBlockAllocBinding map_binding{load->buffer, backing_it->second};
      mlir::Value subview = CreateThreadLocalBlockAllocSubview(map_binding, thread_index);
      llvm::SmallVector<mlir::Value, 4> indices;
      indices.reserve(load->indices.size());
      for (const PrimExpr& index : load->indices) {
        indices.push_back(AsIndex(VisitExpr(index), index.dtype()));
      }
      mlir::Value value = builder_.create<mlir::memref::LoadOp>(loc_, subview, indices);
      return CastValue(value, load->buffer->dtype, result_dtype);
    }

    mlir::Value subview = CreateThreadLocalBlockAllocSubview(*binding, thread_index);
    llvm::SmallVector<mlir::Value, 4> indices;
    indices.reserve(load->indices.size());
    for (const PrimExpr& index : load->indices) {
      indices.push_back(AsIndex(VisitExpr(index), index.dtype()));
    }
    mlir::Value value = builder_.create<mlir::memref::LoadOp>(loc_, subview, indices);
    return CastValue(value, load->buffer->dtype, result_dtype);
  }

  bool ExprIsThreadLocalBindingLoad(
      const PrimExpr& expr, llvm::ArrayRef<ThreadLocalBlockAllocBinding> bindings) const {
    PrimExpr resolved = ResolveBoundPrimExpr(expr);
    const auto* load = resolved.as<tir::BufferLoadNode>();
    if (load == nullptr) {
      resolved = ResolveSerializedWarpReplayExpr(expr);
      load = resolved.as<tir::BufferLoadNode>();
    }
    if (load == nullptr || (load->predicate.defined() && !tir::is_one(load->predicate.value())) ||
        load->buffer->dtype.lanes() != 1) {
      return false;
    }
    for (const ThreadLocalBlockAllocBinding& binding : bindings) {
      if (binding.buffer.get() == load->buffer.get() ||
          binding.buffer->data.get() == load->buffer->data.get()) {
        return true;
      }
    }
    return false;
  }

  PrimExpr ResolveSerializedWarpReplayExpr(const PrimExpr& expr) const {
    class ReplayExprResolver final : public tir::StmtExprMutator {
    public:
      explicit ReplayExprResolver(const TIRToMLIRLowerer* outer) : outer_(outer) {}

      PrimExpr Resolve(const PrimExpr& expr) { return VisitExpr(expr); }

    private:
      PrimExpr VisitExpr_(const tir::VarNode* op) final {
        auto it = outer_->bound_prim_exprs_.find(op);
        if (it == outer_->bound_prim_exprs_.end()) {
          return tvm::ffi::GetRef<PrimExpr>(op);
        }
        if (!active_vars_.insert(op).second) {
          return tvm::ffi::GetRef<PrimExpr>(op);
        }
        PrimExpr resolved = VisitExpr(it->second);
        active_vars_.erase(op);
        return resolved;
      }

      PrimExpr VisitExpr_(const tir::BufferLoadNode* op) final {
        PrimExpr visited = tir::StmtExprMutator::VisitExpr_(op);
        const auto* load = visited.as<tir::BufferLoadNode>();
        if (load == nullptr) {
          return visited;
        }
        if (std::optional<int64_t> tracked_index =
                outer_->GetSerializedWarpReplayStaticBufferLinearIndex(load->buffer,
                                                                       load->indices);
            tracked_index.has_value()) {
          std::optional<PrimExpr> tracked =
              outer_->LookupSerializedWarpReplayTrackedBufferElementExpr(load->buffer,
                                                                         tracked_index.value());
          if (!tracked.has_value()) {
            return visited;
          }
          SerializedWarpReplayBufferElementKey key{load->buffer->data.get(),
                                                   tracked_index.value()};
          if (!active_elements_.insert(key).second) {
            return visited;
          }
          PrimExpr resolved = VisitExpr(tracked.value());
          active_elements_.erase(key);
          return resolved;
        }
        if (!outer_->IsSerializedWarpReplayTrackedScalarLoad(load)) {
          return visited;
        }
        std::optional<PrimExpr> tracked =
            outer_->LookupSerializedWarpReplayTrackedBufferExpr(load->buffer);
        if (!tracked.has_value()) {
          return visited;
        }
        if (!active_buffers_.insert(load->buffer.get()).second) {
          return visited;
        }
        PrimExpr resolved = VisitExpr(tracked.value());
        active_buffers_.erase(load->buffer.get());
        return resolved;
      }

      const TIRToMLIRLowerer* outer_;
      std::unordered_set<const Object*> active_vars_;
      std::unordered_set<const Object*> active_buffers_;
      std::unordered_set<SerializedWarpReplayBufferElementKey,
                         SerializedWarpReplayBufferElementKeyHash>
          active_elements_;
    };

    ReplayExprResolver resolver(this);
    return resolver.Resolve(expr);
  }

  SerializedWarpIndexClass ClassifySerializedWarpReplayIndexExpr(const PrimExpr& expr,
                                                                 const tir::Var& thread_var) const {
    PrimExpr resolved = ResolveBoundPrimExpr(expr);
    if (!ExprUsesVar(resolved, thread_var)) {
      return SerializedWarpIndexClass::kInvariant;
    }

    if (const auto* var = resolved.as<tir::VarNode>()) {
      return var == thread_var.get() ? SerializedWarpIndexClass::kLaneLinear
                                     : SerializedWarpIndexClass::kUnsupported;
    }
    if (const auto* cast = resolved.as<tir::CastNode>()) {
      return ClassifySerializedWarpReplayIndexExpr(cast->value, thread_var);
    }
    auto classify_mod = [&](const PrimExpr& lhs_expr, const PrimExpr& rhs_expr) {
      PrimExpr lhs = ResolveBoundPrimExpr(lhs_expr);
      PrimExpr rhs = ResolveBoundPrimExpr(rhs_expr);
      if (const auto* lhs_var = lhs.as<tir::VarNode>()) {
        if (lhs_var == thread_var.get()) {
          std::optional<int64_t> static_rhs = GetOptionalStaticInt(rhs);
          if (static_rhs.has_value() && static_rhs.value() == 32) {
            return SerializedWarpIndexClass::kLaneLinear;
          }
        }
      }
      return SerializedWarpIndexClass::kUnsupported;
    };
    if (const auto* mod = resolved.as<tir::ModNode>()) {
      return classify_mod(mod->a, mod->b);
    }
    if (const auto* floor_mod = resolved.as<tir::FloorModNode>()) {
      return classify_mod(floor_mod->a, floor_mod->b);
    }

    auto combine_add = [&](const PrimExpr& lhs, const PrimExpr& rhs) {
      SerializedWarpIndexClass lhs_class =
          ClassifySerializedWarpReplayIndexExpr(lhs, thread_var);
      SerializedWarpIndexClass rhs_class =
          ClassifySerializedWarpReplayIndexExpr(rhs, thread_var);
      if (lhs_class == SerializedWarpIndexClass::kUnsupported ||
          rhs_class == SerializedWarpIndexClass::kUnsupported) {
        return SerializedWarpIndexClass::kUnsupported;
      }
      if (lhs_class == SerializedWarpIndexClass::kLaneLinear &&
          rhs_class == SerializedWarpIndexClass::kInvariant) {
        return SerializedWarpIndexClass::kLaneLinear;
      }
      if (lhs_class == SerializedWarpIndexClass::kInvariant &&
          rhs_class == SerializedWarpIndexClass::kLaneLinear) {
        return SerializedWarpIndexClass::kLaneLinear;
      }
      if (lhs_class == SerializedWarpIndexClass::kInvariant &&
          rhs_class == SerializedWarpIndexClass::kInvariant) {
        return SerializedWarpIndexClass::kInvariant;
      }
      return SerializedWarpIndexClass::kUnsupported;
    };

    if (const auto* add = resolved.as<tir::AddNode>()) {
      return combine_add(add->a, add->b);
    }
    if (const auto* sub = resolved.as<tir::SubNode>()) {
      SerializedWarpIndexClass lhs_class =
          ClassifySerializedWarpReplayIndexExpr(sub->a, thread_var);
      SerializedWarpIndexClass rhs_class =
          ClassifySerializedWarpReplayIndexExpr(sub->b, thread_var);
      if (lhs_class == SerializedWarpIndexClass::kUnsupported ||
          rhs_class == SerializedWarpIndexClass::kUnsupported) {
        return SerializedWarpIndexClass::kUnsupported;
      }
      if (rhs_class == SerializedWarpIndexClass::kInvariant) {
        return lhs_class;
      }
      return SerializedWarpIndexClass::kUnsupported;
    }
    return SerializedWarpIndexClass::kUnsupported;
  }

  bool MatchSerializedWarpReplayGuard(const PrimExpr& condition, const PrimExpr& index_expr,
                                      const tir::Var& thread_var,
                                      PrimExpr* upper_bound_expr) const {
    PrimExpr resolved_condition = ResolveBoundPrimExpr(condition);
    PrimExpr resolved_index = ResolveBoundPrimExpr(index_expr);
    const auto* lt = resolved_condition.as<tir::LTNode>();
    if (lt == nullptr) {
      return false;
    }
    PrimExpr lhs = ResolveBoundPrimExpr(lt->a);
    PrimExpr rhs = ResolveBoundPrimExpr(lt->b);
    if (!tir::ExprDeepEqual()(lhs, resolved_index) || ExprUsesVar(rhs, thread_var)) {
      return false;
    }
    *upper_bound_expr = rhs;
    return true;
  }

  bool MatchSerializedWarpReplayLoadValue(const PrimExpr& expr, const tir::Var& thread_var,
                                          tir::Buffer* load_buffer, PrimExpr* index_expr,
                                          DataType* value_dtype) const {
    PrimExpr resolved = ResolveBoundPrimExpr(expr);
    if (const auto* cast = resolved.as<tir::CastNode>()) {
      if (!IsLowerableScalarType(cast->dtype) || cast->dtype.lanes() != 1) {
        return false;
      }
      tir::Buffer nested_buffer;
      PrimExpr nested_index;
      DataType nested_dtype;
      if (!MatchSerializedWarpReplayLoadValue(cast->value, thread_var, &nested_buffer,
                                              &nested_index, &nested_dtype)) {
        return false;
      }
      *load_buffer = nested_buffer;
      *index_expr = nested_index;
      *value_dtype = cast->dtype;
      return true;
    }

    const auto* load = resolved.as<tir::BufferLoadNode>();
    if (load == nullptr || load->indices.size() != 1 ||
        (load->predicate.defined() && !tir::is_one(load->predicate.value())) ||
        load->buffer->dtype.lanes() != 1 || !BufferIsParamBackedForSerializedWarpReplay(load->buffer)) {
      return false;
    }
    if (ClassifySerializedWarpReplayIndexExpr(load->indices[0], thread_var) !=
        SerializedWarpIndexClass::kLaneLinear) {
      return false;
    }
    *load_buffer = load->buffer;
    *index_expr = ResolveBoundPrimExpr(load->indices[0]);
    *value_dtype = resolved.dtype();
    return true;
  }

  bool MatchSerializedWarpMatchReplayPattern(const tir::CallNode* op, const tir::Var& thread_var,
                                             SerializedWarpMatchReplayPattern* pattern) const {
    if (!IsLowerableScalarType(op->dtype) || op->dtype.lanes() != 1) {
      return false;
    }

    std::optional<SerializedWarpMatchKind> kind = GetSerializedWarpMatchKind(op);
    if (!kind.has_value()) {
      return false;
    }

    PrimExpr value_expr;
    if (IsMatchSyncCallExtern(op)) {
      if (op->args.size() != 3U) {
        return false;
      }
      value_expr = op->args[2];
    } else {
      if (op->args.size() != 2U) {
        return false;
      }
      value_expr = op->args[1];
    }

    PrimExpr resolved_value = ResolveBoundPrimExpr(value_expr);
    if (!IsIntegerLikeType(resolved_value.dtype()) || resolved_value.dtype().lanes() != 1) {
      return false;
    }

    tir::Buffer load_buffer;
    PrimExpr index_expr;
    DataType value_dtype;
    if (MatchSerializedWarpReplayLoadValue(resolved_value, thread_var, &load_buffer, &index_expr,
                                           &value_dtype)) {
      *pattern = SerializedWarpMatchReplayPattern{load_buffer, index_expr, value_dtype,
                                                  std::nullopt, std::nullopt};
      return true;
    }

    const auto* select = resolved_value.as<tir::SelectNode>();
    if (select == nullptr) {
      return false;
    }
    if (!MatchSerializedWarpReplayLoadValue(select->true_value, thread_var, &load_buffer,
                                            &index_expr, &value_dtype)) {
      return false;
    }

    PrimExpr upper_bound_expr;
    if (!MatchSerializedWarpReplayGuard(select->condition, index_expr, thread_var,
                                        &upper_bound_expr)) {
      return false;
    }
    PrimExpr fallback_expr = ResolveBoundPrimExpr(select->false_value);
    if (ExprUsesVar(fallback_expr, thread_var) || !IsLowerableScalarType(fallback_expr.dtype()) ||
        fallback_expr.dtype().lanes() != 1) {
      return false;
    }
    *pattern = SerializedWarpMatchReplayPattern{load_buffer, index_expr, value_dtype,
                                                upper_bound_expr, fallback_expr};
    return true;
  }

  bool ExprLoadsOnlySerializedWarpReplayFriendlyBuffers(const PrimExpr& expr) const {
    PrimExpr resolved = ResolveSerializedWarpReplayExpr(expr);
    bool supported = true;
    tir::PostOrderVisit(resolved, [&](const ObjectRef& node) {
      if (!supported) {
        return;
      }
      if (const auto* load = node.as<tir::BufferLoadNode>()) {
        if ((load->predicate.defined() && !tir::is_one(load->predicate.value())) ||
            load->buffer->dtype.lanes() != 1) {
          supported = false;
          return;
        }
        if (BufferIsParamBackedForSerializedWarpReplay(load->buffer) ||
            tl::IsSharedBuffer(load->buffer)) {
          return;
        }
        supported = false;
        return;
      }
      const auto* call = node.as<tir::CallNode>();
      if (call == nullptr) {
        return;
      }
      if (call->op.same_as(tir::builtin::call_extern())) {
        supported = false;
        return;
      }
      const auto* op_node = call->op.as<OpNode>();
      if (op_node != nullptr && IsCooperativeThreadIntrinsicName(op_node->name) &&
          !IsThreadIndexHelperIntrinsicName(op_node->name)) {
        supported = false;
      }
    });
    return supported;
  }

  std::optional<SerializedWarpMatchKind> GetSerializedWarpMatchKind(
      const tir::CallNode* op) const {
    if (IsMatchSyncCallExtern(op)) {
      return GetCallExternName(op) == "__match_any_sync" ? SerializedWarpMatchKind::kAny
                                                          : SerializedWarpMatchKind::kAll;
    }
    const auto* op_node = op == nullptr ? nullptr : op->op.as<OpNode>();
    if (op_node == nullptr) {
      return std::nullopt;
    }
    llvm::StringRef name(op_node->name.c_str());
    if (name == "tl.match_any_sync") {
      return SerializedWarpMatchKind::kAny;
    }
    if (name == "tl.match_all_sync") {
      return SerializedWarpMatchKind::kAll;
    }
    return std::nullopt;
  }

  const ThreadLaunchFrame* GetSingleStaticThreadIdxXReplayFrame() const {
    if (!InNonUnitLogicalThreadRegion()) {
      return nullptr;
    }
    const ThreadLaunchFrame* thread_idx_x = CurrentThreadIdxXFrame();
    if (thread_idx_x == nullptr || thread_idx_x->iter_var == nullptr ||
        !thread_idx_x->extent.has_value()) {
      return nullptr;
    }

    int non_unit_thread_dims = 0;
    for (const ThreadLaunchFrame& frame : thread_launch_stack_) {
      bool non_unit = !frame.extent.has_value() || frame.extent.value() != 1;
      if (!non_unit) {
        continue;
      }
      ++non_unit_thread_dims;
      if (frame.iter_var == nullptr || frame.iter_var->thread_tag != "threadIdx.x") {
        return nullptr;
      }
    }
    if (non_unit_thread_dims != 1) {
      return nullptr;
    }
    return thread_idx_x;
  }

  bool CanLowerMatchSyncCallAsSerializedWarpReplay(const tir::CallNode* op) const {
    if (!GetSerializedWarpMatchKind(op).has_value()) {
      return false;
    }
    const ThreadLaunchFrame* thread_idx_x = GetSingleStaticThreadIdxXReplayFrame();
    if (thread_idx_x == nullptr) {
      return false;
    }

    SerializedWarpMatchReplayPattern pattern;
    return MatchSerializedWarpMatchReplayPattern(
        op, tvm::ffi::GetRef<tir::Var>(thread_idx_x->iter_var->var.get()), &pattern);
  }

  std::optional<SerializedWarpVoteKind> GetSerializedWarpVoteKind(
      const tir::CallNode* op) const {
    const auto* op_node = op == nullptr ? nullptr : op->op.as<OpNode>();
    if (op_node == nullptr) {
      return std::nullopt;
    }
    llvm::StringRef name(op_node->name.c_str());
    if (name == "tl.any_sync") {
      return SerializedWarpVoteKind::kAny;
    }
    if (name == "tl.all_sync") {
      return SerializedWarpVoteKind::kAll;
    }
    if (name == "tl.ballot_sync" || name == "tl.ballot") {
      return SerializedWarpVoteKind::kBallot;
    }
    if (name == "tl.activemask" || name == "tir.tvm_warp_activemask") {
      return SerializedWarpVoteKind::kActiveMask;
    }
    return std::nullopt;
  }

  bool CanLowerVoteCallAsSerializedWarpReplay(const tir::CallNode* op) const {
    std::optional<SerializedWarpVoteKind> kind = GetSerializedWarpVoteKind(op);
    if (!kind.has_value() || !IsLowerableScalarType(op->dtype) || op->dtype.lanes() != 1) {
      return false;
    }
    const ThreadLaunchFrame* thread_idx_x = GetSingleStaticThreadIdxXReplayFrame();
    if (thread_idx_x == nullptr) {
      return false;
    }

    if (kind.value() == SerializedWarpVoteKind::kActiveMask) {
      return op->args.empty();
    }

    PrimExpr predicate_expr;
    PrimExpr mask_expr;
    tir::Var thread_var = tvm::ffi::GetRef<tir::Var>(thread_idx_x->iter_var->var.get());
    if (kind.value() == SerializedWarpVoteKind::kBallot && op->args.size() == 1U) {
      predicate_expr = GetSerializedWarpReplayCandidateExpr(op->args[0]);
      mask_expr =
          IntImm(DataType::UInt(32), static_cast<int64_t>(uint64_t{0xFFFFFFFFu}));
    } else {
      if (op->args.size() != 2U) {
        return false;
      }
      mask_expr = ResolveBoundPrimExpr(op->args[0]);
      predicate_expr = GetSerializedWarpReplayCandidateExpr(op->args[1]);
      if (!IsIntegerLikeType(mask_expr.dtype()) || mask_expr.dtype().lanes() != 1 ||
          ExprUsesVar(mask_expr, thread_var)) {
        return false;
      }
    }

    if (!IsLowerableScalarType(predicate_expr.dtype()) || predicate_expr.dtype().lanes() != 1) {
      return false;
    }
    if (CanUseDirectThreadLocalLoadForSerializedWarpReplay(predicate_expr)) {
      return true;
    }
    return ExprLoadsOnlySerializedWarpReplayFriendlyBuffers(predicate_expr);
  }

  bool CanLowerSyncthreadsOrCallAsSerializedThreadReplay(const tir::CallNode* op) const {
    const auto* op_node = op == nullptr ? nullptr : op->op.as<OpNode>();
    if (op_node == nullptr || op_node->name != "tl.syncthreads_or" ||
        !IsLowerableScalarType(op->dtype) || op->dtype.lanes() != 1 || op->args.size() != 1U) {
      return false;
    }
    if (GetSingleStaticThreadIdxXReplayFrame() == nullptr) {
      return false;
    }

    PrimExpr predicate_expr = GetSerializedWarpReplayCandidateExpr(op->args[0]);
    if (!IsLowerableScalarType(predicate_expr.dtype()) || predicate_expr.dtype().lanes() != 1) {
      return false;
    }
    if (CanUseDirectThreadLocalLoadForSerializedWarpReplay(predicate_expr)) {
      return true;
    }
    return ExprLoadsOnlySerializedWarpReplayFriendlyBuffers(predicate_expr);
  }

  std::optional<SerializedWarpShuffleKind> GetSerializedWarpShuffleKind(
      const tir::CallNode* op) const {
    const auto* op_node = op->op.as<OpNode>();
    if (op_node == nullptr) {
      return std::nullopt;
    }
    llvm::StringRef name(op_node->name.c_str());
    if (name == "tl.shfl_sync") {
      return SerializedWarpShuffleKind::kSync;
    }
    if (name == "tir.tvm_warp_shuffle") {
      return SerializedWarpShuffleKind::kSync;
    }
    if (name == "tl.shfl_xor_sync") {
      return SerializedWarpShuffleKind::kXor;
    }
    if (name == "tir.tvm_warp_shuffle_down") {
      return SerializedWarpShuffleKind::kDown;
    }
    if (name == "tl.shfl_down_sync") {
      return SerializedWarpShuffleKind::kDown;
    }
    if (name == "tir.tvm_warp_shuffle_up") {
      return SerializedWarpShuffleKind::kUp;
    }
    if (name == "tl.shfl_up_sync") {
      return SerializedWarpShuffleKind::kUp;
    }
    return std::nullopt;
  }

  bool CanLowerShuffleCallAsSerializedWarpReplay(const tir::CallNode* op) const {
    if (!InNonUnitLogicalThreadRegion()) {
      return false;
    }
    if (!GetSerializedWarpShuffleKind(op).has_value() ||
        (op->args.size() != 4U && op->args.size() != 5U) ||
        !IsLowerableScalarType(op->dtype) || op->dtype.lanes() != 1) {
      return false;
    }
    const ThreadLaunchFrame* thread_idx_x = CurrentThreadIdxXFrame();
    if (thread_idx_x == nullptr || thread_idx_x->iter_var == nullptr ||
        !thread_idx_x->extent.has_value()) {
      return false;
    }

    int non_unit_thread_dims = 0;
    for (const ThreadLaunchFrame& frame : thread_launch_stack_) {
      bool non_unit = !frame.extent.has_value() || frame.extent.value() != 1;
      if (!non_unit) {
        continue;
      }
      ++non_unit_thread_dims;
      if (frame.iter_var == nullptr || frame.iter_var->thread_tag != "threadIdx.x") {
        return false;
      }
    }
    if (non_unit_thread_dims != 1) {
      return false;
    }

    PrimExpr mask_expr = ResolveBoundPrimExpr(op->args[0]);
    PrimExpr value_expr = GetSerializedWarpReplayCandidateExpr(op->args[1]);
    PrimExpr lane_expr = ResolveBoundPrimExpr(op->args[2]);
    PrimExpr width_expr = ResolveBoundPrimExpr(op->args[3]);
    if (op->args.size() == 5U) {
      std::optional<int64_t> static_warp_size =
          GetOptionalStaticInt(ResolveBoundPrimExpr(op->args[4]));
      if (!static_warp_size.has_value() || static_warp_size.value() != 32) {
        return false;
      }
    }
    tir::Var thread_var = tvm::ffi::GetRef<tir::Var>(thread_idx_x->iter_var->var.get());

    if (!IsIntegerLikeType(mask_expr.dtype()) || mask_expr.dtype().lanes() != 1 ||
        !IsLowerableScalarType(value_expr.dtype()) || value_expr.dtype().lanes() != 1 ||
        !IsIntegerLikeType(lane_expr.dtype()) || lane_expr.dtype().lanes() != 1 ||
        ExprUsesVar(lane_expr, thread_var) || ExprUsesVar(width_expr, thread_var)) {
      return false;
    }

    std::optional<int64_t> static_width = GetOptionalStaticInt(width_expr);
    if (!static_width.has_value() || static_width.value() <= 0 || static_width.value() > 32) {
      return false;
    }

    if (CanUseDirectThreadLocalLoadForSerializedWarpReplay(op->args[1])) {
      return true;
    }
    return ExprLoadsOnlySerializedWarpReplayFriendlyBuffers(value_expr);
  }

  std::optional<ReductionLoopNestMatch::Kind> GetWarpReduceCombineKind(
      const tir::CallNode* op) const {
    const auto* op_node = op->op.as<OpNode>();
    if (op_node == nullptr) {
      return std::nullopt;
    }
    llvm::StringRef name(op_node->name.c_str());
    if (name == "tl.warp_reduce_sum") {
      return ReductionLoopNestMatch::Kind::kAdd;
    }
    if (name == "tl.warp_reduce_max") {
      return ReductionLoopNestMatch::Kind::kMax;
    }
    if (name == "tl.warp_reduce_min") {
      return ReductionLoopNestMatch::Kind::kMin;
    }
    if (name == "tl.warp_reduce_bitand") {
      return ReductionLoopNestMatch::Kind::kBitAnd;
    }
    if (name == "tl.warp_reduce_bitor") {
      return ReductionLoopNestMatch::Kind::kBitOr;
    }
    return std::nullopt;
  }

  bool CanLowerWarpReduceCallAsSerializedWarpReplay(const tir::CallNode* op) const {
    if (!InNonUnitLogicalThreadRegion() || !GetWarpReduceCombineKind(op).has_value() ||
        op->args.size() != 1U || !IsLowerableScalarType(op->dtype) || op->dtype.lanes() != 1) {
      return false;
    }
    const ThreadLaunchFrame* thread_idx_x = CurrentThreadIdxXFrame();
    if (thread_idx_x == nullptr || thread_idx_x->iter_var == nullptr ||
        !thread_idx_x->extent.has_value()) {
      return false;
    }

    int non_unit_thread_dims = 0;
    for (const ThreadLaunchFrame& frame : thread_launch_stack_) {
      bool non_unit = !frame.extent.has_value() || frame.extent.value() != 1;
      if (!non_unit) {
        continue;
      }
      ++non_unit_thread_dims;
      if (frame.iter_var == nullptr || frame.iter_var->thread_tag != "threadIdx.x") {
        return false;
      }
    }
    if (non_unit_thread_dims != 1) {
      return false;
    }

    PrimExpr value_expr = GetSerializedWarpReplayCandidateExpr(op->args[0]);
    if (!IsLowerableScalarType(value_expr.dtype()) || value_expr.dtype().lanes() != 1) {
      return false;
    }
    std::optional<ReductionLoopNestMatch::Kind> kind = GetWarpReduceCombineKind(op);
    if (!kind.has_value()) {
      return false;
    }
    if (!CanUseDirectThreadLocalLoadForSerializedWarpReplay(op->args[0]) &&
        !ExprLoadsOnlySerializedWarpReplayFriendlyBuffers(value_expr)) {
      return false;
    }
    if ((kind.value() == ReductionLoopNestMatch::Kind::kBitAnd ||
         kind.value() == ReductionLoopNestMatch::Kind::kBitOr) &&
        !IsIntegerLikeType(value_expr.dtype())) {
      return false;
    }
    return true;
  }

  bool GetStaticSerializedWarpReplayLoopInfo(const tir::ForNode* op, int64_t* trip_count,
                                             int64_t* min_value,
                                             int64_t* step_value) const {
    if (op == nullptr || op->thread_binding.defined() || op->kind == tir::ForKind::kParallel ||
        !IsSupportedGeneralLoopKind(op->kind)) {
      return false;
    }
    std::optional<int64_t> static_min = GetOptionalStaticInt(ResolveBoundPrimExpr(op->min));
    std::optional<int64_t> static_extent =
        GetOptionalStaticInt(ResolveBoundPrimExpr(op->extent));
    if (!static_min.has_value() || !static_extent.has_value() || static_extent.value() < 0) {
      return false;
    }
    int64_t static_step = 1;
    if (op->step.defined()) {
      std::optional<int64_t> loop_step =
          GetOptionalStaticInt(ResolveBoundPrimExpr(op->step.value()));
      if (!loop_step.has_value() || loop_step.value() <= 0) {
        return false;
      }
      static_step = loop_step.value();
    }
    if (static_extent.value() == 0) {
      *trip_count = 0;
      *min_value = static_min.value();
      *step_value = static_step;
      return true;
    }
    int64_t trips = (static_extent.value() + static_step - 1) / static_step;
    if (trips < 0 || trips > 16) {
      return false;
    }
    *trip_count = trips;
    *min_value = static_min.value();
    *step_value = static_step;
    return true;
  }

  bool StmtMayAffectSerializedWarpReplay(const tir::Stmt& stmt) const {
    bool relevant = false;
    tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
      if (relevant) {
        return;
      }
      if (const auto* store = node.as<tir::BufferStoreNode>()) {
        relevant = CanTrackSerializedWarpReplayBufferElements(store->buffer);
        return;
      }
      if (const auto* load = node.as<tir::BufferLoadNode>()) {
        relevant = CanTrackSerializedWarpReplayBufferElements(load->buffer);
        return;
      }
      const auto* call = node.as<tir::CallNode>();
      if (call == nullptr) {
        return;
      }
      const auto* op_node = call->op.as<OpNode>();
      if (op_node != nullptr) {
        if (IsCooperativeThreadIntrinsicName(op_node->name)) {
          relevant = true;
          return;
        }
        if (op_node->name == "tl.tileop.fill") {
          tir::BufferRegion dst_region = tl::NormalizeToBufferRegion(call->args[0]);
          relevant = CanTrackSerializedWarpReplayBufferElements(dst_region->buffer);
          return;
        }
      }
    });
    return relevant;
  }

  std::vector<tir::Buffer> CollectSerializedWarpReplayTrackedBuffersWritten(
      const tir::Stmt& stmt) const {
    std::vector<tir::Buffer> buffers;
    auto add_buffer = [&](const tir::Buffer& buffer) {
      if (!CanTrackSerializedWarpReplayBufferElements(buffer)) {
        return;
      }
      for (const tir::Buffer& existing : buffers) {
        if (SameBuffer(existing, buffer)) {
          return;
        }
      }
      buffers.push_back(buffer);
    };
    tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
      if (const auto* store = node.as<tir::BufferStoreNode>()) {
        add_buffer(store->buffer);
        return;
      }
      const auto* call = node.as<tir::CallNode>();
      if (call == nullptr) {
        return;
      }
      const auto* op_node = call->op.as<OpNode>();
      if (op_node != nullptr && op_node->name == "tl.tileop.fill") {
        add_buffer(tl::NormalizeToBufferRegion(call->args[0])->buffer);
      }
    });
    return buffers;
  }

  bool ShouldInlineStaticLoopForSerializedWarpReplay(const tir::ForNode* op) const {
    int64_t trip_count = 0;
    int64_t min_value = 0;
    int64_t step_value = 0;
    return InNonUnitLogicalThreadRegion() &&
           GetStaticSerializedWarpReplayLoopInfo(op, &trip_count, &min_value, &step_value) &&
           StmtMayAffectSerializedWarpReplay(op->body);
  }

  void EmitInlineStaticLoopForSerializedWarpReplay(const tir::ForNode* op) {
    int64_t trip_count = 0;
    int64_t min_value = 0;
    int64_t step_value = 0;
    ICHECK(GetStaticSerializedWarpReplayLoopInfo(op, &trip_count, &min_value, &step_value));
    for (int64_t trip = 0; trip < trip_count; ++trip) {
      int64_t loop_value = min_value + trip * step_value;
      PrimExpr loop_value_expr = tir::make_const(op->loop_var.dtype(), loop_value);
      mlir::Value loop_value_mlir =
          ConstantIntLike(loop_value, LowerScalarType(op->loop_var.dtype()));
      std::optional<PrimExpr> saved_loop_expr =
          SaveAndSetPrimExpr(bound_prim_exprs_, op->loop_var.get(), loop_value_expr);
      EmitLoopBodyWithBindings(op, loop_value_mlir, [&]() { VisitStmt(op->body); }, false);
      RestorePrimExprBinding(bound_prim_exprs_, op->loop_var.get(), saved_loop_expr);
    }
  }

  bool IsThreadReturnStmt(const tir::Stmt& stmt) const {
    const auto* eval = stmt.as<tir::EvaluateNode>();
    if (eval == nullptr) {
      return false;
    }
    return IsThreadReturnCall(eval->value.as<tir::CallNode>());
  }

  bool IsSimpleThreadReturnIf(const tir::IfThenElseNode* op) const {
    return op != nullptr && !op->else_case.defined() && IsThreadReturnStmt(op->then_case);
  }

  bool IsThreadReturnGuardedBody(const tir::Stmt& body, PrimExpr* cond) const {
    const auto* if_node = body.as<tir::IfThenElseNode>();
    if (!IsSimpleThreadReturnIf(if_node)) {
      return false;
    }
    *cond = if_node->condition;
    return true;
  }

  bool IsSharedStorageSyncStmt(const tir::Stmt& stmt) const {
    const auto* eval = stmt.as<tir::EvaluateNode>();
    if (eval == nullptr) {
      return false;
    }
    const auto* call = eval->value.as<tir::CallNode>();
    if (call == nullptr || !call->op.same_as(tir::builtin::tvm_storage_sync()) ||
        call->args.empty()) {
      return false;
    }
    const auto* scope = call->args[0].as<tir::StringImmNode>();
    return scope != nullptr && (scope->value == "shared" || scope->value == "shared.dyn");
  }

  bool IsSyncWarpStmt(const tir::Stmt& stmt) const {
    const auto* eval = stmt.as<tir::EvaluateNode>();
    if (eval == nullptr) {
      return false;
    }
    const auto* call = eval->value.as<tir::CallNode>();
    return call != nullptr && call->op.same_as(tvm::tl::sync_warp());
  }

  bool IsSyncGridStmt(const tir::Stmt& stmt) const {
    const auto* eval = stmt.as<tir::EvaluateNode>();
    if (eval == nullptr) {
      return false;
    }
    const auto* call = eval->value.as<tir::CallNode>();
    return call != nullptr && call->op.same_as(tvm::tl::sync_grid());
  }

  bool HasNonUnitBlockLaunchExtent() const { return HasNonUnitLaunchExtent(block_launch_stack_); }

  bool CanLowerSyncGridAsLocalBarrier() const {
    return !HasNonUnitBlockLaunchExtent();
  }

  bool IsPhaseBoundarySyncStmt(const tir::Stmt& stmt) const {
    return IsSharedStorageSyncStmt(stmt) || IsSyncWarpStmt(stmt) ||
           (IsSyncGridStmt(stmt) && CanLowerSyncGridAsLocalBarrier());
  }

  bool StmtEndsWithPhaseBoundarySync(
      const tir::Stmt& stmt,
      const tir::Var* thread_var_for_phase_global = nullptr) const {
    if (!stmt.defined()) {
      return false;
    }
    if (IsPhaseBoundarySyncStmt(stmt)) {
      return true;
    }
    if (const auto* seq = stmt.as<tir::SeqStmtNode>()) {
      for (size_t i = seq->seq.size(); i > 0; --i) {
        if (StmtEndsWithPhaseBoundarySync(seq->seq[i - 1], thread_var_for_phase_global)) {
          return true;
        }
        return false;
      }
      return false;
    }
    if (const auto* attr = stmt.as<tir::AttrStmtNode>()) {
      return StmtEndsWithPhaseBoundarySync(attr->body, thread_var_for_phase_global);
    }
    if (const auto* let = stmt.as<tir::LetStmtNode>()) {
      return StmtEndsWithPhaseBoundarySync(let->body, thread_var_for_phase_global);
    }
    if (const auto* realize = stmt.as<tir::BlockRealizeNode>()) {
      const tir::BlockNode* block = realize->block.as<tir::BlockNode>();
      return block != nullptr &&
             StmtEndsWithPhaseBoundarySync(block->body, thread_var_for_phase_global);
    }
    if (const auto* if_node = stmt.as<tir::IfThenElseNode>()) {
      if (thread_var_for_phase_global == nullptr ||
          ExprUsesVar(if_node->condition, *thread_var_for_phase_global)) {
        return false;
      }
      if (StmtEndsWithPhaseBoundarySync(if_node->then_case, thread_var_for_phase_global)) {
        return true;
      }
      return if_node->else_case.defined() &&
             StmtEndsWithPhaseBoundarySync(if_node->else_case.value(),
                                          thread_var_for_phase_global);
    }
    return false;
  }

  void AppendTrailingNoOpPhaseIfNeeded(std::vector<tir::Stmt>* phases,
                                       bool ends_with_phase_boundary) const {
    if (!ends_with_phase_boundary) {
      return;
    }
    if (phases->empty()) {
      phases->push_back(tir::Evaluate(0));
      phases->push_back(tir::Evaluate(0));
      return;
    }
    phases->push_back(tir::Evaluate(0));
  }

  const tir::CallNode* MatchThreadLaunchPhaseGlobalTileCumsumCall(const tir::Stmt& stmt) const {
    tir::Stmt current = stmt;
    while (current.defined()) {
      if (const auto* eval = current.as<tir::EvaluateNode>()) {
        const auto* call = eval->value.as<tir::CallNode>();
        if (call == nullptr) {
          return nullptr;
        }
        const auto* op_node = call->op.as<OpNode>();
        if (op_node == nullptr) {
          return nullptr;
        }
        if (!IsLowerableTileScanIntrinsicName(op_node->name)) {
          return nullptr;
        }
        return call;
      }
      if (const auto* seq = current.as<tir::SeqStmtNode>()) {
        if (seq->seq.size() != 1) {
          return nullptr;
        }
        current = seq->seq[0];
        continue;
      }
      if (const auto* attr = current.as<tir::AttrStmtNode>()) {
        if (attr->attr_key == tir::attr::thread_extent) {
          std::optional<int64_t> static_extent = GetOptionalStaticInt(attr->value);
          if (!static_extent.has_value() || static_extent.value() != 1) {
            return nullptr;
          }
        }
        current = attr->body;
        continue;
      }
      if (const auto* realize = current.as<tir::BlockRealizeNode>()) {
        const tir::BlockNode* block = realize->block.as<tir::BlockNode>();
        if (block == nullptr || block->init.defined() || !block->match_buffers.empty()) {
          return nullptr;
        }
        for (const tir::Buffer& buffer : block->alloc_buffers) {
          if (!tl::IsSharedBuffer(buffer)) {
            return nullptr;
          }
        }
        current = block->body;
        continue;
      }
      return nullptr;
    }
    return nullptr;
  }

  bool IsThreadLaunchPhaseGlobalTileCumsumStmt(const tir::Stmt& stmt) const {
    return MatchThreadLaunchPhaseGlobalTileCumsumCall(stmt) != nullptr;
  }

  bool CanLowerTileCumsumAsThreadLaunchPhaseGlobalOp(const tir::Stmt& stmt,
                                                     const tir::Var* thread_var) const {
    if (thread_var == nullptr) {
      return false;
    }
    const tir::CallNode* call = MatchThreadLaunchPhaseGlobalTileCumsumCall(stmt);
    if (call == nullptr) {
      return false;
    }
    if (call->args.size() != 4U) {
      return false;
    }
    const auto* input_load = call->args[0].as<tir::BufferLoadNode>();
    const auto* output_load = call->args[1].as<tir::BufferLoadNode>();
    if (input_load == nullptr || output_load == nullptr) {
      return false;
    }
    for (const PrimExpr& arg : call->args) {
      if (ExprUsesVar(arg, *thread_var)) {
        return false;
      }
    }
    if (BufferUsesVar(input_load->buffer, *thread_var) ||
        BufferUsesVar(output_load->buffer, *thread_var)) {
      return false;
    }
    return tl::IsSharedBuffer(input_load->buffer) && tl::IsSharedBuffer(output_load->buffer);
  }

  bool PushNonEmptyPhase(ffi::Array<tir::Stmt>* phase, std::vector<tir::Stmt>* phases) const {
    if (phase->empty()) {
      return false;
    }
    if (phase->size() == 1) {
      phases->push_back(phase->front());
    } else {
      phases->push_back(tir::SeqStmt(*phase));
    }
    phase->clear();
    return true;
  }

  void PushPhaseBoundaryOrNoOp(ffi::Array<tir::Stmt>* current_phase,
                               std::vector<tir::Stmt>* phases) const {
    if (!PushNonEmptyPhase(current_phase, phases)) {
      phases->push_back(tir::Evaluate(0));
    }
  }

  bool SplitAtSharedStorageSync(const tir::Stmt& stmt, std::vector<tir::Stmt>* phases,
                                std::string* unsupported_reason = nullptr,
                                std::vector<tir::Buffer>* cross_phase_local_buffers = nullptr,
                                const tir::Var* thread_var_for_phase_global = nullptr) {
    phases->clear();
    if (IsPhaseBoundarySyncStmt(stmt)) {
      return true;
    }

    if (const auto* seq = stmt.as<tir::SeqStmtNode>()) {
      ffi::Array<tir::Stmt> current_phase;
      std::vector<tir::Buffer> current_phase_shared_writes;
      bool saw_sync = false;
      for (size_t child_index = 0; child_index < seq->seq.size(); ++child_index) {
        const tir::Stmt& child = seq->seq[child_index];
        if (IsPhaseBoundarySyncStmt(child)) {
          saw_sync = true;
          PushPhaseBoundaryOrNoOp(&current_phase, phases);
          current_phase_shared_writes.clear();
          continue;
        }
        if (!current_phase.empty() &&
            StmtReadsAnySharedBuffer(child, current_phase_shared_writes)) {
          saw_sync = true;
          PushNonEmptyPhase(&current_phase, phases);
          current_phase_shared_writes.clear();
        }
        if (!current_phase.empty() && StmtContainsCooperativeThreadIntrinsic(child)) {
          saw_sync = true;
          PushNonEmptyPhase(&current_phase, phases);
          current_phase_shared_writes.clear();
        }
        if (CanLowerTileCumsumAsThreadLaunchPhaseGlobalOp(child, thread_var_for_phase_global)) {
          bool prev_is_phase_boundary =
              child_index > 0 && IsPhaseBoundarySyncStmt(seq->seq[child_index - 1]);
          bool next_is_phase_boundary =
              child_index + 1 < seq->seq.size() &&
              IsPhaseBoundarySyncStmt(seq->seq[child_index + 1]);
          if (prev_is_phase_boundary && next_is_phase_boundary && current_phase.empty()) {
            saw_sync = true;
            phases->push_back(child);
            continue;
          }
        }

        std::vector<tir::Stmt> child_phases;
        if (!SplitAtSharedStorageSync(child, &child_phases, unsupported_reason,
                                      cross_phase_local_buffers,
                                      thread_var_for_phase_global)) {
          return false;
        }
        if (child_phases.empty()) {
          saw_sync = true;
          PushPhaseBoundaryOrNoOp(&current_phase, phases);
          continue;
        }
        if (child_phases.size() == 1) {
          current_phase.push_back(child_phases.front());
          CollectSharedBufferStores(child_phases.front(), &current_phase_shared_writes);
          continue;
        }

        saw_sync = true;
        current_phase.push_back(child_phases.front());
        CollectSharedBufferStores(child_phases.front(), &current_phase_shared_writes);
        PushNonEmptyPhase(&current_phase, phases);
        current_phase_shared_writes.clear();
        for (size_t i = 1; i + 1 < child_phases.size(); ++i) {
          phases->push_back(child_phases[i]);
        }
        current_phase.push_back(child_phases.back());
        CollectSharedBufferStores(child_phases.back(), &current_phase_shared_writes);
      }
      PushNonEmptyPhase(&current_phase, phases);
      if (!saw_sync) {
        ICHECK_EQ(phases->size(), 1);
      }
      return true;
    }

    if (const auto* attr = stmt.as<tir::AttrStmtNode>()) {
      if (attr->attr_key == tir::attr::thread_extent) {
        std::optional<int64_t> static_extent = GetOptionalStaticInt(attr->value);
        if (!static_extent.has_value() || static_extent.value() != 1) {
          phases->push_back(stmt);
          return true;
        }
      }

      std::vector<tir::Stmt> body_phases;
      if (!SplitAtSharedStorageSync(attr->body, &body_phases, unsupported_reason,
                                    cross_phase_local_buffers,
                                    thread_var_for_phase_global)) {
        return false;
      }
      AppendTrailingNoOpPhaseIfNeeded(
          &body_phases,
          StmtEndsWithPhaseBoundarySync(attr->body, thread_var_for_phase_global));
      phases->reserve(body_phases.size());
      for (const tir::Stmt& phase : body_phases) {
        phases->push_back(
            tir::AttrStmt(attr->node, attr->attr_key, attr->value, phase, attr->span));
      }
      return true;
    }

    if (const auto* let = stmt.as<tir::LetStmtNode>()) {
      std::vector<tir::Stmt> body_phases;
      if (!SplitAtSharedStorageSync(let->body, &body_phases, unsupported_reason,
                                    cross_phase_local_buffers,
                                    thread_var_for_phase_global)) {
        return false;
      }
      AppendTrailingNoOpPhaseIfNeeded(
          &body_phases,
          StmtEndsWithPhaseBoundarySync(let->body, thread_var_for_phase_global));
      phases->reserve(body_phases.size());
      for (const tir::Stmt& phase : body_phases) {
        phases->push_back(tir::LetStmt(let->var, let->value, phase, let->span));
      }
      return true;
    }

    if (const auto* realize = stmt.as<tir::BlockRealizeNode>()) {
      const tir::BlockNode* block = realize->block.as<tir::BlockNode>();
      ICHECK(block != nullptr);
      if (block->init.defined() || !block->match_buffers.empty()) {
        return false;
      }

      std::vector<tir::Stmt> body_phases;
      if (!SplitAtSharedStorageSync(block->body, &body_phases, unsupported_reason,
                                    cross_phase_local_buffers,
                                    thread_var_for_phase_global)) {
        return false;
      }
      AppendTrailingNoOpPhaseIfNeeded(
          &body_phases,
          StmtEndsWithPhaseBoundarySync(block->body, thread_var_for_phase_global));

      for (const tir::Buffer& buffer : block->alloc_buffers) {
        if (tl::IsSharedBuffer(buffer)) {
          continue;
        }
        int use_count = 0;
        for (const tir::Stmt& phase : body_phases) {
          if (StmtUsesBuffer(phase, buffer)) {
            ++use_count;
          }
        }
        if (use_count > 1) {
          if (cross_phase_local_buffers != nullptr) {
            AppendUniqueBuffer(buffer, cross_phase_local_buffers);
          } else {
            if (unsupported_reason != nullptr && unsupported_reason->empty()) {
              std::ostringstream os;
              os << "local block allocation '" << buffer->name
                 << "' is used across a shared sync barrier inside a non-unit thread launch; "
                 << "riscv only keeps local allocations that are confined to one "
                 << "barrier-separated phase";
              *unsupported_reason = os.str();
            }
            return false;
          }
        }
      }

      phases->reserve(body_phases.size());
      for (const tir::Stmt& phase : body_phases) {
        ffi::Array<tir::Buffer> phase_alloc_buffers;
        for (const tir::Buffer& buffer : block->alloc_buffers) {
          if (tl::IsSharedBuffer(buffer) || StmtUsesBuffer(phase, buffer)) {
            phase_alloc_buffers.push_back(buffer);
          }
        }
        tir::Block phase_block(block->iter_vars, block->reads, block->writes,
                               block->name_hint, phase, block->init, phase_alloc_buffers,
                               block->match_buffers, block->annotations, block->span);
        phases->push_back(tir::BlockRealize(realize->iter_values, realize->predicate,
                                            phase_block, realize->span));
      }
      return true;
    }

    if (const auto* if_node = stmt.as<tir::IfThenElseNode>()) {
      std::vector<tir::Stmt> then_phases;
      if (!SplitAtSharedStorageSync(if_node->then_case, &then_phases, unsupported_reason,
                                    cross_phase_local_buffers,
                                    thread_var_for_phase_global)) {
        return false;
      }
      std::vector<tir::Stmt> else_phases;
      if (if_node->else_case.defined()) {
        if (!SplitAtSharedStorageSync(if_node->else_case.value(), &else_phases,
                                      unsupported_reason, cross_phase_local_buffers,
                                      thread_var_for_phase_global)) {
          return false;
        }
      } else {
        else_phases.push_back(tir::Evaluate(0));
      }

      AppendTrailingNoOpPhaseIfNeeded(
          &then_phases,
          StmtEndsWithPhaseBoundarySync(if_node->then_case, thread_var_for_phase_global));
      if (if_node->else_case.defined()) {
        AppendTrailingNoOpPhaseIfNeeded(
            &else_phases,
            StmtEndsWithPhaseBoundarySync(if_node->else_case.value(),
                                         thread_var_for_phase_global));
      }

      size_t phase_count = std::max(then_phases.size(), else_phases.size());
      if (phase_count > 1) {
        phases->reserve(phase_count);
        for (size_t i = 0; i < phase_count; ++i) {
          tir::Stmt then_phase =
              i < then_phases.size() ? then_phases[i] : tir::Evaluate(0);
          std::optional<tir::Stmt> else_phase;
          if (if_node->else_case.defined()) {
            else_phase = i < else_phases.size() ? else_phases[i] : tir::Evaluate(0);
          }
          phases->push_back(
              tir::IfThenElse(if_node->condition, then_phase, else_phase, if_node->span));
        }
        return true;
      }
      phases->push_back(stmt);
      return true;
    }

    if (stmt.as<tir::ForNode>() != nullptr || stmt.as<tir::AllocateNode>() != nullptr ||
        stmt.as<tir::BufferRealizeNode>() != nullptr) {
      phases->push_back(stmt);
      return true;
    }

    phases->push_back(stmt);
    return true;
  }

  void PushDeferredLoopBindings(const tir::Var& loop_var, DeferredLoopBindings bindings) {
    deferred_loop_bindings_[loop_var.get()].push_back(std::move(bindings));
  }

  void PopDeferredLoopBindings(const tir::Var& loop_var) {
    auto it = deferred_loop_bindings_.find(loop_var.get());
    ICHECK(it != deferred_loop_bindings_.end() && !it->second.empty());
    it->second.pop_back();
    if (it->second.empty()) {
      deferred_loop_bindings_.erase(it);
    }
  }

  const tir::ForNode* FindDeferredBindingLoop(const tir::Stmt& stmt) const {
    tir::Stmt current = stmt;
    while (current.defined()) {
      if (const auto* for_node = current.as<tir::ForNode>()) {
        return for_node;
      }
      if (const auto* attr = current.as<tir::AttrStmtNode>()) {
        current = attr->body;
        continue;
      }
      if (const auto* seq = current.as<tir::SeqStmtNode>()) {
        if (seq->seq.size() != 1) {
          return nullptr;
        }
        current = seq->seq[0];
        continue;
      }
      return nullptr;
    }
    return nullptr;
  }

  bool MatchesBufferLoad(const PrimExpr& expr, const tir::Buffer& buffer,
                         llvm::ArrayRef<tir::Var> vars) {
    const auto* load = expr.as<tir::BufferLoadNode>();
    return load != nullptr && load->buffer.get() == buffer.get() &&
           IndicesMatchIdentityVars(load->indices, vars);
  }

  bool MatchReductionCombineExpr(const PrimExpr& expr, const tir::Buffer& output_buffer,
                                 llvm::ArrayRef<tir::Var> output_vars,
                                 ReductionLoopNestMatch::Kind* kind,
                                 const PrimExpr** input_expr) {
    auto try_match_binary = [&](const PrimExpr& lhs, const PrimExpr& rhs,
                                ReductionLoopNestMatch::Kind candidate_kind) {
      if (MatchesBufferLoad(lhs, output_buffer, output_vars)) {
        *kind = candidate_kind;
        *input_expr = &rhs;
        return true;
      }
      if (MatchesBufferLoad(rhs, output_buffer, output_vars)) {
        *kind = candidate_kind;
        *input_expr = &lhs;
        return true;
      }
      return false;
    };

    if (const auto* add = expr.as<tir::AddNode>()) {
      return try_match_binary(add->a, add->b, ReductionLoopNestMatch::Kind::kAdd);
    }
    if (const auto* min = expr.as<tir::MinNode>()) {
      return try_match_binary(min->a, min->b, ReductionLoopNestMatch::Kind::kMin);
    }
    if (const auto* max = expr.as<tir::MaxNode>()) {
      return try_match_binary(max->a, max->b, ReductionLoopNestMatch::Kind::kMax);
    }
    return false;
  }

  StructuredIndexPattern FullIdentityPattern(size_t rank) {
    StructuredIndexPattern pattern;
    pattern.dims.reserve(rank);
    for (size_t i = 0; i < rank; ++i) {
      pattern.dims.push_back(static_cast<unsigned>(i));
    }
    return pattern;
  }

  bool IsReductionDim(llvm::ArrayRef<size_t> reduction_dims, size_t dim) {
    for (size_t reduction_dim : reduction_dims) {
      if (reduction_dim == dim) {
        return true;
      }
    }
    return false;
  }

  StructuredIndexPattern ReductionOutputPattern(size_t rank,
                                                llvm::ArrayRef<size_t> reduction_dims) {
    StructuredIndexPattern pattern;
    pattern.dims.reserve(rank >= reduction_dims.size() ? rank - reduction_dims.size() : 0);
    for (size_t i = 0; i < rank; ++i) {
      if (IsReductionDim(reduction_dims, i)) {
        continue;
      }
      pattern.dims.push_back(static_cast<unsigned>(i));
    }
    return pattern;
  }

  mlir::AffineMap PatternMap(size_t rank, const StructuredIndexPattern& pattern) {
    llvm::SmallVector<mlir::AffineExpr, 4> results;
    results.reserve(pattern.dims.size());
    for (unsigned dim : pattern.dims) {
      results.push_back(builder_.getAffineDimExpr(dim));
    }
    return mlir::AffineMap::get(static_cast<unsigned>(rank), 0, results, &context_);
  }

  mlir::Value EmitReductionCombine(ReductionLoopNestMatch::Kind kind, DataType dtype,
                                   mlir::OpBuilder& builder, mlir::Location loc,
                                   mlir::Value acc, mlir::Value value) {
    switch (kind) {
      case ReductionLoopNestMatch::Kind::kAdd:
        if (IsFloatLikeType(dtype)) {
          return builder.create<mlir::arith::AddFOp>(loc, acc, value);
        }
        return builder.create<mlir::arith::AddIOp>(loc, acc, value);
      case ReductionLoopNestMatch::Kind::kMin: {
        mlir::Value cond;
        if (IsFloatLikeType(dtype)) {
          cond = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, acc,
                                                     value);
        } else if (dtype.is_uint() || dtype.is_bool()) {
          cond = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, acc,
                                                     value);
        } else {
          cond = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, acc,
                                                     value);
        }
        return builder.create<mlir::arith::SelectOp>(loc, cond, acc, value);
      }
      case ReductionLoopNestMatch::Kind::kMax: {
        mlir::Value cond;
        if (IsFloatLikeType(dtype)) {
          cond = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, acc,
                                                     value);
        } else if (dtype.is_uint() || dtype.is_bool()) {
          cond = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ugt, acc,
                                                     value);
        } else {
          cond = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, acc,
                                                     value);
        }
        return builder.create<mlir::arith::SelectOp>(loc, cond, acc, value);
      }
      case ReductionLoopNestMatch::Kind::kBitAnd:
        ICHECK(IsIntegerLikeType(dtype))
            << "bitwise reduction combine expects an integer-like dtype";
        return builder.create<mlir::arith::AndIOp>(loc, acc, value);
      case ReductionLoopNestMatch::Kind::kBitOr:
        ICHECK(IsIntegerLikeType(dtype))
            << "bitwise reduction combine expects an integer-like dtype";
        return builder.create<mlir::arith::OrIOp>(loc, acc, value);
      case ReductionLoopNestMatch::Kind::kBitXor:
        ICHECK(IsIntegerLikeType(dtype))
            << "bitwise reduction combine expects an integer-like dtype";
        return builder.create<mlir::arith::XOrIOp>(loc, acc, value);
    }
    LOG(FATAL) << "Unknown reduction combine kind";
    TVM_FFI_UNREACHABLE();
  }

  int64_t GetStaticInt(const PrimExpr& expr, const char* what) {
    const auto* imm = expr.as<IntImmNode>();
    ICHECK(imm != nullptr) << what << " must be a static IntImm in riscv lowering";
    return imm->value;
  }

  std::optional<int64_t> GetOptionalStaticInt(const PrimExpr& expr) const {
    if (const auto* imm = expr.as<IntImmNode>()) {
      return imm->value;
    }
    return std::nullopt;
  }

  bool HasDynamicElemOffset(const tir::Buffer& buffer) {
    return !tir::is_zero(buffer->elem_offset) &&
           !GetOptionalStaticInt(buffer->elem_offset).has_value();
  }

  std::optional<bool> GetOptionalStaticBool(const PrimExpr& expr) {
    if (tir::is_zero(expr)) {
      return false;
    }
    if (tir::is_one(expr)) {
      return true;
    }
    return std::nullopt;
  }

  bool GetStaticBool(const PrimExpr& expr, const char* what) {
    std::optional<bool> value = GetOptionalStaticBool(expr);
    if (value.has_value()) {
      return value.value();
    }
    LOG(FATAL) << what << " must be a static boolean in riscv lowering";
    TVM_FFI_UNREACHABLE();
  }

  bool HasCompactRowMajorLayout(const tir::Buffer& buffer) const {
    if (buffer->strides.empty()) {
      return true;
    }
    if (buffer->strides.size() != buffer->shape.size()) {
      return false;
    }

    arith::Analyzer analyzer;
    PrimExpr expected_stride =
        buffer->shape.empty() ? PrimExpr(Integer(1))
                              : tir::make_const(buffer->shape.back().dtype(), 1);
    for (int i = static_cast<int>(buffer->shape.size()) - 1; i >= 0; --i) {
      if (!analyzer.CanProveEqual(buffer->strides[i], expected_stride)) {
        return false;
      }
      expected_stride = analyzer.Simplify(expected_stride * buffer->shape[i]);
    }
    return true;
  }

  void ValidateContiguousBuffer(const tir::Buffer& buffer) {
    ICHECK(IsLowerableScalarType(buffer->dtype))
        << "Unsupported buffer element dtype in riscv: " << buffer->dtype;
    ICHECK(HasCompactRowMajorLayout(buffer))
        << "Only compact row-major buffers are supported in riscv: " << buffer->name;
    ICHECK(!HasDynamicElemOffset(buffer))
        << "Dynamic elem_offset is not supported yet for riscv: " << buffer->name;
    if (std::optional<int64_t> static_offset = GetOptionalStaticInt(buffer->elem_offset)) {
      ICHECK_GE(static_offset.value(), 0)
          << "Negative elem_offset is not supported for riscv: " << buffer->name;
    }
  }

  void ValidateLowerableBufferLayout(const tir::Buffer& buffer) {
    ICHECK(IsLowerableScalarType(buffer->dtype))
        << "Unsupported buffer element dtype in riscv: " << buffer->dtype;
    ICHECK(buffer->strides.empty() || buffer->strides.size() == buffer->shape.size())
        << "Buffer strides rank must match shape rank in riscv: " << buffer->name;
    for (const PrimExpr& stride : buffer->strides) {
      if (std::optional<int64_t> static_stride = GetOptionalStaticInt(stride)) {
        ICHECK_GT(static_stride.value(), 0)
            << "Static buffer strides must be positive in riscv: " << buffer->name;
      }
    }
    if (std::optional<int64_t> static_offset = GetOptionalStaticInt(buffer->elem_offset)) {
      ICHECK_GE(static_offset.value(), 0)
          << "Negative elem_offset is not supported for riscv: " << buffer->name;
    }
  }

  void ValidateBoundLocalAllocationExpr(const tir::Buffer& buffer, const PrimExpr& expr,
                                        const char* field_name) {
    for (const tir::Var& var : tir::UndefinedVars(expr)) {
      if (scalar_values_.count(var.get()) || buffer_values_.count(var.get())) {
        continue;
      }
      LOG(FATAL) << "Dynamic local buffer " << field_name << " for '" << buffer->name
                 << "' uses unbound symbolic var '" << var->name_hint
                 << "' in riscv lowering. Bind it as a PrimFunc scalar parameter, "
                 << "let var, or loop var before T.alloc_buffer.";
      TVM_FFI_UNREACHABLE();
    }
  }

  void ValidateAllocaBufferLayout(const tir::Buffer& buffer) {
    ValidateLowerableBufferLayout(buffer);
    for (const PrimExpr& dim : buffer->shape) {
      if (!dim.as<IntImmNode>()) {
        ValidateBoundLocalAllocationExpr(buffer, dim, "shape");
      }
    }
    if (HasDynamicElemOffset(buffer)) {
      ValidateBoundLocalAllocationExpr(buffer, buffer->elem_offset, "elem_offset");
    }
    for (const PrimExpr& stride : buffer->strides) {
      if (!stride.as<IntImmNode>()) {
        ValidateBoundLocalAllocationExpr(buffer, stride, "stride");
      }
    }
  }

  struct VectorizedRampAccess {
    const tir::RampNode* ramp{nullptr};
    size_t ramp_dim{0};
  };

  std::optional<VectorizedRampAccess> MatchVectorizedRampAccess(const tir::Buffer& buffer,
                                                                const Array<PrimExpr>& indices,
                                                                DataType access_dtype) {
    if (access_dtype.lanes() <= 1) {
      return std::nullopt;
    }
    if (indices.size() != buffer->shape.size()) {
      return std::nullopt;
    }

    VectorizedRampAccess access;
    bool found_ramp = false;
    for (size_t dim = 0; dim < indices.size(); ++dim) {
      const auto* ramp = ResolveRampExpr(indices[dim]);
      if (ramp == nullptr) {
        continue;
      }
      ICHECK(!found_ramp)
          << "Vectorized Ramp buffer access currently supports only one Ramp dimension in "
             "riscv";
      access.ramp = ramp;
      access.ramp_dim = dim;
      found_ramp = true;
    }
    if (!found_ramp) {
      return std::nullopt;
    }
    ICHECK_EQ(access.ramp_dim + 1, indices.size())
        << "Vectorized Ramp buffer access expects the Ramp dimension to be the last buffer index "
           "in riscv";

    ValidateLowerableBufferLayout(buffer);
    ICHECK(buffer->dtype.element_of() == access_dtype.element_of())
        << "Vectorized Ramp buffer access element dtype mismatch in riscv";
    ICHECK_EQ(buffer->dtype.lanes(), 1)
        << "Vectorized Ramp buffer access expects a scalar element buffer in riscv";
    ICHECK_EQ(access_dtype.lanes(), access.ramp->dtype.lanes())
        << "Vectorized Ramp buffer access lane count mismatch in riscv";
    for (size_t dim = 0; dim < indices.size(); ++dim) {
      if (dim == access.ramp_dim) {
        continue;
      }
      ICHECK_EQ(indices[dim].dtype().lanes(), 1)
          << "Non-Ramp dimensions in vectorized Ramp buffer access must be scalar in riscv";
    }
    return access;
  }

  const tir::RampNode* ResolveRampExpr(const PrimExpr& expr) const {
    if (const auto* ramp = expr.as<tir::RampNode>()) {
      return ramp;
    }
    const auto* var = expr.as<tir::VarNode>();
    if (var == nullptr) {
      return nullptr;
    }
    auto it = bound_prim_exprs_.find(var);
    if (it == bound_prim_exprs_.end()) {
      return nullptr;
    }
    return it->second.as<tir::RampNode>();
  }

  llvm::SmallVector<mlir::Value, 8> LowerRampLaneIndices(const tir::RampNode* ramp) {
    mlir::Value base = AsIndex(VisitExpr(ramp->base), ramp->base.dtype());
    mlir::Value stride = AsIndex(VisitExpr(ramp->stride), ramp->stride.dtype());
    llvm::SmallVector<mlir::Value, 8> lane_indices;
    lane_indices.reserve(ramp->dtype.lanes());
    for (int lane = 0; lane < ramp->dtype.lanes(); ++lane) {
      if (lane == 0) {
        lane_indices.push_back(base);
        continue;
      }
      mlir::Value lane_offset = ConstantIntLike(lane, builder_.getIndexType());
      lane_offset = builder_.create<mlir::arith::MulIOp>(loc_, lane_offset, stride);
      lane_indices.push_back(builder_.create<mlir::arith::AddIOp>(loc_, base, lane_offset));
    }
    return lane_indices;
  }

  llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 8> LowerVectorizedRampAccessIndices(
      const Array<PrimExpr>& indices, const VectorizedRampAccess& access) {
    llvm::SmallVector<mlir::Value, 8> ramp_lane_indices = LowerRampLaneIndices(access.ramp);
    llvm::SmallVector<mlir::Value, 4> scalar_indices(indices.size());
    for (size_t dim = 0; dim < indices.size(); ++dim) {
      if (dim == access.ramp_dim) {
        continue;
      }
      scalar_indices[dim] = AsIndex(VisitExpr(indices[dim]), indices[dim].dtype());
    }

    llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 8> lane_indices;
    lane_indices.reserve(ramp_lane_indices.size());
    for (size_t lane = 0; lane < ramp_lane_indices.size(); ++lane) {
      llvm::SmallVector<mlir::Value, 4> indices_for_lane;
      indices_for_lane.reserve(indices.size());
      for (size_t dim = 0; dim < indices.size(); ++dim) {
        indices_for_lane.push_back(dim == access.ramp_dim ? ramp_lane_indices[lane]
                                                          : scalar_indices[dim]);
      }
      lane_indices.push_back(indices_for_lane);
    }
    return lane_indices;
  }

  bool CanLowerRampAccessWithVectorTransfer(const VectorizedRampAccess& access) const {
    return tir::is_one(access.ramp->stride);
  }

  llvm::SmallVector<mlir::Value, 4> LowerVectorTransferAccessIndices(
      const Array<PrimExpr>& indices, const VectorizedRampAccess& access) {
    llvm::SmallVector<mlir::Value, 4> transfer_indices;
    transfer_indices.reserve(indices.size());
    for (size_t dim = 0; dim < indices.size(); ++dim) {
      const PrimExpr& index = dim == access.ramp_dim ? access.ramp->base : indices[dim];
      transfer_indices.push_back(AsIndex(VisitExpr(index), index.dtype()));
    }
    return transfer_indices;
  }

  mlir::Value LowerRampBufferLoad(const tir::Buffer& buffer, const Array<PrimExpr>& indices,
                                  const VectorizedRampAccess& access, DataType result_dtype) {
    if (auto packed_view = ResolvePackedScalarViewBinding(buffer)) {
      llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 8> lane_indices =
          LowerVectorizedRampAccessIndices(indices, access);
      llvm::SmallVector<mlir::Value, 8> lanes;
      lanes.reserve(lane_indices.size());
      for (llvm::ArrayRef<mlir::Value> indices_for_lane : lane_indices) {
        lanes.push_back(LowerPackedScalarViewLoad(buffer, packed_view.value(), indices_for_lane,
                                                  result_dtype.element_of()));
      }
      return builder_.create<mlir::vector::FromElementsOp>(loc_, LowerScalarType(result_dtype),
                                                           lanes);
    }

    mlir::Value memref = LookupBufferValue(buffer);
    if (CanLowerRampAccessWithVectorTransfer(access)) {
      mlir::VectorType vector_type = mlir::cast<mlir::VectorType>(LowerScalarType(result_dtype));
      mlir::Value padding = CreateZeroValue(result_dtype.element_of());
      std::array<bool, 1> in_bounds = {true};
      return builder_.create<mlir::vector::TransferReadOp>(
          loc_, vector_type, memref, LowerVectorTransferAccessIndices(indices, access),
          std::optional<mlir::Value>(padding), llvm::ArrayRef<bool>(in_bounds));
    }

    llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 8> lane_indices =
        LowerVectorizedRampAccessIndices(indices, access);
    llvm::SmallVector<mlir::Value, 8> lanes;
    lanes.reserve(lane_indices.size());
    for (llvm::ArrayRef<mlir::Value> indices_for_lane : lane_indices) {
      mlir::Value lane_value = builder_.create<mlir::memref::LoadOp>(loc_, memref, indices_for_lane);
      lanes.push_back(CastValue(lane_value, buffer->dtype, result_dtype.element_of()));
    }
    return builder_.create<mlir::vector::FromElementsOp>(loc_, LowerScalarType(result_dtype),
                                                         lanes);
  }

  void LowerRampBufferStore(const tir::Buffer& buffer, const Array<PrimExpr>& indices,
                            const VectorizedRampAccess& access, mlir::Value value,
                            DataType value_dtype) {
    DataType vector_store_dtype = buffer->dtype.element_of().with_lanes(access.ramp->dtype.lanes());
    value = CastValue(value, value_dtype, vector_store_dtype);

    if (auto packed_view = ResolvePackedScalarViewBinding(buffer)) {
      llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 8> lane_indices =
          LowerVectorizedRampAccessIndices(indices, access);
      for (int lane = 0; lane < access.ramp->dtype.lanes(); ++lane) {
        mlir::Value lane_value = builder_.create<mlir::vector::ExtractOp>(loc_, value, lane);
        LowerPackedScalarViewStore(buffer, packed_view.value(), lane_indices[lane], lane_value,
                                   buffer->dtype.element_of());
      }
      return;
    }

    mlir::Value memref = LookupBufferValue(buffer);
    if (CanLowerRampAccessWithVectorTransfer(access)) {
      std::array<bool, 1> in_bounds = {true};
      builder_.create<mlir::vector::TransferWriteOp>(
          loc_, value, memref, LowerVectorTransferAccessIndices(indices, access),
          llvm::ArrayRef<bool>(in_bounds));
      return;
    }

    llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 8> lane_indices =
        LowerVectorizedRampAccessIndices(indices, access);
    for (int lane = 0; lane < access.ramp->dtype.lanes(); ++lane) {
      mlir::Value lane_value = builder_.create<mlir::vector::ExtractOp>(loc_, value, lane);
      lane_value = CastValue(lane_value, buffer->dtype.element_of(), buffer->dtype);
      builder_.create<mlir::memref::StoreOp>(loc_, lane_value, memref, lane_indices[lane]);
    }
  }

  mlir::Value ProductOfExtents(llvm::ArrayRef<PrimExpr> extents) {
    mlir::Value product = ConstantIntLike(1, builder_.getIndexType());
    for (const PrimExpr& extent : extents) {
      product = builder_.create<mlir::arith::MulIOp>(
          loc_, product, AsIndex(VisitExpr(extent), extent.dtype()));
    }
    return product;
  }

  llvm::SmallVector<mlir::Value, 4> RowMajorLinearOffsetToIndices(const tir::Buffer& buffer,
                                                                  mlir::Value linear_offset) {
    ICHECK(!buffer->shape.empty())
        << "Scalar access_ptr buffers are not supported yet in riscv atomic lowering";
    llvm::SmallVector<mlir::Value, 4> indices(buffer->shape.size());
    mlir::Value remaining = linear_offset;
    for (size_t dim = 0; dim + 1 < buffer->shape.size(); ++dim) {
      llvm::SmallVector<PrimExpr, 4> suffix;
      suffix.reserve(buffer->shape.size() - dim - 1);
      for (size_t suffix_dim = dim + 1; suffix_dim < buffer->shape.size(); ++suffix_dim) {
        suffix.push_back(buffer->shape[suffix_dim]);
      }
      mlir::Value stride = ProductOfExtents(suffix);
      indices[dim] = builder_.create<mlir::arith::DivUIOp>(loc_, remaining, stride);
      remaining = builder_.create<mlir::arith::RemUIOp>(loc_, remaining, stride);
    }
    indices.back() = remaining;
    return indices;
  }

  mlir::Value LowerDynamicAccessPtrOffsetValue(mlir::Value memref) {
    auto metadata = builder_.create<mlir::memref::ExtractStridedMetadataOp>(loc_, memref);
    return mlir::getValueOrCreateConstantIndexOp(
        builder_, loc_, metadata.getConstifiedMixedOffset());
  }

  void ValidateDynamicAccessPtrElemOffset(const tir::Buffer& buffer, const char* context) {
    if (!HasDynamicElemOffset(buffer)) {
      return;
    }
    ICHECK(buffer->elem_offset.as<tir::VarNode>() != nullptr)
        << context
        << " currently expects dynamic tvm_access_ptr elem_offset to be a direct variable";
  }

  mlir::Value RemoveElemOffset(const tir::Buffer& buffer, mlir::Value memref,
                               mlir::Value physical_offset, const char* context) {
    std::optional<int64_t> static_offset = GetOptionalStaticInt(buffer->elem_offset);
    if (static_offset.has_value()) {
      if (static_offset.value() == 0) {
        return physical_offset;
      }
      mlir::Value offset = ConstantIntLike(static_offset.value(), builder_.getIndexType());
      return builder_.create<mlir::arith::SubIOp>(loc_, physical_offset, offset);
    }
    if (!HasDynamicElemOffset(buffer)) {
      return physical_offset;
    }
    ValidateDynamicAccessPtrElemOffset(buffer, context);
    mlir::Value offset = LowerDynamicAccessPtrOffsetValue(memref);
    return builder_.create<mlir::arith::SubIOp>(loc_, physical_offset, offset);
  }

  llvm::SmallVector<int64_t, 4> StaticAccessPtrStrides(const tir::Buffer& buffer,
                                                       const char* context) {
    ICHECK_EQ(buffer->strides.size(), buffer->shape.size())
        << context << " expects strides rank to match shape rank for non-compact "
                   << "tvm_access_ptr buffers";
    llvm::SmallVector<int64_t, 4> strides;
    strides.reserve(buffer->strides.size());
    for (const PrimExpr& stride_expr : buffer->strides) {
      std::optional<int64_t> stride = GetOptionalStaticInt(stride_expr);
      ICHECK(stride.has_value())
          << context
          << " currently only supports static strides for non-compact tvm_access_ptr buffers";
      ICHECK_GT(stride.value(), 0)
          << context << " expects positive strides for non-compact tvm_access_ptr buffers";
      strides.push_back(stride.value());
    }
    return strides;
  }

  void ValidateRowMajorLikeAccessPtrStrides(const tir::Buffer& buffer,
                                            llvm::ArrayRef<int64_t> strides,
                                            const char* context) {
    for (size_t dim = 0; dim + 1 < buffer->shape.size(); ++dim) {
      std::optional<int64_t> inner_extent = GetOptionalStaticInt(buffer->shape[dim + 1]);
      ICHECK(inner_extent.has_value())
          << context
          << " currently requires static inner extents for non-compact multi-dimensional "
             "tvm_access_ptr buffers";
      ICHECK_GE(strides[dim], strides[dim + 1] * inner_extent.value())
          << context
          << " currently only supports row-major-like static strides for non-compact "
             "tvm_access_ptr buffers";
    }
  }

  void ValidateRankOneDynamicAccessPtrStride(const tir::Buffer& buffer, const char* context) {
    ICHECK_EQ(buffer->shape.size(), 1U)
        << context
        << " currently only supports rank-1 dynamic strides for non-compact tvm_access_ptr "
           "buffers";
    ICHECK_EQ(buffer->strides.size(), 1U)
        << context << " expects a rank-1 stride for non-compact tvm_access_ptr buffers";
    ICHECK(buffer->strides[0].as<tir::VarNode>() != nullptr)
        << context
        << " currently expects dynamic tvm_access_ptr strides to be direct stride variables";
  }

  bool HasDynamicStride(const tir::Buffer& buffer) {
    for (const PrimExpr& stride : buffer->strides) {
      if (!GetOptionalStaticInt(stride).has_value()) {
        return true;
      }
    }
    return false;
  }

  void ValidateDirectDynamicAccessPtrStrides(const tir::Buffer& buffer, const char* context) {
    ICHECK_EQ(buffer->strides.size(), buffer->shape.size())
        << context << " expects strides rank to match shape rank for non-compact "
                   << "tvm_access_ptr buffers";
    for (const PrimExpr& stride : buffer->strides) {
      if (std::optional<int64_t> static_stride = GetOptionalStaticInt(stride)) {
        ICHECK_GT(static_stride.value(), 0)
            << context << " expects positive strides for non-compact tvm_access_ptr buffers";
        continue;
      }
      ICHECK(stride.as<tir::VarNode>() != nullptr)
          << context
          << " currently expects dynamic tvm_access_ptr strides to be direct stride variables";
    }
  }

  mlir::Value LowerDynamicAccessPtrStrideValue(mlir::Value memref, size_t dim) {
    auto metadata = builder_.create<mlir::memref::ExtractStridedMetadataOp>(loc_, memref);
    llvm::SmallVector<mlir::OpFoldResult, 4> mixed_strides =
        metadata.getConstifiedMixedStrides();
    ICHECK_LT(dim, mixed_strides.size());
    return mlir::getValueOrCreateConstantIndexOp(builder_, loc_, mixed_strides[dim]);
  }

  void BindDynamicAccessPtrElemOffsetVar(
      const tir::Buffer& buffer, mlir::Value memref,
      std::vector<std::pair<const Object*, SavedBinding>>* saved_bindings,
      const char* context) {
    if (!HasDynamicElemOffset(buffer)) {
      return;
    }
    ValidateDynamicAccessPtrElemOffset(buffer, context);
    const auto* offset_var = buffer->elem_offset.as<tir::VarNode>();
    if (offset_var == nullptr || scalar_values_.count(offset_var)) {
      return;
    }
    mlir::Value offset = LowerDynamicAccessPtrOffsetValue(memref);
    saved_bindings->emplace_back(offset_var, SaveAndSet(scalar_values_, offset_var, offset));
  }

  void BindDynamicAccessPtrStrideVars(
      const tir::Buffer& buffer, mlir::Value memref,
      std::vector<std::pair<const Object*, SavedBinding>>* saved_bindings) {
    if (buffer->strides.empty()) {
      return;
    }
    for (size_t i = 0; i < buffer->strides.size(); ++i) {
      if (GetOptionalStaticInt(buffer->strides[i]).has_value()) {
        continue;
      }
      const auto* stride_var = buffer->strides[i].as<tir::VarNode>();
      if (stride_var == nullptr || scalar_values_.count(stride_var)) {
        continue;
      }
      mlir::Value stride = LowerDynamicAccessPtrStrideValue(memref, i);
      saved_bindings->emplace_back(stride_var, SaveAndSet(scalar_values_, stride_var, stride));
    }
  }

  llvm::SmallVector<mlir::Value, 4> StaticStridedLinearOffsetToIndices(
      const tir::Buffer& buffer, mlir::Value logical_offset, llvm::ArrayRef<int64_t> strides) {
    ICHECK(!buffer->shape.empty())
        << "Scalar access_ptr buffers are not supported yet in riscv atomic lowering";

    llvm::SmallVector<mlir::Value, 4> indices;
    indices.reserve(buffer->shape.size());
    mlir::Value remaining = logical_offset;
    for (size_t dim = 0; dim < buffer->shape.size(); ++dim) {
      mlir::Value index = remaining;
      if (strides[dim] != 1) {
        mlir::Value stride = ConstantIntLike(strides[dim], builder_.getIndexType());
        index = builder_.create<mlir::arith::DivUIOp>(loc_, remaining, stride);
      }
      if (dim + 1 < buffer->shape.size()) {
        mlir::Value stride = ConstantIntLike(strides[dim], builder_.getIndexType());
        remaining = builder_.create<mlir::arith::RemUIOp>(loc_, remaining, stride);
      }
      indices.push_back(index);
    }
    return indices;
  }

  llvm::SmallVector<mlir::Value, 4> RankOneLinearOffsetToIndex(
      const tir::Buffer& buffer, mlir::Value memref, mlir::Value logical_offset,
      const char* context) {
    ICHECK_EQ(buffer->shape.size(), 1U)
        << context << " rank-one access_ptr index lowering expected a rank-1 buffer";
    ICHECK_EQ(buffer->strides.size(), 1U)
        << context << " rank-one access_ptr index lowering expected a rank-1 stride";

    mlir::Value stride;
    if (std::optional<int64_t> static_stride = GetOptionalStaticInt(buffer->strides[0])) {
      ICHECK_GT(static_stride.value(), 0)
          << context << " expects positive strides for non-compact tvm_access_ptr buffers";
      if (static_stride.value() == 1) {
        return llvm::SmallVector<mlir::Value, 4>{logical_offset};
      }
      stride = ConstantIntLike(static_stride.value(), builder_.getIndexType());
    } else {
      ValidateRankOneDynamicAccessPtrStride(buffer, context);
      stride = LowerDynamicAccessPtrStrideValue(memref, 0);
    }
    mlir::Value index = builder_.create<mlir::arith::DivUIOp>(loc_, logical_offset, stride);
    return llvm::SmallVector<mlir::Value, 4>{index};
  }

  void CollectAdditiveTerms(const PrimExpr& expr, llvm::SmallVector<PrimExpr, 8>* terms) {
    if (const auto* add = expr.as<tir::AddNode>()) {
      CollectAdditiveTerms(add->a, terms);
      CollectAdditiveTerms(add->b, terms);
      return;
    }
    terms->push_back(expr);
  }

  std::optional<PrimExpr> ExtractIndexFactorForStride(const PrimExpr& term,
                                                      const PrimExpr& stride) {
    arith::Analyzer analyzer;
    if (analyzer.CanProveEqual(term, stride)) {
      return Integer(1);
    }
    if (const auto* mul = term.as<tir::MulNode>()) {
      if (analyzer.CanProveEqual(mul->a, stride)) {
        return mul->b;
      }
      if (analyzer.CanProveEqual(mul->b, stride)) {
        return mul->a;
      }
    }
    return std::nullopt;
  }

  bool IsElemOffsetTerm(const tir::Buffer& buffer, const PrimExpr& term) {
    if (tir::is_zero(buffer->elem_offset)) {
      return false;
    }
    arith::Analyzer analyzer;
    return analyzer.CanProveEqual(term, buffer->elem_offset);
  }

  std::optional<llvm::SmallVector<mlir::Value, 4>> TryDecodeStridedAccessPtrIndicesFromExpr(
      const tir::Buffer& buffer, const PrimExpr& physical_offset, const char* context) {
    if (buffer->strides.empty() || buffer->shape.size() <= 1 || !HasDynamicStride(buffer)) {
      return std::nullopt;
    }
    ValidateDirectDynamicAccessPtrStrides(buffer, context);

    llvm::SmallVector<PrimExpr, 8> terms;
    CollectAdditiveTerms(physical_offset, &terms);

    llvm::SmallVector<mlir::Value, 4> indices;
    indices.reserve(buffer->shape.size());
    std::vector<bool> used(terms.size(), false);
    for (size_t dim = 0; dim < buffer->shape.size(); ++dim) {
      std::optional<PrimExpr> index_expr;
      for (size_t term_index = 0; term_index < terms.size(); ++term_index) {
        if (used[term_index]) {
          continue;
        }
        if (std::optional<PrimExpr> candidate =
                ExtractIndexFactorForStride(terms[term_index], buffer->strides[dim])) {
          index_expr = candidate.value();
          used[term_index] = true;
          break;
        }
      }
      if (!index_expr.has_value()) {
        return std::nullopt;
      }
      indices.push_back(AsIndex(VisitExpr(index_expr.value()), index_expr.value().dtype()));
    }

    for (size_t term_index = 0; term_index < terms.size(); ++term_index) {
      if (used[term_index] || IsElemOffsetTerm(buffer, terms[term_index]) ||
          tir::is_zero(terms[term_index])) {
        continue;
      }
      return std::nullopt;
    }
    return indices;
  }

  llvm::SmallVector<mlir::Value, 4> LinearOffsetToMemRefIndices(const tir::Buffer& buffer,
                                                                mlir::Value memref,
                                                                mlir::Value physical_offset,
                                                                const char* context) {
    mlir::Value logical_offset = RemoveElemOffset(buffer, memref, physical_offset, context);
    if (HasCompactRowMajorLayout(buffer)) {
      return RowMajorLinearOffsetToIndices(buffer, logical_offset);
    }

    if (buffer->shape.size() == 1) {
      return RankOneLinearOffsetToIndex(buffer, memref, logical_offset, context);
    }

    llvm::SmallVector<int64_t, 4> strides = StaticAccessPtrStrides(buffer, context);
    ValidateRowMajorLikeAccessPtrStrides(buffer, strides, context);
    return StaticStridedLinearOffsetToIndices(buffer, logical_offset, strides);
  }

  void ValidateAccessPtrLayoutBeforeOffsetLowering(const tir::Buffer& buffer, const char* context) {
    ValidateDynamicAccessPtrElemOffset(buffer, context);
    if (HasCompactRowMajorLayout(buffer)) {
      return;
    }
    if (buffer->shape.size() == 1 && !GetOptionalStaticInt(buffer->strides[0]).has_value()) {
      ValidateRankOneDynamicAccessPtrStride(buffer, context);
      return;
    }
    if (buffer->shape.size() > 1 && HasDynamicStride(buffer)) {
      ValidateDirectDynamicAccessPtrStrides(buffer, context);
      return;
    }
    llvm::SmallVector<int64_t, 4> strides = StaticAccessPtrStrides(buffer, context);
    ValidateRowMajorLikeAccessPtrStrides(buffer, strides, context);
  }

  AccessPtrBinding DecodeElementAccessPtr(const PrimExpr& expr, const char* context) {
    const auto* call = expr.as<tir::CallNode>();
    ICHECK(call != nullptr && call->op.same_as(tir::builtin::tvm_access_ptr()))
        << context << " expects a tvm_access_ptr destination";
    ICHECK_GE(call->args.size(), 5U) << context << " expects a full tvm_access_ptr";
    const auto* data_var = call->args[1].as<tir::VarNode>();
    ICHECK(data_var != nullptr) << context << " expects tvm_access_ptr data to be a Var";

    auto owner_it = buffer_owner_.find(data_var);
    ICHECK(owner_it != buffer_owner_.end())
        << context << " could not resolve tvm_access_ptr buffer owner";
    tir::Buffer buffer = owner_it->second;
    ValidateLowerableBufferLayout(buffer);
    ValidateAccessPtrLayoutBeforeOffsetLowering(buffer, context);

    std::optional<int64_t> extent = GetOptionalStaticInt(call->args[3]);
    ICHECK(extent.has_value() && extent.value() == 1)
        << context << " currently only supports element access_ptr extent=1";

    mlir::Value memref = LookupBufferValue(buffer);
    std::vector<std::pair<const Object*, SavedBinding>> saved_layout_bindings;
    BindDynamicAccessPtrElemOffsetVar(buffer, memref, &saved_layout_bindings, context);
    if (std::optional<llvm::SmallVector<mlir::Value, 4>> indices =
            TryDecodeStridedAccessPtrIndicesFromExpr(buffer, call->args[2], context)) {
      RestoreBindings(scalar_values_, saved_layout_bindings);
      return AccessPtrBinding{buffer, memref, indices.value()};
    }
    ICHECK(!(buffer->shape.size() > 1 && HasDynamicStride(buffer)))
        << context
        << " could not decode dynamic-strided tvm_access_ptr offset expression into logical "
           "memref indices";
    BindDynamicAccessPtrStrideVars(buffer, memref, &saved_layout_bindings);
    mlir::Value linear_offset = AsIndex(VisitExpr(call->args[2]), call->args[2].dtype());
    RestoreBindings(scalar_values_, saved_layout_bindings);
    return AccessPtrBinding{buffer, memref,
                            LinearOffsetToMemRefIndices(buffer, memref, linear_offset, context)};
  }

  ContiguousAccessPtrBinding DecodeContiguousAccessPtr(const PrimExpr& expr, int lane_count,
                                                       const char* context) {
    const auto* call = expr.as<tir::CallNode>();
    ICHECK(call != nullptr && call->op.same_as(tir::builtin::tvm_access_ptr()))
        << context << " expects a tvm_access_ptr operand";
    ICHECK_GE(call->args.size(), 5U) << context << " expects a full tvm_access_ptr";
    const auto* data_var = call->args[1].as<tir::VarNode>();
    ICHECK(data_var != nullptr) << context << " expects tvm_access_ptr data to be a Var";

    auto owner_it = buffer_owner_.find(data_var);
    ICHECK(owner_it != buffer_owner_.end())
        << context << " could not resolve tvm_access_ptr buffer owner";
    tir::Buffer buffer = owner_it->second;
    ValidateLowerableBufferLayout(buffer);
    ICHECK(HasCompactRowMajorLayout(buffer))
        << context
        << " currently only supports compact row-major buffers for vector access_ptr operands";
    ValidateDynamicAccessPtrElemOffset(buffer, context);

    std::optional<int64_t> extent = GetOptionalStaticInt(call->args[3]);
    ICHECK(extent.has_value() && extent.value() == lane_count)
        << context << " currently expects vector access_ptr extent=" << lane_count;

    mlir::Value memref = LookupBufferValue(buffer);
    std::vector<std::pair<const Object*, SavedBinding>> saved_layout_bindings;
    BindDynamicAccessPtrElemOffsetVar(buffer, memref, &saved_layout_bindings, context);
    mlir::Value base_offset = AsIndex(VisitExpr(call->args[2]), call->args[2].dtype());
    RestoreBindings(scalar_values_, saved_layout_bindings);

    llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4> lane_indices;
    lane_indices.reserve(lane_count);
    for (int lane = 0; lane < lane_count; ++lane) {
      mlir::Value lane_offset = base_offset;
      if (lane != 0) {
        mlir::Value delta = ConstantIntLike(lane, builder_.getIndexType());
        lane_offset = builder_.create<mlir::arith::AddIOp>(loc_, base_offset, delta);
      }
      lane_indices.push_back(LinearOffsetToMemRefIndices(buffer, memref, lane_offset, context));
    }
    return ContiguousAccessPtrBinding{buffer, memref, lane_indices};
  }

  mlir::Value CastValueToType(mlir::Value value, mlir::Type target_type, bool source_unsigned) {
    mlir::Type source_type = value.getType();
    if (source_type == target_type) {
      return value;
    }

    if (auto source_vector = mlir::dyn_cast<mlir::VectorType>(source_type)) {
      auto target_vector = mlir::dyn_cast<mlir::VectorType>(target_type);
      if (target_vector == nullptr) {
        LOG(FATAL) << "Unsupported mixed scalar/vector MLIR cast in riscv lowering";
        TVM_FFI_UNREACHABLE();
      }
      ICHECK(source_vector.getRank() == target_vector.getRank())
          << "Unsupported vector rank cast in riscv lowering";
      if (source_vector.getShape() != target_vector.getShape()) {
        LOG(FATAL) << "Unsupported vector shape cast in riscv lowering";
        TVM_FFI_UNREACHABLE();
      }

      mlir::Type source_element = source_vector.getElementType();
      mlir::Type target_element = target_vector.getElementType();
      if (mlir::isa<mlir::IntegerType>(source_element) &&
          mlir::isa<mlir::IntegerType>(target_element)) {
        unsigned source_width = mlir::cast<mlir::IntegerType>(source_element).getWidth();
        unsigned target_width = mlir::cast<mlir::IntegerType>(target_element).getWidth();
        if (source_width < target_width) {
          if (source_unsigned) {
            return builder_.create<mlir::arith::ExtUIOp>(loc_, target_type, value);
          }
          return builder_.create<mlir::arith::ExtSIOp>(loc_, target_type, value);
        }
        if (source_width > target_width) {
          return builder_.create<mlir::arith::TruncIOp>(loc_, target_type, value);
        }
        return value;
      }
      if (mlir::isa<mlir::FloatType>(source_element) &&
          mlir::isa<mlir::FloatType>(target_element)) {
        unsigned source_width = mlir::cast<mlir::FloatType>(source_element).getWidth();
        unsigned target_width = mlir::cast<mlir::FloatType>(target_element).getWidth();
        if (source_width < target_width) {
          return builder_.create<mlir::arith::ExtFOp>(loc_, target_type, value);
        }
        if (source_width > target_width) {
          return builder_.create<mlir::arith::TruncFOp>(loc_, target_type, value);
        }
        return value;
      }
      if (mlir::isa<mlir::IntegerType>(source_element) &&
          mlir::isa<mlir::FloatType>(target_element)) {
        if (source_unsigned) {
          return builder_.create<mlir::arith::UIToFPOp>(loc_, target_type, value);
        }
        return builder_.create<mlir::arith::SIToFPOp>(loc_, target_type, value);
      }
      if (mlir::isa<mlir::FloatType>(source_element) &&
          mlir::isa<mlir::IntegerType>(target_element)) {
        if (source_unsigned) {
          return builder_.create<mlir::arith::FPToUIOp>(loc_, target_type, value);
        }
        return builder_.create<mlir::arith::FPToSIOp>(loc_, target_type, value);
      }

      int64_t num_elements = source_vector.getNumElements();
      llvm::SmallVector<mlir::Value, 4> lanes;
      lanes.reserve(num_elements);
      for (int64_t lane = 0; lane < num_elements; ++lane) {
        mlir::Value source_lane = builder_.create<mlir::vector::ExtractOp>(loc_, value, lane);
        lanes.push_back(
            CastValueToType(source_lane, target_vector.getElementType(), source_unsigned));
      }
      return builder_.create<mlir::vector::FromElementsOp>(loc_, target_vector, lanes);
    }
    if (auto target_vector = mlir::dyn_cast<mlir::VectorType>(target_type)) {
      ICHECK_EQ(target_vector.getRank(), 1)
          << "Only rank-1 vector splat casts are supported in riscv lowering";
      mlir::Value element = CastValueToType(value, target_vector.getElementType(), source_unsigned);
      return builder_.create<mlir::vector::BroadcastOp>(loc_, target_vector, element);
    }

    if (source_type.isIndex() && target_type.isIndex()) {
      return value;
    }
    if (source_type.isIndex() && mlir::isa<mlir::IntegerType>(target_type)) {
      return builder_.create<mlir::arith::IndexCastOp>(loc_, target_type, value);
    }
    if (mlir::isa<mlir::IntegerType>(source_type) && target_type.isIndex()) {
      return builder_.create<mlir::arith::IndexCastOp>(loc_, target_type, value);
    }

    if (mlir::isa<mlir::IntegerType>(source_type) && mlir::isa<mlir::IntegerType>(target_type)) {
      unsigned source_width = mlir::cast<mlir::IntegerType>(source_type).getWidth();
      unsigned target_width = mlir::cast<mlir::IntegerType>(target_type).getWidth();
      if (source_width < target_width) {
        if (source_unsigned) {
          return builder_.create<mlir::arith::ExtUIOp>(loc_, target_type, value);
        }
        return builder_.create<mlir::arith::ExtSIOp>(loc_, target_type, value);
      }
      if (source_width > target_width) {
        return builder_.create<mlir::arith::TruncIOp>(loc_, target_type, value);
      }
      return value;
    }

    if (mlir::isa<mlir::FloatType>(source_type) && mlir::isa<mlir::FloatType>(target_type)) {
      unsigned source_width = mlir::cast<mlir::FloatType>(source_type).getWidth();
      unsigned target_width = mlir::cast<mlir::FloatType>(target_type).getWidth();
      if (source_width < target_width) {
        return builder_.create<mlir::arith::ExtFOp>(loc_, target_type, value);
      }
      if (source_width > target_width) {
        return builder_.create<mlir::arith::TruncFOp>(loc_, target_type, value);
      }
      return value;
    }

    if (mlir::isa<mlir::IntegerType>(source_type) && mlir::isa<mlir::FloatType>(target_type)) {
      if (source_unsigned) {
        return builder_.create<mlir::arith::UIToFPOp>(loc_, target_type, value);
      }
      return builder_.create<mlir::arith::SIToFPOp>(loc_, target_type, value);
    }

    if (mlir::isa<mlir::FloatType>(source_type) && mlir::isa<mlir::IntegerType>(target_type)) {
      if (source_unsigned) {
        return builder_.create<mlir::arith::FPToUIOp>(loc_, target_type, value);
      }
      return builder_.create<mlir::arith::FPToSIOp>(loc_, target_type, value);
    }

    if (source_type.isIndex() && mlir::isa<mlir::FloatType>(target_type)) {
      mlir::Type i64_type = builder_.getIntegerType(64);
      mlir::Value as_int = builder_.create<mlir::arith::IndexCastOp>(loc_, i64_type, value);
      return builder_.create<mlir::arith::SIToFPOp>(loc_, target_type, as_int);
    }

    if (mlir::isa<mlir::FloatType>(source_type) && target_type.isIndex()) {
      mlir::Type i64_type = builder_.getIntegerType(64);
      mlir::Value as_int = builder_.create<mlir::arith::FPToSIOp>(loc_, i64_type, value);
      return builder_.create<mlir::arith::IndexCastOp>(loc_, target_type, as_int);
    }

    LOG(FATAL) << "Unsupported MLIR cast encountered in riscv lowering";
    TVM_FFI_UNREACHABLE();
  }

  mlir::Value CastValue(mlir::Value value, DataType source_dtype, DataType target_dtype) {
    if (target_dtype.is_bool()) {
      return LowerConditionValue(value, source_dtype);
    }
    mlir::Type target_type = LowerScalarType(target_dtype);
    if (source_dtype == target_dtype && value.getType() == target_type) {
      return value;
    }
    bool source_unsigned = source_dtype.is_uint() || source_dtype.is_bool();
    return CastValueToType(value, target_type, source_unsigned);
  }

  mlir::Value LowerShuffle(const tir::ShuffleNode* op) {
    mlir::Type target_type = LowerScalarType(op->dtype);
    if (op->indices.empty()) {
      LOG(FATAL) << "tir.Shuffle with empty indices is not supported in riscv";
      TVM_FFI_UNREACHABLE();
    }

    llvm::SmallVector<mlir::Value, 8> flattened_lanes;
    size_t flattened_lane_count = 0;
    for (const PrimExpr& vector_expr : op->vectors) {
      flattened_lane_count += static_cast<size_t>(vector_expr.dtype().lanes());
    }
    flattened_lanes.reserve(flattened_lane_count);
    for (const PrimExpr& vector_expr : op->vectors) {
      mlir::Value value = CastValue(VisitExpr(vector_expr), vector_expr.dtype(), vector_expr.dtype());
      if (vector_expr.dtype().lanes() == 1) {
        flattened_lanes.push_back(value);
        continue;
      }
      for (int lane = 0; lane < vector_expr.dtype().lanes(); ++lane) {
        flattened_lanes.push_back(builder_.create<mlir::vector::ExtractOp>(loc_, value, lane));
      }
    }

    if (op->dtype.lanes() == 1) {
      ICHECK_EQ(op->indices.size(), 1);
      int64_t index = GetStaticInt(op->indices[0], "tir.Shuffle scalar lane");
      ICHECK_GE(index, 0);
      ICHECK_LT(static_cast<size_t>(index), flattened_lanes.size());
      return flattened_lanes[static_cast<size_t>(index)];
    }

    llvm::SmallVector<mlir::Value, 8> result_lanes;
    result_lanes.reserve(op->indices.size());
    ICHECK_EQ(static_cast<int>(op->indices.size()), op->dtype.lanes())
        << "tir.Shuffle vector result lane count must match dtype in riscv";
    for (const PrimExpr& index_expr : op->indices) {
      int64_t index = GetStaticInt(index_expr, "tir.Shuffle vector lane");
      ICHECK_GE(index, 0);
      ICHECK_LT(static_cast<size_t>(index), flattened_lanes.size());
      result_lanes.push_back(flattened_lanes[static_cast<size_t>(index)]);
    }
    return builder_.create<mlir::vector::FromElementsOp>(loc_, target_type, result_lanes);
  }

  mlir::Value LowerBroadcast(const tir::BroadcastNode* op) {
    mlir::Type target_type = LowerScalarType(op->dtype);
    mlir::Value scalar = CastValue(VisitExpr(op->value), op->value.dtype(), op->dtype.element_of());
    return builder_.create<mlir::vector::BroadcastOp>(loc_, target_type, scalar);
  }

  mlir::Value LowerRamp(const tir::RampNode* op) {
    DataType element_dtype = op->dtype.element_of();
    ICHECK(element_dtype.is_int() || element_dtype.is_uint() || IsFloatLikeType(element_dtype))
        << "tir.Ramp currently supports int/uint/float element dtypes in riscv: "
        << op->dtype;
    mlir::Type target_type = LowerScalarType(op->dtype);
    mlir::Type element_type = LowerScalarType(element_dtype);
    mlir::Value base = CastValue(VisitExpr(op->base), op->base.dtype(), element_dtype);
    mlir::Value stride = CastValue(VisitExpr(op->stride), op->stride.dtype(), element_dtype);
    llvm::SmallVector<mlir::Value, 8> lanes;
    lanes.reserve(op->dtype.lanes());
    for (int lane = 0; lane < op->dtype.lanes(); ++lane) {
      mlir::Value lane_value;
      if (IsFloatLikeType(element_dtype)) {
        lane_value = builder_.create<mlir::arith::ConstantOp>(
            loc_, builder_.getFloatAttr(mlir::cast<mlir::FloatType>(element_type),
                                        static_cast<double>(lane)));
        mlir::Value offset = builder_.create<mlir::arith::MulFOp>(loc_, lane_value, stride);
        lanes.push_back(builder_.create<mlir::arith::AddFOp>(loc_, base, offset));
      } else {
        lane_value = ConstantIntLike(lane, element_type);
        mlir::Value offset = builder_.create<mlir::arith::MulIOp>(loc_, lane_value, stride);
        lanes.push_back(builder_.create<mlir::arith::AddIOp>(loc_, base, offset));
      }
    }
    return builder_.create<mlir::vector::FromElementsOp>(loc_, target_type, lanes);
  }

  mlir::Value LowerConditionValue(mlir::Value value, DataType source_dtype) {
    mlir::Type value_type = value.getType();
    if (value_type.isInteger(1)) {
      return value;
    }
    if (value_type.isIndex()) {
      return builder_.create<mlir::arith::CmpIOp>(
          loc_, mlir::arith::CmpIPredicate::ne, value, ConstantIntLike(0, builder_.getIndexType()));
    }
    if (mlir::isa<mlir::IntegerType>(value_type)) {
      mlir::Type zero_type = value_type;
      return builder_.create<mlir::arith::CmpIOp>(
          loc_, mlir::arith::CmpIPredicate::ne, value, ConstantIntLike(0, zero_type));
    }
    if (mlir::isa<mlir::FloatType>(value_type)) {
      mlir::Type zero_type = value_type;
      mlir::Value zero = builder_.create<mlir::arith::ConstantFloatOp>(
          loc_, mlir::cast<mlir::FloatType>(zero_type), llvm::APFloat(0.0));
      return builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::UNE, value,
                                                  zero);
    }
    LOG(FATAL) << "Unsupported condition value type in riscv lowering for TIR dtype "
               << source_dtype;
    TVM_FFI_UNREACHABLE();
  }

  mlir::Value LowerLogicalValue(mlir::Value value, DataType source_dtype) {
    if (auto vector_type = mlir::dyn_cast<mlir::VectorType>(value.getType())) {
      ICHECK_EQ(vector_type.getRank(), 1)
          << "RISC-V vector logical values require a fixed rank-1 vector";
      ICHECK(!vector_type.isScalable())
          << "RISC-V vector logical values do not support scalable vectors";
      ICHECK(vector_type.getElementType().isInteger(1))
          << "RISC-V vector logical values require i1 elements";
      return value;
    }
    return LowerConditionValue(value, source_dtype);
  }

  mlir::Value CreateBooleanTrueLike(mlir::Type type) {
    mlir::Value one = ConstantIntLike(1, builder_.getI1Type());
    if (type.isInteger(1)) {
      return one;
    }
    auto vector_type = mlir::dyn_cast<mlir::VectorType>(type);
    ICHECK(vector_type && vector_type.getRank() == 1 && !vector_type.isScalable() &&
           vector_type.getElementType().isInteger(1))
        << "RISC-V boolean true materialization requires i1 or fixed vector<Nxi1>";
    return builder_.create<mlir::vector::BroadcastOp>(loc_, vector_type, one);
  }

  mlir::Value LowerCondition(const PrimExpr& expr) {
    return LowerConditionValue(VisitExpr(expr), expr.dtype());
  }

  mlir::Value LoadSerializedWarpReplayCandidateValue(
      const SerializedWarpMatchReplayPattern& pattern, mlir::Value candidate_index) {
    auto emit_load = [&]() {
      llvm::SmallVector<mlir::Value, 4> indices{candidate_index};
      if (auto packed_view = ResolvePackedScalarViewBinding(pattern.load_buffer)) {
        return LowerPackedScalarViewLoad(pattern.load_buffer, packed_view.value(), indices,
                                         pattern.value_dtype);
      }
      mlir::Value memref = LookupBufferValue(pattern.load_buffer);
      mlir::Value load = builder_.create<mlir::memref::LoadOp>(loc_, memref, indices);
      return CastValue(load, pattern.load_buffer->dtype, pattern.value_dtype);
    };

    if (!pattern.guard_upper_bound.has_value()) {
      return emit_load();
    }

    mlir::Value result_slot = CreateStaticAlloca({1}, pattern.value_dtype);
    mlir::Value slot_index = ZeroIndex();
    llvm::SmallVector<mlir::Value, 1> slot_indices{slot_index};
    mlir::Type compare_type = LowerScalarType(pattern.index_expr.dtype());
    mlir::Value candidate_typed = CastValueToType(candidate_index, compare_type, false);
    mlir::Value upper_bound =
        CastValue(VisitExpr(pattern.guard_upper_bound.value()),
                  pattern.guard_upper_bound.value().dtype(), pattern.index_expr.dtype());
    mlir::arith::CmpIPredicate predicate =
        pattern.index_expr.dtype().is_uint() || pattern.index_expr.dtype().is_bool()
            ? mlir::arith::CmpIPredicate::ult
            : mlir::arith::CmpIPredicate::slt;
    mlir::Value cond =
        builder_.create<mlir::arith::CmpIOp>(loc_, predicate, candidate_typed, upper_bound);
    mlir::scf::IfOp if_op = builder_.create<mlir::scf::IfOp>(loc_, cond, true);
    {
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPoint(if_op.thenYield());
      builder_.create<mlir::memref::StoreOp>(loc_, emit_load(), result_slot, slot_indices);
    }
    {
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPoint(if_op.elseYield());
      mlir::Value fallback =
          CastValue(VisitExpr(pattern.fallback_expr.value()), pattern.fallback_expr.value().dtype(),
                    pattern.value_dtype);
      builder_.create<mlir::memref::StoreOp>(loc_, fallback, result_slot, slot_indices);
    }
    return builder_.create<mlir::memref::LoadOp>(loc_, result_slot, slot_indices);
  }

  mlir::Value LowerSerializedWarpReplayMatchSyncCall(const tir::CallNode* op) {
    const ThreadLaunchFrame* thread_idx_x = CurrentThreadIdxXFrame();
    ICHECK(thread_idx_x != nullptr && thread_idx_x->iter_var != nullptr &&
           thread_idx_x->extent.has_value())
        << "serialized warp replay for tl.match_*_sync requires a static threadIdx.x launch";

    std::optional<SerializedWarpMatchKind> kind = GetSerializedWarpMatchKind(op);
    ICHECK(kind.has_value())
        << "serialized warp replay expects a supported tl.match_*_sync op";

    SerializedWarpMatchReplayPattern pattern;
    ICHECK(MatchSerializedWarpMatchReplayPattern(
        op, tvm::ffi::GetRef<tir::Var>(thread_idx_x->iter_var->var.get()), &pattern));

    PrimExpr mask_expr = IsMatchSyncCallExtern(op) ? op->args[1] : op->args[0];
    mlir::Type result_type = LowerScalarType(op->dtype);
    mlir::Value current_mask = CastValue(VisitExpr(mask_expr), mask_expr.dtype(), op->dtype);
    mlir::Value current_tx =
        AsIndex(CurrentThreadIdxXValue("serialized warp replay for tl.match_*_sync"),
                DataType::Int(32));
    mlir::Value warp_size = ConstantIntLike(32, builder_.getIndexType());
    mlir::Value current_lane = builder_.create<mlir::arith::RemUIOp>(loc_, current_tx, warp_size);
    mlir::Value warp_base = builder_.create<mlir::arith::SubIOp>(loc_, current_tx, current_lane);
    mlir::Value current_index = AsIndex(VisitExpr(pattern.index_expr), pattern.index_expr.dtype());
    mlir::Value current_value = LoadSerializedWarpReplayCandidateValue(pattern, current_index);
    mlir::Value extent_value =
        ConstantIntLike(thread_idx_x->extent.value(), builder_.getIndexType());
    mlir::Value zero_mask = ConstantIntLike(0, result_type);
    if (kind.value() == SerializedWarpMatchKind::kAny) {
      mlir::Value result = CreateZeroValue(op->dtype);
      for (int lane = 0; lane < 32; ++lane) {
        mlir::Value lane_index = ConstantIntLike(lane, builder_.getIndexType());
        mlir::Value candidate_tx =
            builder_.create<mlir::arith::AddIOp>(loc_, warp_base, lane_index);
        mlir::Value candidate_valid = builder_.create<mlir::arith::CmpIOp>(
            loc_, mlir::arith::CmpIPredicate::ult, candidate_tx, extent_value);
        mlir::Value lane_mask = ConstantIntLike(static_cast<int64_t>(uint64_t{1} << lane),
                                                result_type);
        mlir::Value lane_enabled = builder_.create<mlir::arith::CmpIOp>(
            loc_, mlir::arith::CmpIPredicate::ne,
            builder_.create<mlir::arith::AndIOp>(loc_, current_mask, lane_mask), zero_mask);
        mlir::Value candidate_index = builder_.create<mlir::arith::SubIOp>(loc_, current_index,
                                                                           current_lane);
        candidate_index =
            builder_.create<mlir::arith::AddIOp>(loc_, candidate_index, lane_index);
        mlir::Value candidate_slot = CreateStaticAlloca({1}, pattern.value_dtype);
        mlir::Value candidate_slot_index = ZeroIndex();
        llvm::SmallVector<mlir::Value, 1> candidate_slot_indices{candidate_slot_index};
        mlir::scf::IfOp candidate_if =
            builder_.create<mlir::scf::IfOp>(loc_, candidate_valid, true);
        {
          mlir::OpBuilder::InsertionGuard guard(builder_);
          builder_.setInsertionPoint(candidate_if.thenYield());
          mlir::Value candidate_value =
              LoadSerializedWarpReplayCandidateValue(pattern, candidate_index);
          builder_.create<mlir::memref::StoreOp>(loc_, candidate_value, candidate_slot,
                                                 candidate_slot_indices);
        }
        {
          mlir::OpBuilder::InsertionGuard guard(builder_);
          builder_.setInsertionPoint(candidate_if.elseYield());
          builder_.create<mlir::memref::StoreOp>(loc_, current_value, candidate_slot,
                                                 candidate_slot_indices);
        }
        mlir::Value candidate_value =
            builder_.create<mlir::memref::LoadOp>(loc_, candidate_slot, candidate_slot_indices);
        mlir::Value values_equal = builder_.create<mlir::arith::CmpIOp>(
            loc_, mlir::arith::CmpIPredicate::eq, current_value, candidate_value);
        mlir::Value include = builder_.create<mlir::arith::AndIOp>(loc_, candidate_valid,
                                                                   lane_enabled);
        include = builder_.create<mlir::arith::AndIOp>(loc_, include, values_equal);
        mlir::Value updated = builder_.create<mlir::arith::OrIOp>(loc_, result, lane_mask);
        result = builder_.create<mlir::arith::SelectOp>(loc_, include, updated, result);
      }
      return result;
    }

    mlir::Value active_mask = zero_mask;
    mlir::Value mismatch_found = builder_.create<mlir::arith::ConstantOp>(
        loc_, builder_.getBoolAttr(false));
    for (int lane = 0; lane < 32; ++lane) {
      mlir::Value lane_index = ConstantIntLike(lane, builder_.getIndexType());
      mlir::Value candidate_tx =
          builder_.create<mlir::arith::AddIOp>(loc_, warp_base, lane_index);
      mlir::Value candidate_valid = builder_.create<mlir::arith::CmpIOp>(
          loc_, mlir::arith::CmpIPredicate::ult, candidate_tx, extent_value);
      mlir::Value lane_mask = ConstantIntLike(static_cast<int64_t>(uint64_t{1} << lane),
                                              result_type);
      mlir::Value lane_enabled = builder_.create<mlir::arith::CmpIOp>(
          loc_, mlir::arith::CmpIPredicate::ne,
          builder_.create<mlir::arith::AndIOp>(loc_, current_mask, lane_mask), zero_mask);
      mlir::Value candidate_index = builder_.create<mlir::arith::SubIOp>(loc_, current_index,
                                                                         current_lane);
      candidate_index =
          builder_.create<mlir::arith::AddIOp>(loc_, candidate_index, lane_index);
      mlir::Value candidate_slot = CreateStaticAlloca({1}, pattern.value_dtype);
      mlir::Value candidate_slot_index = ZeroIndex();
      llvm::SmallVector<mlir::Value, 1> candidate_slot_indices{candidate_slot_index};
      mlir::scf::IfOp candidate_if = builder_.create<mlir::scf::IfOp>(loc_, candidate_valid, true);
      {
        mlir::OpBuilder::InsertionGuard guard(builder_);
        builder_.setInsertionPoint(candidate_if.thenYield());
        mlir::Value candidate_value =
            LoadSerializedWarpReplayCandidateValue(pattern, candidate_index);
        builder_.create<mlir::memref::StoreOp>(loc_, candidate_value, candidate_slot,
                                               candidate_slot_indices);
      }
      {
        mlir::OpBuilder::InsertionGuard guard(builder_);
        builder_.setInsertionPoint(candidate_if.elseYield());
        builder_.create<mlir::memref::StoreOp>(loc_, current_value, candidate_slot,
                                               candidate_slot_indices);
      }
      mlir::Value candidate_value =
          builder_.create<mlir::memref::LoadOp>(loc_, candidate_slot, candidate_slot_indices);
      mlir::Value values_equal = builder_.create<mlir::arith::CmpIOp>(
          loc_, mlir::arith::CmpIPredicate::eq, current_value, candidate_value);
      mlir::Value include = builder_.create<mlir::arith::AndIOp>(loc_, candidate_valid,
                                                                 lane_enabled);
      mlir::Value updated_mask =
          builder_.create<mlir::arith::OrIOp>(loc_, active_mask, lane_mask);
      active_mask =
          builder_.create<mlir::arith::SelectOp>(loc_, include, updated_mask, active_mask);
      mlir::Value values_differ =
          builder_.create<mlir::arith::XOrIOp>(loc_, values_equal,
                                               builder_.create<mlir::arith::ConstantOp>(
                                                   loc_, builder_.getBoolAttr(true)));
      mlir::Value mismatch_here =
          builder_.create<mlir::arith::AndIOp>(loc_, include, values_differ);
      mismatch_found =
          builder_.create<mlir::arith::OrIOp>(loc_, mismatch_found, mismatch_here);
    }
    return builder_.create<mlir::arith::SelectOp>(loc_, mismatch_found, zero_mask, active_mask);
  }

  mlir::Value LowerSerializedWarpReplayShuffleCall(const tir::CallNode* op) {
    const ThreadLaunchFrame* thread_idx_x = CurrentThreadIdxXFrame();
    ICHECK(thread_idx_x != nullptr && thread_idx_x->iter_var != nullptr &&
           thread_idx_x->extent.has_value())
        << "serialized warp replay for tl.shfl_* requires a static threadIdx.x launch";
    std::optional<SerializedWarpShuffleKind> shuffle_kind = GetSerializedWarpShuffleKind(op);
    ICHECK(shuffle_kind.has_value())
        << "serialized warp replay expects a supported tl.shfl_* or tir.tvm_warp_shuffle* op";
    ICHECK(op->args.size() == 4U || op->args.size() == 5U)
        << "serialized warp replay expects <mask, value, lane_arg, width[, warp_size]>";

    PrimExpr mask_expr = ResolveBoundPrimExpr(op->args[0]);
    PrimExpr value_expr = ResolveBoundPrimExpr(op->args[1]);
    bool uses_direct_thread_local = CanUseDirectThreadLocalLoadForSerializedWarpReplay(op->args[1]);
    PrimExpr replay_value_expr = GetSerializedWarpReplayCandidateExpr(op->args[1]);
    PrimExpr lane_expr = ResolveBoundPrimExpr(op->args[2]);
    PrimExpr width_expr = ResolveBoundPrimExpr(op->args[3]);
    if (op->args.size() == 5U) {
      std::optional<int64_t> static_warp_size =
          GetOptionalStaticInt(ResolveBoundPrimExpr(op->args[4]));
      ICHECK(static_warp_size.has_value() && static_warp_size.value() == 32)
          << "serialized warp replay for tir.tvm_warp_shuffle* requires warp_size=32";
    }
    std::optional<int64_t> static_width = GetOptionalStaticInt(width_expr);
    ICHECK(static_width.has_value() && static_width.value() > 0 && static_width.value() <= 32)
        << "serialized warp replay for tl.shfl_* requires a static width in [1, 32]";
    llvm::SmallVector<ThreadLocalBlockAllocBinding, 4> active_thread_local_bindings(
        CurrentActiveThreadLocalBlockAllocBindings().begin(),
        CurrentActiveThreadLocalBlockAllocBindings().end());

    mlir::Type result_type = LowerScalarType(op->dtype);
    mlir::Type mask_type = LowerScalarType(mask_expr.dtype());
    mlir::Value current_value = CastValue(VisitExpr(value_expr), value_expr.dtype(), op->dtype);
    mlir::Value current_mask = CastValueToType(VisitExpr(mask_expr), mask_type,
                                               mask_expr.dtype().is_uint() ||
                                                   mask_expr.dtype().is_bool());
    mlir::Value current_tx =
        AsIndex(CurrentThreadIdxXValue("serialized warp replay for tl.shfl_*"),
                DataType::Int(32));
    mlir::Value warp_size = ConstantIntLike(32, builder_.getIndexType());
    mlir::Value current_lane = builder_.create<mlir::arith::RemUIOp>(loc_, current_tx, warp_size);
    mlir::Value warp_base = builder_.create<mlir::arith::SubIOp>(loc_, current_tx, current_lane);
    mlir::Value lane_arg = AsIndex(VisitExpr(lane_expr), lane_expr.dtype());
    mlir::Value width_value = ConstantIntLike(static_width.value(), builder_.getIndexType());
    mlir::Value subgroup_base = ZeroIndex();
    if (static_width.value() != 32) {
      mlir::Value subgroup_index =
          builder_.create<mlir::arith::DivUIOp>(loc_, current_lane, width_value);
      subgroup_base = builder_.create<mlir::arith::MulIOp>(loc_, subgroup_index, width_value);
    }
    mlir::Value lane_in_subgroup =
        builder_.create<mlir::arith::SubIOp>(loc_, current_lane, subgroup_base);
    mlir::Value candidate_sub_lane = lane_arg;
    mlir::Value lane_in_width = builder_.create<mlir::arith::CmpIOp>(
        loc_, mlir::arith::CmpIPredicate::ult, lane_arg, width_value);
    if (shuffle_kind.value() == SerializedWarpShuffleKind::kXor) {
      mlir::Type i32_type = builder_.getI32Type();
      mlir::Value lane_in_subgroup_i32 = CastValueToType(lane_in_subgroup, i32_type, false);
      mlir::Value lane_arg_i32 = CastValueToType(lane_arg, i32_type, false);
      mlir::Value xor_lane =
          builder_.create<mlir::arith::XOrIOp>(loc_, lane_in_subgroup_i32, lane_arg_i32);
      candidate_sub_lane = CastValueToType(xor_lane, builder_.getIndexType(), false);
      lane_in_width = builder_.create<mlir::arith::CmpIOp>(
          loc_, mlir::arith::CmpIPredicate::ult, candidate_sub_lane, width_value);
    } else if (shuffle_kind.value() == SerializedWarpShuffleKind::kDown) {
      candidate_sub_lane = builder_.create<mlir::arith::AddIOp>(loc_, lane_in_subgroup, lane_arg);
      lane_in_width = builder_.create<mlir::arith::CmpIOp>(
          loc_, mlir::arith::CmpIPredicate::ult, candidate_sub_lane, width_value);
    } else if (shuffle_kind.value() == SerializedWarpShuffleKind::kUp) {
      mlir::Value lane_has_delta = builder_.create<mlir::arith::CmpIOp>(
          loc_, mlir::arith::CmpIPredicate::uge, lane_in_subgroup, lane_arg);
      mlir::Value up_lane =
          builder_.create<mlir::arith::SubIOp>(loc_, lane_in_subgroup, lane_arg);
      candidate_sub_lane =
          builder_.create<mlir::arith::SelectOp>(loc_, lane_has_delta, up_lane, ZeroIndex());
      lane_in_width = lane_has_delta;
    }
    mlir::Value candidate_lane =
        builder_.create<mlir::arith::AddIOp>(loc_, subgroup_base, candidate_sub_lane);
    mlir::Value candidate_tx = builder_.create<mlir::arith::AddIOp>(loc_, warp_base, candidate_lane);
    mlir::Value extent_value =
        ConstantIntLike(thread_idx_x->extent.value(), builder_.getIndexType());
    mlir::Value source_in_bounds = builder_.create<mlir::arith::CmpIOp>(
        loc_, mlir::arith::CmpIPredicate::ult, candidate_tx, extent_value);
    mlir::Value candidate_lane_typed =
        CastValueToType(candidate_lane, mask_type, false);
    mlir::Value lane_bit =
        builder_.create<mlir::arith::ShLIOp>(loc_, ConstantIntLike(1, mask_type),
                                             candidate_lane_typed);
    mlir::Value zero_mask = ConstantIntLike(0, mask_type);
    mlir::Value lane_enabled = builder_.create<mlir::arith::CmpIOp>(
        loc_, mlir::arith::CmpIPredicate::ne,
        builder_.create<mlir::arith::AndIOp>(loc_, current_mask, lane_bit), zero_mask);
    mlir::Value candidate_valid =
        builder_.create<mlir::arith::AndIOp>(loc_, source_in_bounds, lane_in_width);
    candidate_valid = builder_.create<mlir::arith::AndIOp>(loc_, candidate_valid, lane_enabled);

    mlir::Value result_slot = CreateStaticAlloca({1}, op->dtype);
    mlir::Value slot_index = ZeroIndex();
    llvm::SmallVector<mlir::Value, 1> slot_indices{slot_index};
    builder_.create<mlir::memref::StoreOp>(loc_, current_value, result_slot, slot_indices);

    mlir::scf::IfOp candidate_if = builder_.create<mlir::scf::IfOp>(loc_, candidate_valid, false);
    {
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPoint(candidate_if.thenYield());
        mlir::Value candidate_value = current_value;
        PrimExpr thread_extent_expr = IntImm(DataType::Int(32), thread_idx_x->extent.value());
        if (uses_direct_thread_local) {
          std::optional<mlir::Value> direct_value =
              TryLoadSerializedWarpReplayDirectThreadLocalValue(
                  op->args[1], active_thread_local_bindings, candidate_tx, op->dtype);
          if (direct_value.has_value()) {
            candidate_value = direct_value.value();
          } else {
            EmitWithBoundThreadLaunchValue(
                thread_idx_x->iter_var, thread_extent_expr, candidate_tx,
                active_thread_local_bindings, true, [&]() {
                  candidate_value = CastValue(VisitExpr(replay_value_expr),
                                              replay_value_expr.dtype(), op->dtype);
                });
          }
        } else {
          EmitWithBoundThreadLaunchValue(
              thread_idx_x->iter_var, thread_extent_expr, candidate_tx,
              active_thread_local_bindings, true, [&]() {
                candidate_value =
                    CastValue(VisitExpr(replay_value_expr), replay_value_expr.dtype(), op->dtype);
              });
        }
      builder_.create<mlir::memref::StoreOp>(loc_, candidate_value, result_slot, slot_indices);
    }
    mlir::Value result = builder_.create<mlir::memref::LoadOp>(loc_, result_slot, slot_indices);
    return CastValue(result, op->dtype, op->dtype);
  }

  mlir::Value LowerSerializedWarpReplayWarpReduceCall(const tir::CallNode* op) {
    const ThreadLaunchFrame* thread_idx_x = CurrentThreadIdxXFrame();
    ICHECK(thread_idx_x != nullptr && thread_idx_x->iter_var != nullptr &&
           thread_idx_x->extent.has_value())
        << "serialized warp replay for warp_reduce requires a static threadIdx.x launch";
    std::optional<ReductionLoopNestMatch::Kind> kind = GetWarpReduceCombineKind(op);
    ICHECK(kind.has_value()) << "serialized warp replay expects a supported tl.warp_reduce_* op";
    ICHECK_EQ(op->args.size(), 1U) << "tl.warp_reduce_* expects <value>";

    llvm::SmallVector<ThreadLocalBlockAllocBinding, 4> active_thread_local_bindings(
        CurrentActiveThreadLocalBlockAllocBindings().begin(),
        CurrentActiveThreadLocalBlockAllocBindings().end());
    bool uses_direct_thread_local =
        CanUseDirectThreadLocalLoadForSerializedWarpReplay(op->args[0]) ||
        ExprIsThreadLocalBindingLoad(op->args[0], active_thread_local_bindings);
    PrimExpr replay_value_expr =
        uses_direct_thread_local ? ResolveBoundPrimExpr(op->args[0])
                                 : ResolveSerializedWarpReplayExpr(op->args[0]);
    mlir::Value acc =
        CastValue(VisitExpr(replay_value_expr), replay_value_expr.dtype(), op->dtype);
    mlir::Value current_tx =
        AsIndex(CurrentThreadIdxXValue("serialized warp replay for tl.warp_reduce_*"),
                DataType::Int(32));
    mlir::Value warp_size = ConstantIntLike(32, builder_.getIndexType());
    mlir::Value current_lane = builder_.create<mlir::arith::RemUIOp>(loc_, current_tx, warp_size);
    mlir::Value warp_base = builder_.create<mlir::arith::SubIOp>(loc_, current_tx, current_lane);
    mlir::Value extent_value =
        ConstantIntLike(thread_idx_x->extent.value(), builder_.getIndexType());
    PrimExpr thread_extent_expr = IntImm(DataType::Int(32), thread_idx_x->extent.value());
    mlir::Value acc_slot = CreateStaticAlloca({1}, op->dtype);
    mlir::Value slot_index = ZeroIndex();
    llvm::SmallVector<mlir::Value, 1> slot_indices{slot_index};
    builder_.create<mlir::memref::StoreOp>(loc_, acc, acc_slot, slot_indices);

    for (int64_t lane = 0; lane < 32; ++lane) {
      mlir::Value candidate_lane = ConstantIntLike(lane, builder_.getIndexType());
      mlir::Value candidate_tx = builder_.create<mlir::arith::AddIOp>(loc_, warp_base, candidate_lane);
      mlir::Value candidate_in_bounds = builder_.create<mlir::arith::CmpIOp>(
          loc_, mlir::arith::CmpIPredicate::ult, candidate_tx, extent_value);
      mlir::Value candidate_is_other_lane = builder_.create<mlir::arith::CmpIOp>(
          loc_, mlir::arith::CmpIPredicate::ne, candidate_tx, current_tx);
      mlir::Value candidate_valid = builder_.create<mlir::arith::AndIOp>(
          loc_, candidate_in_bounds, candidate_is_other_lane);

      mlir::scf::IfOp candidate_if = builder_.create<mlir::scf::IfOp>(loc_, candidate_valid, false);
      {
        mlir::OpBuilder::InsertionGuard guard(builder_);
        builder_.setInsertionPoint(candidate_if.thenYield());
        mlir::Value candidate_value = acc;
        if (uses_direct_thread_local) {
          std::optional<mlir::Value> direct_value =
              TryLoadSerializedWarpReplayDirectThreadLocalValue(
                  op->args[0], active_thread_local_bindings, candidate_tx, op->dtype);
          if (direct_value.has_value()) {
            candidate_value = direct_value.value();
          } else {
            EmitWithBoundThreadLaunchValue(
                thread_idx_x->iter_var, thread_extent_expr, candidate_tx,
                active_thread_local_bindings, true, [&]() {
                  candidate_value =
                      CastValue(VisitExpr(replay_value_expr), replay_value_expr.dtype(), op->dtype);
                });
          }
        } else {
          EmitWithBoundThreadLaunchValue(
              thread_idx_x->iter_var, thread_extent_expr, candidate_tx,
              active_thread_local_bindings, true, [&]() {
                candidate_value =
                    CastValue(VisitExpr(replay_value_expr), replay_value_expr.dtype(), op->dtype);
              });
        }
        mlir::Value current_acc =
            builder_.create<mlir::memref::LoadOp>(loc_, acc_slot, slot_indices);
        mlir::Value updated =
            EmitReductionCombine(kind.value(), op->dtype, builder_, loc_, current_acc,
                                 candidate_value);
        builder_.create<mlir::memref::StoreOp>(loc_, updated, acc_slot, slot_indices);
      }
    }
    acc = builder_.create<mlir::memref::LoadOp>(loc_, acc_slot, slot_indices);
    return CastValue(acc, op->dtype, op->dtype);
  }

  mlir::Value LowerSerializedWarpReplayVoteCall(const tir::CallNode* op) {
    const ThreadLaunchFrame* thread_idx_x = CurrentThreadIdxXFrame();
    ICHECK(thread_idx_x != nullptr && thread_idx_x->iter_var != nullptr &&
           thread_idx_x->extent.has_value())
        << "serialized warp replay for vote helpers requires a static threadIdx.x launch";
    std::optional<SerializedWarpVoteKind> kind = GetSerializedWarpVoteKind(op);
    ICHECK(kind.has_value())
        << "serialized warp replay expects a supported vote-like cooperative op";

    mlir::Type result_type = LowerScalarType(op->dtype);
    mlir::Type mask_type = builder_.getIntegerType(64);
    mlir::Value current_tx =
        AsIndex(CurrentThreadIdxXValue("serialized warp replay for vote helpers"),
                DataType::Int(32));
    mlir::Value warp_size = ConstantIntLike(32, builder_.getIndexType());
    mlir::Value current_lane = builder_.create<mlir::arith::RemUIOp>(loc_, current_tx, warp_size);
    mlir::Value warp_base = builder_.create<mlir::arith::SubIOp>(loc_, current_tx, current_lane);
    mlir::Value extent_value =
        ConstantIntLike(thread_idx_x->extent.value(), builder_.getIndexType());

    if (kind.value() == SerializedWarpVoteKind::kActiveMask) {
      ICHECK(op->args.empty()) << "tl.activemask expects no arguments";
      mlir::Value active_mask = ConstantIntLike(0, mask_type);
      for (int lane = 0; lane < 32; ++lane) {
        mlir::Value lane_index = ConstantIntLike(lane, builder_.getIndexType());
        mlir::Value candidate_tx =
            builder_.create<mlir::arith::AddIOp>(loc_, warp_base, lane_index);
        mlir::Value candidate_valid = builder_.create<mlir::arith::CmpIOp>(
            loc_, mlir::arith::CmpIPredicate::ult, candidate_tx, extent_value);
        mlir::Value lane_mask =
            builder_.create<mlir::arith::ShLIOp>(loc_, ConstantIntLike(1, mask_type),
                                                 CastValueToType(lane_index, mask_type, true));
        mlir::Value updated = builder_.create<mlir::arith::OrIOp>(loc_, active_mask, lane_mask);
        active_mask =
            builder_.create<mlir::arith::SelectOp>(loc_, candidate_valid, updated, active_mask);
      }
      return CastValueToType(active_mask, result_type, true);
    }

    PrimExpr predicate_expr;
    mlir::Value current_mask;
    if (kind.value() == SerializedWarpVoteKind::kBallot && op->args.size() == 1U) {
      predicate_expr = GetSerializedWarpReplayCandidateExpr(op->args[0]);
      current_mask = ConstantIntLike(static_cast<int64_t>(uint64_t{0xFFFFFFFFu}), mask_type);
    } else {
      ICHECK_EQ(op->args.size(), 2U);
      predicate_expr = GetSerializedWarpReplayCandidateExpr(op->args[1]);
      current_mask = CastValueToType(VisitExpr(op->args[0]), mask_type, true);
    }

    bool uses_direct_thread_local =
        CanUseDirectThreadLocalLoadForSerializedWarpReplay(predicate_expr);
    PrimExpr replay_predicate_expr = GetSerializedWarpReplayCandidateExpr(predicate_expr);
    PrimExpr thread_extent_expr = IntImm(DataType::Int(32), thread_idx_x->extent.value());
    llvm::SmallVector<ThreadLocalBlockAllocBinding, 4> active_thread_local_bindings(
        CurrentActiveThreadLocalBlockAllocBindings().begin(),
        CurrentActiveThreadLocalBlockAllocBindings().end());

    mlir::Value ballot_mask = ConstantIntLike(0, mask_type);
    mlir::Value any_true = builder_.create<mlir::arith::ConstantOp>(loc_, builder_.getBoolAttr(false));
    mlir::Value all_true = builder_.create<mlir::arith::ConstantOp>(loc_, builder_.getBoolAttr(true));

    for (int lane = 0; lane < 32; ++lane) {
      mlir::Value lane_index = ConstantIntLike(lane, builder_.getIndexType());
      mlir::Value candidate_tx =
          builder_.create<mlir::arith::AddIOp>(loc_, warp_base, lane_index);
      mlir::Value candidate_in_bounds = builder_.create<mlir::arith::CmpIOp>(
          loc_, mlir::arith::CmpIPredicate::ult, candidate_tx, extent_value);
      mlir::Value lane_mask =
          builder_.create<mlir::arith::ShLIOp>(loc_, ConstantIntLike(1, mask_type),
                                               CastValueToType(lane_index, mask_type, true));
      mlir::Value lane_enabled = builder_.create<mlir::arith::CmpIOp>(
          loc_, mlir::arith::CmpIPredicate::ne,
          builder_.create<mlir::arith::AndIOp>(loc_, current_mask, lane_mask),
          ConstantIntLike(0, mask_type));
      mlir::Value candidate_valid = builder_.create<mlir::arith::AndIOp>(
          loc_, candidate_in_bounds, lane_enabled);

      mlir::Value predicate_slot = CreateStaticAlloca({1}, DataType::Bool());
      mlir::Value slot_index = ZeroIndex();
      llvm::SmallVector<mlir::Value, 1> slot_indices{slot_index};
      builder_.create<mlir::memref::StoreOp>(
          loc_, builder_.create<mlir::arith::ConstantOp>(loc_, builder_.getBoolAttr(false)),
          predicate_slot, slot_indices);

      mlir::scf::IfOp candidate_if = builder_.create<mlir::scf::IfOp>(loc_, candidate_valid, false);
      {
        mlir::OpBuilder::InsertionGuard guard(builder_);
        builder_.setInsertionPoint(candidate_if.thenYield());
        mlir::Value predicate_value;
        if (uses_direct_thread_local) {
          std::optional<mlir::Value> direct_value =
              TryLoadSerializedWarpReplayDirectThreadLocalValue(
                  predicate_expr, active_thread_local_bindings, candidate_tx,
                  replay_predicate_expr.dtype());
          if (direct_value.has_value()) {
            predicate_value = direct_value.value();
          } else {
            EmitWithBoundThreadLaunchValue(
                thread_idx_x->iter_var, thread_extent_expr, candidate_tx,
                active_thread_local_bindings, true, [&]() {
                  predicate_value = VisitExpr(replay_predicate_expr);
                });
          }
        } else {
          EmitWithBoundThreadLaunchValue(
              thread_idx_x->iter_var, thread_extent_expr, candidate_tx,
              active_thread_local_bindings, true, [&]() {
                predicate_value = VisitExpr(replay_predicate_expr);
              });
        }
        mlir::Value predicate_bool =
            LowerConditionValue(predicate_value, replay_predicate_expr.dtype());
        builder_.create<mlir::memref::StoreOp>(loc_, predicate_bool, predicate_slot, slot_indices);
      }
      mlir::Value predicate_bool =
          builder_.create<mlir::memref::LoadOp>(loc_, predicate_slot, slot_indices);

      mlir::Value vote_mask =
          builder_.create<mlir::arith::SelectOp>(loc_, predicate_bool, lane_mask,
                                                 ConstantIntLike(0, mask_type));
      ballot_mask = builder_.create<mlir::arith::OrIOp>(loc_, ballot_mask, vote_mask);

      if (kind.value() == SerializedWarpVoteKind::kAny) {
        mlir::Value vote_true =
            builder_.create<mlir::arith::AndIOp>(loc_, candidate_valid, predicate_bool);
        any_true = builder_.create<mlir::arith::OrIOp>(loc_, any_true, vote_true);
      } else if (kind.value() == SerializedWarpVoteKind::kAll) {
        mlir::Value predicate_false =
            builder_.create<mlir::arith::XOrIOp>(
                loc_, predicate_bool,
                builder_.create<mlir::arith::ConstantOp>(loc_, builder_.getBoolAttr(true)));
        mlir::Value violating_lane =
            builder_.create<mlir::arith::AndIOp>(loc_, candidate_valid, predicate_false);
        mlir::Value next_all =
            builder_.create<mlir::arith::XOrIOp>(
                loc_, violating_lane,
                builder_.create<mlir::arith::ConstantOp>(loc_, builder_.getBoolAttr(true)));
        all_true = builder_.create<mlir::arith::AndIOp>(loc_, all_true, next_all);
      }
    }

    if (kind.value() == SerializedWarpVoteKind::kBallot) {
      return CastValueToType(ballot_mask, result_type, true);
    }
    if (kind.value() == SerializedWarpVoteKind::kAny) {
      return CastValue(any_true, DataType::Bool(), op->dtype);
    }
    ICHECK(kind.value() == SerializedWarpVoteKind::kAll);
    return CastValue(all_true, DataType::Bool(), op->dtype);
  }

  mlir::Value LowerSerializedThreadReplaySyncthreadsOrCall(const tir::CallNode* op) {
    const ThreadLaunchFrame* thread_idx_x = GetSingleStaticThreadIdxXReplayFrame();
    ICHECK(thread_idx_x != nullptr && thread_idx_x->iter_var != nullptr &&
           thread_idx_x->extent.has_value())
        << "serialized thread replay for tl.syncthreads_or requires a static threadIdx.x launch";
    ICHECK_EQ(op->args.size(), 1U) << "tl.syncthreads_or expects <predicate>";

    PrimExpr predicate_expr = GetSerializedWarpReplayCandidateExpr(op->args[0]);
    bool uses_direct_thread_local =
        CanUseDirectThreadLocalLoadForSerializedWarpReplay(predicate_expr);
    PrimExpr replay_predicate_expr = GetSerializedWarpReplayCandidateExpr(predicate_expr);
    PrimExpr thread_extent_expr = IntImm(DataType::Int(32), thread_idx_x->extent.value());
    llvm::SmallVector<ThreadLocalBlockAllocBinding, 4> active_thread_local_bindings(
        CurrentActiveThreadLocalBlockAllocBindings().begin(),
        CurrentActiveThreadLocalBlockAllocBindings().end());

    mlir::Value acc_slot = CreateStaticAlloca({1}, DataType::Bool());
    mlir::Value slot_index = ZeroIndex();
    llvm::SmallVector<mlir::Value, 1> slot_indices{slot_index};
    builder_.create<mlir::memref::StoreOp>(
        loc_, builder_.create<mlir::arith::ConstantOp>(loc_, builder_.getBoolAttr(false)),
        acc_slot, slot_indices);

    mlir::Value zero = ZeroIndex();
    mlir::Value upper = ConstantIntLike(thread_idx_x->extent.value(), builder_.getIndexType());
    mlir::Value one = ConstantIntLike(1, builder_.getIndexType());
    mlir::scf::ForOp replay_loop = builder_.create<mlir::scf::ForOp>(loc_, zero, upper, one);
    {
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPoint(replay_loop.getBody()->getTerminator());
      mlir::Value candidate_tx = replay_loop.getInductionVar();
      mlir::Value predicate_value;
      if (uses_direct_thread_local) {
        std::optional<mlir::Value> direct_value =
            TryLoadSerializedWarpReplayDirectThreadLocalValue(
                predicate_expr, active_thread_local_bindings, candidate_tx,
                replay_predicate_expr.dtype());
        if (direct_value.has_value()) {
          predicate_value = direct_value.value();
        } else {
          EmitWithBoundThreadLaunchValue(
              thread_idx_x->iter_var, thread_extent_expr, candidate_tx,
              active_thread_local_bindings, true, [&]() {
                predicate_value = VisitExpr(replay_predicate_expr);
              });
        }
      } else {
        EmitWithBoundThreadLaunchValue(
            thread_idx_x->iter_var, thread_extent_expr, candidate_tx,
            active_thread_local_bindings, true, [&]() {
              predicate_value = VisitExpr(replay_predicate_expr);
            });
      }

      mlir::Value predicate_bool =
          LowerConditionValue(predicate_value, replay_predicate_expr.dtype());
      mlir::Value acc = builder_.create<mlir::memref::LoadOp>(loc_, acc_slot, slot_indices);
      mlir::Value updated = builder_.create<mlir::arith::OrIOp>(loc_, acc, predicate_bool);
      builder_.create<mlir::memref::StoreOp>(loc_, updated, acc_slot, slot_indices);
    }

    mlir::Value acc = builder_.create<mlir::memref::LoadOp>(loc_, acc_slot, slot_indices);
    return CastValue(acc, DataType::Bool(), op->dtype);
  }

  mlir::Value LowerSingleThreadCooperativeCall(const tir::CallNode* op) {
    std::string name_storage;
    llvm::StringRef name;
    if (op->op.same_as(tir::builtin::call_extern())) {
      name_storage = GetCallExternName(op);
      ICHECK(name_storage == "__match_any_sync" || name_storage == "__match_all_sync")
          << "Only __match_any_sync / __match_all_sync call_extern forms participate in "
             "single-thread cooperative lowering";
      ICHECK(!InNonUnitLogicalThreadRegion())
          << name_storage
          << " inside non-unit thread launch is not supported yet in riscv lowering";
      name = name_storage;
    } else {
      const auto* op_node = op->op.as<OpNode>();
      ICHECK(op_node != nullptr);
      ICHECK(!InNonUnitLogicalThreadRegion())
          << op_node->name
          << " inside non-unit thread launch is not supported yet in riscv lowering";
      name = llvm::StringRef(op_node->name.c_str());
    }

    auto cast_bool_result = [&](const PrimExpr& predicate) {
      mlir::Value cond = LowerCondition(predicate);
      return CastValue(cond, DataType::Bool(), op->dtype);
    };
    std::string name_text = name.str();

    if (name == "tl.any_sync" || name == "tl.all_sync") {
      ICHECK_EQ(op->args.size(), 2U) << name_text << " expects <mask, predicate>";
      return cast_bool_result(op->args[1]);
    }
    if (name == "tl.syncthreads_or") {
      ICHECK_EQ(op->args.size(), 1U) << name_text << " expects <predicate>";
      return cast_bool_result(op->args[0]);
    }
    if (name == "tl.ballot_sync") {
      ICHECK_EQ(op->args.size(), 2U) << name_text << " expects <mask, predicate>";
      return cast_bool_result(op->args[1]);
    }
    if (name == "tl.ballot") {
      ICHECK_EQ(op->args.size(), 1U) << name_text << " expects <predicate>";
      return cast_bool_result(op->args[0]);
    }
    if (name == "tl.activemask" || name == "tir.tvm_warp_activemask" ||
        name == "tl.match_any_sync" || name == "tl.match_all_sync" ||
        name == "__match_any_sync" || name == "__match_all_sync" ||
        name == "tl.tl_shuffle_elect") {
      if (name == "tl.activemask" || name == "tir.tvm_warp_activemask") {
        ICHECK(op->args.empty()) << name_text << " expects no arguments";
      } else if (name == "tl.match_any_sync" || name == "tl.match_all_sync") {
        ICHECK_EQ(op->args.size(), 2U) << name_text << " expects <mask, value>";
      } else if (name == "__match_any_sync" || name == "__match_all_sync") {
        ICHECK_EQ(op->args.size(), 3U) << name_text << " expects <name, mask, value>";
      } else {
        ICHECK_EQ(op->args.size(), 1U) << name_text << " expects <thread_extent>";
      }
      return ConstantIntLike(1, LowerScalarType(op->dtype));
    }

    if (const auto* op_node = op->op.as<OpNode>()) {
      if (IsCooperativeThreadIntrinsicName(op_node->name)) {
        RejectUnsupportedCooperativeThreadExpression(op_node->name);
      }
    }
    TVM_FFI_UNREACHABLE();
  }

  bool CanLowerCooperativeCallAsSingleThread(const tir::CallNode* op) const {
    if (InNonUnitLogicalThreadRegion()) {
      return false;
    }
    if (IsMatchSyncCallExtern(op)) {
      return true;
    }
    const auto* op_node = op->op.as<OpNode>();
    if (op_node == nullptr) {
      return false;
    }
    llvm::StringRef name(op_node->name.c_str());
    return name == "tl.any_sync" || name == "tl.all_sync" || name == "tl.ballot_sync" ||
           name == "tl.ballot" || name == "tl.activemask" ||
           name == "tir.tvm_warp_activemask" || name == "tl.syncthreads_or" ||
           name == "tl.match_any_sync" || name == "tl.match_all_sync" ||
           name == "tl.tl_shuffle_elect";
  }

  mlir::Value LowerThreadIndexHelperCall(const tir::CallNode* op) {
    const auto* op_node = op->op.as<OpNode>();
    ICHECK(op_node != nullptr && IsThreadIndexHelperIntrinsicName(op_node->name));
    llvm::StringRef name(op_node->name.c_str());
    size_t max_args = name == "tl.get_warp_group_idx" ? 2 : 1;
    ICHECK_LE(op->args.size(), max_args)
        << op_node->name << " received too many arguments in riscv lowering";

    auto index_arg_or_default = [&](size_t arg_index, int64_t fallback) {
      if (arg_index >= op->args.size()) {
        return ConstantIntLike(fallback, builder_.getIndexType());
      }
      return AsIndex(VisitExpr(op->args[arg_index]), op->args[arg_index].dtype());
    };

    mlir::Value tx = AsIndex(CurrentThreadIdxXValue(name), DataType::Int(32));
    mlir::Value warp_size = index_arg_or_default(0, 32);
    mlir::Value result;
    if (name == "tl.get_lane_idx") {
      result = builder_.create<mlir::arith::RemUIOp>(loc_, tx, warp_size);
    } else if (name == "tl.get_warp_idx" || name == "tl.get_warp_idx_sync") {
      result = builder_.create<mlir::arith::DivUIOp>(loc_, tx, warp_size);
    } else {
      mlir::Value warps_per_group = index_arg_or_default(1, 4);
      mlir::Value threads_per_group =
          builder_.create<mlir::arith::MulIOp>(loc_, warp_size, warps_per_group);
      result = builder_.create<mlir::arith::DivUIOp>(loc_, tx, threads_per_group);
    }
    return CastValueToType(result, LowerScalarType(op->dtype), false);
  }

  mlir::Value AsIndex(mlir::Value value, DataType source_dtype) {
    if (value.getType().isIndex()) {
      return value;
    }
    return CastValueToType(value, builder_.getIndexType(), source_dtype.is_uint() || source_dtype.is_bool());
  }

  mlir::Value LowerBitcastCall(const tir::CallNode* op, mlir::Value arg) {
    ICHECK(IsSupportedBitcastCall(op))
        << "Unsupported TIR reinterpret in riscv lowering: " << op->op;
    mlir::Type target_type = LowerScalarType(op->dtype);
    if (arg.getType() == target_type) {
      return arg;
    }
    return builder_.create<mlir::arith::BitcastOp>(loc_, target_type, arg);
  }

  mlir::Value LookupVarValue(const tir::Var& var) const {
    auto scalar_it = scalar_values_.find(var.get());
    if (scalar_it != scalar_values_.end()) {
      return scalar_it->second;
    }
    auto buffer_it = buffer_values_.find(var.get());
    if (buffer_it != buffer_values_.end()) {
      return buffer_it->second;
    }
    LOG(FATAL) << "Unbound TIR var during riscv lowering: " << var->name_hint;
    TVM_FFI_UNREACHABLE();
  }

  mlir::Value LookupBufferValue(const tir::Buffer& buffer) {
    auto it = buffer_values_.find(buffer.get());
    if (it != buffer_values_.end()) {
      if (auto source_type = mlir::dyn_cast<mlir::BaseMemRefType>(it->second.getType())) {
        mlir::MemRefType expected_type = LowerBufferMemRefType(buffer);
        if (source_type.getElementType() == expected_type.getElementType() &&
            it->second.getType() != expected_type) {
          if (IsPreboundThreadLocalBlockBuffer(buffer)) {
            return it->second;
          }
          return CreateBufferAliasView(buffer, it->second);
        }
      }
      return it->second;
    }
    it = buffer_values_.find(buffer->data.get());
    if (it != buffer_values_.end()) {
      if (auto source_type = mlir::dyn_cast<mlir::BaseMemRefType>(it->second.getType())) {
        mlir::MemRefType expected_type = LowerBufferMemRefType(buffer);
        if (source_type.getElementType() == expected_type.getElementType() &&
            it->second.getType() != expected_type) {
          if (IsPreboundThreadLocalBlockBuffer(buffer)) {
            return it->second;
          }
          return CreateBufferAliasView(buffer, it->second);
        }
      }
      return it->second;
    }
    LOG(FATAL) << "Unbound TIR buffer during riscv lowering: " << buffer->name;
    TVM_FFI_UNREACHABLE();
  }

  llvm::SmallVector<mlir::Value, 4> LowerScalarIndices(const Array<PrimExpr>& indices) {
    llvm::SmallVector<mlir::Value, 4> lowered;
    lowered.reserve(indices.size());
    for (const PrimExpr& index : indices) {
      lowered.push_back(AsIndex(VisitExpr(index), index.dtype()));
    }
    return lowered;
  }

  void RestoreBindings(ValueMap& map,
                       const std::vector<std::pair<const Object*, SavedBinding>>& saved_bindings) {
    for (auto it = saved_bindings.rbegin(); it != saved_bindings.rend(); ++it) {
      RestoreBinding(map, it->first, it->second);
    }
  }

  void RestorePackedDataOwnerBindings(
      const std::vector<std::pair<const Object*, std::optional<tir::Buffer>>>& saved_bindings) {
    for (auto it = saved_bindings.rbegin(); it != saved_bindings.rend(); ++it) {
      RestorePackedDataOwnerBinding(packed_data_owner_, it->first, it->second);
    }
  }

  void RestoreBufferOwnerBindings(
      const std::vector<std::pair<const Object*, std::optional<tir::Buffer>>>& saved_bindings) {
    for (auto it = saved_bindings.rbegin(); it != saved_bindings.rend(); ++it) {
      RestoreBufferOwnerBinding(buffer_owner_, it->first, it->second);
    }
  }

  bool IsPackedScalarViewAlias(const tir::Buffer& buffer, const tir::Buffer& source_buffer) const {
    if (buffer->dtype.lanes() != 1 || source_buffer->dtype.lanes() <= 1) {
      return false;
    }
    if (buffer->dtype.element_of() != source_buffer->dtype.element_of()) {
      return false;
    }
    if (!HasCompactRowMajorLayout(buffer) || !HasCompactRowMajorLayout(source_buffer)) {
      return false;
    }
    if (!tir::is_zero(buffer->elem_offset) || !tir::is_zero(source_buffer->elem_offset)) {
      return false;
    }

    arith::Analyzer analyzer;
    PrimExpr source_elements = Integer(1);
    for (const PrimExpr& dim : source_buffer->shape) {
      source_elements = analyzer.Simplify(source_elements * dim);
    }
    PrimExpr view_elements = Integer(1);
    for (const PrimExpr& dim : buffer->shape) {
      view_elements = analyzer.Simplify(view_elements * dim);
    }
    source_elements = analyzer.Simplify(
        source_elements * IntImm(DataType::Int(32), source_buffer->dtype.lanes()));
    return analyzer.CanProveEqual(source_elements, view_elements);
  }

  std::optional<PackedScalarViewBinding> SaveAndBindPackedScalarView(const tir::Buffer& view_buffer,
                                                                     const tir::Buffer& source_buffer) {
    auto it = packed_scalar_view_bindings_.find(view_buffer.get());
    std::optional<PackedScalarViewBinding> saved;
    if (it != packed_scalar_view_bindings_.end()) {
      saved = it->second;
    }
    packed_scalar_view_bindings_[view_buffer.get()] = PackedScalarViewBinding{
        source_buffer, source_buffer->dtype, view_buffer->dtype};
    return saved;
  }

  std::optional<tir::Buffer> SaveAndSetPackedDataOwner(const Object* key,
                                                       const tir::Buffer& buffer) {
    auto it = packed_data_owner_.find(key);
    std::optional<tir::Buffer> saved;
    if (it != packed_data_owner_.end()) {
      saved = it->second;
    }
    packed_data_owner_[key] = buffer;
    return saved;
  }

  std::optional<tir::Buffer> SaveAndSetBufferOwner(const Object* key, const tir::Buffer& buffer) {
    auto it = buffer_owner_.find(key);
    std::optional<tir::Buffer> saved;
    if (it != buffer_owner_.end()) {
      saved = it->second;
    }
    buffer_owner_[key] = buffer;
    return saved;
  }

  void BindBufferAliases(const tir::Buffer& buffer, mlir::Value value,
                         std::vector<std::pair<const Object*, SavedBinding>>* saved_bindings,
                         std::vector<std::pair<const Object*, std::optional<tir::Buffer>>>*
                             saved_buffer_owners = nullptr) {
    saved_bindings->emplace_back(buffer.get(), SaveAndSet(buffer_values_, buffer.get(), value));
    saved_bindings->emplace_back(buffer->data.get(),
                                 SaveAndSet(buffer_values_, buffer->data.get(), value));
    if (saved_buffer_owners != nullptr) {
      saved_buffer_owners->emplace_back(buffer.get(), SaveAndSetBufferOwner(buffer.get(), buffer));
      saved_buffer_owners->emplace_back(buffer->data.get(),
                                        SaveAndSetBufferOwner(buffer->data.get(), buffer));
    } else {
      buffer_owner_[buffer.get()] = buffer;
      buffer_owner_[buffer->data.get()] = buffer;
    }
  }

  std::optional<PackedScalarViewBinding> LookupPackedScalarViewBinding(
      const tir::Buffer& buffer) const {
    auto it = packed_scalar_view_bindings_.find(buffer.get());
    if (it != packed_scalar_view_bindings_.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  std::optional<PackedScalarViewBinding> ResolvePackedScalarViewBinding(
      const tir::Buffer& buffer) const {
    if (auto binding = LookupPackedScalarViewBinding(buffer)) {
      return binding;
    }
    auto source_it = packed_data_owner_.find(buffer->data.get());
    if (source_it != packed_data_owner_.end() &&
        IsPackedScalarViewAlias(buffer, source_it->second)) {
      return PackedScalarViewBinding{source_it->second, source_it->second->dtype, buffer->dtype};
    }
    return std::nullopt;
  }

  mlir::Value CreateBufferAliasView(const tir::Buffer& buffer, mlir::Value source) {
    ValidateLowerableBufferLayout(buffer);

    mlir::MemRefType result_type = LowerBufferMemRefType(buffer);
    if (source.getType() == result_type) {
      return source;
    }

    auto source_type = mlir::dyn_cast<mlir::BaseMemRefType>(source.getType());
    ICHECK(source_type) << "DeclBuffer alias expects a memref source in riscv lowering: "
                        << buffer->name;
    ICHECK(source_type.getElementType() == result_type.getElementType())
        << "DeclBuffer alias changes element type without a supported packed-scalar view in riscv: "
        << buffer->name;

    return builder_
        .create<mlir::memref::ReinterpretCastOp>(
            loc_, result_type, source, LowerIndexOpFoldResult(buffer->elem_offset),
            LowerBufferShapeOpFoldResults(buffer), LowerBufferStrideOpFoldResults(buffer))
        .getResult();
  }

  void BindDynamicShapeVars(const tir::Buffer& buffer, mlir::Value value) {
    for (size_t i = 0; i < buffer->shape.size(); ++i) {
      const auto* dim_var = buffer->shape[i].as<tir::VarNode>();
      if (dim_var == nullptr || scalar_values_.count(dim_var)) {
        continue;
      }
      mlir::Value index = ConstantIntLike(static_cast<int64_t>(i), builder_.getIndexType());
      mlir::Value dim = builder_.create<mlir::memref::DimOp>(loc_, value, index);
      scalar_values_[dim_var] = dim;
    }
  }

  mlir::Value CreateSubview(const tir::MatchBufferRegion& match_buffer) {
    const tir::Buffer& target_buffer = match_buffer->buffer;
    const tir::BufferRegion& source_region = match_buffer->source;
    const tir::Buffer& source_buffer = source_region->buffer;

    ValidateLowerableBufferLayout(target_buffer);

    mlir::Value source = LookupBufferValue(source_buffer);
    mlir::MemRefType source_type = mlir::cast<mlir::MemRefType>(source.getType());
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets = LowerRegionOffsets(source_region->region);
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes = LowerRegionSizes(source_region->region);
    llvm::SmallVector<mlir::OpFoldResult, 4> strides = UnitStrides(source_region->region.size());

    mlir::MemRefType result_type;
    if (target_buffer->shape.size() == source_region->region.size()) {
      result_type = mlir::memref::SubViewOp::inferResultType(source_type, offsets, sizes, strides);
    } else {
      result_type = mlir::memref::SubViewOp::inferRankReducedResultType(
          LowerStaticShape(target_buffer->shape), source_type, offsets, sizes, strides);
    }

    return builder_
        .create<mlir::memref::SubViewOp>(loc_, result_type, source, offsets, sizes, strides)
        .getResult();
  }

  mlir::Value LowerPackedScalarViewLoad(const tir::Buffer& view_buffer,
                                        const PackedScalarViewBinding& binding,
                                        llvm::ArrayRef<mlir::Value> view_indices,
                                        DataType result_dtype) {
    ICHECK_GT(binding.source_dtype.lanes(), 1)
        << "Packed scalar view source buffer must have a vector element dtype in riscv";
    mlir::Value memref = LookupBufferValue(binding.source_buffer);
    mlir::Value linear_index = LinearizeCompactBufferIndices(view_buffer, view_indices);
    mlir::Value lane_count =
        ConstantIntLike(binding.source_dtype.lanes(), builder_.getIndexType());
    mlir::Value packed_index =
        builder_.create<mlir::arith::DivUIOp>(loc_, linear_index, lane_count);
    mlir::Value lane_index =
        builder_.create<mlir::arith::RemUIOp>(loc_, linear_index, lane_count);
    mlir::Value packed_value =
        builder_.create<mlir::memref::LoadOp>(loc_, memref, llvm::SmallVector<mlir::Value, 1>{packed_index});
    llvm::SmallVector<mlir::OpFoldResult, 1> lane_position{lane_index};
    mlir::Value scalar_value =
        builder_.create<mlir::vector::ExtractOp>(loc_, packed_value, lane_position);
    return CastValue(scalar_value, binding.view_dtype, result_dtype);
  }

  void LowerPackedScalarViewStore(const tir::Buffer& view_buffer,
                                  const PackedScalarViewBinding& binding,
                                  llvm::ArrayRef<mlir::Value> view_indices,
                                  mlir::Value value, DataType value_dtype) {
    ICHECK_GT(binding.source_dtype.lanes(), 1)
        << "Packed scalar view source buffer must have a vector element dtype in riscv";
    mlir::Value memref = LookupBufferValue(binding.source_buffer);
    mlir::Value linear_index = LinearizeCompactBufferIndices(view_buffer, view_indices);
    mlir::Value lane_count =
        ConstantIntLike(binding.source_dtype.lanes(), builder_.getIndexType());
    mlir::Value packed_index =
        builder_.create<mlir::arith::DivUIOp>(loc_, linear_index, lane_count);
    mlir::Value lane_index =
        builder_.create<mlir::arith::RemUIOp>(loc_, linear_index, lane_count);
    llvm::SmallVector<mlir::Value, 1> packed_indices{packed_index};
    mlir::Value packed_value = builder_.create<mlir::memref::LoadOp>(loc_, memref, packed_indices);
    mlir::Value lane_value = CastValue(value, value_dtype, binding.view_dtype);
    llvm::SmallVector<mlir::OpFoldResult, 1> lane_position{lane_index};
    mlir::Value updated =
        builder_.create<mlir::vector::InsertOp>(loc_, lane_value, packed_value, lane_position);
    builder_.create<mlir::memref::StoreOp>(loc_, updated, memref, packed_indices);
  }

  mlir::Value CreateSubview(const tir::BufferRegion& region) {
    const tir::Buffer& source_buffer = region->buffer;
    mlir::Value source = LookupBufferValue(source_buffer);
    mlir::MemRefType source_type = mlir::cast<mlir::MemRefType>(source.getType());
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets = LowerRegionOffsets(region->region);
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes = LowerRegionSizes(region->region);
    llvm::SmallVector<mlir::OpFoldResult, 4> strides = UnitStrides(region->region.size());
    mlir::MemRefType result_type =
        mlir::memref::SubViewOp::inferResultType(source_type, offsets, sizes, strides);
    return builder_
        .create<mlir::memref::SubViewOp>(loc_, result_type, source, offsets, sizes, strides)
        .getResult();
  }

  mlir::Value CreateLogicalSubview(const tir::BufferRegion& region) {
    Array<PrimExpr> logical_extents;
    for (const Range& range : region->region) {
      if (!IsStaticOne(range->extent)) {
        logical_extents.push_back(range->extent);
      }
    }
    if (logical_extents.size() == region->region.size()) {
      return CreateSubview(region);
    }

    const tir::Buffer& source_buffer = region->buffer;
    mlir::Value source = LookupBufferValue(source_buffer);
    mlir::MemRefType source_type = mlir::cast<mlir::MemRefType>(source.getType());
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets = LowerRegionOffsets(region->region);
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes = LowerRegionSizes(region->region);
    llvm::SmallVector<mlir::OpFoldResult, 4> strides = UnitStrides(region->region.size());
    mlir::MemRefType result_type = mlir::memref::SubViewOp::inferRankReducedResultType(
        LowerStaticShape(logical_extents), source_type, offsets, sizes, strides);
    return builder_
        .create<mlir::memref::SubViewOp>(loc_, result_type, source, offsets, sizes, strides)
        .getResult();
  }

  mlir::Value CreateAlloca(const Array<PrimExpr>& shape_exprs, DataType element_dtype) {
    mlir::MemRefType memref_type = LowerMemRefType(element_dtype, shape_exprs);
    llvm::SmallVector<mlir::Value, 4> dynamic_sizes = LowerDynamicSizes(shape_exprs);
    return builder_.create<mlir::memref::AllocaOp>(loc_, memref_type, dynamic_sizes);
  }

  mlir::Value CreateAlloca(const tir::Buffer& buffer) {
    mlir::MemRefType memref_type = LowerBufferMemRefType(buffer);
    llvm::SmallVector<mlir::Value, 4> dynamic_sizes = LowerDynamicSizes(buffer->shape);
    llvm::SmallVector<mlir::Value, 4> symbol_operands = LowerDynamicLayoutSymbols(buffer);
    return builder_.create<mlir::memref::AllocaOp>(loc_, memref_type, dynamic_sizes,
                                                   symbol_operands);
  }

  mlir::Value CreateStaticAlloca(llvm::ArrayRef<int64_t> shape, DataType element_dtype) {
    mlir::MemRefType memref_type = mlir::MemRefType::get(shape, LowerScalarType(element_dtype));
    return builder_.create<mlir::memref::AllocaOp>(loc_, memref_type);
  }

  std::optional<PrimExpr> LookupLocalVarInitForDataVar(const tir::Var& data_var) const {
    if (!local_var_init_map_.defined()) {
      return std::nullopt;
    }
    auto it = local_var_init_map_.find(data_var);
    if (it == local_var_init_map_.end()) {
      return std::nullopt;
    }
    return (*it).second;
  }

  std::optional<PrimExpr> LookupLocalVarInitFromAnnotations(
      const Map<String, ffi::Any>& annotations, const tir::Var& data_var,
      bool allow_plain_primexpr = false) const {
    auto init_it = annotations.find(tl::attr::kLocalVarInit);
    if (init_it == annotations.end()) {
      return std::nullopt;
    }
    if (auto local_init_map = (*init_it).second.try_cast<Map<tir::Var, PrimExpr>>()) {
      auto value_it = local_init_map.value().find(data_var);
      if (value_it != local_init_map.value().end()) {
        return (*value_it).second;
      }
      return std::nullopt;
    }
    if (allow_plain_primexpr) {
      if (auto init_expr = (*init_it).second.try_cast<PrimExpr>()) {
        return init_expr.value();
      }
    }
    return std::nullopt;
  }

  void InitializeAllocaFromExpr(mlir::Value alloc, const Array<PrimExpr>& shape, DataType dtype,
                                const PrimExpr& init_expr) {
    llvm::SmallVector<mlir::Value, 4> zero_indices;
    zero_indices.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      zero_indices.push_back(ZeroIndex());
    }
    mlir::Value init_value = VisitExpr(init_expr);
    init_value = CastValue(init_value, init_expr.dtype(), dtype);
    builder_.create<mlir::memref::StoreOp>(loc_, init_value, alloc, zero_indices);
  }

  mlir::Value CreateZeroValueBeforeCanonicalHelper(DataType dtype) {
    mlir::Type type = LowerScalarType(dtype);
    if (dtype.lanes() > 1) {
      mlir::Type element_type = LowerScalarType(dtype.element_of());
      llvm::SmallVector<mlir::Value, 8> lanes;
      lanes.reserve(dtype.lanes());
      for (int lane = 0; lane < dtype.lanes(); ++lane) {
        if (IsFloatLikeType(dtype)) {
          lanes.push_back(builder_.create<mlir::arith::ConstantOp>(
              loc_, builder_.getFloatAttr(mlir::cast<mlir::FloatType>(element_type), 0.0)));
        } else {
          lanes.push_back(ConstantIntLike(0, element_type));
        }
      }
      return builder_.create<mlir::vector::FromElementsOp>(loc_, type, lanes);
    }
    if (IsFloatLikeType(dtype)) {
      return builder_.create<mlir::arith::ConstantOp>(
          loc_, builder_.getFloatAttr(mlir::cast<mlir::FloatType>(type), 0.0));
    }
    return ConstantIntLike(0, type);
  }

  void InitializeThreadLocalBackingFromExpr(mlir::Value backing, const tir::Buffer& buffer,
                                            const PrimExpr& thread_extent,
                                            std::optional<PrimExpr> init_expr) {
    mlir::Value zero = ZeroIndex();
    mlir::Value upper = AsIndex(VisitExpr(thread_extent), thread_extent.dtype());
    mlir::Value one = ConstantIntLike(1, builder_.getIndexType());
    mlir::scf::ForOp for_op = builder_.create<mlir::scf::ForOp>(loc_, zero, upper, one);

    mlir::OpBuilder::InsertionGuard guard(builder_);
    builder_.setInsertionPoint(for_op.getBody()->getTerminator());
    llvm::SmallVector<mlir::Value, 4> indices;
    indices.push_back(for_op.getInductionVar());
    if (!IsScalarLocalVarBuffer(buffer)) {
      for (size_t i = 0; i < buffer->shape.size(); ++i) {
        indices.push_back(ZeroIndex());
      }
    }
    mlir::Value init_value;
    if (init_expr.has_value()) {
      init_value = VisitExpr(init_expr.value());
      init_value = CastValue(init_value, init_expr.value().dtype(), buffer->dtype);
    } else {
      init_value = CreateZeroValueBeforeCanonicalHelper(buffer->dtype);
    }
    builder_.create<mlir::memref::StoreOp>(loc_, init_value, backing, indices);
  }

  void MaybeInitializeAllocaFromLocalVarInit(mlir::Value alloc, const tir::Var& data_var,
                                             const Array<PrimExpr>& shape, DataType dtype) {
    std::optional<PrimExpr> init = LookupLocalVarInitForDataVar(data_var);
    if (!init.has_value()) {
      return;
    }
    InitializeAllocaFromExpr(alloc, shape, dtype, init.value());
  }

  mlir::Value CreateInfinityValue(DataType dtype) {
    mlir::Type type = LowerScalarType(dtype);
    if (auto vector_type = mlir::dyn_cast<mlir::VectorType>(type)) {
      mlir::Value scalar =
          CreateInfinityConstant(builder_, loc_, vector_type.getElementType());
      return builder_.create<mlir::vector::BroadcastOp>(loc_, vector_type, scalar);
    }
    return CreateInfinityConstant(builder_, loc_, type);
  }

  mlir::Value MaterializeTranspose2D(mlir::Value source, const PrimExpr& rows,
                                     const PrimExpr& cols, DataType element_dtype) {
    mlir::Value transposed = CreateAlloca(Array<PrimExpr>{cols, rows}, element_dtype);
    mlir::Value zero = ZeroIndex();
    mlir::Value one = ConstantIntLike(1, builder_.getIndexType());
    mlir::Value row_upper = AsIndex(VisitExpr(rows), rows.dtype());
    mlir::Value col_upper = AsIndex(VisitExpr(cols), cols.dtype());

    mlir::scf::ForOp row_loop = builder_.create<mlir::scf::ForOp>(loc_, zero, row_upper, one);
    {
      mlir::OpBuilder::InsertionGuard row_guard(builder_);
      builder_.setInsertionPoint(row_loop.getBody()->getTerminator());
      mlir::Value row_iv = row_loop.getInductionVar();

      mlir::scf::ForOp col_loop = builder_.create<mlir::scf::ForOp>(loc_, zero, col_upper, one);
      mlir::OpBuilder::InsertionGuard col_guard(builder_);
      builder_.setInsertionPoint(col_loop.getBody()->getTerminator());
      mlir::Value col_iv = col_loop.getInductionVar();

      mlir::Value value = builder_.create<mlir::memref::LoadOp>(loc_, source,
                                                                llvm::SmallVector<mlir::Value, 2>{row_iv, col_iv});
      builder_.create<mlir::memref::StoreOp>(loc_, value, transposed,
                                             llvm::SmallVector<mlir::Value, 2>{col_iv, row_iv});
    }
    return transposed;
  }

  template <typename F>
  void EmitConditionalRegion(const PrimExpr& predicate, F&& body_builder) {
    if (tir::is_one(predicate)) {
      body_builder();
      return;
    }
    mlir::Value cond = LowerCondition(predicate);
    mlir::scf::IfOp if_op = builder_.create<mlir::scf::IfOp>(loc_, cond, false);
    mlir::OpBuilder::InsertionGuard guard(builder_);
    builder_.setInsertionPoint(if_op.thenYield());
    body_builder();
  }

  template <typename ThenBuilder, typename ElseBuilder>
  mlir::Value EmitConditionalValue(const PrimExpr& predicate, DataType result_dtype,
                                   ThenBuilder&& then_builder, ElseBuilder&& else_builder) {
    if (tir::is_one(predicate)) {
      return then_builder();
    }
    if (tir::is_zero(predicate)) {
      return else_builder();
    }

    mlir::Value result_slot = CreateStaticAlloca({1}, result_dtype);
    mlir::Value slot_index = ZeroIndex();
    llvm::SmallVector<mlir::Value, 1> slot_indices{slot_index};
    mlir::Value cond = LowerCondition(predicate);
    mlir::scf::IfOp if_op = builder_.create<mlir::scf::IfOp>(loc_, cond, true);

    {
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPoint(if_op.thenYield());
      builder_.create<mlir::memref::StoreOp>(loc_, then_builder(), result_slot, slot_indices);
    }
    {
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPoint(if_op.elseYield());
      builder_.create<mlir::memref::StoreOp>(loc_, else_builder(), result_slot, slot_indices);
    }
    return builder_.create<mlir::memref::LoadOp>(loc_, result_slot, slot_indices);
  }

  template <typename F>
  void EmitLoopNest(const Array<Range>& region, F&& body_builder) {
    llvm::SmallVector<mlir::Value, 4> coords;
    std::function<void(size_t)> emit = [&](size_t dim) {
      if (dim == region.size()) {
        body_builder(coords);
        return;
      }

      const Range& current = region[dim];
      mlir::Value lower = ZeroIndex();
      mlir::Value extent = AsIndex(VisitExpr(current->extent), current->extent.dtype());
      mlir::Value upper = extent;
      mlir::Value step = ConstantIntLike(1, builder_.getIndexType());
      mlir::scf::ForOp for_op = builder_.create<mlir::scf::ForOp>(loc_, lower, upper, step);

      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPoint(for_op.getBody()->getTerminator());
      coords.push_back(for_op.getInductionVar());
      emit(dim + 1);
      coords.pop_back();
    };
    emit(0);
  }

  mlir::Value OffsetIndex(mlir::Value coord, const PrimExpr& min) {
    if (tir::is_zero(min)) {
      return coord;
    }
    return builder_.create<mlir::arith::AddIOp>(loc_, coord, AsIndex(VisitExpr(min), min.dtype()));
  }

  mlir::Value CreateZeroValue(DataType dtype) {
    if (dtype.lanes() > 1) {
      mlir::Type type = LowerScalarType(dtype);
      mlir::Type element_type = LowerScalarType(dtype.element_of());
      llvm::SmallVector<mlir::Value, 8> lanes;
      lanes.reserve(dtype.lanes());
      if (IsFloatLikeType(dtype)) {
        for (int lane = 0; lane < dtype.lanes(); ++lane) {
          lanes.push_back(builder_.create<mlir::arith::ConstantOp>(
              loc_, builder_.getFloatAttr(mlir::cast<mlir::FloatType>(element_type), 0.0)));
        }
      } else {
        for (int lane = 0; lane < dtype.lanes(); ++lane) {
          lanes.push_back(ConstantIntLike(0, element_type));
        }
      }
      return builder_.create<mlir::vector::FromElementsOp>(loc_, type, lanes);
    }
    if (IsFloatLikeType(dtype)) {
      mlir::FloatType type = mlir::cast<mlir::FloatType>(LowerScalarType(dtype));
      return builder_.create<mlir::arith::ConstantOp>(loc_, builder_.getFloatAttr(type, 0.0));
    }
    return ConstantIntLike(0, LowerScalarType(dtype));
  }

  struct TileReduceAccess {
    tir::Buffer buffer;
    llvm::SmallVector<llvm::SmallVector<mlir::Value, 8>, 4> dim_lane_indices;
    llvm::SmallVector<size_t, 4> lane_counts;
  };

  ReductionLoopNestMatch::Kind TileReductionCombineKind(TileReductionKind kind) {
    switch (kind) {
      case TileReductionKind::kSum:
      case TileReductionKind::kAbsSum:
        return ReductionLoopNestMatch::Kind::kAdd;
      case TileReductionKind::kMax:
      case TileReductionKind::kAbsMax:
        return ReductionLoopNestMatch::Kind::kMax;
      case TileReductionKind::kMin:
        return ReductionLoopNestMatch::Kind::kMin;
      case TileReductionKind::kBitAnd:
        return ReductionLoopNestMatch::Kind::kBitAnd;
      case TileReductionKind::kBitOr:
        return ReductionLoopNestMatch::Kind::kBitOr;
      case TileReductionKind::kBitXor:
        return ReductionLoopNestMatch::Kind::kBitXor;
    }
    TVM_FFI_UNREACHABLE();
  }

  llvm::SmallVector<mlir::Value, 8> LowerTileReduceIndexLanes(const PrimExpr& index,
                                                              const char* context) {
    if (const auto* ramp = ResolveRampExpr(index)) {
      ICHECK_GT(ramp->dtype.lanes(), 0) << context << " expects non-empty Ramp lanes";
      return LowerRampLaneIndices(ramp);
    }
    ICHECK_EQ(index.dtype().lanes(), 1)
        << context << " only supports scalar indices or static Ramp slice indices";
    return llvm::SmallVector<mlir::Value, 8>{
        AsIndex(VisitExpr(index), index.dtype())};
  }

  TileReduceAccess DecodeTileReduceAccess(const tir::BufferLoadNode* load, const char* context) {
    ICHECK(load != nullptr) << context << " expects a BufferLoad slice argument";
    ICHECK(!load->predicate.defined() || tir::is_one(load->predicate.value()))
        << context << " does not support predicated BufferLoad slices";
    ValidateLowerableBufferLayout(load->buffer);
    ICHECK_EQ(load->buffer->dtype.lanes(), 1)
        << context << " currently supports only scalar element buffers";
    ICHECK_EQ(load->indices.size(), load->buffer->shape.size())
        << context << " slice rank must match buffer rank";

    TileReduceAccess access;
    access.buffer = load->buffer;
    access.dim_lane_indices.reserve(load->indices.size());
    access.lane_counts.reserve(load->indices.size());
    for (const PrimExpr& index : load->indices) {
      llvm::SmallVector<mlir::Value, 8> lanes = LowerTileReduceIndexLanes(index, context);
      ICHECK(!lanes.empty()) << context << " produced an empty index lane list";
      access.lane_counts.push_back(lanes.size());
      access.dim_lane_indices.push_back(std::move(lanes));
    }
    return access;
  }

  template <typename F>
  void EnumerateStaticTileCoordinates(llvm::ArrayRef<size_t> lane_counts, F&& body_builder) {
    llvm::SmallVector<size_t, 4> coords;
    coords.reserve(lane_counts.size());
    std::function<void(size_t)> emit = [&](size_t dim) {
      if (dim == lane_counts.size()) {
        body_builder(coords);
        return;
      }
      for (size_t lane = 0; lane < lane_counts[dim]; ++lane) {
        coords.push_back(lane);
        emit(dim + 1);
        coords.pop_back();
      }
    };
    emit(0);
  }

  llvm::SmallVector<mlir::Value, 4> BuildTileReduceInputIndices(
      const TileReduceAccess& input, size_t reduce_dim,
      llvm::ArrayRef<size_t> output_coords, size_t reduce_lane) {
    llvm::SmallVector<mlir::Value, 4> indices;
    indices.reserve(input.dim_lane_indices.size());
    size_t output_coord_dim = 0;
    for (size_t dim = 0; dim < input.dim_lane_indices.size(); ++dim) {
      size_t lane = reduce_lane;
      if (dim != reduce_dim) {
        ICHECK_LT(output_coord_dim, output_coords.size());
        lane = output_coords[output_coord_dim++];
      }
      ICHECK_LT(lane, input.dim_lane_indices[dim].size());
      indices.push_back(input.dim_lane_indices[dim][lane]);
    }
    ICHECK_EQ(output_coord_dim, output_coords.size());
    return indices;
  }

  llvm::SmallVector<mlir::Value, 4> BuildTileReduceOutputIndices(
      const TileReduceAccess& output, size_t input_rank, size_t reduce_dim,
      llvm::ArrayRef<size_t> output_coords, bool output_keeps_reduced_dim) {
    llvm::SmallVector<mlir::Value, 4> indices;
    indices.reserve(output.dim_lane_indices.size());
    if (output_keeps_reduced_dim) {
      ICHECK_EQ(output.dim_lane_indices.size(), input_rank);
      size_t output_coord_dim = 0;
      for (size_t dim = 0; dim < output.dim_lane_indices.size(); ++dim) {
        size_t lane = 0;
        if (dim != reduce_dim) {
          ICHECK_LT(output_coord_dim, output_coords.size());
          lane = output_coords[output_coord_dim++];
        }
        ICHECK_LT(lane, output.dim_lane_indices[dim].size());
        indices.push_back(output.dim_lane_indices[dim][lane]);
      }
      ICHECK_EQ(output_coord_dim, output_coords.size());
      return indices;
    }

    ICHECK_EQ(output.dim_lane_indices.size() + 1, input_rank);
    for (size_t dim = 0; dim < output.dim_lane_indices.size(); ++dim) {
      ICHECK_LT(dim, output_coords.size());
      size_t lane = output_coords[dim];
      ICHECK_LT(lane, output.dim_lane_indices[dim].size());
      indices.push_back(output.dim_lane_indices[dim][lane]);
    }
    return indices;
  }

  mlir::Value LoadScalarTileReduceValue(const tir::Buffer& buffer,
                                        llvm::ArrayRef<mlir::Value> indices,
                                        DataType result_dtype) {
    if (auto packed_view = ResolvePackedScalarViewBinding(buffer)) {
      return LowerPackedScalarViewLoad(buffer, packed_view.value(), indices, result_dtype);
    }
    mlir::Value memref = LookupBufferValue(buffer);
    mlir::Value value = builder_.create<mlir::memref::LoadOp>(loc_, memref, indices);
    return CastValue(value, buffer->dtype, result_dtype);
  }

  void StoreScalarTileReduceValue(const tir::Buffer& buffer,
                                  llvm::ArrayRef<mlir::Value> indices,
                                  mlir::Value value, DataType source_dtype) {
    if (auto packed_view = ResolvePackedScalarViewBinding(buffer)) {
      LowerPackedScalarViewStore(buffer, packed_view.value(), indices, value, source_dtype);
      return;
    }
    mlir::Value memref = LookupBufferValue(buffer);
    value = CastValue(value, source_dtype, buffer->dtype);
    builder_.create<mlir::memref::StoreOp>(loc_, value, memref, indices);
  }

  mlir::Value EmitAbsValue(mlir::Value value, DataType dtype) {
    if (dtype.is_uint() || dtype.is_bool()) {
      return value;
    }
    mlir::Value zero = CreateZeroValue(dtype);
    mlir::Value negative;
    mlir::Value cond;
    if (IsFloatLikeType(dtype)) {
      negative = builder_.create<mlir::arith::NegFOp>(loc_, value);
      cond = builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::OLT, value,
                                                  zero);
    } else {
      negative = builder_.create<mlir::arith::SubIOp>(loc_, zero, value);
      cond = builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::slt, value,
                                                  zero);
    }
    return builder_.create<mlir::arith::SelectOp>(loc_, cond, negative, value);
  }

  mlir::Value LoadTileReduceInputValue(const TileReduceAccess& input, size_t reduce_dim,
                                       llvm::ArrayRef<size_t> output_coords,
                                       size_t reduce_lane, TileReductionKind kind,
                                       DataType result_dtype) {
    llvm::SmallVector<mlir::Value, 4> input_indices =
        BuildTileReduceInputIndices(input, reduce_dim, output_coords, reduce_lane);
    mlir::Value value = LoadScalarTileReduceValue(input.buffer, input_indices, result_dtype);
    if (TileReductionNeedsAbs(kind)) {
      value = EmitAbsValue(value, result_dtype);
    }
    return value;
  }

  void ValidateTileReduceOutputShape(const TileReduceAccess& input,
                                     const TileReduceAccess& output, size_t reduce_dim,
                                     bool output_keeps_reduced_dim) {
    if (output_keeps_reduced_dim) {
      ICHECK_EQ(output.lane_counts.size(), input.lane_counts.size())
          << "tile reduction output rank with a kept reduced dimension must match input rank";
      for (size_t dim = 0; dim < output.lane_counts.size(); ++dim) {
        size_t expected_lanes = dim == reduce_dim ? 1 : input.lane_counts[dim];
        ICHECK_EQ(output.lane_counts[dim], expected_lanes)
            << "tile reduction output slice lanes do not match input non-reduction lanes";
      }
      return;
    }

    ICHECK_EQ(output.lane_counts.size() + 1, input.lane_counts.size())
        << "tile reduction output rank must either drop the reduced dimension or keep it as size 1";
    size_t output_dim = 0;
    for (size_t input_dim = 0; input_dim < input.lane_counts.size(); ++input_dim) {
      if (input_dim == reduce_dim) {
        continue;
      }
      ICHECK_LT(output_dim, output.lane_counts.size());
      ICHECK_EQ(output.lane_counts[output_dim], input.lane_counts[input_dim])
          << "tile reduction output slice lanes do not match input non-reduction lanes";
      ++output_dim;
    }
    ICHECK_EQ(output_dim, output.lane_counts.size());
  }

  void LowerTileReduce(const tir::CallNode* op) {
    ICHECK_EQ(op->args.size(), 5U)
        << "tl.tileop.reduce expects input, output, kind, dim, and clear arguments";
    const auto* input_load = op->args[0].as<tir::BufferLoadNode>();
    const auto* output_load = op->args[1].as<tir::BufferLoadNode>();
    const auto* kind_imm = op->args[2].as<tir::StringImmNode>();
    ICHECK(kind_imm != nullptr) << "tl.tileop.reduce kind must be a static string";
    llvm::StringRef kind_name(kind_imm->value.c_str());
    std::optional<TileReductionKind> maybe_kind = ParseTileReductionKind(kind_name);
    ICHECK(maybe_kind.has_value())
        << "Unsupported tile reduction kind in riscv lowering: " << kind_name.str()
        << ". Supported static fragment reductions are sum, abssum, max, absmax, min, "
           "bitand, bitor, and bitxor.";
    TileReductionKind kind = maybe_kind.value();

    TileReduceAccess input = DecodeTileReduceAccess(input_load, "tl.tileop.reduce input");
    TileReduceAccess output = DecodeTileReduceAccess(output_load, "tl.tileop.reduce output");
    ICHECK(!input.lane_counts.empty())
        << "tl.tileop.reduce expects an input rank of at least one";

    int64_t reduce_dim_i64 = GetStaticInt(op->args[3], "tl.tileop.reduce dim");
    if (reduce_dim_i64 < 0) {
      reduce_dim_i64 += static_cast<int64_t>(input.lane_counts.size());
    }
    ICHECK_GE(reduce_dim_i64, 0) << "tl.tileop.reduce dim is out of range";
    ICHECK_LT(static_cast<size_t>(reduce_dim_i64), input.lane_counts.size())
        << "tl.tileop.reduce dim is out of range";
    size_t reduce_dim = static_cast<size_t>(reduce_dim_i64);
    const size_t reduce_lanes = input.lane_counts[reduce_dim];
    ICHECK_GT(reduce_lanes, 0) << "tl.tileop.reduce expects a non-empty reduction slice";
    bool clear = GetStaticBool(op->args[4], "tl.tileop.reduce clear");

    bool output_keeps_reduced_dim = false;
    if (output.lane_counts.size() == input.lane_counts.size()) {
      output_keeps_reduced_dim = true;
    } else {
      ICHECK_EQ(output.lane_counts.size() + 1, input.lane_counts.size())
          << "tl.tileop.reduce output rank must drop the reduced dimension or keep it as size 1";
    }
    ValidateTileReduceOutputShape(input, output, reduce_dim, output_keeps_reduced_dim);

    llvm::SmallVector<size_t, 4> output_lane_counts;
    output_lane_counts.reserve(input.lane_counts.size() - 1);
    for (size_t dim = 0; dim < input.lane_counts.size(); ++dim) {
      if (dim != reduce_dim) {
        output_lane_counts.push_back(input.lane_counts[dim]);
      }
    }

    DataType result_dtype = output.buffer->dtype;
    if (TileReductionIsBitwise(kind)) {
      ICHECK(IsIntegerLikeType(input.buffer->dtype) && IsIntegerLikeType(result_dtype))
          << "Bitwise tile reductions require integer-like input and output dtypes";
    }

    const auto* reduce_ramp = ResolveRampExpr(input_load->indices[reduce_dim]);
    bool supports_fixed_vector_max =
        input.buffer->dtype == result_dtype &&
        (input.buffer->dtype == DataType::Float(32) ||
         input.buffer->dtype == DataType::Float(64));
    bool use_fixed_vector_max =
        kind == TileReductionKind::kMax && clear && input.dim_lane_indices.size() == 1 &&
        reduce_dim == 0 && reduce_lanes > 1 && reduce_ramp != nullptr &&
        tir::is_one(reduce_ramp->stride) && supports_fixed_vector_max &&
        output.lane_counts.size() == 1 &&
        output.lane_counts[0] == 1;
    if (use_fixed_vector_max) {
      mlir::Value input_vector = VisitExpr(tvm::ffi::GetRef<PrimExpr>(input_load));
      auto vector_type = mlir::dyn_cast<mlir::VectorType>(input_vector.getType());
      ICHECK(vector_type && vector_type.getRank() == 1 && !vector_type.isScalable())
          << "RISC-V fixed max reduction expected a fixed rank-1 vector load";
      ICHECK_EQ(static_cast<size_t>(vector_type.getNumElements()), reduce_lanes)
          << "RISC-V fixed max reduction lane count mismatch";
      mlir::Value reduced = builder_.create<mlir::vector::ReductionOp>(
          loc_, mlir::vector::CombiningKind::MAXIMUMF, input_vector);
      llvm::SmallVector<mlir::Value, 4> output_indices =
          BuildTileReduceOutputIndices(output, input.lane_counts.size(), reduce_dim, {},
                                       output_keeps_reduced_dim);
      StoreScalarTileReduceValue(output.buffer, output_indices, reduced, result_dtype);
      return;
    }

    ReductionLoopNestMatch::Kind combine_kind = TileReductionCombineKind(kind);
    EnumerateStaticTileCoordinates(output_lane_counts, [&](llvm::ArrayRef<size_t> output_coords) {
      llvm::SmallVector<mlir::Value, 4> output_indices =
          BuildTileReduceOutputIndices(output, input.lane_counts.size(), reduce_dim,
                                       output_coords, output_keeps_reduced_dim);
      mlir::Value acc;
      size_t start_lane = 0;
      if (clear && TileReductionCanUseZeroIdentity(kind)) {
        acc = CreateZeroValue(result_dtype);
      } else if (clear) {
        acc = LoadTileReduceInputValue(input, reduce_dim, output_coords, 0, kind, result_dtype);
        start_lane = 1;
      } else {
        acc = LoadScalarTileReduceValue(output.buffer, output_indices, result_dtype);
      }

      for (size_t reduce_lane = start_lane; reduce_lane < reduce_lanes; ++reduce_lane) {
        mlir::Value value = LoadTileReduceInputValue(input, reduce_dim, output_coords,
                                                     reduce_lane, kind, result_dtype);
        acc = EmitReductionCombine(combine_kind, result_dtype, builder_, loc_, acc, value);
      }
      StoreScalarTileReduceValue(output.buffer, output_indices, acc, result_dtype);
    });
  }

  void ValidateTileCumsumShape(const TileReduceAccess& input, const TileReduceAccess& output) {
    ICHECK_EQ(output.lane_counts.size(), input.lane_counts.size())
        << "tl.tileop.cumsum input and output ranks must match";
    for (size_t dim = 0; dim < input.lane_counts.size(); ++dim) {
      ICHECK_EQ(output.lane_counts[dim], input.lane_counts[dim])
          << "tl.tileop.cumsum input and output slice lanes must match";
    }
  }

  void LowerTileCumsum(const tir::CallNode* op) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.tileop.cumsum expects input, output, dim, and reverse arguments";
    if (InNonUnitLogicalThreadRegion()) {
      RejectUnsupportedTileScanIntrinsic(ffi::String("tl.tileop.cumsum"));
    }

    const auto* input_load = op->args[0].as<tir::BufferLoadNode>();
    const auto* output_load = op->args[1].as<tir::BufferLoadNode>();
    TileReduceAccess input = DecodeTileReduceAccess(input_load, "tl.tileop.cumsum input");
    TileReduceAccess output = DecodeTileReduceAccess(output_load, "tl.tileop.cumsum output");
    ICHECK(!input.lane_counts.empty())
        << "tl.tileop.cumsum expects an input rank of at least one";
    ValidateTileCumsumShape(input, output);

    int64_t scan_dim_i64 = GetStaticInt(op->args[2], "tl.tileop.cumsum dim");
    if (scan_dim_i64 < 0) {
      scan_dim_i64 += static_cast<int64_t>(input.lane_counts.size());
    }
    ICHECK_GE(scan_dim_i64, 0) << "tl.tileop.cumsum dim is out of range";
    ICHECK_LT(static_cast<size_t>(scan_dim_i64), input.lane_counts.size())
        << "tl.tileop.cumsum dim is out of range";
    size_t scan_dim = static_cast<size_t>(scan_dim_i64);
    const size_t scan_lanes = input.lane_counts[scan_dim];
    ICHECK_GT(scan_lanes, 0) << "tl.tileop.cumsum expects a non-empty scan slice";
    bool reverse = GetStaticBool(op->args[3], "tl.tileop.cumsum reverse");

    DataType result_dtype = output.buffer->dtype;
    ICHECK(IsFloatLikeType(input.buffer->dtype) || input.buffer->dtype.is_int() ||
           input.buffer->dtype.is_uint())
        << "tl.tileop.cumsum currently supports only int/uint/float input dtypes";
    ICHECK(IsFloatLikeType(result_dtype) || result_dtype.is_int() || result_dtype.is_uint())
        << "tl.tileop.cumsum currently supports only int/uint/float output dtypes";

    llvm::SmallVector<size_t, 4> outer_lane_counts;
    outer_lane_counts.reserve(input.lane_counts.size() - 1);
    for (size_t dim = 0; dim < input.lane_counts.size(); ++dim) {
      if (dim != scan_dim) {
        outer_lane_counts.push_back(input.lane_counts[dim]);
      }
    }

    EnumerateStaticTileCoordinates(outer_lane_counts, [&](llvm::ArrayRef<size_t> outer_coords) {
      mlir::Value acc;
      bool has_acc = false;
      for (size_t step = 0; step < scan_lanes; ++step) {
        size_t scan_lane = reverse ? (scan_lanes - 1 - step) : step;
        llvm::SmallVector<mlir::Value, 4> input_indices =
            BuildTileReduceInputIndices(input, scan_dim, outer_coords, scan_lane);
        llvm::SmallVector<mlir::Value, 4> output_indices =
            BuildTileReduceInputIndices(output, scan_dim, outer_coords, scan_lane);
        mlir::Value value = LoadScalarTileReduceValue(input.buffer, input_indices, result_dtype);
        if (!has_acc) {
          acc = value;
          has_acc = true;
        } else {
          acc = EmitReductionCombine(ReductionLoopNestMatch::Kind::kAdd, result_dtype, builder_,
                                     loc_, acc, value);
        }
        StoreScalarTileReduceValue(output.buffer, output_indices, acc, result_dtype);
      }
    });
  }

  void LowerFinalizeReducer(const tir::CallNode* op) {
    ICHECK_GE(op->args.size(), 1U) << "tl.tileop.finalize_reducer expects a reducer argument";
    if (InNonUnitLogicalThreadRegionThatUsesThreadVar()) {
      RejectUnsupportedTileReductionIntrinsic(ffi::String("tl.tileop.finalize_reducer"));
    }
    // The TileLang operator lowering emits a no-op when the reducer layout has
    // ReplicateExtent == 1. In the serialized RISC-V path, a finalize outside a
    // non-unit threadIdx launch, or inside a non-unit launch whose whole body is
    // thread-invariant, has no thread-partitioned participants to merge.
  }

  llvm::SmallVector<mlir::Value, 4> LowerRegionIndices(
      const tir::BufferRegion& region, llvm::ArrayRef<mlir::Value> coords) {
    ICHECK_EQ(region->region.size(), coords.size());
    llvm::SmallVector<mlir::Value, 4> indices;
    indices.reserve(coords.size());
    for (size_t i = 0; i < coords.size(); ++i) {
      indices.push_back(OffsetIndex(coords[i], region->region[i]->min));
    }
    return indices;
  }

  Array<Range> LogicalRegion(const Array<Range>& region) {
    Array<Range> logical_region;
    for (const Range& range : region) {
      if (!IsStaticOne(range->extent)) {
        logical_region.push_back(range);
      }
    }
    return logical_region;
  }

  Array<PrimExpr> LogicalRegionExtents(const tir::BufferRegion& region) {
    Array<PrimExpr> extents;
    for (const Range& range : region->region) {
      if (!IsStaticOne(range->extent)) {
        extents.push_back(range->extent);
      }
    }
    return extents;
  }

  Array<PrimExpr> RegionExtents(const Array<Range>& region) {
    Array<PrimExpr> extents;
    extents.reserve(region.size());
    for (const Range& range : region) {
      extents.push_back(range->extent);
    }
    return extents;
  }

  Array<PrimExpr> GemmRegionExtents(const tir::BufferRegion& region) {
    if (region->region.size() == 2) {
      return RegionExtents(region->region);
    }
    return LogicalRegionExtents(region);
  }

  mlir::Value CreateGemmSubview(const tir::BufferRegion& region) {
    if (region->region.size() == 2) {
      return CreateSubview(region);
    }
    return CreateLogicalSubview(region);
  }

  llvm::SmallVector<mlir::Value, 4> LowerLogicalRegionIndices(
      const tir::BufferRegion& region, llvm::ArrayRef<mlir::Value> coords) {
    llvm::SmallVector<mlir::Value, 4> indices;
    indices.reserve(region->region.size());
    size_t logical_dim = 0;
    for (const Range& range : region->region) {
      if (IsStaticOne(range->extent)) {
        indices.push_back(AsIndex(VisitExpr(range->min), range->min.dtype()));
        continue;
      }
      ICHECK_LT(logical_dim, coords.size());
      indices.push_back(OffsetIndex(coords[logical_dim], range->min));
      ++logical_dim;
    }
    ICHECK_EQ(logical_dim, coords.size());
    return indices;
  }

  bool RegionsHaveSameExtents(const tir::BufferRegion& lhs, const tir::BufferRegion& rhs) {
    if (lhs->region.size() != rhs->region.size()) {
      return false;
    }
    for (size_t i = 0; i < lhs->region.size(); ++i) {
      if (!analyzer_.CanProveEqual(lhs->region[i]->extent, rhs->region[i]->extent)) {
        return false;
      }
    }
    return true;
  }

  bool RegionSubviewFitsStaticShape(const tir::BufferRegion& region) {
    const tir::Buffer& buffer = region->buffer;
    if (region->region.size() > buffer->shape.size()) {
      return false;
    }
    for (size_t i = 0; i < region->region.size(); ++i) {
      std::optional<int64_t> shape_extent = GetOptionalStaticInt(buffer->shape[i]);
      std::optional<int64_t> region_min = GetOptionalStaticInt(region->region[i]->min);
      std::optional<int64_t> region_extent = GetOptionalStaticInt(region->region[i]->extent);
      if (!shape_extent.has_value() || !region_extent.has_value()) {
        continue;
      }
      if (region_extent.value() > shape_extent.value()) {
        return false;
      }
      if (region_min.has_value() &&
          region_min.value() + region_extent.value() > shape_extent.value()) {
        return false;
      }
    }
    return true;
  }

  bool IsStaticOneBufferDim(const tir::Buffer& buffer, size_t dim) {
    return dim < buffer->shape.size() && IsStaticOne(buffer->shape[dim]);
  }

  bool IsBroadcastCopySourceDim(const tir::BufferRegion& src_region, size_t dim) {
    return IsStaticOne(src_region->region[dim]->extent) ||
           IsStaticOneBufferDim(src_region->buffer, dim);
  }

  void FillBufferRegion(const tir::BufferRegion& dst_region, mlir::Value fill_value,
                        DataType source_dtype) {
    const tir::Buffer& dst_buffer = dst_region->buffer;
    mlir::Value dst_memref = LookupBufferValue(dst_buffer);

    EmitLoopNest(dst_region->region, [&](llvm::ArrayRef<mlir::Value> coords) {
      llvm::SmallVector<mlir::Value, 4> dst_indices = LowerRegionIndices(dst_region, coords);
      mlir::Value value = CastValue(fill_value, source_dtype, dst_buffer->dtype);
      builder_.create<mlir::memref::StoreOp>(loc_, value, dst_memref, dst_indices);
    });
  }

  void SerializedAtomicAddStore(const tir::Buffer& dst_buffer, mlir::Value dst_memref,
                                llvm::ArrayRef<mlir::Value> dst_indices, mlir::Value operand,
                                DataType source_dtype) {
    ICHECK(IsFloatLikeType(dst_buffer->dtype) || dst_buffer->dtype.is_int() ||
           dst_buffer->dtype.is_uint())
        << "tl.tileop.atomicadd currently only supports int/uint/float destination buffers in "
           "riscv lowering";
    operand = CastValue(operand, source_dtype, dst_buffer->dtype);
    mlir::Value old_value = builder_.create<mlir::memref::LoadOp>(loc_, dst_memref, dst_indices);
    mlir::Value updated;
    if (IsFloatLikeType(dst_buffer->dtype)) {
      updated = builder_.create<mlir::arith::AddFOp>(loc_, old_value, operand);
    } else {
      updated = builder_.create<mlir::arith::AddIOp>(loc_, old_value, operand);
    }
    builder_.create<mlir::memref::StoreOp>(loc_, updated, dst_memref, dst_indices);
  }

  void LowerTileAtomicAdd(const tir::CallNode* op) {
    ICHECK(op != nullptr && op->args.size() == 2U)
        << "tl.tileop.atomicadd expects source and destination regions";

    tir::BufferRegion dst_region = tl::NormalizeToBufferRegion(op->args[1]);
    const tir::Buffer& dst_buffer = dst_region->buffer;
    mlir::Value dst_memref = LookupBufferValue(dst_buffer);

    if (tl::IsBufferLikeExpr(op->args[0])) {
      tir::BufferRegion src_region = tl::NormalizeToBufferRegion(op->args[0]);
      const tir::Buffer& src_buffer = src_region->buffer;
      mlir::Value src_memref = LookupBufferValue(src_buffer);

      if (src_region->region.size() == dst_region->region.size()) {
        EmitLoopNest(dst_region->region, [&](llvm::ArrayRef<mlir::Value> coords) {
          llvm::SmallVector<mlir::Value, 4> src_indices;
          src_indices.reserve(coords.size());
          for (size_t i = 0; i < coords.size(); ++i) {
            if (IsBroadcastCopySourceDim(src_region, i)) {
              src_indices.push_back(
                  AsIndex(VisitExpr(src_region->region[i]->min), src_region->region[i]->min.dtype()));
              continue;
            }
            ICHECK(analyzer_.CanProveEqual(src_region->region[i]->extent, dst_region->region[i]->extent))
                << "tl.tileop.atomicadd currently requires matching extents, except for static-1 "
                   "broadcast on the source side";
            src_indices.push_back(OffsetIndex(coords[i], src_region->region[i]->min));
          }

          llvm::SmallVector<mlir::Value, 4> dst_indices = LowerRegionIndices(dst_region, coords);
          mlir::Value operand = builder_.create<mlir::memref::LoadOp>(loc_, src_memref, src_indices);
          SerializedAtomicAddStore(dst_buffer, dst_memref, dst_indices, operand, src_buffer->dtype);
        });
        return;
      }

      Array<PrimExpr> src_logical_extents = LogicalRegionExtents(src_region);
      Array<PrimExpr> dst_logical_extents = LogicalRegionExtents(dst_region);
      ICHECK_EQ(src_logical_extents.size(), dst_logical_extents.size())
          << "rank-changing tl.tileop.atomicadd only supports dropping static-1 dimensions in "
             "riscv; logical ranks must match after dropping static-1 dims";
      for (size_t i = 0; i < src_logical_extents.size(); ++i) {
        ICHECK(analyzer_.CanProveEqual(src_logical_extents[i], dst_logical_extents[i]))
            << "rank-reduced tl.tileop.atomicadd extents must match after dropping static-1 "
               "dimensions in riscv";
      }

      EmitLoopNest(LogicalRegion(dst_region->region), [&](llvm::ArrayRef<mlir::Value> coords) {
        llvm::SmallVector<mlir::Value, 4> src_indices = LowerLogicalRegionIndices(src_region, coords);
        llvm::SmallVector<mlir::Value, 4> dst_indices = LowerLogicalRegionIndices(dst_region, coords);
        mlir::Value operand = builder_.create<mlir::memref::LoadOp>(loc_, src_memref, src_indices);
        SerializedAtomicAddStore(dst_buffer, dst_memref, dst_indices, operand, src_buffer->dtype);
      });
      return;
    }

    mlir::Value operand = VisitExpr(op->args[0]);
    EmitLoopNest(LogicalRegion(dst_region->region), [&](llvm::ArrayRef<mlir::Value> coords) {
      llvm::SmallVector<mlir::Value, 4> dst_indices = LowerLogicalRegionIndices(dst_region, coords);
      SerializedAtomicAddStore(dst_buffer, dst_memref, dst_indices, operand, op->args[0].dtype());
    });
  }

  void TrackSerializedWarpReplayTileFill(const tir::BufferRegion& dst_region,
                                         const PrimExpr& fill_expr) {
    const tir::Buffer& buffer = dst_region->buffer;
    if (!CanTrackSerializedWarpReplayBufferElements(buffer)) {
      return;
    }
    PrimExpr resolved_fill_expr = ResolveSerializedWarpReplayExpr(fill_expr);
    llvm::SmallVector<int64_t, 4> mins;
    llvm::SmallVector<int64_t, 4> extents;
    mins.reserve(dst_region->region.size());
    extents.reserve(dst_region->region.size());
    for (const Range& range : dst_region->region) {
      std::optional<int64_t> static_min = GetOptionalStaticInt(ResolveBoundPrimExpr(range->min));
      std::optional<int64_t> static_extent =
          GetOptionalStaticInt(ResolveBoundPrimExpr(range->extent));
      if (!static_min.has_value() || !static_extent.has_value() || static_extent.value() < 0) {
        ClearSerializedWarpReplayTrackedBufferElementExprs(buffer);
        if (IsSerializedWarpReplayTrackedScalarBuffer(buffer)) {
          ClearSerializedWarpReplayTrackedBufferExpr(buffer);
        }
        return;
      }
      mins.push_back(static_min.value());
      extents.push_back(static_extent.value());
    }
    llvm::SmallVector<int64_t, 4> coords(mins.size(), 0);
    auto track_region = [&](auto&& self, size_t dim) -> void {
      if (dim >= mins.size()) {
        Array<PrimExpr> indices;
        for (size_t i = 0; i < coords.size(); ++i) {
          indices.push_back(tir::make_const(buffer->shape[i].dtype(), mins[i] + coords[i]));
        }
        std::optional<int64_t> linear_index =
            GetSerializedWarpReplayStaticBufferLinearIndex(buffer, indices);
        if (!linear_index.has_value()) {
          ClearSerializedWarpReplayTrackedBufferElementExprs(buffer);
          if (IsSerializedWarpReplayTrackedScalarBuffer(buffer)) {
            ClearSerializedWarpReplayTrackedBufferExpr(buffer);
          }
          return;
        }
        if (IsSerializedWarpReplayTrackedScalarBuffer(buffer) && linear_index.value() == 0) {
          SetSerializedWarpReplayTrackedBufferExpr(buffer, resolved_fill_expr);
        } else {
          SetSerializedWarpReplayTrackedBufferElementExpr(buffer, linear_index.value(),
                                                          resolved_fill_expr);
        }
        return;
      }
      for (int64_t value = 0; value < extents[dim]; ++value) {
        coords[dim] = value;
        self(self, dim + 1);
      }
    };
    track_region(track_region, 0);
  }

  void LowerTileFill(const tir::CallNode* op) {
    tir::BufferRegion dst_region = tl::NormalizeToBufferRegion(op->args[0]);
    FillBufferRegion(dst_region, VisitExpr(op->args[1]), op->args[1].dtype());
  }

  void LowerBufferRegionCopy(const tir::BufferRegion& src_region, const tir::BufferRegion& dst_region) {
    const tir::Buffer& src_buffer = src_region->buffer;
    const tir::Buffer& dst_buffer = dst_region->buffer;
    mlir::Value src_memref = LookupBufferValue(src_buffer);
    mlir::Value dst_memref = LookupBufferValue(dst_buffer);

    if (src_region->region.size() == dst_region->region.size()) {
      if (src_buffer->dtype == dst_buffer->dtype && RegionsHaveSameExtents(src_region, dst_region) &&
          RegionSubviewFitsStaticShape(src_region) && RegionSubviewFitsStaticShape(dst_region)) {
        builder_.create<mlir::memref::CopyOp>(loc_, CreateSubview(src_region), CreateSubview(dst_region));
        return;
      }

      EmitLoopNest(dst_region->region, [&](llvm::ArrayRef<mlir::Value> coords) {
        llvm::SmallVector<mlir::Value, 4> src_indices;
        src_indices.reserve(coords.size());
        for (size_t i = 0; i < coords.size(); ++i) {
          if (IsBroadcastCopySourceDim(src_region, i)) {
            src_indices.push_back(AsIndex(VisitExpr(src_region->region[i]->min),
                                          src_region->region[i]->min.dtype()));
            continue;
          }
          ICHECK(analyzer_.CanProveEqual(src_region->region[i]->extent, dst_region->region[i]->extent))
              << "tl.copy currently requires matching extents, except for static-1 broadcast";
          src_indices.push_back(OffsetIndex(coords[i], src_region->region[i]->min));
        }

        llvm::SmallVector<mlir::Value, 4> dst_indices = LowerRegionIndices(dst_region, coords);
        mlir::Value value = builder_.create<mlir::memref::LoadOp>(loc_, src_memref, src_indices);
        value = CastValue(value, src_buffer->dtype, dst_buffer->dtype);
        builder_.create<mlir::memref::StoreOp>(loc_, value, dst_memref, dst_indices);
      });
      return;
    }

    Array<PrimExpr> src_logical_extents = LogicalRegionExtents(src_region);
    Array<PrimExpr> dst_logical_extents = LogicalRegionExtents(dst_region);
    ICHECK_EQ(src_logical_extents.size(), dst_logical_extents.size())
        << "rank-changing tl.copy only supports dropping static-1 dimensions in "
           "riscv; logical ranks must match after dropping static-1 dims";
    for (size_t i = 0; i < src_logical_extents.size(); ++i) {
      ICHECK(analyzer_.CanProveEqual(src_logical_extents[i], dst_logical_extents[i]))
          << "rank-reduced tl.copy extents must match after dropping static-1 dimensions "
             "in riscv";
    }

    if (src_buffer->dtype == dst_buffer->dtype) {
      builder_.create<mlir::memref::CopyOp>(loc_, CreateLogicalSubview(src_region),
                                            CreateLogicalSubview(dst_region));
      return;
    }

    EmitLoopNest(LogicalRegion(dst_region->region), [&](llvm::ArrayRef<mlir::Value> coords) {
      llvm::SmallVector<mlir::Value, 4> src_indices = LowerLogicalRegionIndices(src_region, coords);
      llvm::SmallVector<mlir::Value, 4> dst_indices = LowerLogicalRegionIndices(dst_region, coords);
      mlir::Value value = builder_.create<mlir::memref::LoadOp>(loc_, src_memref, src_indices);
      value = CastValue(value, src_buffer->dtype, dst_buffer->dtype);
      builder_.create<mlir::memref::StoreOp>(loc_, value, dst_memref, dst_indices);
    });
  }

  void LowerTileCopy(const tir::CallNode* op) {
    tir::BufferRegion src_region = tl::NormalizeToBufferRegion(op->args[0]);
    tir::BufferRegion dst_region = tl::NormalizeToBufferRegion(op->args[1]);
    LowerBufferRegionCopy(src_region, dst_region);
  }

  void LowerTileAsyncCopy(const tir::CallNode* op) {
    // The serialized RISC-V backend has no async pipeline runtime. Model
    // explicit async_copy conservatively as an ordinary synchronous copy and
    // lower matching PTX wait/commit markers as no-ops.
    LowerTileCopy(op);
  }

  const tir::CallNode* ResolveBoundCallExpr(const PrimExpr& expr) const {
    if (const auto* call = expr.as<tir::CallNode>()) {
      return call;
    }
    const auto* var = expr.as<tir::VarNode>();
    if (var == nullptr) {
      return nullptr;
    }
    auto it = bound_prim_exprs_.find(var);
    if (it == bound_prim_exprs_.end()) {
      return nullptr;
    }
    return it->second.as<tir::CallNode>();
  }

  std::optional<tir::Buffer> ResolveBufferOwnerFromExpr(const PrimExpr& expr) const {
    if (const auto* load = expr.as<tir::BufferLoadNode>()) {
      return load->buffer;
    }
    const auto* var = expr.as<tir::VarNode>();
    if (var == nullptr) {
      return std::nullopt;
    }
    auto it = buffer_owner_.find(var);
    if (it == buffer_owner_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  std::optional<tir::Buffer> ResolveBufferOwnerFromAccessPtrExpr(const PrimExpr& expr) const {
    const auto* access = expr.as<tir::CallNode>();
    if (access == nullptr || !access->op.same_as(tir::builtin::tvm_access_ptr()) ||
        access->args.size() < 2U) {
      return std::nullopt;
    }
    const auto* data_var = access->args[1].as<tir::VarNode>();
    if (data_var == nullptr) {
      return std::nullopt;
    }
    auto it = buffer_owner_.find(data_var);
    if (it == buffer_owner_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  tir::BufferRegion MakeFullBufferRegion(const tir::Buffer& buffer) const {
    Array<Range> ranges;
    for (const PrimExpr& dim : buffer->shape) {
      ranges.push_back(Range::FromMinExtent(0, dim));
    }
    return tir::BufferRegion(buffer, ranges);
  }

  mlir::Value CreateSerializedOpaqueHandlePlaceholder() {
    return ConstantIntLike(0, builder_.getI64Type());
  }

  bool LowerSerializedFp8TmaLoadCallExtern(const tir::CallNode* op) {
    ICHECK(op != nullptr && op->op.same_as(tir::builtin::call_extern()));
    ICHECK_EQ(GetCallExternName(op), "tl::fp8_tma_load_4d_ptx");
    ICHECK_EQ(op->args.size(), 8U)
        << "tl::fp8_tma_load_4d_ptx call_extern expects <name, desc, barrier, dst, dim, head, seq, batch>";

    const tir::CallNode* descriptor_call = ResolveBoundCallExpr(op->args[1]);
    ICHECK(descriptor_call != nullptr && descriptor_call->op.same_as(tvm::tl::create_tma_descriptor()))
        << "serialized tl::fp8_tma_load_4d_ptx expects its descriptor to come from tl.create_tma_descriptor";
    ICHECK_GE(descriptor_call->args.size(), 4U)
        << "tl.create_tma_descriptor expects at least <dtype, rank, global_addr, ...>";

    std::optional<tir::Buffer> src_buffer = ResolveBufferOwnerFromExpr(descriptor_call->args[2]);
    ICHECK(src_buffer.has_value())
        << "serialized tl::fp8_tma_load_4d_ptx could not resolve the source buffer backing its descriptor";

    const auto* dst_access = op->args[3].as<tir::CallNode>();
    ICHECK(dst_access != nullptr && dst_access->op.same_as(tir::builtin::tvm_access_ptr()))
        << "serialized tl::fp8_tma_load_4d_ptx expects a tvm_access_ptr destination";
    const auto* dst_data_var = dst_access->args[1].as<tir::VarNode>();
    ICHECK(dst_data_var != nullptr)
        << "serialized tl::fp8_tma_load_4d_ptx expects tvm_access_ptr data to be a Var";
    auto dst_owner_it = buffer_owner_.find(dst_data_var);
    ICHECK(dst_owner_it != buffer_owner_.end())
        << "serialized tl::fp8_tma_load_4d_ptx could not resolve the destination buffer owner";
    const tir::Buffer& dst_buffer = dst_owner_it->second;

    ICHECK_EQ(src_buffer.value()->shape.size(), 4U)
        << "serialized tl::fp8_tma_load_4d_ptx currently expects a rank-4 source tensor";
    ICHECK_EQ(dst_buffer->shape.size(), 2U)
        << "serialized tl::fp8_tma_load_4d_ptx currently expects a rank-2 destination tile";

    Array<Range> src_ranges = {
        Range::FromMinExtent(op->args[7], 1),
        Range::FromMinExtent(op->args[6], dst_buffer->shape[0]),
        Range::FromMinExtent(op->args[5], 1),
        Range::FromMinExtent(op->args[4], dst_buffer->shape[1]),
    };
    Array<Range> dst_ranges = {
        Range::FromMinExtent(0, dst_buffer->shape[0]),
        Range::FromMinExtent(0, dst_buffer->shape[1]),
    };
    LowerBufferRegionCopy(tir::BufferRegion(src_buffer.value(), src_ranges),
                          tir::BufferRegion(dst_buffer, dst_ranges));
    return true;
  }

  bool LowerSerializedFp8ZeroRawAccCallExtern(const tir::CallNode* op) {
    ICHECK(op != nullptr && op->op.same_as(tir::builtin::call_extern()));
    ICHECK_EQ(GetCallExternName(op), "tl::fp8_zero_raw_acc_64");
    ICHECK_EQ(op->args.size(), 2U) << "tl::fp8_zero_raw_acc_64 expects <name, acc_buffer>";
    std::optional<tir::Buffer> buffer = ResolveBufferOwnerFromExpr(op->args[1]);
    ICHECK(buffer.has_value())
        << "serialized tl::fp8_zero_raw_acc_64 could not resolve its accumulator buffer";
    FillBufferRegion(MakeFullBufferRegion(buffer.value()), CreateZeroValue(buffer.value()->dtype),
                     buffer.value()->dtype);
    return true;
  }

  bool LowerSerializedFp8RawAccStoreCallExtern(const tir::CallNode* op) {
    ICHECK(op != nullptr && op->op.same_as(tir::builtin::call_extern()));
    ICHECK_EQ(GetCallExternName(op), "tl::fp8_fa3_raw_acc_store_smem_cute_64x128");
    ICHECK_EQ(op->args.size(), 5U)
        << "tl::fp8_fa3_raw_acc_store_smem_cute_64x128 expects <name, acc, lse, warp_count, dst>";
    std::optional<tir::Buffer> src_buffer = ResolveBufferOwnerFromExpr(op->args[1]);
    std::optional<tir::Buffer> dst_buffer = ResolveBufferOwnerFromAccessPtrExpr(op->args[4]);
    ICHECK(src_buffer.has_value() && dst_buffer.has_value())
        << "serialized tl::fp8_fa3_raw_acc_store_smem_cute_64x128 could not resolve src/dst buffers";
    LowerBufferRegionCopy(MakeFullBufferRegion(src_buffer.value()), MakeFullBufferRegion(dst_buffer.value()));
    return true;
  }

  bool LowerSerializedFp8OutputStoreCallExtern(const tir::CallNode* op) {
    ICHECK(op != nullptr && op->op.same_as(tir::builtin::call_extern()));
    ICHECK_EQ(GetCallExternName(op), "tl::fp8_fa3_o_smem_store_global_cute_64x128");
    ICHECK_EQ(op->args.size(), 4U)
        << "tl::fp8_fa3_o_smem_store_global_cute_64x128 expects <name, src, dst, stride>";
    std::optional<tir::Buffer> src_buffer = ResolveBufferOwnerFromAccessPtrExpr(op->args[1]);
    ICHECK(src_buffer.has_value())
        << "serialized tl::fp8_fa3_o_smem_store_global_cute_64x128 could not resolve its source buffer";
    std::optional<AddressOfBufferLoadAccess> dst_access = MatchAddressOfBufferLoad(op->args[2]);
    ICHECK(dst_access.has_value())
        << "serialized tl::fp8_fa3_o_smem_store_global_cute_64x128 expects an address_of(buffer_load) destination";
    Array<Range> dst_ranges = {
        Range::FromMinExtent(dst_access->indices[0], 1),
        Range::FromMinExtent(dst_access->indices[1], src_buffer.value()->shape[0]),
        Range::FromMinExtent(dst_access->indices[2], 1),
        Range::FromMinExtent(dst_access->indices[3], src_buffer.value()->shape[1]),
    };
    LowerBufferRegionCopy(MakeFullBufferRegion(src_buffer.value()),
                          tir::BufferRegion(dst_access->buffer, dst_ranges));
    return true;
  }

  void LowerTileGemmPy(const tir::CallNode* op) {
    tir::BufferRegion a_region = tl::NormalizeToBufferRegion(op->args[0]);
    tir::BufferRegion b_region = tl::NormalizeToBufferRegion(op->args[1]);
    tir::BufferRegion c_region = tl::NormalizeToBufferRegion(op->args[2]);

    bool transpose_a = GetStaticBool(op->args[3], "tl.gemm transpose_a");
    bool transpose_b = GetStaticBool(op->args[4], "tl.gemm transpose_b");
    PrimExpr m = op->args[5];
    PrimExpr n = op->args[6];
    PrimExpr k = op->args[7];
    std::optional<bool> clear_accum = GetOptionalStaticBool(op->args[9]);

    Array<PrimExpr> a_extents = GemmRegionExtents(a_region);
    Array<PrimExpr> b_extents = GemmRegionExtents(b_region);
    Array<PrimExpr> c_extents = GemmRegionExtents(c_region);

    ICHECK_EQ(a_extents.size(), 2) << "Only logical 2D tl.gemm A operands are supported";
    ICHECK_EQ(b_extents.size(), 2) << "Only logical 2D tl.gemm B operands are supported";
    ICHECK_EQ(c_extents.size(), 2) << "Only logical 2D tl.gemm C operands are supported";

    if (transpose_a) {
      ICHECK(analyzer_.CanProveEqual(a_extents[0], k))
          << "tl.gemm A K extent must match K";
      ICHECK(analyzer_.CanProveEqual(a_extents[1], m))
          << "tl.gemm A M extent must match M";
    } else {
      ICHECK(analyzer_.CanProveEqual(a_extents[0], m))
          << "tl.gemm A M extent must match M";
      ICHECK(analyzer_.CanProveEqual(a_extents[1], k))
          << "tl.gemm A K extent must match K";
    }
    if (transpose_b) {
      ICHECK(analyzer_.CanProveEqual(b_extents[0], n))
          << "tl.gemm B N extent must match N";
      ICHECK(analyzer_.CanProveEqual(b_extents[1], k))
          << "tl.gemm B K extent must match K";
    } else {
      ICHECK(analyzer_.CanProveEqual(b_extents[0], k))
          << "tl.gemm B K extent must match K";
      ICHECK(analyzer_.CanProveEqual(b_extents[1], n))
          << "tl.gemm B N extent must match N";
    }
    ICHECK(analyzer_.CanProveEqual(c_extents[0], m))
        << "tl.gemm C M extent must match M";
    ICHECK(analyzer_.CanProveEqual(c_extents[1], n))
        << "tl.gemm C N extent must match N";

    if (clear_accum.value_or(false)) {
      FillBufferRegion(c_region, CreateZeroValue(c_region->buffer->dtype), c_region->buffer->dtype);
    } else if (!clear_accum.has_value()) {
      EmitConditionalRegion(op->args[9], [&]() {
        FillBufferRegion(c_region, CreateZeroValue(c_region->buffer->dtype),
                         c_region->buffer->dtype);
      });
    }

    mlir::Value a_view = CreateGemmSubview(a_region);
    mlir::Value b_view = CreateGemmSubview(b_region);
    mlir::Value c_view = CreateGemmSubview(c_region);
    if (transpose_a && transpose_b) {
      mlir::Value a_materialized = MaterializeTranspose2D(a_view, k, m, a_region->buffer->dtype);
      builder_.create<mlir::linalg::MatmulTransposeBOp>(loc_,
                                                        mlir::ValueRange{a_materialized, b_view},
                                                        mlir::ValueRange{c_view});
      return;
    }
    if (transpose_a) {
      builder_.create<mlir::linalg::MatmulTransposeAOp>(loc_, mlir::ValueRange{a_view, b_view},
                                                        mlir::ValueRange{c_view});
      return;
    }
    if (transpose_b) {
      builder_.create<mlir::linalg::MatmulTransposeBOp>(loc_, mlir::ValueRange{a_view, b_view},
                                                        mlir::ValueRange{c_view});
      return;
    }
    builder_.create<mlir::linalg::MatmulOp>(loc_, mlir::ValueRange{a_view, b_view},
                                            mlir::ValueRange{c_view});
  }

  mlir::Value LowerScalarAtomicRMW(const tir::CallNode* op) {
    std::optional<ScalarAtomicRMWKind> maybe_kind = GetScalarAtomicRMWKind(op);
    ICHECK(maybe_kind.has_value()) << "LowerScalarAtomicRMW expects scalar atomic RMW";
    ScalarAtomicRMWKind kind = maybe_kind.value();
    const char* context = ScalarAtomicRMWContext(kind);
    ICHECK(op->args.size() == 2 || op->args.size() == 3)
        << context << " expects destination, value, and optional memory order";
    AccessPtrBinding dst = DecodeElementAccessPtr(op->args[0], context);
    ICHECK_EQ(dst.buffer->dtype.lanes(), 1)
        << context << " currently only supports scalar element buffers in riscv";
    ICHECK(IsFloatLikeType(dst.buffer->dtype) || dst.buffer->dtype.is_int() ||
           dst.buffer->dtype.is_uint())
        << context << " currently only supports int/uint/float buffers in riscv";

    mlir::Value old_value = builder_.create<mlir::memref::LoadOp>(loc_, dst.memref, dst.indices);
    mlir::Value operand = VisitExpr(op->args[1]);
    operand = CastValue(operand, op->args[1].dtype(), dst.buffer->dtype);
    mlir::Value updated;
    if (kind == ScalarAtomicRMWKind::kAdd) {
      if (IsFloatLikeType(dst.buffer->dtype)) {
        updated = builder_.create<mlir::arith::AddFOp>(loc_, old_value, operand);
      } else {
        updated = builder_.create<mlir::arith::AddIOp>(loc_, old_value, operand);
      }
    } else if (IsFloatLikeType(dst.buffer->dtype)) {
      mlir::arith::CmpFPredicate predicate = kind == ScalarAtomicRMWKind::kMax
                                                 ? mlir::arith::CmpFPredicate::OGT
                                                 : mlir::arith::CmpFPredicate::OLT;
      mlir::Value cond = builder_.create<mlir::arith::CmpFOp>(loc_, predicate, operand, old_value);
      updated = builder_.create<mlir::arith::SelectOp>(loc_, cond, operand, old_value);
    } else {
      const bool is_unsigned = dst.buffer->dtype.is_uint() || dst.buffer->dtype.is_bool();
      mlir::arith::CmpIPredicate predicate;
      if (kind == ScalarAtomicRMWKind::kMax) {
        predicate = is_unsigned ? mlir::arith::CmpIPredicate::ugt
                                : mlir::arith::CmpIPredicate::sgt;
      } else {
        predicate = is_unsigned ? mlir::arith::CmpIPredicate::ult
                                : mlir::arith::CmpIPredicate::slt;
      }
      mlir::Value cond = builder_.create<mlir::arith::CmpIOp>(loc_, predicate, operand, old_value);
      updated = builder_.create<mlir::arith::SelectOp>(loc_, cond, operand, old_value);
    }
    builder_.create<mlir::memref::StoreOp>(loc_, updated, dst.memref, dst.indices);
    if (ScalarAtomicRMWReturnsValue(op)) {
      return CastValue(old_value, dst.buffer->dtype, op->dtype);
    }
    return old_value;
  }

  mlir::Value LowerScalarAtomicLoad(const tir::CallNode* op) {
    ICHECK(IsScalarAtomicLoadCall(op)) << "LowerScalarAtomicLoad expects atomic load";
    ICHECK(op->args.size() == 1 || op->args.size() == 2)
        << "atomic_load expects source and optional memory order";
    AccessPtrBinding src = DecodeElementAccessPtr(op->args[0], "atomic_load");
    ICHECK_EQ(src.buffer->dtype.lanes(), 1)
        << "atomic_load currently only supports scalar element buffers in riscv";
    ICHECK(IsFloatLikeType(src.buffer->dtype) || src.buffer->dtype.is_int() ||
           src.buffer->dtype.is_uint())
        << "atomic_load currently only supports int/uint/float buffers in riscv";

    mlir::Value loaded = builder_.create<mlir::memref::LoadOp>(loc_, src.memref, src.indices);
    return CastValue(loaded, src.buffer->dtype, op->dtype);
  }

  mlir::Value LowerAtomicAddOffsetCallExtern(const tir::CallNode* op) {
    ICHECK(IsAtomicAddOffsetCallExtern(op))
        << "LowerAtomicAddOffsetCallExtern expects tl_atomic_add_offset";
    ICHECK_EQ(op->args.size(), 4U)
        << "tl_atomic_add_offset expects <name, base_ptr, offset, value>";
    std::optional<AddressOfBufferLoadAccess> base_ptr = MatchAddressOfBufferLoad(op->args[1]);
    ICHECK(base_ptr.has_value())
        << "tl_atomic_add_offset expects address_of(BufferLoad(...)) as its base pointer";

    const tir::Buffer& buffer = base_ptr->buffer;
    ValidateContiguousBuffer(buffer);
    ICHECK(!buffer->shape.empty())
        << "tl_atomic_add_offset expects a non-scalar compact buffer destination";
    ICHECK(buffer->dtype.is_int() || buffer->dtype.is_uint())
        << "tl_atomic_add_offset currently only supports integer buffers in riscv lowering";

    mlir::Value memref = LookupBufferValue(buffer);
    llvm::SmallVector<mlir::Value, 4> base_indices = LowerScalarIndices(base_ptr->indices);
    mlir::Value base_linear_index = LinearizeCompactBufferIndices(buffer, base_indices);
    mlir::Value linear_offset = AsIndex(VisitExpr(op->args[2]), op->args[2].dtype());
    mlir::Value logical_index =
        builder_.create<mlir::arith::AddIOp>(loc_, base_linear_index, linear_offset);
    llvm::SmallVector<mlir::Value, 4> slot_indices =
        RowMajorLinearOffsetToIndices(buffer, logical_index);

    mlir::Value old_value = builder_.create<mlir::memref::LoadOp>(loc_, memref, slot_indices);
    mlir::Value delta = CastValue(VisitExpr(op->args[3]), op->args[3].dtype(), buffer->dtype);
    mlir::Value updated = builder_.create<mlir::arith::AddIOp>(loc_, old_value, delta);
    builder_.create<mlir::memref::StoreOp>(loc_, updated, memref, slot_indices);
    return CastValue(old_value, buffer->dtype, op->dtype);
  }

  void LowerRngInit(const tir::CallNode* op) {
    ICHECK(op->op.same_as(tvm::tl::rng_init())) << "LowerRngInit expects tl.rng_init";
    ICHECK_EQ(op->args.size(), 4U) << "tl.rng_init expects <seed, seq, offset, generator>";
    const auto* generator = op->args[3].as<tir::StringImmNode>();
    ICHECK(generator != nullptr) << "tl.rng_init expects generator as a string literal";
    llvm::StringRef generator_name(generator->value.c_str());
    ICHECK(generator_name == "curandStatePhilox4_32_10_t" ||
           generator_name == "curandStateMRG32k3a_t" ||
           generator_name == "curandStateXORWOW_t")
        << "Unsupported tl.rng_init generator in riscv lowering: " << generator->value;

    if (!active_rng_state_.has_value()) {
      active_rng_state_ = CreateStaticAlloca({1}, DataType::UInt(64));
    }

    mlir::Type i64_type = builder_.getI64Type();
    mlir::Value seed = CastValue(VisitExpr(op->args[0]), op->args[0].dtype(), DataType::UInt(64));
    mlir::Value seq = CastValue(VisitExpr(op->args[1]), op->args[1].dtype(), DataType::UInt(64));
    mlir::Value offset =
        CastValue(VisitExpr(op->args[2]), op->args[2].dtype(), DataType::UInt(64));
    mlir::Value seq_mix = builder_.create<mlir::arith::MulIOp>(
        loc_, seq, ConstantIntLike(6364136223846793005LL, i64_type));
    mlir::Value offset_mix = builder_.create<mlir::arith::MulIOp>(
        loc_, offset, ConstantIntLike(1442695040888963407LL, i64_type));
    mlir::Value state =
        builder_.create<mlir::arith::AddIOp>(loc_, seed, ConstantIntLike(1, i64_type));
    state = builder_.create<mlir::arith::AddIOp>(loc_, state, seq_mix);
    state = builder_.create<mlir::arith::AddIOp>(loc_, state, offset_mix);
    builder_.create<mlir::memref::StoreOp>(loc_, state, active_rng_state_.value(), ZeroIndex());
  }

  mlir::Value AdvanceRngState() {
    ICHECK(active_rng_state_.has_value())
        << "tl.rng_rand/tl.rng_rand_float require a preceding tl.rng_init in riscv lowering";
    mlir::Type i64_type = builder_.getI64Type();
    mlir::Value state =
        builder_.create<mlir::memref::LoadOp>(loc_, active_rng_state_.value(), ZeroIndex());
    mlir::Value next = builder_.create<mlir::arith::MulIOp>(
        loc_, state, ConstantIntLike(6364136223846793005LL, i64_type));
    next = builder_.create<mlir::arith::AddIOp>(
        loc_, next, ConstantIntLike(1442695040888963407LL, i64_type));
    builder_.create<mlir::memref::StoreOp>(loc_, next, active_rng_state_.value(), ZeroIndex());
    return next;
  }

  mlir::Value LowerRngRand(const tir::CallNode* op) {
    ICHECK(op->op.same_as(tvm::tl::rng_rand())) << "LowerRngRand expects tl.rng_rand";
    ICHECK(op->args.empty()) << "tl.rng_rand expects no arguments";
    mlir::Value next = AdvanceRngState();
    mlir::Value high = builder_.create<mlir::arith::ShRUIOp>(
        loc_, next, ConstantIntLike(32, builder_.getI64Type()));
    mlir::Value raw = builder_.create<mlir::arith::TruncIOp>(loc_, builder_.getI32Type(), high);
    return CastValue(raw, DataType::UInt(32), op->dtype);
  }

  mlir::Value LowerRngRandFloat(const tir::CallNode* op) {
    ICHECK(op->op.same_as(tvm::tl::rng_rand_float()))
        << "LowerRngRandFloat expects tl.rng_rand_float";
    ICHECK_EQ(op->args.size(), 1U) << "tl.rng_rand_float expects <distribution>";
    const auto* dist = op->args[0].as<tir::StringImmNode>();
    ICHECK(dist != nullptr) << "tl.rng_rand_float expects its distribution as a string literal";
    ICHECK_EQ(dist->value, "uniform")
        << "tl.rng_rand_float currently only supports uniform distribution in riscv lowering";
    ICHECK(op->dtype.is_float() && op->dtype.lanes() == 1 &&
           (op->dtype.bits() == 32 || op->dtype.bits() == 64))
        << "tl.rng_rand_float currently only supports scalar float32/float64 in riscv lowering";

    mlir::Value next = AdvanceRngState();
    mlir::Type result_type = LowerScalarType(op->dtype);
    if (op->dtype.bits() == 32) {
      mlir::Value mantissa = builder_.create<mlir::arith::ShRUIOp>(
          loc_, next, ConstantIntLike(40, builder_.getI64Type()));
      mlir::Value fp = builder_.create<mlir::arith::UIToFPOp>(loc_, result_type, mantissa);
      mlir::Value scale = CreateFloatConstant(builder_, loc_, result_type, 1.0 / 16777216.0);
      return builder_.create<mlir::arith::MulFOp>(loc_, fp, scale);
    }

    mlir::Value mantissa = builder_.create<mlir::arith::ShRUIOp>(
        loc_, next, ConstantIntLike(11, builder_.getI64Type()));
    mlir::Value fp = builder_.create<mlir::arith::UIToFPOp>(loc_, result_type, mantissa);
    mlir::Value scale =
        CreateFloatConstant(builder_, loc_, result_type, 1.0 / 9007199254740992.0);
    return builder_.create<mlir::arith::MulFOp>(loc_, fp, scale);
  }

  void LowerScalarAtomicStore(const tir::CallNode* op) {
    ICHECK(IsScalarAtomicStoreCall(op)) << "LowerScalarAtomicStore expects atomic store";
    ICHECK(op->args.size() == 2 || op->args.size() == 3)
        << "atomic_store expects destination, value, and optional memory order";
    AccessPtrBinding dst = DecodeElementAccessPtr(op->args[0], "atomic_store");
    ICHECK_EQ(dst.buffer->dtype.lanes(), 1)
        << "atomic_store currently only supports scalar element buffers in riscv";
    ICHECK(IsFloatLikeType(dst.buffer->dtype) || dst.buffer->dtype.is_int() ||
           dst.buffer->dtype.is_uint())
        << "atomic_store currently only supports int/uint/float buffers in riscv";

    mlir::Value value = VisitExpr(op->args[1]);
    value = CastValue(value, op->args[1].dtype(), dst.buffer->dtype);
    builder_.create<mlir::memref::StoreOp>(loc_, value, dst.memref, dst.indices);
  }

  mlir::Value LowerVectorAtomicAdd(const tir::CallNode* op) {
    ICHECK(IsVectorAtomicAddCall(op)) << "LowerVectorAtomicAdd expects atomic_addx2/addx4";
    const int lane_count = VectorAtomicAddLaneCount(op);
    const char* context = VectorAtomicAddContext(op);
    ICHECK(op->args.size() == 2 || op->args.size() == 3)
        << context << " expects destination, source, and optional memory order";

    ContiguousAccessPtrBinding dst = DecodeContiguousAccessPtr(op->args[0], lane_count, context);
    ContiguousAccessPtrBinding src = DecodeContiguousAccessPtr(op->args[1], lane_count, context);
    ICHECK_EQ(dst.buffer->dtype.lanes(), 1)
        << context << " currently only supports scalar element destination buffers";
    ICHECK_EQ(src.buffer->dtype.lanes(), 1)
        << context << " currently only supports scalar element source buffers";
    ICHECK(IsFloatLikeType(dst.buffer->dtype) || dst.buffer->dtype.is_int() ||
           dst.buffer->dtype.is_uint())
        << context << " currently only supports int/uint/float destination buffers";
    ICHECK(IsFloatLikeType(src.buffer->dtype) || src.buffer->dtype.is_int() ||
           src.buffer->dtype.is_uint())
        << context << " currently only supports int/uint/float source buffers";
    ICHECK_EQ(dst.lane_indices.size(), static_cast<size_t>(lane_count));
    ICHECK_EQ(src.lane_indices.size(), static_cast<size_t>(lane_count));
    const bool returns_value = !op->dtype.is_handle();
    llvm::SmallVector<mlir::Value, 4> old_values;
    if (returns_value) {
      ICHECK_EQ(op->dtype.lanes(), lane_count)
          << context << " return-value form expects a vector dtype with " << lane_count
          << " lanes in riscv lowering";
      old_values.reserve(static_cast<size_t>(lane_count));
    }

    for (int lane = 0; lane < lane_count; ++lane) {
      mlir::Value old_value =
          builder_.create<mlir::memref::LoadOp>(loc_, dst.memref, dst.lane_indices[lane]);
      if (returns_value) {
        old_values.push_back(CastValue(old_value, dst.buffer->dtype, op->dtype.element_of()));
      }
      mlir::Value operand =
          builder_.create<mlir::memref::LoadOp>(loc_, src.memref, src.lane_indices[lane]);
      operand = CastValue(operand, src.buffer->dtype, dst.buffer->dtype);
      mlir::Value updated;
      if (IsFloatLikeType(dst.buffer->dtype)) {
        updated = builder_.create<mlir::arith::AddFOp>(loc_, old_value, operand);
      } else {
        updated = builder_.create<mlir::arith::AddIOp>(loc_, old_value, operand);
      }
      builder_.create<mlir::memref::StoreOp>(loc_, updated, dst.memref, dst.lane_indices[lane]);
    }
    if (returns_value) {
      return builder_.create<mlir::vector::FromElementsOp>(loc_, LowerScalarType(op->dtype),
                                                           old_values);
    }
    return mlir::Value();
  }

  bool MatchElementwiseLoopNest(const tir::ForNode* outer, ElementwiseLoopNestMatch* match) {
    const tir::ForNode* current = outer;
    while (true) {
      if (!IsStructuredLoopKind(current->kind) || current->thread_binding.defined() ||
          !tir::is_zero(current->min) ||
          (current->step.defined() && !tir::is_one(current->step.value()))) {
        return false;
      }
      match->loops.push_back(current);

      if (const auto* inner = current->body.as<tir::ForNode>()) {
        current = inner;
        continue;
      }
      match->block_realize = current->body.as<tir::BlockRealizeNode>();
      if (match->block_realize == nullptr || !tir::is_one(match->block_realize->predicate)) {
        return false;
      }
      break;
    }

    match->block = match->block_realize->block.get();
    if (match->block->init.defined() || !match->block->alloc_buffers.empty() ||
        !match->block->match_buffers.empty() ||
        match->block->iter_vars.size() != match->loops.size() ||
        match->block_realize->iter_values.size() != match->loops.size()) {
      return false;
    }

    for (size_t i = 0; i < match->loops.size(); ++i) {
      const tir::IterVar& iter_var = match->block->iter_vars[i];
      if (iter_var->iter_type != tir::IterVarType::kDataPar) {
        return false;
      }
      const auto* iter_value_var = match->block_realize->iter_values[i].as<tir::VarNode>();
      if (iter_value_var == nullptr || iter_value_var != match->loops[i]->loop_var.get()) {
        return false;
      }
      match->block_vars.push_back(iter_var->var);
    }

    match->store = match->block->body.as<tir::BufferStoreNode>();
    if (match->store == nullptr || match->store->predicate.defined() ||
        !IndicesMatchIdentityVars(match->store->indices, match->block_vars)) {
      return false;
    }

    const tir::Buffer& output_buffer = match->store->buffer;
    ValidateLowerableBufferLayout(output_buffer);
    if (output_buffer->shape.size() != match->loops.size()) {
      return false;
    }
    for (size_t i = 0; i < match->loops.size(); ++i) {
      if (!analyzer_.CanProveEqual(output_buffer->shape[i], match->loops[i]->extent)) {
        return false;
      }
    }

    StructuredExprAnalyzer analyzer(match->block_vars);
    if (!analyzer.Analyze(match->store->value)) {
      return false;
    }
    match->input_buffers = analyzer.input_buffers();

    for (const tir::Buffer& input_buffer : match->input_buffers) {
      if (input_buffer.get() == output_buffer.get()) {
        return false;
      }
      if (!ValidateBufferForPattern(input_buffer, analyzer.input_pattern(input_buffer),
                                    match->loops)) {
        return false;
      }
    }
    return true;
  }

  bool MatchReductionLoopNest(const tir::ForNode* outer, ReductionLoopNestMatch* match) {
    const tir::ForNode* current = outer;
    while (true) {
      if (!IsStructuredLoopKind(current->kind) || current->thread_binding.defined() ||
          !tir::is_zero(current->min) ||
          (current->step.defined() && !tir::is_one(current->step.value()))) {
        return false;
      }
      match->loops.push_back(current);

      if (const auto* inner = current->body.as<tir::ForNode>()) {
        current = inner;
        continue;
      }
      match->block_realize = current->body.as<tir::BlockRealizeNode>();
      if (match->block_realize == nullptr || !tir::is_one(match->block_realize->predicate)) {
        return false;
      }
      break;
    }

    match->block = match->block_realize->block.get();
    if (match->block->init.defined() || !match->block->alloc_buffers.empty() ||
        !match->block->match_buffers.empty() ||
        match->block->iter_vars.size() != match->loops.size() ||
        match->block_realize->iter_values.size() != match->loops.size()) {
      return false;
    }

    for (size_t i = 0; i < match->loops.size(); ++i) {
      const tir::IterVar& iter_var = match->block->iter_vars[i];
      const auto* iter_value_var = match->block_realize->iter_values[i].as<tir::VarNode>();
      if (iter_value_var == nullptr || iter_value_var != match->loops[i]->loop_var.get()) {
        return false;
      }
      match->block_vars.push_back(iter_var->var);
      if (iter_var->iter_type == tir::IterVarType::kCommReduce) {
        match->reduction_dims.push_back(i);
      } else if (iter_var->iter_type == tir::IterVarType::kDataPar) {
        match->output_vars.push_back(iter_var->var);
      } else {
        return false;
      }
    }
    if (match->reduction_dims.empty()) {
      return false;
    }

    const auto* seq = match->block->body.as<tir::SeqStmtNode>();
    if (seq == nullptr || seq->seq.size() != 2) {
      return false;
    }
    match->init_if = seq->seq[0].as<tir::IfThenElseNode>();
    match->update_store = seq->seq[1].as<tir::BufferStoreNode>();
    if (match->init_if == nullptr || match->init_if->else_case.defined() ||
        match->update_store == nullptr || match->update_store->predicate.defined()) {
      return false;
    }

    llvm::SmallVector<tir::Var, 4> reduction_vars;
    reduction_vars.reserve(match->reduction_dims.size());
    for (size_t reduction_dim : match->reduction_dims) {
      reduction_vars.push_back(match->block_vars[reduction_dim]);
    }
    if (!MatchesReductionInitCondition(match->init_if->condition, reduction_vars)) {
      return false;
    }

    const auto* init_store = match->init_if->then_case.as<tir::BufferStoreNode>();
    if (init_store == nullptr || init_store->predicate.defined() ||
        !IndicesMatchIdentityVars(init_store->indices, match->output_vars) ||
        !IndicesMatchIdentityVars(match->update_store->indices, match->output_vars)) {
      return false;
    }

    match->output_buffer = match->update_store->buffer;
    if (init_store->buffer.get() != match->output_buffer.get()) {
      return false;
    }
    match->init_value = init_store->value;

    const PrimExpr* input_expr = nullptr;
    if (!MatchReductionCombineExpr(match->update_store->value, match->output_buffer,
                                   match->output_vars, &match->kind, &input_expr)) {
      return false;
    }
    ValidateLowerableBufferLayout(match->output_buffer);
    match->reduction_expr = *input_expr;
    if (match->output_buffer->shape.size() != match->output_vars.size()) {
      return false;
    }
    for (size_t i = 0, out_i = 0; i < match->loops.size(); ++i) {
      if (IsReductionDim(match->reduction_dims, i)) {
        continue;
      }
      if (!analyzer_.CanProveEqual(match->output_buffer->shape[out_i], match->loops[i]->extent)) {
        return false;
      }
      ++out_i;
    }
    return true;
  }

  bool ValidateBufferForPattern(const tir::Buffer& buffer, const StructuredIndexPattern& pattern,
                                llvm::ArrayRef<const tir::ForNode*> loops) {
    ValidateLowerableBufferLayout(buffer);
    if (buffer->shape.size() != pattern.dims.size()) {
      return false;
    }
    for (size_t i = 0; i < pattern.dims.size(); ++i) {
      if (pattern.dims[i] >= loops.size() ||
          !analyzer_.CanProveEqual(buffer->shape[i], loops[pattern.dims[i]]->extent)) {
        return false;
      }
    }
    return true;
  }

  bool TryLowerReductionLoopNestAsGeneric(const ReductionLoopNestMatch& match, mlir::Value init_value,
                                          mlir::Value output_value) {
    StructuredExprAnalyzer analyzer(match.block_vars);
    if (!analyzer.Analyze(match.reduction_expr)) {
      return false;
    }

    llvm::SmallVector<mlir::Value, 4> input_values;
    llvm::SmallVector<StructuredIndexPattern, 4> input_patterns;
    input_values.reserve(analyzer.input_buffers().size());
    input_patterns.reserve(analyzer.input_buffers().size());
    for (const tir::Buffer& input_buffer : analyzer.input_buffers()) {
      if (input_buffer.get() == match.output_buffer.get()) {
        return false;
      }
      StructuredIndexPattern pattern = analyzer.input_pattern(input_buffer);
      if (!ValidateBufferForPattern(input_buffer, pattern, match.loops)) {
        return false;
      }
      input_values.push_back(LookupBufferValue(input_buffer));
      input_patterns.push_back(pattern);
    }

    size_t rank = match.loops.size();
    StructuredIndexPattern output_pattern = ReductionOutputPattern(rank, match.reduction_dims);
    llvm::SmallVector<mlir::AffineMap, 4> indexing_maps;
    indexing_maps.reserve(input_values.size() + 1);
    for (const StructuredIndexPattern& pattern : input_patterns) {
      indexing_maps.push_back(PatternMap(rank, pattern));
    }
    indexing_maps.push_back(PatternMap(rank, output_pattern));

    llvm::SmallVector<mlir::utils::IteratorType, 4> iterator_types;
    iterator_types.reserve(rank);
    for (size_t i = 0; i < rank; ++i) {
      iterator_types.push_back(IsReductionDim(match.reduction_dims, i)
                                   ? mlir::utils::IteratorType::reduction
                                   : mlir::utils::IteratorType::parallel);
    }

    builder_.create<mlir::linalg::FillOp>(loc_, mlir::ValueRange{init_value},
                                          mlir::ValueRange{output_value});
    mlir::linalg::GenericOp generic = builder_.create<mlir::linalg::GenericOp>(
        loc_, mlir::ValueRange(input_values), mlir::ValueRange{output_value}, indexing_maps,
        iterator_types);

    mlir::Block* body = new mlir::Block();
    generic.getRegion().push_back(body);
    for (const tir::Buffer& input_buffer : analyzer.input_buffers()) {
      body->addArgument(LowerScalarType(input_buffer->dtype), loc_);
    }
    body->addArgument(LowerScalarType(match.output_buffer->dtype), loc_);

    {
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPointToStart(body);

      llvm::SmallVector<mlir::Value, 4> element_args;
      llvm::SmallVector<StructuredIndexPattern, 4> element_patterns;
      element_args.reserve(analyzer.input_buffers().size());
      element_patterns.reserve(analyzer.input_buffers().size());
      for (size_t i = 0; i < analyzer.input_buffers().size(); ++i) {
        element_args.push_back(body->getArgument(i));
        element_patterns.push_back(input_patterns[i]);
      }

      StructuredRegionExprLowerer lowerer(this, match.block_vars, analyzer.input_buffers(),
                                          element_args, element_patterns);
      mlir::Value value = lowerer.Lower(match.reduction_expr);
      value = CastValue(value, match.reduction_expr.dtype(), match.output_buffer->dtype);
      mlir::Value acc = body->getArgument(static_cast<unsigned>(analyzer.input_buffers().size()));
      mlir::Value reduced = EmitReductionCombine(match.kind, match.output_buffer->dtype, builder_,
                                                loc_, acc, value);
      builder_.create<mlir::linalg::YieldOp>(loc_, reduced);
    }
    return true;
  }

  bool TryLowerReductionLoopNest(const tir::ForNode* outer) {
    ReductionLoopNestMatch match;
    if (!MatchReductionLoopNest(outer, &match)) {
      return false;
    }

    mlir::Value init_value = VisitExpr(match.init_value);
    init_value = CastValue(init_value, match.init_value.dtype(), match.output_buffer->dtype);
    mlir::Value output_value = LookupBufferValue(match.output_buffer);

    const auto* input_load = match.reduction_expr.as<tir::BufferLoadNode>();
    if (input_load == nullptr || !IndicesMatchIdentityVars(input_load->indices, match.block_vars)) {
      return TryLowerReductionLoopNestAsGeneric(match, init_value, output_value);
    }
    if (!ValidateBufferForPattern(input_load->buffer, FullIdentityPattern(match.loops.size()),
                                  match.loops)) {
      return TryLowerReductionLoopNestAsGeneric(match, init_value, output_value);
    }
    mlir::Value input_value = LookupBufferValue(input_load->buffer);

    builder_.create<mlir::linalg::FillOp>(loc_, mlir::ValueRange{init_value},
                                          mlir::ValueRange{output_value});
    llvm::SmallVector<int64_t, 4> reduction_dims;
    reduction_dims.reserve(match.reduction_dims.size());
    for (size_t reduction_dim : match.reduction_dims) {
      reduction_dims.push_back(static_cast<int64_t>(reduction_dim));
    }
    builder_.create<mlir::linalg::ReduceOp>(
        loc_, mlir::ValueRange{input_value}, mlir::ValueRange{output_value}, reduction_dims,
        [&](mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange args) {
          ICHECK_EQ(args.size(), 2);
          mlir::Value in = args[0];
          mlir::Value acc = args[1];
          mlir::Value reduced =
              EmitReductionCombine(match.kind, match.output_buffer->dtype, b, loc, acc, in);
          b.create<mlir::linalg::YieldOp>(loc, reduced);
        });
    return true;
  }

  bool TryLowerElementwiseLoopNest(const tir::ForNode* outer) {
    ElementwiseLoopNestMatch match;
    if (!MatchElementwiseLoopNest(outer, &match)) {
      return false;
    }

    llvm::SmallVector<mlir::Value, 4> input_values;
    input_values.reserve(match.input_buffers.size());
    llvm::SmallVector<StructuredIndexPattern, 4> input_patterns;
    input_patterns.reserve(match.input_buffers.size());
    StructuredExprAnalyzer analyzer(match.block_vars);
    ICHECK(analyzer.Analyze(match.store->value));
    for (const tir::Buffer& input_buffer : match.input_buffers) {
      input_values.push_back(LookupBufferValue(input_buffer));
      input_patterns.push_back(analyzer.input_pattern(input_buffer));
    }

    mlir::Value output_value = LookupBufferValue(match.store->buffer);
    size_t rank = match.loops.size();
    mlir::AffineMap identity_map = mlir::AffineMap::getMultiDimIdentityMap(rank, &context_);
    llvm::SmallVector<mlir::AffineMap, 4> indexing_maps;
    indexing_maps.reserve(match.input_buffers.size() + 1);
    for (const StructuredIndexPattern& pattern : input_patterns) {
      indexing_maps.push_back(PatternMap(rank, pattern));
    }
    indexing_maps.push_back(identity_map);
    llvm::SmallVector<mlir::utils::IteratorType, 4> iterator_types(
        rank, mlir::utils::IteratorType::parallel);

    mlir::linalg::GenericOp generic = builder_.create<mlir::linalg::GenericOp>(
        loc_, mlir::ValueRange(input_values), mlir::ValueRange{output_value}, indexing_maps,
        iterator_types);

    mlir::Block* body = new mlir::Block();
    generic.getRegion().push_back(body);
    for (const tir::Buffer& input_buffer : match.input_buffers) {
      body->addArgument(LowerScalarType(input_buffer->dtype), loc_);
    }
    body->addArgument(LowerScalarType(match.store->buffer->dtype), loc_);

    {
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPointToStart(body);
      llvm::SmallVector<mlir::Value, 4> element_args;
      llvm::SmallVector<StructuredIndexPattern, 4> element_patterns;
      element_args.reserve(match.input_buffers.size());
      element_patterns.reserve(match.input_buffers.size());
      for (size_t i = 0; i < match.input_buffers.size(); ++i) {
        element_args.push_back(body->getArgument(i));
        element_patterns.push_back(input_patterns[i]);
      }
      StructuredRegionExprLowerer lowerer(this, match.block_vars, match.input_buffers,
                                          element_args, element_patterns);
      mlir::Value result = lowerer.Lower(match.store->value);
      result = CastValue(result, match.store->value.dtype(), match.store->buffer->dtype);
      builder_.create<mlir::linalg::YieldOp>(loc_, result);
    }
    return true;
  }

  bool LowerStatementLikeCall(const tir::CallNode* op) {
    const auto* op_node = op->op.as<OpNode>();
    if (op_node == nullptr) {
      return false;
    }

    if (op->op.same_as(tir::builtin::call_extern())) {
      std::string extern_name = GetCallExternName(op);
      if (IsSerializedFp8TmaLoadCallExternName(extern_name)) {
        return LowerSerializedFp8TmaLoadCallExtern(op);
      }
      if (extern_name == "tl::fp8_zero_raw_acc_64") {
        return LowerSerializedFp8ZeroRawAccCallExtern(op);
      }
      if (extern_name == "tl::fp8_fa3_raw_acc_store_smem_cute_64x128") {
        return LowerSerializedFp8RawAccStoreCallExtern(op);
      }
      if (extern_name == "tl::fp8_fa3_o_smem_store_global_cute_64x128") {
        return LowerSerializedFp8OutputStoreCallExtern(op);
      }
      if (IsSerializedNoOpCallExternName(extern_name)) {
        return true;
      }
      RejectUnsupportedCallExtern(op);
    }
    if (op->op.same_as(tvm::tl::rng_init())) {
      LowerRngInit(op);
      return true;
    }
    if (op_node->name == "tl.tileop.async_copy") {
      LowerTileAsyncCopy(op);
      return true;
    }
    if (op_node->name == "tl.tileop.tma_copy") {
      // The serialized RISC-V backend has no TMA runtime, so TMA-marked tile
      // copies become ordinary synchronous region copies.
      LowerTileCopy(op);
      return true;
    }
    if (op->op.same_as(tir::builtin::ptx_wait_group()) ||
        op_node->name == "tir.ptx_commit_group" || IsSerializedNoOpTargetSyncCall(op)) {
      // async copy, TMA barriers, WGMMA waits, and register-hint / warpgroup
      // markers all become no-ops in the serialized backend.
      return true;
    }
    if (IsCudaPipelineOrTargetSyncIntrinsicName(op_node->name)) {
      RejectCudaPipelineOrTargetSyncIntrinsic(op_node->name);
    }
    if (IsLowerableTileReductionIntrinsicName(op_node->name)) {
      LowerTileReduce(op);
      return true;
    }
    if (IsFinalizeReducerIntrinsicName(op_node->name)) {
      LowerFinalizeReducer(op);
      return true;
    }
    if (IsLowerableTileScanIntrinsicName(op_node->name)) {
      LowerTileCumsum(op);
      return true;
    }
    if (IsUnsupportedTileReductionIntrinsicName(op_node->name)) {
      RejectUnsupportedTileReductionIntrinsic(op_node->name);
    }
    if (IsUnsupportedTileScanIntrinsicName(op_node->name)) {
      RejectUnsupportedTileScanIntrinsic(op_node->name);
    }

    if (op->op.same_as(tir::builtin::tvm_storage_sync())) {
      if (InNonUnitLogicalThreadRegion()) {
        LOG(FATAL) << "tvm_storage_sync inside non-unit thread launch is not supported yet "
                   << "in riscv lowering unless it is handled by serialized phase splitting";
      }
      return true;
    }
    if (IsThreadReturnCall(op)) {
      ICHECK(!InLogicalThreadRegion())
          << "thread_return must be handled via a surrounding if-guard in riscv lowering";
      return true;
    }
    if (op->op.same_as(tvm::tl::pdl_sync())) {
      // Programmatic dependent launch is a CUDA launch-order feature. The RISC-V
      // MLIR backend currently emits a single-kernel module, so the in-kernel
      // wait marker has no runtime counterpart and is lowered as metadata/no-op.
      return true;
    }
    if (op->op.same_as(tvm::tl::sync_grid())) {
      if (!CanLowerSyncGridAsLocalBarrier()) {
        LOG(FATAL) << "tl.sync_grid inside non-unit launch is not supported yet in "
                   << "riscv lowering because it requires a cooperative grid runtime. "
                   << "Only the static single-block case is lowered as a serialized local "
                   << "phase boundary.";
      }
      return true;
    }
    if (op->op.same_as(tvm::tl::sync_warp())) {
      if (InNonUnitLogicalThreadRegion()) {
        LOG(FATAL) << "tl.sync_warp inside non-unit thread launch must be handled by "
                   << "phase splitting in riscv lowering";
      }
      return true;
    }
    if (IsCooperativeThreadIntrinsicName(op_node->name)) {
      if (InNonUnitLogicalThreadRegion()) {
        RejectUnsupportedCooperativeThreadStatement(op_node->name);
      }
      return true;
    }
    if (op->op.same_as(tvm::tl::device_assert()) ||
        op->op.same_as(tvm::tl::device_assert_with_msg())) {
      ICHECK(!op->args.empty()) << "device_assert expects at least one argument";
      (void)LowerCondition(op->args[0]);
      return true;
    }
    if (op->op.same_as(tir::builtin::assume())) {
      ICHECK_EQ(op->args.size(), 1) << "tir.assume expects exactly one condition argument";
      (void)LowerCondition(op->args[0]);
      return true;
    }
    if (IsLoopBreakCall(op)) {
      ICHECK(InBreakLoopRegion())
          << "tl.loop_break/break_loop is only supported inside an enclosing serialized loop in riscv lowering";
      StoreBreakLoopFlag(CurrentBreakLoopFlag(), true);
      return true;
    }
    if (op_node->name == "tl.tileop.atomicadd") {
      LowerTileAtomicAdd(op);
      return true;
    }

    if (IsScalarAtomicRMWCall(op)) {
      (void)LowerScalarAtomicRMW(op);
      return true;
    }
    if (IsScalarAtomicStoreCall(op)) {
      LowerScalarAtomicStore(op);
      return true;
    }
    if (IsScalarAtomicLoadCall(op)) {
      (void)LowerScalarAtomicLoad(op);
      return true;
    }
    if (IsVectorAtomicAddCall(op)) {
      (void)LowerVectorAtomicAdd(op);
      return true;
    }
    if (IsAtomicIntrinsicName(op_node->name)) {
      LOG(FATAL) << "Unsupported atomic intrinsic in riscv lowering: " << op->op;
      TVM_FFI_UNREACHABLE();
    }

    if (op_node->name == "tl.tileop.copy") {
      LowerTileCopy(op);
      return true;
    }
    if (op_node->name == "tl.tileop.fill") {
      TrackSerializedWarpReplayTileFill(tl::NormalizeToBufferRegion(op->args[0]), op->args[1]);
      LowerTileFill(op);
      return true;
    }
    if (op_node->name == "tl.tileop.gemm" || op_node->name == "tl.tileop.gemm_py" ||
        op_node->name == "tl.tileop.wgmma_gemm") {
      LowerTileGemmPy(op);
      return true;
    }
    return false;
  }

  void LowerFunction(const std::string& name, const tir::PrimFunc& func) {
    scalar_values_.clear();
    bound_prim_exprs_.clear();
    buffer_values_.clear();
    break_loop_stack_.clear();
    serialized_warp_replay_buffer_exprs_.clear();
    serialized_warp_replay_buffer_element_exprs_.clear();
    function_param_buffers_.clear();
    function_param_buffer_data_.clear();
    buffer_owner_.clear();
    active_rng_state_.reset();
    if (auto local_var_init =
            func->attrs.GetAttr<Map<tir::Var, PrimExpr>>(tl::attr::kLocalVarInit)) {
      local_var_init_map_ = local_var_init.value();
    } else {
      local_var_init_map_ = Map<tir::Var, PrimExpr>();
    }

    llvm::SmallVector<mlir::Type, 8> input_types;
    input_types.reserve(func->params.size());

    for (const tir::Var& param : func->params) {
      if (func->buffer_map.count(param)) {
        const tir::Buffer& buffer = func->buffer_map[param];
        ValidateLowerableBufferLayout(buffer);
        input_types.push_back(LowerBufferMemRefType(buffer));
      } else {
        ICHECK(!param.dtype().is_handle())
            << "Handle scalar params without buffer_map are not supported yet: " << param;
        input_types.push_back(LowerScalarType(param.dtype()));
      }
    }

    mlir::func::FuncOp func_op = builder_.create<mlir::func::FuncOp>(
        loc_, name, builder_.getFunctionType(input_types, llvm::ArrayRef<mlir::Type>{}));
    mlir::Block* entry_block = func_op.addEntryBlock();

    {
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPointToStart(entry_block);

      for (size_t i = 0; i < func->params.size(); ++i) {
        const tir::Var& param = func->params[i];
        mlir::Value arg = entry_block->getArgument(static_cast<unsigned>(i));
        if (func->buffer_map.count(param)) {
          const tir::Buffer& buffer = func->buffer_map[param];
          buffer_values_[param.get()] = arg;
          buffer_values_[buffer.get()] = arg;
          buffer_values_[buffer->data.get()] = arg;
          function_param_buffers_.insert(buffer.get());
          function_param_buffer_data_.insert(buffer->data.get());
          buffer_owner_[param.get()] = buffer;
          buffer_owner_[buffer.get()] = buffer;
          buffer_owner_[buffer->data.get()] = buffer;
          BindDynamicShapeVars(buffer, arg);
        } else {
          scalar_values_[param.get()] = arg;
        }
      }

      VisitStmt(func->body);
      builder_.create<mlir::func::ReturnOp>(loc_);
    }
  }

  void VisitSeqWithOptionalBreakGuards(const Array<tir::Stmt>& seq, size_t start_index = 0) {
    for (size_t i = start_index; i < seq.size(); ++i) {
      VisitStmt(seq[i]);
      if (InBreakLoopRegion() && i + 1 < seq.size() && ContainsDirectLoopBreak(seq[i])) {
        EmitIfCurrentLoopNotBroken([&]() { VisitSeqWithOptionalBreakGuards(seq, i + 1); });
        return;
      }
    }
  }

  void VisitStmt_(const tir::SeqStmtNode* op) final {
    if (InLogicalThreadRegion() && !op->seq.empty()) {
      if (const auto* if_node = op->seq.front().as<tir::IfThenElseNode>();
          IsSimpleThreadReturnIf(if_node)) {
        EmitConditionalRegion(tir::Not(if_node->condition), [&]() {
          VisitSeqWithOptionalBreakGuards(op->seq, 1);
        });
        return;
      }
    }
    VisitSeqWithOptionalBreakGuards(op->seq);
  }

  void VisitStmt_(const tir::EvaluateNode* op) final {
    if (const auto* call = op->value.as<tir::CallNode>()) {
      if (LowerStatementLikeCall(call)) {
        return;
      }
    }
    if (!op->value.as<IntImmNode>() || !tir::is_zero(op->value)) {
      (void)VisitExpr(op->value);
    }
  }

  template <typename BodyEmitter>
  void EmitLoopBodyWithBindings(const tir::ForNode* op, mlir::Value induction_var,
                                BodyEmitter&& body_emitter,
                                bool restore_replay_exprs = true) {
    SavedBinding saved_loop_var = SaveAndSet(scalar_values_, op->loop_var.get(), induction_var);
    std::vector<std::pair<const Object*, SavedBinding>> saved_bindings;
    std::vector<std::pair<const Object*, std::optional<tir::Buffer>>> saved_buffer_owners;
    std::vector<std::pair<const Object*, std::optional<tir::Buffer>>> saved_packed_owners;
    auto deferred_it = deferred_loop_bindings_.find(op->loop_var.get());
    if (deferred_it != deferred_loop_bindings_.end() && !deferred_it->second.empty()) {
      const DeferredLoopBindings& deferred = deferred_it->second.back();
      saved_bindings.reserve(deferred.alloc_buffers.size() * 2 +
                             deferred.match_buffers.size() * 2);
      saved_packed_owners.reserve(deferred.alloc_buffers.size());
      for (const tir::Buffer& buffer : deferred.alloc_buffers) {
        ValidateAllocaBufferLayout(buffer);
        mlir::Value alloc = CreateAlloca(buffer);
        BindBufferAliases(buffer, alloc, &saved_bindings, &saved_buffer_owners);
        if (buffer->dtype.lanes() == 2) {
          saved_packed_owners.emplace_back(
              buffer->data.get(), SaveAndSetPackedDataOwner(buffer->data.get(), buffer));
        }
      }
      for (const tir::MatchBufferRegion& match_buffer : deferred.match_buffers) {
        mlir::Value subview = CreateSubview(match_buffer);
        BindBufferAliases(match_buffer->buffer, subview, &saved_bindings, &saved_buffer_owners);
      }
    }
    std::vector<tir::Buffer> invalidated_replay_buffers =
        restore_replay_exprs ? CollectSerializedWarpReplayTrackedBuffersWritten(op->body)
                             : std::vector<tir::Buffer>();
    PrimExprMap saved_replay_exprs = serialized_warp_replay_buffer_exprs_;
    SerializedWarpReplayBufferElementExprMap saved_replay_element_exprs =
        serialized_warp_replay_buffer_element_exprs_;
    body_emitter();
    if (restore_replay_exprs) {
      serialized_warp_replay_buffer_exprs_ = std::move(saved_replay_exprs);
      serialized_warp_replay_buffer_element_exprs_ = std::move(saved_replay_element_exprs);
      for (const tir::Buffer& buffer : invalidated_replay_buffers) {
        ClearSerializedWarpReplayTrackedBufferElementExprs(buffer);
        if (IsSerializedWarpReplayTrackedScalarBuffer(buffer)) {
          ClearSerializedWarpReplayTrackedBufferExpr(buffer);
        }
      }
    }
    RestoreBindings(buffer_values_, saved_bindings);
    RestoreBufferOwnerBindings(saved_buffer_owners);
    RestorePackedDataOwnerBindings(saved_packed_owners);
    RestoreBinding(scalar_values_, op->loop_var.get(), saved_loop_var);
  }

  void LowerLoopBodyStmt(const tir::ForNode* op, const tir::Stmt& body,
                         mlir::Value induction_var) {
    EmitLoopBodyWithBindings(op, induction_var, [&]() { VisitStmt(body); });
  }

  void LowerLoopBody(const tir::ForNode* op, mlir::Value induction_var) {
    LowerLoopBodyStmt(op, op->body, induction_var);
  }

  template <typename BodyEmitter>
  void EmitLaunchSerialLoop(const tir::IterVarNode* iter_var, const PrimExpr& extent,
                            LaunchTrackingKind tracking, BodyEmitter&& body_emitter,
                            bool body_uses_iter_var = false,
                            const tir::Stmt* body_for_break = nullptr) {
    const bool has_direct_break =
        body_for_break != nullptr && ContainsDirectLoopBreak(*body_for_break);
    if (has_direct_break) {
      mlir::Value break_flag = CreateStaticAlloca({1}, DataType::Bool());
      StoreBreakLoopFlag(break_flag, false);
      break_loop_stack_.push_back(BreakLoopFrame{break_flag});
    }
    mlir::Value lower = ConstantIntLike(0, builder_.getIndexType());
    mlir::Value upper = AsIndex(VisitExpr(extent), extent.dtype());
    mlir::Value step = ConstantIntLike(1, builder_.getIndexType());
    mlir::scf::ForOp for_op = builder_.create<mlir::scf::ForOp>(loc_, lower, upper, step);
    mlir::OpBuilder::InsertionGuard guard(builder_);
    builder_.setInsertionPoint(for_op.getBody()->getTerminator());
    if (tracking == LaunchTrackingKind::kThread) {
      thread_launch_stack_.push_back(
          ThreadLaunchFrame{iter_var, GetOptionalStaticInt(extent), body_uses_iter_var});
      active_thread_local_bindings_stack_.emplace_back();
    } else if (tracking == LaunchTrackingKind::kBlock) {
      block_launch_stack_.push_back(
          ThreadLaunchFrame{iter_var, GetOptionalStaticInt(extent), body_uses_iter_var});
    }
    SavedBinding saved = SaveAndSet(scalar_values_, iter_var->var.get(), for_op.getInductionVar());
    PrimExprMap saved_replay_exprs = serialized_warp_replay_buffer_exprs_;
    SerializedWarpReplayBufferElementExprMap saved_replay_element_exprs =
        serialized_warp_replay_buffer_element_exprs_;
    if (has_direct_break) {
      EmitIfCurrentLoopNotBroken([&]() { body_emitter(for_op.getInductionVar()); });
    } else {
      body_emitter(for_op.getInductionVar());
    }
    serialized_warp_replay_buffer_exprs_ = std::move(saved_replay_exprs);
    serialized_warp_replay_buffer_element_exprs_ = std::move(saved_replay_element_exprs);
    RestoreBinding(scalar_values_, iter_var->var.get(), saved);
    if (tracking == LaunchTrackingKind::kThread) {
      active_thread_local_bindings_stack_.pop_back();
      thread_launch_stack_.pop_back();
    } else if (tracking == LaunchTrackingKind::kBlock) {
      block_launch_stack_.pop_back();
    }
    if (has_direct_break) {
      break_loop_stack_.pop_back();
    }
  }

  template <typename BodyEmitter>
  void EmitSingleExecutionThreadLaunch(const tir::IterVarNode* iter_var, const PrimExpr& extent,
                                       const tir::Stmt& body, BodyEmitter&& body_emitter) {
    std::vector<tir::Buffer> shared_buffers;
    CollectSharedBlockAllocBuffers(body, &shared_buffers);
    ScopedBufferBindings saved_shared_bindings;
    if (!shared_buffers.empty()) {
      BindSharedBlockAllocBuffersForThreadLaunch(shared_buffers, iter_var,
                                                 &saved_shared_bindings);
    }

    thread_launch_stack_.push_back(ThreadLaunchFrame{iter_var, GetOptionalStaticInt(extent),
                                                     /*body_uses_iter_var=*/false});
    active_thread_local_bindings_stack_.emplace_back();
    mlir::Value zero_thread = ConstantIntLike(0, LowerScalarType(iter_var->var.dtype()));
    SavedBinding saved = SaveAndSet(scalar_values_, iter_var->var.get(), zero_thread);
    PrimExprMap saved_replay_exprs = serialized_warp_replay_buffer_exprs_;
    SerializedWarpReplayBufferElementExprMap saved_replay_element_exprs =
        serialized_warp_replay_buffer_element_exprs_;

    body_emitter(zero_thread);

    serialized_warp_replay_buffer_exprs_ = std::move(saved_replay_exprs);
    serialized_warp_replay_buffer_element_exprs_ = std::move(saved_replay_element_exprs);
    RestoreBinding(scalar_values_, iter_var->var.get(), saved);
    active_thread_local_bindings_stack_.pop_back();
    thread_launch_stack_.pop_back();
    RestoreScopedBufferBindings(&saved_shared_bindings);
  }

  template <typename BodyEmitter>
  void EmitThreadLaunchBodyWithBindings(
      const tir::Stmt& body, const tir::IterVarNode* iter_var, const PrimExpr& extent,
      llvm::ArrayRef<ThreadLocalBlockAllocBinding> existing_thread_local_bindings,
      BodyEmitter&& body_emitter, bool body_uses_iter_var = false,
      const tir::Stmt* body_for_break = nullptr) {
    std::vector<tir::Buffer> replay_local_buffers;
    std::optional<int64_t> static_extent = GetOptionalStaticInt(extent);
    bool should_materialize_replay_backings =
        iter_var != nullptr && iter_var->thread_tag == "threadIdx.x" &&
        (!static_extent.has_value() || static_extent.value() != 1);
    if (should_materialize_replay_backings) {
      CollectSerializedWarpReplayThreadLocalBlockAllocBuffers(
          body, existing_thread_local_bindings, &replay_local_buffers);
    }
    std::vector<ThreadLocalBlockAllocBinding> thread_local_bindings(
        existing_thread_local_bindings.begin(), existing_thread_local_bindings.end());
    std::vector<ThreadLocalBlockAllocBinding> new_thread_local_bindings =
        CreateThreadLocalBlockAllocBackings(replay_local_buffers, iter_var, extent);
    thread_local_bindings.insert(thread_local_bindings.end(), new_thread_local_bindings.begin(),
                                 new_thread_local_bindings.end());

    EmitLaunchSerialLoop(
        iter_var, extent, LaunchTrackingKind::kThread,
        [&](mlir::Value induction_var) {
          ICHECK(!active_thread_local_bindings_stack_.empty())
              << "thread-local replay backing requires an active thread launch scope";
          size_t binding_base = active_thread_local_bindings_stack_.back().size();
          active_thread_local_bindings_stack_.back().insert(
              active_thread_local_bindings_stack_.back().end(),
              thread_local_bindings.begin(), thread_local_bindings.end());
          ScopedBufferBindings saved_thread_local_bindings;
          BindThreadLocalBlockAllocBuffersForPhase(thread_local_bindings, induction_var,
                                                  &saved_thread_local_bindings);
          body_emitter(induction_var);
          RestoreScopedBufferBindings(&saved_thread_local_bindings);
          active_thread_local_bindings_stack_.back().resize(binding_base);
        },
        body_uses_iter_var, body_for_break);
  }

  template <typename PhaseEmitter>
  void EmitPerThreadLaunchPhase(
      const tir::Stmt& phase, const tir::IterVarNode* iter_var, const PrimExpr& extent,
      llvm::ArrayRef<ThreadLocalBlockAllocBinding> thread_local_bindings,
      PhaseEmitter&& phase_emitter) {
    EmitThreadLaunchBodyWithBindings(
        phase, iter_var, extent, thread_local_bindings,
        [&](mlir::Value induction_var) { phase_emitter(phase, induction_var); },
        StmtUsesVar(phase, iter_var->var), &phase);
  }

  bool CanSplitThreadInvariantSerialLoopAtSharedSync(
      const tir::ForNode* loop, const tir::IterVarNode* thread_iter_var,
      std::vector<tir::Stmt>* body_phases,
      std::vector<tir::Buffer>* cross_phase_local_buffers_out = nullptr) {
    ICHECK(loop != nullptr);
    if (!IsSupportedGeneralLoopKind(loop->kind) || loop->kind == tir::ForKind::kParallel ||
        loop->kind == tir::ForKind::kVectorized) {
      return false;
    }
    if (ExprUsesVar(loop->min, thread_iter_var->var) ||
        ExprUsesVar(loop->extent, thread_iter_var->var) ||
        (loop->step.defined() && ExprUsesVar(loop->step.value(), thread_iter_var->var))) {
      return false;
    }

    std::vector<tir::Buffer> cross_phase_local_buffers;
    std::string unsupported_reason;
    if (!SplitAtSharedStorageSync(loop->body, body_phases, &unsupported_reason,
                                  &cross_phase_local_buffers, &thread_iter_var->var)) {
      return false;
    }
    bool has_phase_boundary =
        body_phases->size() > 1 ||
        StmtEndsWithPhaseBoundarySync(loop->body, &thread_iter_var->var);
    if (!has_phase_boundary) {
      return false;
    }
    if (cross_phase_local_buffers_out != nullptr) {
      *cross_phase_local_buffers_out = std::move(cross_phase_local_buffers);
      return true;
    }
    return cross_phase_local_buffers.empty();
  }

  bool LoopBodyContainsNonWarpPhaseBoundarySync(const tir::Stmt& stmt) const {
    bool found = false;
    tir::PostOrderVisit(stmt, [&](const ObjectRef& node) {
      if (found) {
        return;
      }
      const auto* eval = node.as<tir::EvaluateNode>();
      if (eval == nullptr) {
        return;
      }
      const auto* call = eval->value.as<tir::CallNode>();
      if (call == nullptr) {
        return;
      }
      found = call->op.same_as(tir::builtin::tvm_storage_sync()) ||
              call->op.same_as(tvm::tl::sync_grid());
    });
    return found;
  }

  bool CanSplitWarpSynchronousSerialLoopAtSharedSync(
      const tir::ForNode* loop, const tir::IterVarNode* thread_iter_var,
      std::vector<tir::Stmt>* body_phases, int64_t* loop_step_value,
      std::vector<tir::Buffer>* cross_phase_local_buffers_out = nullptr) {
    ICHECK(loop != nullptr);
    ICHECK(loop_step_value != nullptr);
    if (thread_iter_var == nullptr || thread_iter_var->thread_tag != "threadIdx.x") {
      return false;
    }
    if (!IsSupportedGeneralLoopKind(loop->kind) || loop->kind == tir::ForKind::kParallel ||
        loop->kind == tir::ForKind::kVectorized) {
      return false;
    }
    int64_t static_step = 1;
    if (loop->step.defined()) {
      std::optional<int64_t> loop_step = GetOptionalStaticInt(loop->step.value());
      if (!loop_step.has_value() || loop_step.value() <= 0) {
        return false;
      }
      static_step = loop_step.value();
    }
    if (LoopBodyContainsNonWarpPhaseBoundarySync(loop->body)) {
      return false;
    }

    std::vector<tir::Buffer> cross_phase_local_buffers;
    std::string unsupported_reason;
    if (!SplitAtSharedStorageSync(loop->body, body_phases, &unsupported_reason,
                                  &cross_phase_local_buffers, nullptr)) {
      return false;
    }
    if (body_phases->empty()) {
      return false;
    }
    bool has_phase_boundary =
        body_phases->size() > 1 || StmtEndsWithPhaseBoundarySync(loop->body);
    if (!has_phase_boundary) {
      return false;
    }
    if (cross_phase_local_buffers_out != nullptr) {
      *cross_phase_local_buffers_out = std::move(cross_phase_local_buffers);
    } else if (!cross_phase_local_buffers.empty()) {
      return false;
    }
    *loop_step_value = static_step;
    return true;
  }

  bool LoopsShareCommonLeafThreadLaunchSignature(const tir::ForNode* loop,
                                                 const tir::ForNode* reference) const {
    if (loop == nullptr || reference == nullptr) {
      return false;
    }
    if (loop->thread_binding.defined() || reference->thread_binding.defined()) {
      return false;
    }
    if (!IsSupportedGeneralLoopKind(loop->kind) ||
        loop->kind == tir::ForKind::kParallel ||
        loop->kind == tir::ForKind::kVectorized ||
        loop->kind != reference->kind) {
      return false;
    }
    StructuralEqual structural_equal;
    if (!structural_equal(loop->min, reference->min) ||
        !structural_equal(loop->extent, reference->extent)) {
      return false;
    }
    if (loop->step.defined() != reference->step.defined()) {
      return false;
    }
    if (loop->step.defined() &&
        !structural_equal(loop->step.value(), reference->step.value())) {
      return false;
    }
    return true;
  }

  const tir::ForNode* MatchCommonLeafThreadLaunchLoop(
      const tir::Stmt& stmt, const tir::ForNode* reference = nullptr) const {
    if (!stmt.defined()) {
      return nullptr;
    }
    if (const auto* seq = stmt.as<tir::SeqStmtNode>()) {
      if (seq->seq.empty()) {
        return nullptr;
      }
      return MatchCommonLeafThreadLaunchLoop(seq->seq.back(), reference);
    }
    if (const auto* attr = stmt.as<tir::AttrStmtNode>()) {
      if (attr->attr_key == tir::attr::tilelang_assume ||
          (attr->attr_key == tir::attr::thread_extent &&
           IsThreadLaunchIterVar(attr->node.as<tir::IterVarNode>()) &&
           GetOptionalStaticInt(attr->value).value_or(0) == 1)) {
        return MatchCommonLeafThreadLaunchLoop(attr->body, reference);
      }
      return nullptr;
    }
    if (const auto* let = stmt.as<tir::LetStmtNode>()) {
      return MatchCommonLeafThreadLaunchLoop(let->body, reference);
    }
    if (const auto* realize = stmt.as<tir::BlockRealizeNode>()) {
      const tir::BlockNode* block = realize->block.as<tir::BlockNode>();
      if (block == nullptr || block->init.defined() || !block->match_buffers.empty() ||
          !tir::is_one(realize->predicate)) {
        return nullptr;
      }
      return MatchCommonLeafThreadLaunchLoop(block->body, reference);
    }
    if (const auto* loop = stmt.as<tir::ForNode>()) {
      if (reference != nullptr) {
        return LoopsShareCommonLeafThreadLaunchSignature(loop, reference)
                   ? reference
                   : nullptr;
      }
      if (loop->thread_binding.defined() || !IsSupportedGeneralLoopKind(loop->kind) ||
          loop->kind == tir::ForKind::kParallel ||
          loop->kind == tir::ForKind::kVectorized) {
        return nullptr;
      }
      return loop;
    }
    const auto* if_node = stmt.as<tir::IfThenElseNode>();
    if (if_node == nullptr) {
      return nullptr;
    }
    if (!if_node->else_case.defined()) {
      return MatchCommonLeafThreadLaunchLoop(if_node->then_case, reference);
    }
    const tir::ForNode* matched = MatchCommonLeafThreadLaunchLoop(if_node->then_case, reference);
    if (matched == nullptr) {
      return nullptr;
    }
    return MatchCommonLeafThreadLaunchLoop(if_node->else_case.value(), matched);
  }

  tir::Stmt StripCommonLeafThreadLaunchLoop(const tir::Stmt& stmt,
                                            const tir::ForNode* reference) const {
    ICHECK(reference != nullptr);
    if (const auto* seq = stmt.as<tir::SeqStmtNode>()) {
      ICHECK(!seq->seq.empty());
      tir::Stmt stripped_tail =
          StripCommonLeafThreadLaunchLoop(seq->seq.back(), reference);
      if (seq->seq.size() == 1) {
        return stripped_tail;
      }
      ffi::Array<tir::Stmt> rebuilt_seq;
      rebuilt_seq.reserve(seq->seq.size());
      for (size_t i = 0; i + 1 < seq->seq.size(); ++i) {
        rebuilt_seq.push_back(seq->seq[i]);
      }
      rebuilt_seq.push_back(stripped_tail);
      return tir::SeqStmt(rebuilt_seq, seq->span);
    }
    if (const auto* attr = stmt.as<tir::AttrStmtNode>()) {
      if (attr->attr_key == tir::attr::tilelang_assume ||
          (attr->attr_key == tir::attr::thread_extent &&
           IsThreadLaunchIterVar(attr->node.as<tir::IterVarNode>()) &&
           GetOptionalStaticInt(attr->value).value_or(0) == 1)) {
        return StripCommonLeafThreadLaunchLoop(attr->body, reference);
      }
      ICHECK(false) << "unexpected attribute while stripping common leaf thread-launch loop";
    }
    if (const auto* let = stmt.as<tir::LetStmtNode>()) {
      return tir::LetStmt(let->var, let->value,
                          StripCommonLeafThreadLaunchLoop(let->body, reference), let->span);
    }
    if (const auto* realize = stmt.as<tir::BlockRealizeNode>()) {
      const tir::BlockNode* block = realize->block.as<tir::BlockNode>();
      ICHECK(block != nullptr && !block->init.defined() && block->match_buffers.empty() &&
             tir::is_one(realize->predicate));
      tir::Block stripped_block(block->iter_vars, block->reads, block->writes, block->name_hint,
                                StripCommonLeafThreadLaunchLoop(block->body, reference),
                                block->init, block->alloc_buffers, block->match_buffers,
                                block->annotations, block->span);
      return tir::BlockRealize(realize->iter_values, realize->predicate, stripped_block,
                               realize->span);
    }
    if (const auto* loop = stmt.as<tir::ForNode>()) {
      ICHECK(LoopsShareCommonLeafThreadLaunchSignature(loop, reference));
      if (loop->loop_var.same_as(reference->loop_var)) {
        return loop->body;
      }
      ffi::Map<tir::Var, PrimExpr> var_remap;
      var_remap.Set(loop->loop_var, reference->loop_var);
      return tir::Substitute(loop->body, var_remap);
    }
    const auto* if_node = stmt.as<tir::IfThenElseNode>();
    ICHECK(if_node != nullptr);
    tir::Stmt then_body = StripCommonLeafThreadLaunchLoop(if_node->then_case, reference);
    std::optional<tir::Stmt> else_body;
    if (if_node->else_case.defined()) {
      else_body = StripCommonLeafThreadLaunchLoop(if_node->else_case.value(), reference);
    } else {
      else_body = tir::Evaluate(0);
    }
    return tir::IfThenElse(if_node->condition, then_body, else_body, if_node->span);
  }

  tir::For LiftCommonLeafThreadLaunchLoop(const tir::Stmt& stmt,
                                          const tir::ForNode* reference) const {
    ICHECK(reference != nullptr);
    tir::Stmt lifted_body = StripCommonLeafThreadLaunchLoop(stmt, reference);
    return tir::For(reference->loop_var, reference->min, reference->extent, reference->kind,
                    lifted_body, reference->thread_binding, reference->annotations,
                    reference->step);
  }

  mlir::Value ComputePositiveCeilDivIndex(mlir::Value extent, int64_t divisor) {
    ICHECK_GT(divisor, 0);
    if (divisor == 1) {
      return extent;
    }
    mlir::Value divisor_value = ConstantIntLike(divisor, builder_.getIndexType());
    mlir::Value adjusted =
        builder_.create<mlir::arith::AddIOp>(loc_, extent, ConstantIntLike(divisor - 1,
                                                                           builder_.getIndexType()));
    return builder_.create<mlir::arith::DivUIOp>(loc_, adjusted, divisor_value);
  }

  template <typename BodyEmitter>
  void EmitWithBoundThreadLaunchValue(
      const tir::IterVarNode* iter_var, const PrimExpr& extent, mlir::Value thread_value,
      llvm::ArrayRef<ThreadLocalBlockAllocBinding> thread_local_bindings,
      bool body_uses_iter_var, BodyEmitter&& body_emitter) {
    mlir::OpBuilder::InsertionGuard guard(builder_);
    SavedBinding saved_thread = SaveAndSet(scalar_values_, iter_var->var.get(), thread_value);
    thread_launch_stack_.push_back(
        ThreadLaunchFrame{iter_var, GetOptionalStaticInt(extent), body_uses_iter_var});
    active_thread_local_bindings_stack_.emplace_back(thread_local_bindings.begin(),
                                                     thread_local_bindings.end());
    ScopedBufferBindings saved_thread_local_bindings;
    BindThreadLocalBlockAllocBuffersForPhase(thread_local_bindings, thread_value,
                                            &saved_thread_local_bindings);
    PrimExprMap saved_replay_exprs = serialized_warp_replay_buffer_exprs_;
    SerializedWarpReplayBufferElementExprMap saved_replay_element_exprs =
        serialized_warp_replay_buffer_element_exprs_;
    body_emitter();
    serialized_warp_replay_buffer_exprs_ = std::move(saved_replay_exprs);
    serialized_warp_replay_buffer_element_exprs_ = std::move(saved_replay_element_exprs);
    RestoreScopedBufferBindings(&saved_thread_local_bindings);
    active_thread_local_bindings_stack_.pop_back();
    thread_launch_stack_.pop_back();
    RestoreBinding(scalar_values_, iter_var->var.get(), saved_thread);
  }

  template <typename BodyEmitter>
  void EmitWithLetBinding(const tir::LetStmtNode* let, BodyEmitter&& body_emitter) {
    ICHECK(let != nullptr);
    if (TryEmitPointerBackedHandleLet(let, body_emitter)) {
      return;
    }
    mlir::Value value = VisitExpr(let->value);
    SavedBinding saved = SaveAndSet(scalar_values_, let->var.get(), value);
    std::optional<PrimExpr> saved_expr =
        SaveAndSetPrimExpr(bound_prim_exprs_, let->var.get(), let->value);
    body_emitter();
    RestorePrimExprBinding(bound_prim_exprs_, let->var.get(), saved_expr);
    RestoreBinding(scalar_values_, let->var.get(), saved);
  }

  template <typename BodyEmitter>
  void EmitWithLetBindings(llvm::ArrayRef<const tir::LetStmtNode*> let_bindings,
                           BodyEmitter&& body_emitter) {
    auto emit = [&](auto&& self, size_t index) -> void {
      if (index >= let_bindings.size()) {
        body_emitter();
        return;
      }
      EmitWithLetBinding(let_bindings[index], [&]() { self(self, index + 1); });
    };
    emit(emit, 0);
  }

  template <typename PhaseEmitter>
  void EmitThreadInvariantSerialLoopWithThreadLaunchPhases(
      const tir::ForNode* loop, const std::vector<tir::Stmt>& body_phases,
      const tir::IterVarNode* thread_iter_var, const PrimExpr& thread_extent,
      llvm::ArrayRef<ThreadLocalBlockAllocBinding> thread_local_bindings,
      llvm::ArrayRef<const tir::LetStmtNode*> let_bindings,
      PhaseEmitter&& phase_emitter) {
    ICHECK(loop != nullptr);
    mlir::Value lower;
    mlir::Value extent;
    mlir::Value step;
    EmitWithLetBindings(let_bindings, [&]() {
      lower = AsIndex(VisitExpr(loop->min), loop->min.dtype());
      extent = AsIndex(VisitExpr(loop->extent), loop->extent.dtype());
      step =
          loop->step.defined()
              ? AsIndex(VisitExpr(loop->step.value()), loop->step.value().dtype())
              : ConstantIntLike(1, builder_.getIndexType());
    });
    mlir::Value upper = builder_.create<mlir::arith::AddIOp>(loc_, lower, extent);

    mlir::scf::ForOp for_op = builder_.create<mlir::scf::ForOp>(loc_, lower, upper, step);
    mlir::OpBuilder::InsertionGuard guard(builder_);
    builder_.setInsertionPoint(for_op.getBody()->getTerminator());
    EmitLoopBodyWithBindings(loop, for_op.getInductionVar(), [&]() {
      for (const tir::Stmt& phase : body_phases) {
        if (IsThreadLaunchPhaseGlobalTileCumsumStmt(phase)) {
          EmitWithLetBindings(let_bindings, [&]() { VisitStmt(phase); });
          continue;
        }
        if (TryEmitThreadLaunchNestedSerialLoopPhases(
                phase, thread_iter_var, thread_extent, phase_emitter,
                thread_local_bindings, let_bindings)) {
          continue;
        }
        EmitPerThreadLaunchPhase(
            phase, thread_iter_var, thread_extent, thread_local_bindings,
            [&](const tir::Stmt& phase_stmt, mlir::Value induction_var) {
              EmitWithLetBindings(let_bindings,
                                  [&]() { phase_emitter(phase_stmt, induction_var); });
            });
      }
    });
  }

  bool ThreadLocalBindingsContainBuffer(
      llvm::ArrayRef<ThreadLocalBlockAllocBinding> bindings, const tir::Buffer& buffer) const {
    for (const ThreadLocalBlockAllocBinding& binding : bindings) {
      if (SameBuffer(binding.buffer, buffer)) {
        return true;
      }
    }
    return false;
  }

  bool HasNestedThreadLaunchSplitLoop(const tir::Stmt& stmt,
                                      const tir::IterVarNode* thread_iter_var) {
    if (!stmt.defined()) {
      return false;
    }
    if (const auto* if_node = stmt.as<tir::IfThenElseNode>()) {
      if (ExprUsesVar(if_node->condition, thread_iter_var->var)) {
        const auto* common_leaf_loop = MatchCommonLeafThreadLaunchLoop(stmt);
        if (common_leaf_loop != nullptr) {
          tir::For lifted_loop = LiftCommonLeafThreadLaunchLoop(stmt, common_leaf_loop);
          std::vector<tir::Stmt> body_phases;
          std::vector<tir::Buffer> cross_phase_local_buffers;
          int64_t loop_step = 0;
          return CanSplitThreadInvariantSerialLoopAtSharedSync(
                     lifted_loop.as<tir::ForNode>(), thread_iter_var, &body_phases,
                     &cross_phase_local_buffers) ||
                 CanSplitWarpSynchronousSerialLoopAtSharedSync(
                     lifted_loop.as<tir::ForNode>(), thread_iter_var, &body_phases,
                     &loop_step, &cross_phase_local_buffers);
        }
      }
    }
    if (const auto* attr = stmt.as<tir::AttrStmtNode>()) {
      if (attr->attr_key == tir::attr::tilelang_assume) {
        return HasNestedThreadLaunchSplitLoop(attr->body, thread_iter_var);
      }
      if (attr->attr_key != tir::attr::thread_extent) {
        return false;
      }
      const auto* unit_iter_var = attr->node.as<tir::IterVarNode>();
      if (unit_iter_var == nullptr || !IsThreadLaunchIterVar(unit_iter_var)) {
        return false;
      }
      std::optional<int64_t> static_extent = GetOptionalStaticInt(attr->value);
      return static_extent.has_value() && static_extent.value() == 1 &&
             HasNestedThreadLaunchSplitLoop(attr->body, thread_iter_var);
    }
    if (const auto* let = stmt.as<tir::LetStmtNode>()) {
      return HasNestedThreadLaunchSplitLoop(let->body, thread_iter_var);
    }
    if (const auto* loop = stmt.as<tir::ForNode>()) {
      std::vector<tir::Stmt> body_phases;
      int64_t loop_step = 0;
      std::vector<tir::Buffer> cross_phase_local_buffers;
      return CanSplitThreadInvariantSerialLoopAtSharedSync(
                 loop, thread_iter_var, &body_phases, &cross_phase_local_buffers) ||
             CanSplitWarpSynchronousSerialLoopAtSharedSync(
                 loop, thread_iter_var, &body_phases, &loop_step,
                 &cross_phase_local_buffers);
    }
    if (const auto* realize = stmt.as<tir::BlockRealizeNode>()) {
      const tir::BlockNode* block = realize->block.as<tir::BlockNode>();
      if (block == nullptr || block->init.defined() || !block->match_buffers.empty() ||
          !tir::is_one(realize->predicate)) {
        return false;
      }
      return HasNestedThreadLaunchSplitLoop(block->body, thread_iter_var);
    }
    if (const auto* if_node = stmt.as<tir::IfThenElseNode>()) {
      return HasNestedThreadLaunchSplitLoop(if_node->then_case, thread_iter_var) ||
             (if_node->else_case.defined() &&
              HasNestedThreadLaunchSplitLoop(if_node->else_case.value(),
                                             thread_iter_var));
    }
    if (const auto* seq = stmt.as<tir::SeqStmtNode>()) {
      for (const tir::Stmt& child : seq->seq) {
        if (HasNestedThreadLaunchSplitLoop(child, thread_iter_var)) {
          return true;
        }
      }
    }
    return false;
  }

  void EmitWarpSynchronousSerialLoopWithThreadLaunchPhases(
      const tir::ForNode* loop, const std::vector<tir::Stmt>& body_phases, int64_t loop_step,
      const tir::IterVarNode* thread_iter_var, const PrimExpr& thread_extent,
      llvm::ArrayRef<ThreadLocalBlockAllocBinding> thread_local_bindings,
      llvm::ArrayRef<const tir::LetStmtNode*> let_bindings) {
    ICHECK(loop != nullptr);
    ICHECK_GT(loop_step, 0);
    mlir::Value zero = ZeroIndex();
    mlir::Value warp_size = ConstantIntLike(32, builder_.getIndexType());
    mlir::Value one = ConstantIntLike(1, builder_.getIndexType());
    mlir::Value thread_upper = AsIndex(VisitExpr(thread_extent), thread_extent.dtype());

    mlir::scf::ForOp warp_loop =
        builder_.create<mlir::scf::ForOp>(loc_, zero, thread_upper, warp_size);
    mlir::OpBuilder::InsertionGuard warp_guard(builder_);
    builder_.setInsertionPoint(warp_loop.getBody()->getTerminator());
    mlir::Value warp_base = warp_loop.getInductionVar();

    mlir::Type index_type = builder_.getIndexType();
    mlir::Value max_rounds = zero;
    for (int lane = 0; lane < 32; ++lane) {
      mlir::Value lane_offset = ConstantIntLike(lane, index_type);
      mlir::Value thread_value =
          builder_.create<mlir::arith::AddIOp>(loc_, warp_base, lane_offset);
      mlir::Value lane_valid = builder_.create<mlir::arith::CmpIOp>(
          loc_, mlir::arith::CmpIPredicate::ult, thread_value, thread_upper);
      mlir::MemRefType lane_round_slot_type = mlir::MemRefType::get({1}, index_type);
      mlir::Value lane_round_slot =
          builder_.create<mlir::memref::AllocaOp>(loc_, lane_round_slot_type);
      llvm::SmallVector<mlir::Value, 1> lane_round_slot_indices{zero};
      builder_.create<mlir::memref::StoreOp>(loc_, zero, lane_round_slot,
                                             lane_round_slot_indices);
      mlir::scf::IfOp lane_round_if = builder_.create<mlir::scf::IfOp>(loc_, lane_valid, false);
      {
        mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPoint(lane_round_if.thenYield());
      mlir::Value lane_rounds = zero;
      EmitWithBoundThreadLaunchValue(
          thread_iter_var, thread_extent, thread_value, thread_local_bindings, true, [&]() {
            EmitWithLetBindings(let_bindings, [&]() {
              mlir::Value lane_extent =
                  AsIndex(VisitExpr(loop->extent), loop->extent.dtype());
              lane_rounds = ComputePositiveCeilDivIndex(lane_extent, loop_step);
            });
          });
        builder_.create<mlir::memref::StoreOp>(loc_, lane_rounds, lane_round_slot,
                                               lane_round_slot_indices);
      }
      mlir::Value lane_rounds =
          builder_.create<mlir::memref::LoadOp>(loc_, lane_round_slot, lane_round_slot_indices);
      mlir::Value round_gt = builder_.create<mlir::arith::CmpIOp>(
          loc_, mlir::arith::CmpIPredicate::ugt, lane_rounds, max_rounds);
      max_rounds = builder_.create<mlir::arith::SelectOp>(loc_, round_gt, lane_rounds, max_rounds);
    }

    mlir::scf::ForOp round_loop = builder_.create<mlir::scf::ForOp>(loc_, zero, max_rounds, one);
    mlir::OpBuilder::InsertionGuard round_guard(builder_);
    builder_.setInsertionPoint(round_loop.getBody()->getTerminator());
    mlir::Value round_iv = round_loop.getInductionVar();
    mlir::Value round_step =
        builder_.create<mlir::arith::MulIOp>(loc_, round_iv, ConstantIntLike(loop_step, index_type));

    for (const tir::Stmt& phase : body_phases) {
      for (int lane = 0; lane < 32; ++lane) {
        mlir::Value lane_offset = ConstantIntLike(lane, index_type);
        mlir::Value thread_value =
            builder_.create<mlir::arith::AddIOp>(loc_, warp_base, lane_offset);
        mlir::Value lane_valid = builder_.create<mlir::arith::CmpIOp>(
            loc_, mlir::arith::CmpIPredicate::ult, thread_value, thread_upper);
        mlir::scf::IfOp lane_if = builder_.create<mlir::scf::IfOp>(loc_, lane_valid, false);
        mlir::OpBuilder::InsertionGuard lane_guard(builder_);
        builder_.setInsertionPoint(lane_if.thenYield());
        EmitWithBoundThreadLaunchValue(
            thread_iter_var, thread_extent, thread_value, thread_local_bindings, true, [&]() {
              EmitWithLetBindings(let_bindings, [&]() {
                mlir::Value lane_extent =
                    AsIndex(VisitExpr(loop->extent), loop->extent.dtype());
                mlir::Value lane_active = builder_.create<mlir::arith::CmpIOp>(
                    loc_, mlir::arith::CmpIPredicate::ult, round_step, lane_extent);
                mlir::scf::IfOp active_if =
                    builder_.create<mlir::scf::IfOp>(loc_, lane_active, false);
                mlir::OpBuilder::InsertionGuard active_guard(builder_);
                builder_.setInsertionPoint(active_if.thenYield());
                mlir::Value loop_lower = AsIndex(VisitExpr(loop->min), loop->min.dtype());
                mlir::Value loop_iter =
                    builder_.create<mlir::arith::AddIOp>(loc_, loop_lower, round_step);
                EmitLoopBodyWithBindings(loop, loop_iter, [&]() { VisitStmt(phase); });
              });
            });
      }
    }
  }

  template <typename PhaseEmitter>
  bool TryEmitThreadLaunchNestedSerialLoopPhases(
      const tir::Stmt& stmt, const tir::IterVarNode* thread_iter_var,
      const PrimExpr& thread_extent, PhaseEmitter&& phase_emitter,
      llvm::ArrayRef<ThreadLocalBlockAllocBinding> existing_thread_local_bindings = {},
      llvm::ArrayRef<const tir::LetStmtNode*> active_let_bindings = {}) {
    if (const auto* attr = stmt.as<tir::AttrStmtNode>()) {
      if (attr->attr_key == tir::attr::tilelang_assume) {
        return TryEmitThreadLaunchNestedSerialLoopPhases(attr->body, thread_iter_var,
                                                         thread_extent, phase_emitter,
                                                         existing_thread_local_bindings,
                                                         active_let_bindings);
      }
      if (attr->attr_key != tir::attr::thread_extent) {
        return false;
      }
      const auto* unit_iter_var = attr->node.as<tir::IterVarNode>();
      if (unit_iter_var == nullptr || !IsThreadLaunchIterVar(unit_iter_var)) {
        return false;
      }
      std::optional<int64_t> static_extent = GetOptionalStaticInt(attr->value);
      if (!static_extent.has_value() || static_extent.value() != 1) {
        return false;
      }
      SavedBinding saved = SaveAndSet(
          scalar_values_, unit_iter_var->var.get(),
          ConstantIntLike(0, LowerScalarType(unit_iter_var->var.dtype())));
      bool emitted = TryEmitThreadLaunchNestedSerialLoopPhases(attr->body, thread_iter_var,
                                                               thread_extent, phase_emitter,
                                                               existing_thread_local_bindings,
                                                               active_let_bindings);
      RestoreBinding(scalar_values_, unit_iter_var->var.get(), saved);
      return emitted;
    }
    if (const auto* let = stmt.as<tir::LetStmtNode>()) {
      llvm::SmallVector<const tir::LetStmtNode*, 4> next_let_bindings(
          active_let_bindings.begin(), active_let_bindings.end());
      next_let_bindings.push_back(let);
      return TryEmitThreadLaunchNestedSerialLoopPhases(let->body, thread_iter_var,
                                                       thread_extent, phase_emitter,
                                                       existing_thread_local_bindings,
                                                       next_let_bindings);
    }
    if (const auto* if_node = stmt.as<tir::IfThenElseNode>()) {
      if (ExprUsesVar(if_node->condition, thread_iter_var->var)) {
        const auto* common_leaf_loop = MatchCommonLeafThreadLaunchLoop(stmt);
        if (common_leaf_loop != nullptr) {
          tir::For lifted_loop = LiftCommonLeafThreadLaunchLoop(stmt, common_leaf_loop);
          return TryEmitThreadLaunchNestedSerialLoopPhases(
              lifted_loop, thread_iter_var, thread_extent, phase_emitter,
              existing_thread_local_bindings, active_let_bindings);
        }
      }
    }
    if (const auto* loop = stmt.as<tir::ForNode>()) {
      std::vector<tir::Stmt> body_phases;
      std::vector<tir::Buffer> cross_phase_local_buffers;
      if (CanSplitThreadInvariantSerialLoopAtSharedSync(
              loop, thread_iter_var, &body_phases, &cross_phase_local_buffers)) {
        std::vector<ThreadLocalBlockAllocBinding> thread_local_bindings(
            existing_thread_local_bindings.begin(), existing_thread_local_bindings.end());
        std::vector<tir::Buffer> new_cross_phase_local_buffers;
        for (const tir::Buffer& buffer : cross_phase_local_buffers) {
          if (!ThreadLocalBindingsContainBuffer(existing_thread_local_bindings, buffer)) {
            new_cross_phase_local_buffers.push_back(buffer);
          }
        }
        std::vector<ThreadLocalBlockAllocBinding> new_thread_local_bindings =
            CreateThreadLocalBlockAllocBackings(new_cross_phase_local_buffers, thread_iter_var,
                                                thread_extent);
        thread_local_bindings.insert(thread_local_bindings.end(),
                                     new_thread_local_bindings.begin(),
                                     new_thread_local_bindings.end());
        EmitThreadInvariantSerialLoopWithThreadLaunchPhases(
            loop, body_phases, thread_iter_var, thread_extent, thread_local_bindings,
            active_let_bindings, phase_emitter);
        return true;
      }
      int64_t loop_step = 0;
      cross_phase_local_buffers.clear();
      if (CanSplitWarpSynchronousSerialLoopAtSharedSync(
              loop, thread_iter_var, &body_phases, &loop_step,
              &cross_phase_local_buffers)) {
        std::vector<ThreadLocalBlockAllocBinding> thread_local_bindings(
            existing_thread_local_bindings.begin(), existing_thread_local_bindings.end());
        std::vector<tir::Buffer> new_cross_phase_local_buffers;
        for (const tir::Buffer& buffer : cross_phase_local_buffers) {
          if (!ThreadLocalBindingsContainBuffer(existing_thread_local_bindings, buffer)) {
            new_cross_phase_local_buffers.push_back(buffer);
          }
        }
        std::vector<ThreadLocalBlockAllocBinding> new_thread_local_bindings =
            CreateThreadLocalBlockAllocBackings(new_cross_phase_local_buffers, thread_iter_var,
                                                thread_extent);
        thread_local_bindings.insert(thread_local_bindings.end(),
                                     new_thread_local_bindings.begin(),
                                     new_thread_local_bindings.end());
        EmitWarpSynchronousSerialLoopWithThreadLaunchPhases(
            loop, body_phases, loop_step, thread_iter_var, thread_extent,
            thread_local_bindings, active_let_bindings);
        return true;
      }
      return false;
    }
    if (const auto* allocate = stmt.as<tir::AllocateNode>()) {
      if (!HasNestedThreadLaunchSplitLoop(allocate->body, thread_iter_var)) {
        return false;
      }
      tir::Buffer owner_buffer(allocate->buffer_var, allocate->dtype, allocate->extents,
                               ffi::Array<PrimExpr>(), PrimExpr(),
                               allocate->buffer_var->name_hint, runtime::kAllocAlignment, 1,
                               tir::BufferType::kDefault);
      if (tl::IsSharedBuffer(owner_buffer)) {
        return false;
      }
      std::vector<ThreadLocalBlockAllocBinding> thread_local_bindings(
          existing_thread_local_bindings.begin(), existing_thread_local_bindings.end());
      if (!ThreadLocalBindingsContainBuffer(existing_thread_local_bindings, owner_buffer)) {
        std::vector<ThreadLocalBlockAllocBinding> new_thread_local_bindings =
            CreateThreadLocalBlockAllocBackings({owner_buffer}, thread_iter_var, thread_extent);
        thread_local_bindings.insert(thread_local_bindings.end(),
                                     new_thread_local_bindings.begin(),
                                     new_thread_local_bindings.end());
      }
      EmitConditionalRegion(allocate->condition, [&]() {
        TryEmitThreadLaunchNestedSerialLoopPhases(
            allocate->body, thread_iter_var, thread_extent, phase_emitter,
            thread_local_bindings, active_let_bindings);
      });
      return true;
    }
    if (const auto* seq = stmt.as<tir::SeqStmtNode>()) {
      bool found_nested_split_loop = false;
      for (const tir::Stmt& child : seq->seq) {
        if (IsPhaseBoundarySyncStmt(child)) {
          return false;
        }
        if (HasNestedThreadLaunchSplitLoop(child, thread_iter_var)) {
          found_nested_split_loop = true;
        }
      }
      if (!found_nested_split_loop) {
        return false;
      }
      for (const tir::Stmt& child : seq->seq) {
        if (TryEmitThreadLaunchNestedSerialLoopPhases(
                child, thread_iter_var, thread_extent, phase_emitter,
                existing_thread_local_bindings, active_let_bindings)) {
          continue;
        }
        EmitPerThreadLaunchPhase(
            child, thread_iter_var, thread_extent, existing_thread_local_bindings,
            [&](const tir::Stmt& phase, mlir::Value induction_var) {
              EmitWithLetBindings(active_let_bindings,
                                  [&]() { phase_emitter(phase, induction_var); });
            });
      }
      return true;
    }
    const auto* realize = stmt.as<tir::BlockRealizeNode>();
    if (realize == nullptr) {
      return false;
    }
    const tir::BlockNode* block = realize->block.as<tir::BlockNode>();
    if (block == nullptr || block->init.defined() || !block->match_buffers.empty() ||
        !tir::is_one(realize->predicate)) {
      return false;
    }
    std::vector<tir::Stmt> root_children;
    if (const auto* seq = block->body.as<tir::SeqStmtNode>()) {
      root_children.assign(seq->seq.begin(), seq->seq.end());
    } else {
      root_children.push_back(block->body);
    }
    bool found_nested_split_loop = false;
    for (const tir::Stmt& child : root_children) {
      if (IsPhaseBoundarySyncStmt(child)) {
        return false;
      }
      if (HasNestedThreadLaunchSplitLoop(child, thread_iter_var)) {
        found_nested_split_loop = true;
      }
    }
    if (!found_nested_split_loop) {
      return false;
    }

    std::vector<tir::Buffer> shared_buffers;
    std::vector<tir::Buffer> local_buffers;
    for (const tir::Buffer& buffer : block->alloc_buffers) {
      if (tl::IsSharedBuffer(buffer)) {
        if (!BufferIsBound(buffer)) {
          shared_buffers.push_back(buffer);
        }
      } else {
        if (!ThreadLocalBindingsContainBuffer(existing_thread_local_bindings, buffer)) {
          local_buffers.push_back(buffer);
        }
      }
    }

    ScopedBufferBindings saved_shared_bindings;
    BindSharedBlockAllocBuffersForThreadLaunch(shared_buffers, thread_iter_var,
                                               &saved_shared_bindings);
    std::vector<ThreadLocalBlockAllocBinding> thread_local_bindings(
        existing_thread_local_bindings.begin(), existing_thread_local_bindings.end());
    std::vector<ThreadLocalBlockAllocBinding> new_thread_local_bindings =
        CreateThreadLocalBlockAllocBackings(local_buffers, thread_iter_var, thread_extent);
    thread_local_bindings.insert(thread_local_bindings.end(), new_thread_local_bindings.begin(),
                                 new_thread_local_bindings.end());

    for (const tir::Stmt& child : root_children) {
      if (TryEmitThreadLaunchNestedSerialLoopPhases(child, thread_iter_var, thread_extent,
                                                    phase_emitter, thread_local_bindings,
                                                    active_let_bindings)) {
        continue;
      }
      EmitPerThreadLaunchPhase(
          child, thread_iter_var, thread_extent, thread_local_bindings,
          [&](const tir::Stmt& phase, mlir::Value induction_var) {
            EmitWithLetBindings(active_let_bindings,
                                [&]() { phase_emitter(phase, induction_var); });
          });
    }

    RestoreScopedBufferBindings(&saved_shared_bindings);
    return true;
  }

  template <typename PhaseEmitter>
  void EmitSplitThreadLaunchPhases(
      const tir::IterVarNode* iter_var, const PrimExpr& extent,
      const tir::Stmt& original_body, const std::vector<tir::Stmt>& phases,
      llvm::ArrayRef<tir::Buffer> cross_phase_local_buffers, PhaseEmitter&& phase_emitter) {
    std::vector<tir::Buffer> shared_buffers;
    CollectSharedBlockAllocBuffers(original_body, &shared_buffers);
    ScopedBufferBindings saved_shared_bindings;
    BindSharedBlockAllocBuffersForThreadLaunch(shared_buffers, iter_var,
                                               &saved_shared_bindings);
    std::vector<tir::Buffer> thread_local_storage_buffers(cross_phase_local_buffers.begin(),
                                                          cross_phase_local_buffers.end());
    std::vector<tir::Buffer> local_var_buffers;
    CollectLocalVarBlockAllocBuffers(original_body, &local_var_buffers);
    for (const tir::Buffer& buffer : local_var_buffers) {
      AppendUniqueBuffer(buffer, &thread_local_storage_buffers);
    }
    std::vector<ThreadLocalBlockAllocBinding> thread_local_bindings =
        CreateThreadLocalBlockAllocBackings(thread_local_storage_buffers, iter_var, extent);
    for (const tir::Stmt& phase : phases) {
      if (IsThreadLaunchPhaseGlobalTileCumsumStmt(phase)) {
        VisitStmt(phase);
        continue;
      }
      if (TryEmitThreadLaunchNestedSerialLoopPhases(phase, iter_var, extent, phase_emitter,
                                                    thread_local_bindings)) {
        continue;
      }
      EmitPerThreadLaunchPhase(phase, iter_var, extent, thread_local_bindings, phase_emitter);
    }
    RestoreScopedBufferBindings(&saved_shared_bindings);
  }

  void VisitStmt_(const tir::ForNode* op) final {
    if (op->thread_binding.defined()) {
      const auto* iter_var = op->thread_binding.as<tir::IterVarNode>();
      ICHECK(iter_var != nullptr)
          << "thread_binding is expected to bind an IterVar in riscv lowering";
      const bool is_thread_launch = IsThreadLaunchIterVar(iter_var);
      const bool is_block_launch = IsBlockLaunchIterVar(iter_var);
      ICHECK(is_thread_launch || is_block_launch)
          << "Only threadIdx.* and blockIdx.* bindings are supported in riscv lowering";
      ICHECK(tir::is_zero(op->min))
          << "Thread binding loops must start at zero in riscv lowering";
      ICHECK(op->step.defined() ? tir::is_one(op->step.value()) : true)
          << "Thread binding loops must have unit step in riscv lowering";
      std::optional<int64_t> static_extent = GetOptionalStaticInt(op->extent);
      if (is_thread_launch && (!static_extent.has_value() || static_extent.value() != 1)) {
        if (ShouldCollapseThreadInvariantLaunchBody(op->body, iter_var->var)) {
          EmitSingleExecutionThreadLaunch(iter_var, op->extent, op->body,
                                          [&](mlir::Value induction_var) {
                                            LowerLoopBody(op, induction_var);
                                          });
          return;
        }
        std::vector<tir::Stmt> phases;
        std::vector<tir::Buffer> cross_phase_local_buffers;
        std::string split_unsupported_reason;
        if (SplitAtSharedStorageSync(op->body, &phases, &split_unsupported_reason,
                                     &cross_phase_local_buffers, &iter_var->var)) {
          if (phases.empty()) {
            return;
          }
          if (phases.size() > 1) {
            EmitSplitThreadLaunchPhases(
                iter_var, op->extent, op->body, phases, cross_phase_local_buffers,
                [&](const tir::Stmt& phase, mlir::Value induction_var) {
                  LowerLoopBodyStmt(op, phase, induction_var);
                });
            return;
          }
          if (TryEmitThreadLaunchNestedSerialLoopPhases(
                  op->body, iter_var, op->extent,
                  [&](const tir::Stmt& stmt, mlir::Value induction_var) {
                    LowerLoopBodyStmt(op, stmt, induction_var);
                  })) {
            return;
          }
          std::vector<tir::Buffer> shared_buffers;
          CollectSharedBlockAllocBuffers(op->body, &shared_buffers);
          if (!shared_buffers.empty()) {
            ScopedBufferBindings saved_shared_bindings;
            BindSharedBlockAllocBuffersForThreadLaunch(shared_buffers, iter_var,
                                                       &saved_shared_bindings);
            std::vector<tir::Buffer> local_var_buffers;
            CollectLocalVarBlockAllocBuffers(op->body, &local_var_buffers);
            std::vector<ThreadLocalBlockAllocBinding> thread_local_bindings =
                CreateThreadLocalBlockAllocBackings(local_var_buffers, iter_var, op->extent);
            EmitThreadLaunchBodyWithBindings(
                phases.front(), iter_var, op->extent, thread_local_bindings,
                [&](mlir::Value induction_var) {
                  LowerLoopBodyStmt(op, phases.front(), induction_var);
                },
                StmtUsesVar(op->body, iter_var->var), &phases.front());
            RestoreScopedBufferBindings(&saved_shared_bindings);
            return;
          }
        }
        ICHECK(split_unsupported_reason.empty()) << split_unsupported_reason;
      }
      if (is_thread_launch) {
        EmitThreadLaunchBodyWithBindings(
            op->body, iter_var, op->extent, {},
            [&](mlir::Value induction_var) { LowerLoopBody(op, induction_var); },
            StmtUsesVar(op->body, iter_var->var), &op->body);
      } else {
        EmitLaunchSerialLoop(iter_var, op->extent, LaunchTrackingKind::kBlock,
                             [&](mlir::Value induction_var) {
                               LowerLoopBody(op, induction_var);
                             });
      }
      return;
    }
    if (ShouldInlineStaticLoopForSerializedWarpReplay(op)) {
      EmitInlineStaticLoopForSerializedWarpReplay(op);
      return;
    }
    if (TryLowerReductionLoopNest(op)) {
      return;
    }
    if (TryLowerElementwiseLoopNest(op)) {
      return;
    }

    ICHECK(IsSupportedGeneralLoopKind(op->kind))
        << "Only serial/unrolled/parallel/vectorized loops are supported in the current riscv lowering";
    const bool has_direct_break = ContainsDirectLoopBreak(op->body);
    ICHECK(!(has_direct_break && op->kind == tir::ForKind::kParallel))
        << "loop_break inside T.parallel is not supported in riscv lowering";
    if (has_direct_break) {
      mlir::Value break_flag = CreateStaticAlloca({1}, DataType::Bool());
      StoreBreakLoopFlag(break_flag, false);
      break_loop_stack_.push_back(BreakLoopFrame{break_flag});
    }
    mlir::Value lower = AsIndex(VisitExpr(op->min), op->min.dtype());
    mlir::Value extent = AsIndex(VisitExpr(op->extent), op->extent.dtype());
    mlir::Value step =
        op->step.defined() ? AsIndex(VisitExpr(op->step.value()), op->step.value().dtype())
                           : ConstantIntLike(1, builder_.getIndexType());
    mlir::Value upper = builder_.create<mlir::arith::AddIOp>(loc_, lower, extent);

    if (op->kind == tir::ForKind::kParallel) {
      mlir::scf::ParallelOp parallel_op = builder_.create<mlir::scf::ParallelOp>(
          loc_, mlir::ValueRange{lower}, mlir::ValueRange{upper}, mlir::ValueRange{step});
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPoint(parallel_op.getBody()->getTerminator());
      LowerLoopBody(op, parallel_op.getInductionVars()[0]);
      if (has_direct_break) {
        break_loop_stack_.pop_back();
      }
      return;
    }

    mlir::scf::ForOp for_op = builder_.create<mlir::scf::ForOp>(loc_, lower, upper, step);
    mlir::OpBuilder::InsertionGuard guard(builder_);
    builder_.setInsertionPoint(for_op.getBody()->getTerminator());
    if (has_direct_break) {
      EmitIfCurrentLoopNotBroken([&]() { LowerLoopBody(op, for_op.getInductionVar()); });
      break_loop_stack_.pop_back();
      return;
    }
    LowerLoopBody(op, for_op.getInductionVar());
  }

  void VisitStmt_(const tir::IfThenElseNode* op) final {
    if (InLogicalThreadRegion() && IsSimpleThreadReturnIf(op)) {
      return;
    }
    mlir::Value cond = LowerCondition(op->condition);
    bool has_else = op->else_case.defined();
    PrimExprMap saved_replay_exprs = serialized_warp_replay_buffer_exprs_;
    SerializedWarpReplayBufferElementExprMap saved_replay_element_exprs =
        serialized_warp_replay_buffer_element_exprs_;
    mlir::scf::IfOp if_op = builder_.create<mlir::scf::IfOp>(loc_, cond, has_else);

    {
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPoint(if_op.thenYield());
      VisitStmt(op->then_case);
    }

    if (has_else) {
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPoint(if_op.elseYield());
      VisitStmt(op->else_case.value());
      serialized_warp_replay_buffer_exprs_ = std::move(saved_replay_exprs);
      serialized_warp_replay_buffer_element_exprs_ = std::move(saved_replay_element_exprs);
    }
  }

  void VisitStmt_(const tir::BufferStoreNode* op) final {
    if (std::optional<VectorizedRampAccess> ramp_access =
            MatchVectorizedRampAccess(op->buffer, op->indices, op->value.dtype())) {
      if (CanTrackSerializedWarpReplayBufferElements(op->buffer)) {
        ClearSerializedWarpReplayTrackedBufferElementExprs(op->buffer);
      }
      if (IsSerializedWarpReplayTrackedScalarBuffer(op->buffer)) {
        ClearSerializedWarpReplayTrackedBufferExpr(op->buffer);
      }
      auto emit_ramp_store = [&]() {
        mlir::Value value = VisitExpr(op->value);
        LowerRampBufferStore(op->buffer, op->indices, ramp_access.value(), value,
                             op->value.dtype());
      };
      if (op->predicate.defined()) {
        EmitConditionalRegion(op->predicate.value(), emit_ramp_store);
      } else {
        emit_ramp_store();
      }
      return;
    }

    llvm::SmallVector<mlir::Value, 4> indices;
    indices.reserve(op->indices.size());
    for (const PrimExpr& index : op->indices) {
      indices.push_back(AsIndex(VisitExpr(index), index.dtype()));
    }

    std::optional<int64_t> tracked_linear_index;
    bool track_replay_expr = !op->predicate.defined() &&
                             (tracked_linear_index = GetSerializedWarpReplayStaticBufferLinearIndex(
                                  op->buffer, op->indices))
                                 .has_value();
    std::optional<PrimExpr> tracked_expr;
    if (track_replay_expr) {
      tracked_expr = ResolveSerializedWarpReplayExpr(op->value);
    }
    bool clear_replay_exprs = !track_replay_expr && CanTrackSerializedWarpReplayBufferElements(op->buffer);

    auto emit_store = [&]() {
      mlir::Value value = VisitExpr(op->value);
      if (auto packed_view = ResolvePackedScalarViewBinding(op->buffer)) {
        LowerPackedScalarViewStore(op->buffer, packed_view.value(), indices, value, op->value.dtype());
        return;
      }
      mlir::Value memref = LookupBufferValue(op->buffer);
      value = CastValue(value, op->value.dtype(), op->buffer->dtype);
      builder_.create<mlir::memref::StoreOp>(loc_, value, memref, indices);
    };

    if (op->predicate.defined()) {
      EmitConditionalRegion(op->predicate.value(), emit_store);
    } else {
      emit_store();
    }

    if (track_replay_expr) {
      ICHECK(tracked_expr.has_value());
      if (IsSerializedWarpReplayTrackedScalarBuffer(op->buffer) &&
          tracked_linear_index.value() == 0) {
        SetSerializedWarpReplayTrackedBufferExpr(op->buffer, tracked_expr.value());
      } else {
        SetSerializedWarpReplayTrackedBufferElementExpr(op->buffer, tracked_linear_index.value(),
                                                        tracked_expr.value());
      }
    } else if (clear_replay_exprs) {
      ClearSerializedWarpReplayTrackedBufferElementExprs(op->buffer);
      if (IsSerializedWarpReplayTrackedScalarBuffer(op->buffer)) {
        ClearSerializedWarpReplayTrackedBufferExpr(op->buffer);
      }
    }
  }

  void VisitStmt_(const tir::DeclBufferNode* op) final {
    std::vector<std::pair<const Object*, SavedBinding>> saved_bindings;
    std::vector<std::pair<const Object*, std::optional<tir::Buffer>>> saved_buffer_owners;
    std::optional<PackedScalarViewBinding> saved_packed_view;
    auto it = buffer_values_.find(op->buffer->data.get());
    ICHECK(it != buffer_values_.end())
        << "DeclBuffer lowered before its data binding was materialized: " << op->buffer->name;
    mlir::Value alias_value = it->second;
    if (auto packed_view = ResolvePackedScalarViewBinding(op->buffer)) {
      saved_packed_view = SaveAndBindPackedScalarView(op->buffer, packed_view->source_buffer);
    } else {
      alias_value = CreateBufferAliasView(op->buffer, it->second);
    }
    BindBufferAliases(op->buffer, alias_value, &saved_bindings, &saved_buffer_owners);
    VisitStmt(op->body);
    RestoreBindings(buffer_values_, saved_bindings);
    RestoreBufferOwnerBindings(saved_buffer_owners);
    RestorePackedScalarViewBinding(packed_scalar_view_bindings_, op->buffer.get(), saved_packed_view);
  }

  void VisitStmt_(const tir::AllocateNode* op) final {
    auto emit_body = [&]() {
      tir::Buffer owner_buffer(op->buffer_var, op->dtype, op->extents, ffi::Array<PrimExpr>(),
                               PrimExpr(), op->buffer_var->name_hint, runtime::kAllocAlignment, 1,
                               tir::BufferType::kDefault);
      ICHECK(!(InNonUnitLogicalThreadRegion() && tl::IsSharedBuffer(owner_buffer)))
          << "shared allocations inside non-unit thread launch are not supported yet in riscv lowering";
      std::vector<std::pair<const Object*, SavedBinding>> saved_bindings;
      std::vector<std::pair<const Object*, std::optional<tir::Buffer>>> saved_buffer_owners;
      std::optional<tir::Buffer> saved_owner;
      std::optional<tir::Buffer> saved_buffer_owner;
      mlir::Value alloc = CreateAlloca(op->extents, op->dtype);
      if (std::optional<PrimExpr> init =
              LookupLocalVarInitFromAnnotations(op->annotations, op->buffer_var,
                                                /*allow_plain_primexpr=*/true)) {
        InitializeAllocaFromExpr(alloc, op->extents, op->dtype, init.value());
      } else {
        MaybeInitializeAllocaFromLocalVarInit(alloc, op->buffer_var, op->extents, op->dtype);
      }
      saved_bindings.emplace_back(op->buffer_var.get(),
                                  SaveAndSet(buffer_values_, op->buffer_var.get(), alloc));
      saved_buffer_owner = SaveAndSetBufferOwner(op->buffer_var.get(), owner_buffer);
      if (op->dtype.lanes() > 1) {
        saved_owner = SaveAndSetPackedDataOwner(op->buffer_var.get(), owner_buffer);
      }
      VisitStmt(op->body);
      RestoreBindings(buffer_values_, saved_bindings);
      RestorePackedDataOwnerBinding(packed_data_owner_, op->buffer_var.get(), saved_owner);
      RestoreBufferOwnerBinding(buffer_owner_, op->buffer_var.get(), saved_buffer_owner);
    };
    EmitConditionalRegion(op->condition, emit_body);
  }

  void VisitStmt_(const tir::BufferRealizeNode* op) final {
    ValidateContiguousBuffer(op->buffer);
    ICHECK(!(InNonUnitLogicalThreadRegion() && tl::IsSharedBuffer(op->buffer)))
        << "shared buffer realize inside non-unit thread launch is not supported yet in riscv lowering";
    Array<PrimExpr> extents;
    for (const Range& range : op->bounds) {
      extents.push_back(range->extent);
    }

    std::vector<std::pair<const Object*, SavedBinding>> saved_bindings;
    std::vector<std::pair<const Object*, std::optional<tir::Buffer>>> saved_buffer_owners;
    mlir::Value alloc = CreateAlloca(extents, op->buffer->dtype);
    BindBufferAliases(op->buffer, alloc, &saved_bindings, &saved_buffer_owners);
    EmitConditionalRegion(op->condition, [&]() { VisitStmt(op->body); });
    RestoreBindings(buffer_values_, saved_bindings);
    RestoreBufferOwnerBindings(saved_buffer_owners);
  }

  void VisitStmt_(const tir::AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::tilelang_assume) {
      VisitStmt(op->body);
      return;
    }
    if (op->attr_key == tir::attr::thread_extent) {
      const auto* iter_var = op->node.as<tir::IterVarNode>();
      ICHECK(iter_var != nullptr)
          << "thread_extent is expected to bind an IterVar in riscv lowering";
      std::optional<int64_t> static_extent = GetOptionalStaticInt(op->value);
      if (IsBlockLaunchIterVar(iter_var)) {
        mlir::Type block_type = LowerScalarType(iter_var->var.dtype());
        if (static_extent.has_value() && static_extent.value() == 1) {
          SavedBinding saved =
              SaveAndSet(scalar_values_, iter_var->var.get(), ConstantIntLike(0, block_type));
          VisitStmt(op->body);
          RestoreBinding(scalar_values_, iter_var->var.get(), saved);
          return;
        }
        EmitLaunchSerialLoop(iter_var, op->value, LaunchTrackingKind::kBlock,
                             [&](mlir::Value) { VisitStmt(op->body); }, false, &op->body);
        return;
      }
      ICHECK(IsThreadLaunchIterVar(iter_var))
          << "Only threadIdx.* and blockIdx.* extents are supported in riscv lowering";
      if (!static_extent.has_value() || static_extent.value() != 1) {
        if (ShouldCollapseThreadInvariantLaunchBody(op->body, iter_var->var)) {
          EmitSingleExecutionThreadLaunch(
              iter_var, op->value, op->body, [&](mlir::Value) { VisitStmt(op->body); });
          return;
        }
        std::vector<tir::Stmt> phases;
        std::vector<tir::Buffer> cross_phase_local_buffers;
        std::string split_unsupported_reason;
        if (SplitAtSharedStorageSync(op->body, &phases, &split_unsupported_reason,
                                     &cross_phase_local_buffers, &iter_var->var)) {
          if (phases.empty()) {
            return;
          }
          if (phases.size() > 1) {
            EmitSplitThreadLaunchPhases(
                iter_var, op->value, op->body, phases, cross_phase_local_buffers,
                [&](const tir::Stmt& phase, mlir::Value) { VisitStmt(phase); });
            return;
          }
          if (TryEmitThreadLaunchNestedSerialLoopPhases(
                  op->body, iter_var, op->value,
                  [&](const tir::Stmt& stmt, mlir::Value) { VisitStmt(stmt); })) {
            return;
          }
          std::vector<tir::Buffer> shared_buffers;
          CollectSharedBlockAllocBuffers(op->body, &shared_buffers);
          if (!shared_buffers.empty()) {
            ScopedBufferBindings saved_shared_bindings;
            BindSharedBlockAllocBuffersForThreadLaunch(shared_buffers, iter_var,
                                                       &saved_shared_bindings);
            std::vector<tir::Buffer> local_var_buffers;
            CollectLocalVarBlockAllocBuffers(op->body, &local_var_buffers);
            std::vector<ThreadLocalBlockAllocBinding> thread_local_bindings =
                CreateThreadLocalBlockAllocBackings(local_var_buffers, iter_var, op->value);
            EmitThreadLaunchBodyWithBindings(
                phases.front(), iter_var, op->value, thread_local_bindings,
                [&](mlir::Value) { VisitStmt(phases.front()); },
                StmtUsesVar(op->body, iter_var->var), &phases.front());
            RestoreScopedBufferBindings(&saved_shared_bindings);
            return;
          }
        }
        ICHECK(split_unsupported_reason.empty()) << split_unsupported_reason;
      }
      EmitThreadLaunchBodyWithBindings(
          op->body, iter_var, op->value, {}, [&](mlir::Value) { VisitStmt(op->body); },
          StmtUsesVar(op->body, iter_var->var), &op->body);
      return;
    }
    VisitStmt(op->body);
  }

  void VisitStmt_(const tir::LetStmtNode* op) final {
    if (TryEmitPointerBackedHandleLet(op, [&]() { VisitStmt(op->body); })) {
      return;
    }
    mlir::Value value = VisitExpr(op->value);
    SavedBinding saved = SaveAndSet(scalar_values_, op->var.get(), value);
    std::optional<PrimExpr> saved_expr =
        SaveAndSetPrimExpr(bound_prim_exprs_, op->var.get(), op->value);
    VisitStmt(op->body);
    RestorePrimExprBinding(bound_prim_exprs_, op->var.get(), saved_expr);
    RestoreBinding(scalar_values_, op->var.get(), saved);
  }

  void VisitStmt_(const tir::BlockNode* op) final {
    ICHECK(!op->init.defined()) << "Reduction blocks are not supported yet in riscv lowering";

    if (InNonUnitLogicalThreadRegion()) {
      for (const tir::Buffer& buffer : op->alloc_buffers) {
        if (tl::IsSharedBuffer(buffer) && !BufferIsBound(buffer)) {
          LOG(FATAL) << "shared block allocation '" << buffer->name
                     << "' inside non-unit thread launch is not supported without "
                     << "serialized phase prebinding in riscv lowering. Add a "
                     << "supported sync_threads/sync_warp phase boundary, make the launch "
                     << "thread-invariant, or keep the buffer local/private.";
        }
        ICHECK(!IsPreboundThreadLocalBlockBuffer(buffer) || BufferIsBound(buffer))
            << "prebound thread-local block allocation is missing its phase subview in "
            << "riscv lowering: " << buffer->name;
      }
    }
    std::unordered_set<const Object*> cooperative_replay_buffers;
    if (InSingleNonUnitThreadIdxXRegion()) {
      cooperative_replay_buffers = CollectCooperativeReplayOperandBuffers(op->body);
    }

    const auto* body_for = FindDeferredBindingLoop(op->body);
    std::vector<std::pair<const Object*, SavedBinding>> saved_bindings;
    std::vector<std::pair<const Object*, std::optional<tir::Buffer>>> saved_buffer_owners;
    std::vector<std::pair<const Object*, std::optional<tir::Buffer>>> saved_packed_owners;
    std::vector<const Object*> scoped_thread_local_keys;
    size_t scoped_thread_local_binding_base =
        active_thread_local_bindings_stack_.empty() ? 0
                                                    : active_thread_local_bindings_stack_.back().size();
    DeferredLoopBindings deferred;
    saved_bindings.reserve(op->alloc_buffers.size() * 2 + op->match_buffers.size() * 2);
    saved_packed_owners.reserve(op->alloc_buffers.size());
    for (const tir::Buffer& buffer : op->alloc_buffers) {
      if (InNonUnitLogicalThreadRegion() &&
          (tl::IsSharedBuffer(buffer) || IsPreboundThreadLocalBlockBuffer(buffer))) {
        mlir::Value alloc = LookupBufferValue(buffer);
        BindBufferAliases(buffer, alloc, &saved_bindings, &saved_buffer_owners);
        if (buffer->dtype.lanes() > 1) {
          saved_packed_owners.emplace_back(buffer->data.get(),
                                           SaveAndSetPackedDataOwner(buffer->data.get(), buffer));
        }
        continue;
      }
      if (body_for != nullptr && scalar_values_.count(body_for->loop_var.get()) == 0 &&
          BufferUsesVar(buffer, body_for->loop_var)) {
        deferred.alloc_buffers.push_back(buffer);
        continue;
      }
      if (InSingleNonUnitThreadIdxXRegion() && buffer.defined() &&
          buffer.scope() == "local.var" &&
          CanMaterializeThreadLocalBlockAllocForSerializedWarpReplay(buffer)) {
        const ThreadLaunchFrame* thread_idx_x = CurrentThreadIdxXFrame();
        ICHECK(thread_idx_x != nullptr && thread_idx_x->extent.has_value())
            << "local.var backing requires a static threadIdx.x launch";
        ICHECK(!active_thread_local_bindings_stack_.empty())
            << "local.var backing requires an active thread-local binding scope";
        Array<PrimExpr> backing_shape;
        backing_shape.push_back(IntImm(DataType::Int(32), thread_idx_x->extent.value()));
        if (!IsScalarLocalVarBuffer(buffer)) {
          for (const PrimExpr& dim : buffer->shape) {
            backing_shape.push_back(dim);
          }
        }
        ThreadLocalBlockAllocBinding binding{buffer, CreateAlloca(backing_shape, buffer->dtype)};
        if (std::optional<PrimExpr> init =
                LookupLocalVarInitFromAnnotations(op->annotations, buffer->data)) {
          InitializeThreadLocalBackingFromExpr(
              binding.backing, buffer, backing_shape[0], init);
        } else if (std::optional<PrimExpr> init = LookupLocalVarInitForDataVar(buffer->data)) {
          InitializeThreadLocalBackingFromExpr(
              binding.backing, buffer, backing_shape[0], init);
        } else {
          InitializeThreadLocalBackingFromExpr(binding.backing, buffer, backing_shape[0],
                                              std::nullopt);
        }
        active_thread_local_bindings_stack_.back().push_back(binding);
        mlir::Value subview = CreateThreadLocalBlockAllocSubview(
            binding, CurrentThreadIdxXValue("local.var thread-local backing"));
        BindBufferAliases(buffer, subview, &saved_bindings, &saved_buffer_owners);
        ++prebound_thread_local_block_buffers_[buffer.get()];
        ++prebound_thread_local_block_buffers_[buffer->data.get()];
        scoped_thread_local_keys.push_back(buffer.get());
        scoped_thread_local_keys.push_back(buffer->data.get());
        continue;
      }
      if (InSingleNonUnitThreadIdxXRegion() &&
          (cooperative_replay_buffers.count(buffer.get()) != 0 ||
           cooperative_replay_buffers.count(buffer->data.get()) != 0) &&
          CanMaterializeThreadLocalBlockAllocForSerializedWarpReplay(buffer)) {
        const ThreadLaunchFrame* thread_idx_x = CurrentThreadIdxXFrame();
        ICHECK(thread_idx_x != nullptr && thread_idx_x->extent.has_value())
            << "thread-local replay backing requires a static threadIdx.x launch";
        ICHECK(!active_thread_local_bindings_stack_.empty())
            << "thread-local replay backing requires an active thread-local binding scope";
        Array<PrimExpr> backing_shape;
        backing_shape.push_back(IntImm(DataType::Int(32), thread_idx_x->extent.value()));
        if (!IsScalarLocalVarBuffer(buffer)) {
          for (const PrimExpr& dim : buffer->shape) {
            backing_shape.push_back(dim);
          }
        }
        ThreadLocalBlockAllocBinding binding{buffer, CreateAlloca(backing_shape, buffer->dtype)};
        if (std::optional<PrimExpr> init =
                LookupLocalVarInitFromAnnotations(op->annotations, buffer->data)) {
          InitializeThreadLocalBackingFromExpr(
              binding.backing, buffer, backing_shape[0], init);
        } else if (std::optional<PrimExpr> init = LookupLocalVarInitForDataVar(buffer->data)) {
          InitializeThreadLocalBackingFromExpr(
              binding.backing, buffer, backing_shape[0], init);
        } else {
          InitializeThreadLocalBackingFromExpr(binding.backing, buffer, backing_shape[0],
                                              std::nullopt);
        }
        active_thread_local_bindings_stack_.back().push_back(binding);
        mlir::Value subview = CreateThreadLocalBlockAllocSubview(
            binding, CurrentThreadIdxXValue("thread-local replay block allocation"));
        BindBufferAliases(buffer, subview, &saved_bindings, &saved_buffer_owners);
        ++prebound_thread_local_block_buffers_[buffer.get()];
        ++prebound_thread_local_block_buffers_[buffer->data.get()];
        scoped_thread_local_keys.push_back(buffer.get());
        scoped_thread_local_keys.push_back(buffer->data.get());
        continue;
      }
      ValidateAllocaBufferLayout(buffer);
      mlir::Value alloc = CreateAlloca(buffer);
      if (std::optional<PrimExpr> init =
              LookupLocalVarInitFromAnnotations(op->annotations, buffer->data)) {
        InitializeAllocaFromExpr(alloc, buffer->shape, buffer->dtype, init.value());
      } else {
        MaybeInitializeAllocaFromLocalVarInit(alloc, buffer->data, buffer->shape, buffer->dtype);
      }
      BindBufferAliases(buffer, alloc, &saved_bindings, &saved_buffer_owners);
      if (buffer->dtype.lanes() > 1) {
        saved_packed_owners.emplace_back(buffer->data.get(),
                                         SaveAndSetPackedDataOwner(buffer->data.get(), buffer));
      }
    }
    for (const tir::MatchBufferRegion& match_buffer : op->match_buffers) {
      if (body_for != nullptr && scalar_values_.count(body_for->loop_var.get()) == 0 &&
          MatchBufferUsesVar(match_buffer, body_for->loop_var)) {
        deferred.match_buffers.push_back(match_buffer);
        continue;
      }
      mlir::Value subview = CreateSubview(match_buffer);
      BindBufferAliases(match_buffer->buffer, subview, &saved_bindings, &saved_buffer_owners);
    }

    if (body_for != nullptr &&
        (!deferred.alloc_buffers.empty() || !deferred.match_buffers.empty())) {
      PushDeferredLoopBindings(body_for->loop_var, deferred);
    }
    VisitStmt(op->body);
    if (body_for != nullptr &&
        (!deferred.alloc_buffers.empty() || !deferred.match_buffers.empty())) {
      PopDeferredLoopBindings(body_for->loop_var);
    }
    RestoreBindings(buffer_values_, saved_bindings);
    RestoreBufferOwnerBindings(saved_buffer_owners);
    RestorePackedDataOwnerBindings(saved_packed_owners);
    for (const Object* key : scoped_thread_local_keys) {
      auto it = prebound_thread_local_block_buffers_.find(key);
      ICHECK(it != prebound_thread_local_block_buffers_.end() && it->second > 0)
          << "missing scoped thread-local replay backing entry in riscv lowering";
      if (--it->second == 0) {
        prebound_thread_local_block_buffers_.erase(it);
      }
    }
    if (!active_thread_local_bindings_stack_.empty()) {
      active_thread_local_bindings_stack_.back().resize(scoped_thread_local_binding_base);
    }
  }

  void VisitStmt_(const tir::BlockRealizeNode* op) final {
    ICHECK_EQ(op->iter_values.size(), op->block->iter_vars.size())
        << "BlockRealize iter_values must match block iter_vars";

    if (InLogicalThreadRegion()) {
      PrimExpr thread_guard;
      if (IsThreadReturnGuardedBody(op->block->body, &thread_guard)) {
        EmitConditionalRegion(tir::Not(thread_guard), [&]() { VisitStmt(op->block); });
        return;
      }
    }

    std::vector<std::pair<const Object*, SavedBinding>> saved_bindings;
    saved_bindings.reserve(op->iter_values.size());
    for (size_t i = 0; i < op->iter_values.size(); ++i) {
      const tir::IterVar& iter_var = op->block->iter_vars[i];
      mlir::Value iter_value = VisitExpr(op->iter_values[i]);
      saved_bindings.emplace_back(iter_var->var.get(),
                                  SaveAndSet(scalar_values_, iter_var->var.get(), iter_value));
    }

    EmitConditionalRegion(op->predicate, [&]() { VisitStmt(op->block); });
    RestoreBindings(scalar_values_, saved_bindings);
  }

  void VisitStmtDefault_(const Object* op) final {
    LOG(FATAL) << "Unsupported TIR stmt for riscv MLIR lowering: " << op->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

  mlir::Value VisitExpr_(const tir::VarNode* op) final {
    return LookupVarValue(tvm::ffi::GetRef<tir::Var>(op));
  }

  mlir::Value VisitExpr_(const tir::LetNode* op) final {
    mlir::Value value = VisitExpr(op->value);
    SavedBinding saved = SaveAndSet(scalar_values_, op->var.get(), value);
    std::optional<PrimExpr> saved_expr =
        SaveAndSetPrimExpr(bound_prim_exprs_, op->var.get(), op->value);
    mlir::Value result = VisitExpr(op->body);
    RestorePrimExprBinding(bound_prim_exprs_, op->var.get(), saved_expr);
    RestoreBinding(scalar_values_, op->var.get(), saved);
    return result;
  }

  mlir::Value VisitExpr_(const tir::BufferLoadNode* op) final {
    if (std::optional<VectorizedRampAccess> ramp_access =
            MatchVectorizedRampAccess(op->buffer, op->indices, op->dtype)) {
      auto emit_ramp_load = [&]() {
        return LowerRampBufferLoad(op->buffer, op->indices, ramp_access.value(), op->dtype);
      };
      if (!op->predicate.defined() || tir::is_one(op->predicate.value())) {
        return emit_ramp_load();
      }
      return EmitConditionalValue(op->predicate.value(), op->dtype, emit_ramp_load,
                                  [&]() { return CreateZeroValue(op->dtype); });
    }

    llvm::SmallVector<mlir::Value, 4> indices;
    indices.reserve(op->indices.size());
    for (const PrimExpr& index : op->indices) {
      indices.push_back(AsIndex(VisitExpr(index), index.dtype()));
    }

    auto emit_load = [&]() {
      if (auto packed_view = ResolvePackedScalarViewBinding(op->buffer)) {
        return LowerPackedScalarViewLoad(op->buffer, packed_view.value(), indices, op->dtype);
      }
      mlir::Value memref = LookupBufferValue(op->buffer);
      mlir::Value load = builder_.create<mlir::memref::LoadOp>(loc_, memref, indices);
      return CastValue(load, op->buffer->dtype, op->dtype);
    };

    if (!op->predicate.defined() || tir::is_one(op->predicate.value())) {
      return emit_load();
    }
    return EmitConditionalValue(op->predicate.value(), op->dtype, emit_load,
                                [&]() { return CreateZeroValue(op->dtype); });
  }

  mlir::Value VisitExpr_(const tir::AddNode* op) final {
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
    if (IsFloatLikeType(op->dtype)) {
      return builder_.create<mlir::arith::AddFOp>(loc_, lhs, rhs);
    }
    return builder_.create<mlir::arith::AddIOp>(loc_, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::SubNode* op) final {
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
    if (IsFloatLikeType(op->dtype)) {
      return builder_.create<mlir::arith::SubFOp>(loc_, lhs, rhs);
    }
    return builder_.create<mlir::arith::SubIOp>(loc_, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::MulNode* op) final {
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
    if (IsFloatLikeType(op->dtype)) {
      return builder_.create<mlir::arith::MulFOp>(loc_, lhs, rhs);
    }
    return builder_.create<mlir::arith::MulIOp>(loc_, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::DivNode* op) final {
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
    if (IsFloatLikeType(op->dtype)) {
      return builder_.create<mlir::arith::DivFOp>(loc_, lhs, rhs);
    }
    if (op->dtype.is_uint() || op->dtype.is_bool()) {
      return builder_.create<mlir::arith::DivUIOp>(loc_, lhs, rhs);
    }
    return builder_.create<mlir::arith::DivSIOp>(loc_, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::ModNode* op) final {
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
    if (IsFloatLikeType(op->dtype)) {
      return builder_.create<mlir::arith::RemFOp>(loc_, lhs, rhs);
    }
    if (op->dtype.is_uint() || op->dtype.is_bool()) {
      return builder_.create<mlir::arith::RemUIOp>(loc_, lhs, rhs);
    }
    return builder_.create<mlir::arith::RemSIOp>(loc_, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::FloorDivNode* op) final {
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
    ICHECK(!IsFloatLikeType(op->dtype))
        << "tir.FloorDiv on floating-point dtype is not supported yet";
    if (op->dtype.is_uint() || op->dtype.is_bool()) {
      return builder_.create<mlir::arith::DivUIOp>(loc_, lhs, rhs);
    }
    return builder_.create<mlir::arith::FloorDivSIOp>(loc_, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::FloorModNode* op) final {
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
    ICHECK(!IsFloatLikeType(op->dtype))
        << "tir.FloorMod on floating-point dtype is not supported yet";
    if (op->dtype.is_uint() || op->dtype.is_bool()) {
      return builder_.create<mlir::arith::RemUIOp>(loc_, lhs, rhs);
    }
    mlir::Value quotient = builder_.create<mlir::arith::FloorDivSIOp>(loc_, lhs, rhs);
    mlir::Value product = builder_.create<mlir::arith::MulIOp>(loc_, quotient, rhs);
    return builder_.create<mlir::arith::SubIOp>(loc_, lhs, product);
  }

  mlir::Value VisitExpr_(const tir::MinNode* op) final {
    DataType compare_dtype = op->dtype;
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
    mlir::Value cond;
    if (IsFloatLikeType(compare_dtype)) {
      cond = builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::OLT, lhs,
                                                  rhs);
    } else if (compare_dtype.is_uint() || compare_dtype.is_bool()) {
      cond = builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::ult, lhs,
                                                  rhs);
    } else {
      cond = builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::slt, lhs,
                                                  rhs);
    }
    return builder_.create<mlir::arith::SelectOp>(loc_, cond, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::MaxNode* op) final {
    DataType compare_dtype = op->dtype;
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
    mlir::Value cond;
    if (IsFloatLikeType(compare_dtype)) {
      cond = builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::OGT, lhs,
                                                  rhs);
    } else if (compare_dtype.is_uint() || compare_dtype.is_bool()) {
      cond = builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::ugt, lhs,
                                                  rhs);
    } else {
      cond = builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::sgt, lhs,
                                                  rhs);
    }
    return builder_.create<mlir::arith::SelectOp>(loc_, cond, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::CastNode* op) final {
    mlir::Value value = VisitExpr(op->value);
    return CastValue(value, op->value.dtype(), op->dtype);
  }

  mlir::Value VisitExpr_(const tir::EQNode* op) final {
    DataType compare_dtype = op->a.dtype();
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
    if (IsFloatLikeType(compare_dtype)) {
      return builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::OEQ, lhs,
                                                  rhs);
    }
    return builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::eq, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::NENode* op) final {
    DataType compare_dtype = op->a.dtype();
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
    if (IsFloatLikeType(compare_dtype)) {
      return builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::UNE, lhs,
                                                  rhs);
    }
    return builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::ne, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::LTNode* op) final {
    DataType compare_dtype = op->a.dtype();
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
    if (IsFloatLikeType(compare_dtype)) {
      return builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::OLT, lhs,
                                                  rhs);
    }
    mlir::arith::CmpIPredicate predicate =
        compare_dtype.is_uint() || compare_dtype.is_bool() ? mlir::arith::CmpIPredicate::ult
                                                           : mlir::arith::CmpIPredicate::slt;
    return builder_.create<mlir::arith::CmpIOp>(loc_, predicate, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::LENode* op) final {
    DataType compare_dtype = op->a.dtype();
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
    if (IsFloatLikeType(compare_dtype)) {
      return builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::OLE, lhs,
                                                  rhs);
    }
    mlir::arith::CmpIPredicate predicate =
        compare_dtype.is_uint() || compare_dtype.is_bool() ? mlir::arith::CmpIPredicate::ule
                                                           : mlir::arith::CmpIPredicate::sle;
    return builder_.create<mlir::arith::CmpIOp>(loc_, predicate, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::GTNode* op) final {
    DataType compare_dtype = op->a.dtype();
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
    if (IsFloatLikeType(compare_dtype)) {
      return builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::OGT, lhs,
                                                  rhs);
    }
    mlir::arith::CmpIPredicate predicate =
        compare_dtype.is_uint() || compare_dtype.is_bool() ? mlir::arith::CmpIPredicate::ugt
                                                           : mlir::arith::CmpIPredicate::sgt;
    return builder_.create<mlir::arith::CmpIOp>(loc_, predicate, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::GENode* op) final {
    DataType compare_dtype = op->a.dtype();
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
    if (IsFloatLikeType(compare_dtype)) {
      return builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::OGE, lhs,
                                                  rhs);
    }
    mlir::arith::CmpIPredicate predicate =
        compare_dtype.is_uint() || compare_dtype.is_bool() ? mlir::arith::CmpIPredicate::uge
                                                           : mlir::arith::CmpIPredicate::sge;
    return builder_.create<mlir::arith::CmpIOp>(loc_, predicate, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::AndNode* op) final {
    mlir::Value lhs = LowerLogicalValue(VisitExpr(op->a), op->a.dtype());
    mlir::Value rhs = LowerLogicalValue(VisitExpr(op->b), op->b.dtype());
    return builder_.create<mlir::arith::AndIOp>(loc_, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::OrNode* op) final {
    mlir::Value lhs = LowerLogicalValue(VisitExpr(op->a), op->a.dtype());
    mlir::Value rhs = LowerLogicalValue(VisitExpr(op->b), op->b.dtype());
    return builder_.create<mlir::arith::OrIOp>(loc_, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::NotNode* op) final {
    mlir::Value value = LowerLogicalValue(VisitExpr(op->a), op->a.dtype());
    mlir::Value one = CreateBooleanTrueLike(value.getType());
    return builder_.create<mlir::arith::XOrIOp>(loc_, value, one);
  }

  mlir::Value VisitExpr_(const tir::SelectNode* op) final {
    if (op->condition.dtype().lanes() > 1) {
      mlir::Value cond = LowerLogicalValue(VisitExpr(op->condition), op->condition.dtype());
      mlir::Value true_value =
          CastValue(VisitExpr(op->true_value), op->true_value.dtype(), op->dtype);
      mlir::Value false_value =
          CastValue(VisitExpr(op->false_value), op->false_value.dtype(), op->dtype);
      return builder_.create<mlir::arith::SelectOp>(loc_, cond, true_value, false_value);
    }
    return EmitConditionalValue(
        op->condition, op->dtype,
        [&]() { return CastValue(VisitExpr(op->true_value), op->true_value.dtype(), op->dtype); },
        [&]() { return CastValue(VisitExpr(op->false_value), op->false_value.dtype(), op->dtype); });
  }

  mlir::Value VisitExpr_(const tir::BroadcastNode* op) final { return LowerBroadcast(op); }

  mlir::Value VisitExpr_(const tir::RampNode* op) final { return LowerRamp(op); }

  mlir::Value VisitExpr_(const IntImmNode* op) final {
    mlir::Type type = LowerScalarType(op->dtype);
    return ConstantIntLike(op->value, type);
  }

  mlir::Value VisitExpr_(const FloatImmNode* op) final {
    mlir::FloatType type = mlir::cast<mlir::FloatType>(LowerScalarType(op->dtype));
    return builder_.create<mlir::arith::ConstantOp>(loc_, builder_.getFloatAttr(type, op->value));
  }

  mlir::Value VisitExpr_(const tir::ShuffleNode* op) final { return LowerShuffle(op); }

  mlir::Value VisitExpr_(const tir::CallNode* op) final {
    if (const auto* op_node = op->op.as<OpNode>()) {
      if (IsInfinityCall(op)) {
        return CreateInfinityValue(op->dtype);
      }
      if (op->op.same_as(tir::builtin::call_extern())) {
        if (IsFloat2HalfRZCallExtern(op)) {
          ICHECK_EQ(op->args.size(), 2U)
              << "__float2half_rz call_extern expects one value argument";
          mlir::Value arg = VisitExpr(op->args[1]);
          return CastValue(arg, op->args[1].dtype(), op->dtype);
        }
        if (IsAtomicAddOffsetCallExtern(op)) {
          return LowerAtomicAddOffsetCallExtern(op);
        }
        if (IsMatchSyncCallExtern(op) && CanLowerCooperativeCallAsSingleThread(op)) {
          return LowerSingleThreadCooperativeCall(op);
        }
        if (CanLowerMatchSyncCallAsSerializedWarpReplay(op)) {
          return LowerSerializedWarpReplayMatchSyncCall(op);
        }
        RejectUnsupportedCallExtern(op);
      }
      if (IsThreadIndexHelperIntrinsicName(op_node->name)) {
        return LowerThreadIndexHelperCall(op);
      }
      if (op->op.same_as(tvm::tl::rng_rand())) {
        return LowerRngRand(op);
      }
      if (op->op.same_as(tvm::tl::rng_rand_float())) {
        return LowerRngRandFloat(op);
      }
      if (op->op.same_as(tvm::tl::create_tma_descriptor())) {
        // The serialized backend has no TMA runtime. Descriptor values are kept
        // only as opaque placeholders so statement-level helper lowering can
        // recover the original source buffer and emit synchronous fallbacks.
        return CreateSerializedOpaqueHandlePlaceholder();
      }
      if (IsCudaPipelineOrTargetSyncIntrinsicName(op_node->name)) {
        RejectCudaPipelineOrTargetSyncIntrinsic(op_node->name);
      }
      if (IsUnsupportedTileReductionIntrinsicName(op_node->name)) {
        RejectUnsupportedTileReductionIntrinsic(op_node->name);
      }
      if (IsUnsupportedTileScanIntrinsicName(op_node->name)) {
        RejectUnsupportedTileScanIntrinsic(op_node->name);
      }
      if (CanLowerCooperativeCallAsSingleThread(op)) {
        return LowerSingleThreadCooperativeCall(op);
      }
      if (CanLowerMatchSyncCallAsSerializedWarpReplay(op)) {
        return LowerSerializedWarpReplayMatchSyncCall(op);
      }
      if (CanLowerVoteCallAsSerializedWarpReplay(op)) {
        return LowerSerializedWarpReplayVoteCall(op);
      }
      if (CanLowerSyncthreadsOrCallAsSerializedThreadReplay(op)) {
        return LowerSerializedThreadReplaySyncthreadsOrCall(op);
      }
      if (CanLowerShuffleCallAsSerializedWarpReplay(op)) {
        return LowerSerializedWarpReplayShuffleCall(op);
      }
      if (CanLowerWarpReduceCallAsSerializedWarpReplay(op)) {
        return LowerSerializedWarpReplayWarpReduceCall(op);
      }
      if (IsCooperativeThreadIntrinsicName(op_node->name)) {
        RejectUnsupportedCooperativeThreadExpression(op_node->name);
      }
      if (IsScalarAtomicRMWCall(op)) {
        ICHECK(ScalarAtomicRMWReturnsValue(op))
            << op_node->name
            << " is statement-only in riscv lowering unless the previous value is requested";
        return LowerScalarAtomicRMW(op);
      }
      if (IsScalarAtomicLoadCall(op)) {
        return LowerScalarAtomicLoad(op);
      }
      ICHECK(!IsScalarAtomicStoreCall(op))
          << "tl.atomic_store_elem_op is statement-only in riscv lowering";
      if (IsVectorAtomicAddCall(op)) {
        ICHECK(!op->dtype.is_handle())
            << op_node->name
            << " is statement-only in riscv lowering unless the previous value is requested";
        return LowerVectorAtomicAdd(op);
      }
      ICHECK(!IsAtomicIntrinsicName(op_node->name))
          << "Unsupported atomic intrinsic expression in riscv lowering: " << op->op;
    }
    if (IsIfThenElseCall(op)) {
      return EmitConditionalValue(
          op->args[0], op->dtype,
          [&]() { return CastValue(VisitExpr(op->args[1]), op->args[1].dtype(), op->dtype); },
          [&]() { return CastValue(VisitExpr(op->args[2]), op->args[2].dtype(), op->dtype); });
    }
    if (IsSupportedBitcastCall(op)) {
      mlir::Value arg =
          CastValue(VisitExpr(op->args[0]), op->args[0].dtype(), op->args[0].dtype());
      return LowerBitcastCall(op, arg);
    }
    if (IsReinterpretCall(op)) {
      RejectUnsupportedReinterpretCall(op);
    }
    if (IsSupportedPackedX2IntrinsicCall(op)) {
      mlir::Value lhs = CastValue(VisitExpr(op->args[0]), op->args[0].dtype(), op->dtype);
      if (op->args.size() == 1) {
        return LowerSupportedPackedX2IntrinsicCall(builder_, loc_, op, lhs);
      }
      mlir::Value rhs = CastValue(VisitExpr(op->args[1]), op->args[1].dtype(), op->dtype);
      if (op->args.size() == 2) {
        return LowerSupportedPackedX2IntrinsicCall(builder_, loc_, op, lhs, rhs);
      }
      mlir::Value extra = CastValue(VisitExpr(op->args[2]), op->args[2].dtype(), op->dtype);
      return LowerSupportedPackedX2IntrinsicCall(builder_, loc_, op, lhs, rhs, extra);
    }
    if (IsSupportedUnaryMathCall(op)) {
      mlir::Value arg = CastValue(VisitExpr(op->args[0]), op->args[0].dtype(), op->dtype);
      return LowerSupportedUnaryMathCall(builder_, loc_, op, arg);
    }
    if (IsSupportedUnaryIntrinsicCall(op)) {
      mlir::Value arg =
          CastValue(VisitExpr(op->args[0]), op->args[0].dtype(), op->args[0].dtype());
      return LowerSupportedUnaryIntrinsicCall(builder_, loc_, op, arg);
    }
    if (IsSupportedBinaryIntrinsicCall(op)) {
      mlir::Value lhs = CastValue(VisitExpr(op->args[0]), op->args[0].dtype(), op->dtype);
      mlir::Value rhs = CastValue(VisitExpr(op->args[1]), op->args[1].dtype(), op->dtype);
      return LowerSupportedBinaryIntrinsicCall(builder_, loc_, op, lhs, rhs);
    }

    ICHECK(false) << "Unsupported TIR expr for riscv MLIR lowering: " << op->op;
    TVM_FFI_UNREACHABLE();
  }

  mlir::Value VisitExprDefault_(const Object* op) final {
    LOG(FATAL) << "Unsupported TIR expr for riscv MLIR lowering: " << op->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

  mlir::DialectRegistry registry_;
  arith::Analyzer analyzer_;
  mlir::MLIRContext context_;
  mlir::OpBuilder builder_;
  mlir::Location loc_;
  mlir::ModuleOp module_;
  ValueMap scalar_values_;
  PrimExprMap bound_prim_exprs_;
  ValueMap buffer_values_;
  Map<tir::Var, PrimExpr> local_var_init_map_;
  PrimExprMap serialized_warp_replay_buffer_exprs_;
  SerializedWarpReplayBufferElementExprMap serialized_warp_replay_buffer_element_exprs_;
  std::unordered_set<const Object*> function_param_buffers_;
  std::unordered_set<const Object*> function_param_buffer_data_;
  PackedScalarViewMap packed_scalar_view_bindings_;
  PackedDataOwnerMap packed_data_owner_;
  BufferOwnerMap buffer_owner_;
  std::unordered_map<const Object*, std::vector<DeferredLoopBindings>> deferred_loop_bindings_;
  std::unordered_map<const Object*, int> prebound_thread_local_block_buffers_;
  std::unordered_map<const Object*, mlir::Value> active_thread_local_backings_;
  std::unordered_set<std::string> pointer_backed_buffer_view_helpers_;
  std::optional<mlir::Value> active_rng_state_;
  std::vector<std::vector<ThreadLocalBlockAllocBinding>> active_thread_local_bindings_stack_;
  std::vector<BreakLoopFrame> break_loop_stack_;
  const tir::IterVarNode* thread_binding_var_{nullptr};
  const tir::IterVarNode* block_binding_var_{nullptr};
  std::vector<ThreadLaunchFrame> block_launch_stack_;
  std::vector<ThreadLaunchFrame> thread_launch_stack_;
};

std::string BuildStructuredMLIRModule(const std::vector<FunctionEntry>& functions) {
  TIRToMLIRLowerer lowerer;
  return lowerer.Lower(functions);
}
#endif

}  // namespace

void CodeGenTileLangRISCV::AddFunction(const GlobalVar& gvar, const tir::PrimFunc& func) {
  std::string name;
  if (auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
    name = global_symbol.value();
  } else {
    name = gvar->name_hint;
  }
  function_names_.push_back(name);
  functions_.emplace_back(name, func);
}

std::string CodeGenTileLangRISCV::Finish() const {
#if TILELANG_ENABLE_RISCV_MLIR
  return BuildStructuredMLIRModule(functions_);
#else
  return BuildPlaceholderModule(functions_);
#endif
}

}  // namespace codegen
}  // namespace tvm
