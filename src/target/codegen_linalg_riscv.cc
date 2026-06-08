#include "codegen_linalg_riscv.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef TILELANG_ENABLE_LINALG_RISCV_MLIR
#define TILELANG_ENABLE_LINALG_RISCV_MLIR 0
#endif

#if TILELANG_ENABLE_LINALG_RISCV_MLIR
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#endif

#include <tvm/arith/analyzer.h>
#include <tvm/ir/attrs.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../op/utils.h"

namespace tvm {
namespace codegen {

namespace {

using FunctionEntry = std::pair<std::string, tir::PrimFunc>;

std::string BuildPlaceholderModule(const std::vector<FunctionEntry>& functions) {
  std::ostringstream os;
  os << "module {\n";
  os << "  // Placeholder MLIR module for the linalg_riscv backend.\n";
  os << "  // Rebuild TileLang with TILELANG_RISCV_MLIR_MODE=ON after the vendored\n";
  os << "  // LLVM/MLIR toolchain is installed to enable the real C++ MLIR builder.\n";
  for (const auto& [name, func] : functions) {
    (void)func;
    os << "  // pending lowering for @" << name << "\n";
  }
  os << "}\n";
  return os.str();
}

#if TILELANG_ENABLE_LINALG_RISCV_MLIR
bool IsSupportedUnaryMathCall(const tir::CallNode* op) {
  const auto* op_node = op->op.as<OpNode>();
  if (op_node == nullptr || op->args.size() != 1 || !op->dtype.is_float()) {
    return false;
  }
  return op_node->name == "tir.sqrt" || op_node->name == "tir.rsqrt" ||
         op_node->name == "tir.exp2" || op_node->name == "tir.log2";
}

mlir::Value LowerSupportedUnaryMathCall(mlir::OpBuilder& builder, mlir::Location loc,
                                        const tir::CallNode* op, mlir::Value arg) {
  ICHECK(IsSupportedUnaryMathCall(op))
      << "Unsupported TIR call in linalg_riscv lowering: " << op->op;
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
  LOG(FATAL) << "Unsupported TIR math call in linalg_riscv lowering: " << op_node->name;
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
                     mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
    context_.appendDialectRegistry(registry_);
    context_.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                         mlir::linalg::LinalgDialect, mlir::math::MathDialect,
                         mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
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

  struct StructuredIndexPattern {
    llvm::SmallVector<unsigned, 4> dims;

    bool operator==(const StructuredIndexPattern& other) const { return dims == other.dims; }
  };

  struct ReductionLoopNestMatch {
    enum class Kind {
      kAdd,
      kMin,
      kMax,
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
    bool VisitExpr_(const IntImmNode* op) final {
      (void)op;
      return true;
    }
    bool VisitExpr_(const FloatImmNode* op) final {
      (void)op;
      return true;
    }
    bool VisitExpr_(const tir::CallNode* op) final {
      if (!IsSupportedUnaryMathCall(op)) {
        return false;
      }
      return VisitExpr(op->args[0]);
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
      if (op->dtype.is_float()) {
        return outer_->builder_.create<mlir::arith::AddFOp>(outer_->loc_, lhs, rhs);
      }
      return outer_->builder_.create<mlir::arith::AddIOp>(outer_->loc_, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::SubNode* op) final {
      mlir::Value lhs = outer_->CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
      mlir::Value rhs = outer_->CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
      if (op->dtype.is_float()) {
        return outer_->builder_.create<mlir::arith::SubFOp>(outer_->loc_, lhs, rhs);
      }
      return outer_->builder_.create<mlir::arith::SubIOp>(outer_->loc_, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::MulNode* op) final {
      mlir::Value lhs = outer_->CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
      mlir::Value rhs = outer_->CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
      if (op->dtype.is_float()) {
        return outer_->builder_.create<mlir::arith::MulFOp>(outer_->loc_, lhs, rhs);
      }
      return outer_->builder_.create<mlir::arith::MulIOp>(outer_->loc_, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::DivNode* op) final {
      mlir::Value lhs = outer_->CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
      mlir::Value rhs = outer_->CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
      if (op->dtype.is_float()) {
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
      if (op->dtype.is_float()) {
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
      ICHECK(!op->dtype.is_float()) << "tir.FloorDiv on floating-point dtype is not supported yet";
      if (op->dtype.is_uint() || op->dtype.is_bool()) {
        return outer_->builder_.create<mlir::arith::DivUIOp>(outer_->loc_, lhs, rhs);
      }
      return outer_->builder_.create<mlir::arith::FloorDivSIOp>(outer_->loc_, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::FloorModNode* op) final {
      mlir::Value lhs = outer_->CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
      mlir::Value rhs = outer_->CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
      ICHECK(!op->dtype.is_float()) << "tir.FloorMod on floating-point dtype is not supported yet";
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
      if (compare_dtype.is_float()) {
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
      if (compare_dtype.is_float()) {
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

    mlir::Value VisitExpr_(const tir::CallNode* op) final {
      ICHECK(IsSupportedUnaryMathCall(op))
          << "Unsupported TIR call in elementwise linalg.generic lowering: " << op->op;
      mlir::Value arg = outer_->CastValue(VisitExpr(op->args[0]), op->args[0].dtype(), op->dtype);
      return LowerSupportedUnaryMathCall(outer_->builder_, outer_->loc_, op, arg);
    }

    mlir::Value VisitExpr_(const tir::EQNode* op) final {
      DataType compare_dtype = op->a.dtype();
      mlir::Value lhs = outer_->CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
      mlir::Value rhs = outer_->CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
      if (compare_dtype.is_float()) {
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
      if (compare_dtype.is_float()) {
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
      if (compare_dtype.is_float()) {
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
      if (compare_dtype.is_float()) {
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
      if (compare_dtype.is_float()) {
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
      if (compare_dtype.is_float()) {
        return outer_->builder_.create<mlir::arith::CmpFOp>(
            outer_->loc_, mlir::arith::CmpFPredicate::OGE, lhs, rhs);
      }
      mlir::arith::CmpIPredicate predicate =
          compare_dtype.is_uint() || compare_dtype.is_bool() ? mlir::arith::CmpIPredicate::uge
                                                             : mlir::arith::CmpIPredicate::sge;
      return outer_->builder_.create<mlir::arith::CmpIOp>(outer_->loc_, predicate, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::AndNode* op) final {
      mlir::Value lhs = outer_->LowerConditionValue(VisitExpr(op->a), op->a.dtype());
      mlir::Value rhs = outer_->LowerConditionValue(VisitExpr(op->b), op->b.dtype());
      return outer_->builder_.create<mlir::arith::AndIOp>(outer_->loc_, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::OrNode* op) final {
      mlir::Value lhs = outer_->LowerConditionValue(VisitExpr(op->a), op->a.dtype());
      mlir::Value rhs = outer_->LowerConditionValue(VisitExpr(op->b), op->b.dtype());
      return outer_->builder_.create<mlir::arith::OrIOp>(outer_->loc_, lhs, rhs);
    }

    mlir::Value VisitExpr_(const tir::NotNode* op) final {
      mlir::Value value = outer_->LowerConditionValue(VisitExpr(op->a), op->a.dtype());
      mlir::Value one = outer_->ConstantIntLike(1, outer_->builder_.getI1Type());
      return outer_->builder_.create<mlir::arith::XOrIOp>(outer_->loc_, value, one);
    }

    mlir::Value VisitExpr_(const tir::SelectNode* op) final {
      mlir::Value cond = outer_->LowerConditionValue(VisitExpr(op->condition), op->condition.dtype());
      mlir::Value true_value =
          outer_->CastValue(VisitExpr(op->true_value), op->true_value.dtype(), op->dtype);
      mlir::Value false_value =
          outer_->CastValue(VisitExpr(op->false_value), op->false_value.dtype(), op->dtype);
      return outer_->builder_.create<mlir::arith::SelectOp>(outer_->loc_, cond, true_value,
                                                            false_value);
    }

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

  mlir::Type LowerScalarType(DataType dtype) {
    ICHECK_EQ(dtype.lanes(), 1) << "Vector lanes are not supported yet for linalg_riscv";
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
    if (dtype.is_float() && dtype.bits() == 32) {
      return builder_.getF32Type();
    }
    if (dtype.is_float() && dtype.bits() == 64) {
      return builder_.getF64Type();
    }
    LOG(FATAL) << "Unsupported scalar dtype in linalg_riscv MLIR lowering: " << dtype;
    TVM_FFI_UNREACHABLE();
  }

  mlir::Value ConstantIntLike(int64_t value, mlir::Type type) {
    if (type.isIndex()) {
      return builder_.create<mlir::arith::ConstantIndexOp>(loc_, value);
    }
    return builder_.create<mlir::arith::ConstantIntOp>(loc_, value, type);
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
        if (dtype.is_float()) {
          return builder.create<mlir::arith::AddFOp>(loc, acc, value);
        }
        return builder.create<mlir::arith::AddIOp>(loc, acc, value);
      case ReductionLoopNestMatch::Kind::kMin: {
        mlir::Value cond;
        if (dtype.is_float()) {
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
        if (dtype.is_float()) {
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
    }
    LOG(FATAL) << "Unknown reduction combine kind";
    TVM_FFI_UNREACHABLE();
  }

  int64_t GetStaticInt(const PrimExpr& expr, const char* what) {
    const auto* imm = expr.as<IntImmNode>();
    ICHECK(imm != nullptr) << what << " must be a static IntImm in linalg_riscv lowering";
    return imm->value;
  }

  bool GetStaticBool(const PrimExpr& expr, const char* what) {
    if (tir::is_zero(expr)) {
      return false;
    }
    if (tir::is_one(expr)) {
      return true;
    }
    LOG(FATAL) << what << " must be a static boolean in linalg_riscv lowering";
    TVM_FFI_UNREACHABLE();
  }

  bool HasCompactRowMajorLayout(const tir::Buffer& buffer) {
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
    ICHECK_EQ(buffer->dtype.lanes(), 1)
        << "Vector element buffers are not supported yet for linalg_riscv";
    ICHECK(HasCompactRowMajorLayout(buffer))
        << "Only compact row-major buffers are supported in linalg_riscv: " << buffer->name;
    ICHECK(tir::is_zero(buffer->elem_offset))
        << "Non-zero elem_offset is not supported yet for linalg_riscv: " << buffer->name;
  }

  mlir::Value CastValueToType(mlir::Value value, mlir::Type target_type, bool source_unsigned) {
    mlir::Type source_type = value.getType();
    if (source_type == target_type) {
      return value;
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

    LOG(FATAL) << "Unsupported MLIR cast encountered in linalg_riscv lowering";
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
          loc_, llvm::APFloat(0.0), mlir::cast<mlir::FloatType>(zero_type));
      return builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::UNE, value,
                                                  zero);
    }
    LOG(FATAL) << "Unsupported condition value type in linalg_riscv lowering for TIR dtype "
               << source_dtype;
    TVM_FFI_UNREACHABLE();
  }

  mlir::Value LowerCondition(const PrimExpr& expr) {
    return LowerConditionValue(VisitExpr(expr), expr.dtype());
  }

  mlir::Value AsIndex(mlir::Value value, DataType source_dtype) {
    if (value.getType().isIndex()) {
      return value;
    }
    return CastValueToType(value, builder_.getIndexType(), source_dtype.is_uint() || source_dtype.is_bool());
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
    LOG(FATAL) << "Unbound TIR var during linalg_riscv lowering: " << var->name_hint;
    TVM_FFI_UNREACHABLE();
  }

  mlir::Value LookupBufferValue(const tir::Buffer& buffer) const {
    auto it = buffer_values_.find(buffer.get());
    if (it != buffer_values_.end()) {
      return it->second;
    }
    it = buffer_values_.find(buffer->data.get());
    if (it != buffer_values_.end()) {
      return it->second;
    }
    LOG(FATAL) << "Unbound TIR buffer during linalg_riscv lowering: " << buffer->name;
    TVM_FFI_UNREACHABLE();
  }

  void RestoreBindings(ValueMap& map,
                       const std::vector<std::pair<const Object*, SavedBinding>>& saved_bindings) {
    for (auto it = saved_bindings.rbegin(); it != saved_bindings.rend(); ++it) {
      RestoreBinding(map, it->first, it->second);
    }
  }

  void BindBufferAliases(const tir::Buffer& buffer, mlir::Value value,
                         std::vector<std::pair<const Object*, SavedBinding>>* saved_bindings) {
    saved_bindings->emplace_back(buffer.get(), SaveAndSet(buffer_values_, buffer.get(), value));
    saved_bindings->emplace_back(buffer->data.get(),
                                 SaveAndSet(buffer_values_, buffer->data.get(), value));
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

    ValidateContiguousBuffer(target_buffer);

    mlir::Value source = LookupBufferValue(source_buffer);
    mlir::MemRefType source_type = mlir::cast<mlir::MemRefType>(source.getType());
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets = LowerRegionOffsets(source_region->region);
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes = LowerRegionSizes(source_region->region);
    llvm::SmallVector<mlir::OpFoldResult, 4> strides = UnitStrides(source_region->region.size());

    mlir::MemRefType result_type;
    if (target_buffer->shape.size() == source_region->region.size()) {
      result_type = mlir::cast<mlir::MemRefType>(
          mlir::memref::SubViewOp::inferResultType(source_type, offsets, sizes, strides));
    } else {
      result_type = mlir::cast<mlir::MemRefType>(mlir::memref::SubViewOp::inferRankReducedResultType(
          LowerStaticShape(target_buffer->shape), source_type, offsets, sizes, strides));
    }

    return builder_
        .create<mlir::memref::SubViewOp>(loc_, result_type, source, offsets, sizes, strides)
        .getResult();
  }

  mlir::Value CreateSubview(const tir::BufferRegion& region) {
    const tir::Buffer& source_buffer = region->buffer;
    mlir::Value source = LookupBufferValue(source_buffer);
    mlir::MemRefType source_type = mlir::cast<mlir::MemRefType>(source.getType());
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets = LowerRegionOffsets(region->region);
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes = LowerRegionSizes(region->region);
    llvm::SmallVector<mlir::OpFoldResult, 4> strides = UnitStrides(region->region.size());
    mlir::MemRefType result_type = mlir::cast<mlir::MemRefType>(
        mlir::memref::SubViewOp::inferResultType(source_type, offsets, sizes, strides));
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
    mlir::MemRefType result_type = mlir::cast<mlir::MemRefType>(
        mlir::memref::SubViewOp::inferRankReducedResultType(LowerStaticShape(logical_extents),
                                                            source_type, offsets, sizes, strides));
    return builder_
        .create<mlir::memref::SubViewOp>(loc_, result_type, source, offsets, sizes, strides)
        .getResult();
  }

  mlir::Value CreateAlloca(const Array<PrimExpr>& shape_exprs, DataType element_dtype) {
    mlir::MemRefType memref_type = LowerMemRefType(element_dtype, shape_exprs);
    llvm::SmallVector<mlir::Value, 4> dynamic_sizes = LowerDynamicSizes(shape_exprs);
    return builder_.create<mlir::memref::AllocaOp>(loc_, memref_type, dynamic_sizes);
  }

  mlir::Value CreateStaticAlloca(llvm::ArrayRef<int64_t> shape, DataType element_dtype) {
    mlir::MemRefType memref_type = mlir::MemRefType::get(shape, LowerScalarType(element_dtype));
    return builder_.create<mlir::memref::AllocaOp>(loc_, memref_type);
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
    if (dtype.is_float()) {
      mlir::FloatType type = mlir::cast<mlir::FloatType>(LowerScalarType(dtype));
      return builder_.create<mlir::arith::ConstantOp>(loc_, builder_.getFloatAttr(type, 0.0));
    }
    return ConstantIntLike(0, LowerScalarType(dtype));
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

  void LowerTileFill(const tir::CallNode* op) {
    tir::BufferRegion dst_region = tl::NormalizeToBufferRegion(op->args[0]);
    FillBufferRegion(dst_region, VisitExpr(op->args[1]), op->args[1].dtype());
  }

  void LowerTileCopy(const tir::CallNode* op) {
    tir::BufferRegion src_region = tl::NormalizeToBufferRegion(op->args[0]);
    tir::BufferRegion dst_region = tl::NormalizeToBufferRegion(op->args[1]);
    const tir::Buffer& src_buffer = src_region->buffer;
    const tir::Buffer& dst_buffer = dst_region->buffer;
    mlir::Value src_memref = LookupBufferValue(src_buffer);
    mlir::Value dst_memref = LookupBufferValue(dst_buffer);

    if (src_region->region.size() == dst_region->region.size()) {
      if (src_buffer->dtype == dst_buffer->dtype && RegionsHaveSameExtents(src_region, dst_region)) {
        builder_.create<mlir::memref::CopyOp>(loc_, CreateSubview(src_region), CreateSubview(dst_region));
        return;
      }

      EmitLoopNest(dst_region->region, [&](llvm::ArrayRef<mlir::Value> coords) {
        llvm::SmallVector<mlir::Value, 4> src_indices;
        src_indices.reserve(coords.size());
        for (size_t i = 0; i < coords.size(); ++i) {
          if (IsStaticOne(src_region->region[i]->extent)) {
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
        << "tl.copy currently requires matching logical ranks after dropping static-1 dims";
    for (size_t i = 0; i < src_logical_extents.size(); ++i) {
      ICHECK(analyzer_.CanProveEqual(src_logical_extents[i], dst_logical_extents[i]))
          << "tl.copy rank-reduced extents must match";
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

  void LowerTileGemmPy(const tir::CallNode* op) {
    tir::BufferRegion a_region = tl::NormalizeToBufferRegion(op->args[0]);
    tir::BufferRegion b_region = tl::NormalizeToBufferRegion(op->args[1]);
    tir::BufferRegion c_region = tl::NormalizeToBufferRegion(op->args[2]);

    bool transpose_a = GetStaticBool(op->args[3], "tl.gemm transpose_a");
    bool transpose_b = GetStaticBool(op->args[4], "tl.gemm transpose_b");
    PrimExpr m = op->args[5];
    PrimExpr n = op->args[6];
    PrimExpr k = op->args[7];
    bool clear_accum = GetStaticBool(op->args[9], "tl.gemm clear_accum");

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

    if (clear_accum) {
      FillBufferRegion(c_region, CreateZeroValue(c_region->buffer->dtype), c_region->buffer->dtype);
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

  bool MatchElementwiseLoopNest(const tir::ForNode* outer, ElementwiseLoopNestMatch* match) {
    const tir::ForNode* current = outer;
    while (true) {
      if (!(current->kind == tir::ForKind::kSerial || current->kind == tir::ForKind::kUnrolled) ||
          current->thread_binding.defined() || !tir::is_zero(current->min) ||
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
    ValidateContiguousBuffer(output_buffer);
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
      if (!(current->kind == tir::ForKind::kSerial || current->kind == tir::ForKind::kUnrolled) ||
          current->thread_binding.defined() || !tir::is_zero(current->min) ||
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
    ValidateContiguousBuffer(match->output_buffer);
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
    ValidateContiguousBuffer(buffer);
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

    if (op_node->name == "tl.tileop.copy") {
      LowerTileCopy(op);
      return true;
    }
    if (op_node->name == "tl.tileop.fill") {
      LowerTileFill(op);
      return true;
    }
    if (op_node->name == "tl.tileop.gemm" || op_node->name == "tl.tileop.gemm_py") {
      LowerTileGemmPy(op);
      return true;
    }
    return false;
  }

  void LowerFunction(const std::string& name, const tir::PrimFunc& func) {
    scalar_values_.clear();
    buffer_values_.clear();

    llvm::SmallVector<mlir::Type, 8> input_types;
    input_types.reserve(func->params.size());

    for (const tir::Var& param : func->params) {
      if (func->buffer_map.count(param)) {
        const tir::Buffer& buffer = func->buffer_map[param];
        ValidateContiguousBuffer(buffer);
        input_types.push_back(LowerMemRefType(buffer->dtype, buffer->shape));
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
          BindDynamicShapeVars(buffer, arg);
        } else {
          scalar_values_[param.get()] = arg;
        }
      }

      VisitStmt(func->body);
      builder_.create<mlir::func::ReturnOp>(loc_);
    }
  }

  void VisitStmt_(const tir::SeqStmtNode* op) final {
    for (const tir::Stmt& stmt : op->seq) {
      VisitStmt(stmt);
    }
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

  void VisitStmt_(const tir::ForNode* op) final {
    if (TryLowerReductionLoopNest(op)) {
      return;
    }
    if (TryLowerElementwiseLoopNest(op)) {
      return;
    }

    ICHECK(op->kind == tir::ForKind::kSerial || op->kind == tir::ForKind::kUnrolled)
        << "Only serial/unrolled loops are supported in the current linalg_riscv lowering";
    ICHECK(!op->thread_binding.defined())
        << "Thread-bound loops are not supported in linalg_riscv lowering";

    mlir::Value lower = AsIndex(VisitExpr(op->min), op->min.dtype());
    mlir::Value extent = AsIndex(VisitExpr(op->extent), op->extent.dtype());
    mlir::Value step =
        op->step.defined() ? AsIndex(VisitExpr(op->step.value()), op->step.value().dtype())
                           : ConstantIntLike(1, builder_.getIndexType());
    mlir::Value upper = builder_.create<mlir::arith::AddIOp>(loc_, lower, extent);
    mlir::scf::ForOp for_op = builder_.create<mlir::scf::ForOp>(loc_, lower, upper, step);

    SavedBinding saved_loop_var = SaveAndSet(scalar_values_, op->loop_var.get(), for_op.getInductionVar());
    {
      mlir::OpBuilder::InsertionGuard guard(builder_);
      builder_.setInsertionPoint(for_op.getBody()->getTerminator());
      std::vector<std::pair<const Object*, SavedBinding>> saved_bindings;
      auto deferred_it = deferred_loop_bindings_.find(op->loop_var.get());
      if (deferred_it != deferred_loop_bindings_.end() && !deferred_it->second.empty()) {
        const DeferredLoopBindings& deferred = deferred_it->second.back();
        saved_bindings.reserve(deferred.alloc_buffers.size() * 2 + deferred.match_buffers.size() * 2);
        for (const tir::Buffer& buffer : deferred.alloc_buffers) {
          ValidateContiguousBuffer(buffer);
          mlir::Value alloc = CreateAlloca(buffer->shape, buffer->dtype);
          BindBufferAliases(buffer, alloc, &saved_bindings);
        }
        for (const tir::MatchBufferRegion& match_buffer : deferred.match_buffers) {
          mlir::Value subview = CreateSubview(match_buffer);
          BindBufferAliases(match_buffer->buffer, subview, &saved_bindings);
        }
      }
      VisitStmt(op->body);
      RestoreBindings(buffer_values_, saved_bindings);
    }
    RestoreBinding(scalar_values_, op->loop_var.get(), saved_loop_var);
  }

  void VisitStmt_(const tir::IfThenElseNode* op) final {
    mlir::Value cond = LowerCondition(op->condition);
    bool has_else = op->else_case.defined();
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
    }
  }

  void VisitStmt_(const tir::BufferStoreNode* op) final {
    mlir::Value memref = LookupBufferValue(op->buffer);
    llvm::SmallVector<mlir::Value, 4> indices;
    indices.reserve(op->indices.size());
    for (const PrimExpr& index : op->indices) {
      indices.push_back(AsIndex(VisitExpr(index), index.dtype()));
    }

    auto emit_store = [&]() {
      mlir::Value value = VisitExpr(op->value);
      value = CastValue(value, op->value.dtype(), op->buffer->dtype);
      builder_.create<mlir::memref::StoreOp>(loc_, value, memref, indices);
    };

    if (op->predicate.defined()) {
      EmitConditionalRegion(op->predicate.value(), emit_store);
    } else {
      emit_store();
    }
  }

  void VisitStmt_(const tir::DeclBufferNode* op) final {
    std::vector<std::pair<const Object*, SavedBinding>> saved_bindings;
    auto it = buffer_values_.find(op->buffer->data.get());
    ICHECK(it != buffer_values_.end())
        << "DeclBuffer lowered before its data binding was materialized: " << op->buffer->name;
    BindBufferAliases(op->buffer, it->second, &saved_bindings);
    VisitStmt(op->body);
    RestoreBindings(buffer_values_, saved_bindings);
  }

  void VisitStmt_(const tir::AllocateNode* op) final {
    auto emit_body = [&]() {
      std::vector<std::pair<const Object*, SavedBinding>> saved_bindings;
      mlir::Value alloc = CreateAlloca(op->extents, op->dtype);
      saved_bindings.emplace_back(op->buffer_var.get(),
                                  SaveAndSet(buffer_values_, op->buffer_var.get(), alloc));
      VisitStmt(op->body);
      RestoreBindings(buffer_values_, saved_bindings);
    };
    EmitConditionalRegion(op->condition, emit_body);
  }

  void VisitStmt_(const tir::BufferRealizeNode* op) final {
    ValidateContiguousBuffer(op->buffer);
    Array<PrimExpr> extents;
    for (const Range& range : op->bounds) {
      extents.push_back(range->extent);
    }

    std::vector<std::pair<const Object*, SavedBinding>> saved_bindings;
    mlir::Value alloc = CreateAlloca(extents, op->buffer->dtype);
    BindBufferAliases(op->buffer, alloc, &saved_bindings);
    EmitConditionalRegion(op->condition, [&]() { VisitStmt(op->body); });
    RestoreBindings(buffer_values_, saved_bindings);
  }

  void VisitStmt_(const tir::AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      const auto* iter_var = op->node.as<tir::IterVarNode>();
      ICHECK(iter_var != nullptr)
          << "thread_extent is expected to bind an IterVar in linalg_riscv lowering";
      ICHECK(op->value.as<IntImmNode>() && op->value.as<IntImmNode>()->value == 1)
          << "Only unit thread extents are supported in linalg_riscv lowering";
      mlir::Type thread_type = LowerScalarType(iter_var->var.dtype());
      SavedBinding saved =
          SaveAndSet(scalar_values_, iter_var->var.get(), ConstantIntLike(0, thread_type));
      VisitStmt(op->body);
      RestoreBinding(scalar_values_, iter_var->var.get(), saved);
      return;
    }
    VisitStmt(op->body);
  }

  void VisitStmt_(const tir::BlockNode* op) final {
    ICHECK(!op->init.defined()) << "Reduction blocks are not supported yet in linalg_riscv lowering";

    const auto* body_for = FindDeferredBindingLoop(op->body);
    std::vector<std::pair<const Object*, SavedBinding>> saved_bindings;
    DeferredLoopBindings deferred;
    saved_bindings.reserve(op->alloc_buffers.size() * 2 + op->match_buffers.size() * 2);
    for (const tir::Buffer& buffer : op->alloc_buffers) {
      if (body_for != nullptr && scalar_values_.count(body_for->loop_var.get()) == 0 &&
          BufferUsesVar(buffer, body_for->loop_var)) {
        deferred.alloc_buffers.push_back(buffer);
        continue;
      }
      ValidateContiguousBuffer(buffer);
      mlir::Value alloc = CreateAlloca(buffer->shape, buffer->dtype);
      BindBufferAliases(buffer, alloc, &saved_bindings);
    }
    for (const tir::MatchBufferRegion& match_buffer : op->match_buffers) {
      if (body_for != nullptr && scalar_values_.count(body_for->loop_var.get()) == 0 &&
          MatchBufferUsesVar(match_buffer, body_for->loop_var)) {
        deferred.match_buffers.push_back(match_buffer);
        continue;
      }
      mlir::Value subview = CreateSubview(match_buffer);
      BindBufferAliases(match_buffer->buffer, subview, &saved_bindings);
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
  }

  void VisitStmt_(const tir::BlockRealizeNode* op) final {
    ICHECK_EQ(op->iter_values.size(), op->block->iter_vars.size())
        << "BlockRealize iter_values must match block iter_vars";

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
    LOG(FATAL) << "Unsupported TIR stmt for linalg_riscv MLIR lowering: " << op->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

  mlir::Value VisitExpr_(const tir::VarNode* op) final {
    return LookupVarValue(tvm::ffi::GetRef<tir::Var>(op));
  }

  mlir::Value VisitExpr_(const tir::BufferLoadNode* op) final {
    ICHECK(!op->predicate.defined() || tir::is_one(op->predicate.value()))
        << "Predicated buffer loads are not supported yet in linalg_riscv lowering";

    mlir::Value memref = LookupBufferValue(op->buffer);
    llvm::SmallVector<mlir::Value, 4> indices;
    indices.reserve(op->indices.size());
    for (const PrimExpr& index : op->indices) {
      indices.push_back(AsIndex(VisitExpr(index), index.dtype()));
    }
    mlir::Value load = builder_.create<mlir::memref::LoadOp>(loc_, memref, indices);
    return CastValue(load, op->buffer->dtype, op->dtype);
  }

  mlir::Value VisitExpr_(const tir::AddNode* op) final {
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
    if (op->dtype.is_float()) {
      return builder_.create<mlir::arith::AddFOp>(loc_, lhs, rhs);
    }
    return builder_.create<mlir::arith::AddIOp>(loc_, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::SubNode* op) final {
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
    if (op->dtype.is_float()) {
      return builder_.create<mlir::arith::SubFOp>(loc_, lhs, rhs);
    }
    return builder_.create<mlir::arith::SubIOp>(loc_, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::MulNode* op) final {
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
    if (op->dtype.is_float()) {
      return builder_.create<mlir::arith::MulFOp>(loc_, lhs, rhs);
    }
    return builder_.create<mlir::arith::MulIOp>(loc_, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::DivNode* op) final {
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
    if (op->dtype.is_float()) {
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
    if (op->dtype.is_float()) {
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
    ICHECK(!op->dtype.is_float()) << "tir.FloorDiv on floating-point dtype is not supported yet";
    if (op->dtype.is_uint() || op->dtype.is_bool()) {
      return builder_.create<mlir::arith::DivUIOp>(loc_, lhs, rhs);
    }
    return builder_.create<mlir::arith::FloorDivSIOp>(loc_, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::FloorModNode* op) final {
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), op->dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), op->dtype);
    ICHECK(!op->dtype.is_float()) << "tir.FloorMod on floating-point dtype is not supported yet";
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
    if (compare_dtype.is_float()) {
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
    if (compare_dtype.is_float()) {
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
    if (compare_dtype.is_float()) {
      return builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::OEQ, lhs,
                                                  rhs);
    }
    return builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::eq, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::NENode* op) final {
    DataType compare_dtype = op->a.dtype();
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
    if (compare_dtype.is_float()) {
      return builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::UNE, lhs,
                                                  rhs);
    }
    return builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::ne, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::LTNode* op) final {
    DataType compare_dtype = op->a.dtype();
    mlir::Value lhs = CastValue(VisitExpr(op->a), op->a.dtype(), compare_dtype);
    mlir::Value rhs = CastValue(VisitExpr(op->b), op->b.dtype(), compare_dtype);
    if (compare_dtype.is_float()) {
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
    if (compare_dtype.is_float()) {
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
    if (compare_dtype.is_float()) {
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
    if (compare_dtype.is_float()) {
      return builder_.create<mlir::arith::CmpFOp>(loc_, mlir::arith::CmpFPredicate::OGE, lhs,
                                                  rhs);
    }
    mlir::arith::CmpIPredicate predicate =
        compare_dtype.is_uint() || compare_dtype.is_bool() ? mlir::arith::CmpIPredicate::uge
                                                           : mlir::arith::CmpIPredicate::sge;
    return builder_.create<mlir::arith::CmpIOp>(loc_, predicate, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::AndNode* op) final {
    mlir::Value lhs = LowerCondition(op->a);
    mlir::Value rhs = LowerCondition(op->b);
    return builder_.create<mlir::arith::AndIOp>(loc_, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::OrNode* op) final {
    mlir::Value lhs = LowerCondition(op->a);
    mlir::Value rhs = LowerCondition(op->b);
    return builder_.create<mlir::arith::OrIOp>(loc_, lhs, rhs);
  }

  mlir::Value VisitExpr_(const tir::NotNode* op) final {
    mlir::Value value = LowerCondition(op->a);
    mlir::Value one = ConstantIntLike(1, builder_.getI1Type());
    return builder_.create<mlir::arith::XOrIOp>(loc_, value, one);
  }

  mlir::Value VisitExpr_(const tir::SelectNode* op) final {
    mlir::Value cond = LowerCondition(op->condition);
    mlir::Value true_value = CastValue(VisitExpr(op->true_value), op->true_value.dtype(), op->dtype);
    mlir::Value false_value =
        CastValue(VisitExpr(op->false_value), op->false_value.dtype(), op->dtype);
    return builder_.create<mlir::arith::SelectOp>(loc_, cond, true_value, false_value);
  }

  mlir::Value VisitExpr_(const IntImmNode* op) final {
    mlir::Type type = LowerScalarType(op->dtype);
    return ConstantIntLike(op->value, type);
  }

  mlir::Value VisitExpr_(const FloatImmNode* op) final {
    mlir::FloatType type = mlir::cast<mlir::FloatType>(LowerScalarType(op->dtype));
    return builder_.create<mlir::arith::ConstantOp>(loc_, builder_.getFloatAttr(type, op->value));
  }

  mlir::Value VisitExpr_(const tir::CallNode* op) final {
    ICHECK(IsSupportedUnaryMathCall(op))
        << "Unsupported TIR expr for linalg_riscv MLIR lowering: " << op->op;
    mlir::Value arg = CastValue(VisitExpr(op->args[0]), op->args[0].dtype(), op->dtype);
    return LowerSupportedUnaryMathCall(builder_, loc_, op, arg);
  }

  mlir::Value VisitExprDefault_(const Object* op) final {
    LOG(FATAL) << "Unsupported TIR expr for linalg_riscv MLIR lowering: " << op->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

  mlir::DialectRegistry registry_;
  arith::Analyzer analyzer_;
  mlir::MLIRContext context_;
  mlir::OpBuilder builder_;
  mlir::Location loc_;
  mlir::ModuleOp module_;
  ValueMap scalar_values_;
  ValueMap buffer_values_;
  std::unordered_map<const Object*, std::vector<DeferredLoopBindings>> deferred_loop_bindings_;
};

std::string BuildStructuredMLIRModule(const std::vector<FunctionEntry>& functions) {
  TIRToMLIRLowerer lowerer;
  return lowerer.Lower(functions);
}
#endif

}  // namespace

void CodeGenTileLangLinalgRISCV::AddFunction(const GlobalVar& gvar, const tir::PrimFunc& func) {
  std::string name;
  if (auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
    name = global_symbol.value();
  } else {
    name = gvar->name_hint;
  }
  function_names_.push_back(name);
  functions_.emplace_back(name, func);
}

std::string CodeGenTileLangLinalgRISCV::Finish() const {
#if TILELANG_ENABLE_LINALG_RISCV_MLIR
  return BuildStructuredMLIRModule(functions_);
#else
  return BuildPlaceholderModule(functions_);
#endif
}

}  // namespace codegen
}  // namespace tvm
