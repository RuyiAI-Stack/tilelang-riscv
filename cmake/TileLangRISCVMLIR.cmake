include_guard(GLOBAL)

set(TILELANG_RISCV_MLIR_MODE "AUTO" CACHE STRING
    "Enable the external LLVM/MLIR integration for the RISC-V backend (AUTO, ON, OFF)")
set_property(CACHE TILELANG_RISCV_MLIR_MODE PROPERTY STRINGS AUTO ON OFF)
set(TILELANG_RISCV_LLVM_ROOT "" CACHE PATH
    "LLVM/MLIR install prefix used by the RISC-V backend")

function(tilelang_configure_riscv_mlir)
  string(TOUPPER "${TILELANG_RISCV_MLIR_MODE}" _tilelang_riscv_mlir_mode)
  if(_tilelang_riscv_mlir_mode STREQUAL "OFF")
    set(TILELANG_RISCV_MLIR_FOUND FALSE PARENT_SCOPE)
    set(TILELANG_RISCV_MLIR_ENABLED FALSE PARENT_SCOPE)
    set(TILELANG_RISCV_MLIR_ROOT "" PARENT_SCOPE)
    set(TILELANG_RISCV_MLIR_INCLUDE_DIRS "" PARENT_SCOPE)
    set(TILELANG_RISCV_MLIR_LINK_LIBS "" PARENT_SCOPE)
    return()
  endif()

  set(_candidate_roots)
  foreach(_value
      "${TILELANG_RISCV_LLVM_ROOT}"
      "$ENV{TILELANG_RISCV_LLVM_ROOT}"
      "$ENV{TILELANG_LLVM_INSTALL_DIR}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../buddy-mlir/llvm/build"
      "${CMAKE_CURRENT_SOURCE_DIR}/../buddy-mlir/llvm/install"
      "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llvm-project/install"
      "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llvm-project/build-host/install"
      "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llvm-project/build/install")
    if(_value AND IS_DIRECTORY "${_value}")
      cmake_path(ABSOLUTE_PATH _value NORMALIZE OUTPUT_VARIABLE _resolved)
      list(APPEND _candidate_roots "${_resolved}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _candidate_roots)

  set(_found_root "")
  set(_llvm_dir "")
  set(_mlir_dir "")
  foreach(_root IN LISTS _candidate_roots)
    foreach(_lib_dir IN ITEMS lib lib64)
      if(EXISTS "${_root}/${_lib_dir}/cmake/llvm/LLVMConfig.cmake"
         AND EXISTS "${_root}/${_lib_dir}/cmake/mlir/MLIRConfig.cmake")
        set(_found_root "${_root}")
        set(_llvm_dir "${_root}/${_lib_dir}/cmake/llvm")
        set(_mlir_dir "${_root}/${_lib_dir}/cmake/mlir")
        break()
      endif()
    endforeach()
    if(_found_root)
      break()
    endif()
  endforeach()

  if(NOT _found_root)
    if(_tilelang_riscv_mlir_mode STREQUAL "ON")
      message(FATAL_ERROR
        "TILELANG_RISCV_MLIR_MODE=ON but no LLVM/MLIR install was found. "
        "Build Buddy MLIR release/v0.1.0 as described in "
        "docs/get_started/BuildOnSG2044.md or set TILELANG_RISCV_LLVM_ROOT.")
    endif()
    message(STATUS
      "LLVM/MLIR toolchain for RISC-V not found; keeping placeholder backend build. "
      "Set TILELANG_RISCV_LLVM_ROOT to a Buddy MLIR v0.1.0 LLVM/MLIR install "
      "as described in docs/get_started/BuildOnSG2044.md to enable it.")
    set(TILELANG_RISCV_MLIR_FOUND FALSE PARENT_SCOPE)
    set(TILELANG_RISCV_MLIR_ENABLED FALSE PARENT_SCOPE)
    set(TILELANG_RISCV_MLIR_ROOT "" PARENT_SCOPE)
    set(TILELANG_RISCV_MLIR_INCLUDE_DIRS "" PARENT_SCOPE)
    set(TILELANG_RISCV_MLIR_LINK_LIBS "" PARENT_SCOPE)
    return()
  endif()

  # Buddy MLIR release/v0.1.0 is based on LLVM/MLIR 24.0.0git. Require the
  # supported major/minor version so an older, API-incompatible checkout fails
  # during configuration instead of later in a C++ compile or at runtime.
  find_package(LLVM 24.0 REQUIRED CONFIG PATHS "${_llvm_dir}" NO_DEFAULT_PATH)
  find_package(MLIR 24.0 REQUIRED CONFIG PATHS "${_mlir_dir}" NO_DEFAULT_PATH)

  set(_include_dirs ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
  list(REMOVE_DUPLICATES _include_dirs)

  message(STATUS "Using LLVM/MLIR toolchain for riscv: ${_found_root}")

  set(TILELANG_RISCV_MLIR_FOUND TRUE PARENT_SCOPE)
  set(TILELANG_RISCV_MLIR_ENABLED TRUE PARENT_SCOPE)
  set(TILELANG_RISCV_MLIR_ROOT "${_found_root}" PARENT_SCOPE)
  set(TILELANG_RISCV_MLIR_INCLUDE_DIRS "${_include_dirs}" PARENT_SCOPE)
  set(TILELANG_RISCV_MLIR_LINK_LIBS
      "MLIRIR;MLIRFuncDialect;MLIRArithDialect;MLIRLinalgDialect;MLIRMemRefDialect;MLIRSCFDialect;MLIRVectorDialect"
      PARENT_SCOPE)
endfunction()
