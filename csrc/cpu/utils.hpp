// Copyright 2026 The Spyre-Inference Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Portions adapted from vLLM (https://github.com/vllm-project/vllm),
// Copyright contributors to the vLLM project, Apache-2.0.

// Vendored from vLLM csrc/cpu/{utils.hpp,cpu_types*.hpp} (vllm#61995), trimmed to
// what the MRV2 host kernels use; each kept line is unchanged from vLLM.
#ifndef UTILS_HPP
#define UTILS_HPP

#include <atomic>
#include <cstdint>

#include "cpu/cpu_types.hpp"

#include <ATen/Dispatch.h>
#include <c10/util/Exception.h>

#ifndef FORCE_INLINE
  #define FORCE_INLINE __attribute__((always_inline)) inline
#endif

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)            \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)    \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

namespace cpu_utils {
enum class ISA { AMX, VEC, RVV, NEON, VSX };

// Without OpenMP the omp pragmas compile to serial loops, so report 1: kernels
// that barrier on the thread count would otherwise deadlock.
inline int get_max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  TORCH_WARN_ONCE(
      "vLLM CPU was built without OpenMP; running single-threaded.");
  return 1;
#endif
}

struct Counter {
  std::atomic<int64_t> counter;
  char _padding[56];

  Counter() : counter(0) {}

  void reset_counter() { counter.store(0); }

  int64_t acquire_counter() { return counter++; }
};
}  // namespace cpu_utils

#endif
