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

#pragma once

#include "common.h"

namespace cpu_sampling {

template <typename scalar_t>
class SamplingImpl<cpu_utils::ISA::VEC, scalar_t> {
 public:
  using scalar_type = scalar_t;
  static constexpr int64_t RangeAlignment = 1;

  void greedy_argmax(const scalar_t* logits, int64_t token_begin,
                     int64_t token_end, float& best_score,
                     int64_t& best_idx) const {
    scalar_greedy_argmax(logits, token_begin, token_end, best_score, best_idx);
  }

  template <typename score_t, bool APPLY_TEMPERATURE>
  void gumbel_argmax(const scalar_t* logits, int64_t token_begin,
                     int64_t token_end, int64_t seed, int64_t position,
                     float temperature, score_t& best_score,
                     int64_t& best_idx) const {
    scalar_gumbel_argmax<scalar_t, score_t, APPLY_TEMPERATURE>(
        logits, token_begin, token_end, seed, position, temperature, best_score,
        best_idx);
  }
};

}  // namespace cpu_sampling
