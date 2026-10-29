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

// MRV2 host sampling ops from vLLM csrc/cpu/sampling_kernels.cpp (vllm#61995).
#include "cpu_types.hpp"
#include "utils.hpp"
#include "sampling/vec.h"

#include <torch/all.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

template <typename scalar_t>
using SelectedSamplingImpl =
    cpu_sampling::SamplingImpl<cpu_utils::ISA::VEC, scalar_t>;

torch::Tensor gumbel_sample_cpu(const torch::Tensor& logits,
                                const torch::Tensor& expanded_idx_mapping,
                                const torch::Tensor& temperature,
                                const torch::Tensor& seeds,
                                const torch::Tensor& positions,
                                bool apply_temperature, bool is_drafting,
                                bool use_fp64) {
  TORCH_CHECK(logits.dim() == 2, "logits must be 2-D [num_tokens, vocab]");
  TORCH_CHECK(logits.stride(1) == 1,
              "the logits vocabulary dimension must be contiguous");

  auto output =
      torch::empty({logits.size(0)}, logits.options().dtype(torch::kInt64));
  VLLM_DISPATCH_FLOATING_TYPES(logits.scalar_type(), "gumbel_sample_cpu", [&] {
    using sampling_impl_t = SelectedSamplingImpl<scalar_t>;
    cpu_sampling::SamplingMainLoop<sampling_impl_t> mainloop;
    mainloop(output.data_ptr<int64_t>(), logits.data_ptr<scalar_t>(),
             logits.stride(0), expanded_idx_mapping.data_ptr<int64_t>(),
             temperature.data_ptr<float>(), seeds.data_ptr<int64_t>(),
             positions.data_ptr<int64_t>(), logits.size(0), logits.size(1),
             apply_temperature, is_drafting, use_fp64);
  });
  return output;
}

FORCE_INLINE void fill_logits(float* logits, int64_t size, float value) {
  constexpr int64_t VEC_SIZE = vec_op::FP32Vec16::VEC_ELEM_NUM;
  const vec_op::FP32Vec16 values(value);
  const int64_t vectorized_size = size - size % VEC_SIZE;
  int64_t offset = 0;
  for (; offset < vectorized_size; offset += VEC_SIZE) {
    values.save(logits + offset);
  }
  std::fill(logits + offset, logits + size, value);
}

void apply_logit_bias_cpu(
    torch::Tensor& logits, const torch::Tensor& expanded_idx_mapping,
    const torch::Tensor& positions, const torch::Tensor& num_allowed_token_ids,
    const torch::Tensor& allowed_token_ids, const torch::Tensor& num_logit_bias,
    const torch::Tensor& logit_bias_token_ids, const torch::Tensor& logit_bias,
    const torch::Tensor& min_lens, const torch::Tensor& num_stop_token_ids,
    const torch::Tensor& restore_when_all_masked,
    const torch::Tensor& stop_token_ids, bool check_all_masked_rows) {
  constexpr int64_t MAX_ALLOWED_TOKEN_IDS = 1024;
  constexpr int64_t MAX_STOP_TOKEN_IDS = 128;
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  const int64_t num_tokens = logits.size(0);
  const int64_t vocab_size = logits.size(1);

  float* logits_ptr = logits.data_ptr<float>();
  const int64_t* idx_mapping_ptr = expanded_idx_mapping.data_ptr<int64_t>();
  const int64_t* positions_ptr = positions.data_ptr<int64_t>();
  const int32_t* num_allowed_ptr = num_allowed_token_ids.data_ptr<int32_t>();
  const int32_t* allowed_ptr = allowed_token_ids.data_ptr<int32_t>();
  const int32_t* num_bias_ptr = num_logit_bias.data_ptr<int32_t>();
  const int32_t* bias_token_ids_ptr = logit_bias_token_ids.data_ptr<int32_t>();
  const float* bias_ptr = logit_bias.data_ptr<float>();
  const int32_t* min_lens_ptr = min_lens.data_ptr<int32_t>();
  const int32_t* num_stop_ptr = num_stop_token_ids.data_ptr<int32_t>();
  const int32_t* restore_ptr = restore_when_all_masked.data_ptr<int32_t>();
  const int32_t* stop_token_ids_ptr = stop_token_ids.data_ptr<int32_t>();

  const int64_t logits_stride = logits.stride(0);
  const int64_t allowed_stride = allowed_token_ids.stride(0);
  const int64_t bias_ids_stride = logit_bias_token_ids.stride(0);
  const int64_t bias_stride = logit_bias.stride(0);
  const int64_t stop_stride = stop_token_ids.stride(0);

#pragma omp parallel for schedule(static) if (num_tokens > 1)
  for (int64_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
    const int64_t req_idx = idx_mapping_ptr[token_idx];
    float* row = logits_ptr + token_idx * logits_stride;

    const int32_t num_allowed = num_allowed_ptr[req_idx];
    if (num_allowed > 0) {
      std::array<float, MAX_ALLOWED_TOKEN_IDS> saved_logits;
      const int32_t* row_allowed = allowed_ptr + req_idx * allowed_stride;
      for (int32_t i = 0; i < num_allowed; ++i) {
        saved_logits[i] = row[row_allowed[i]];
      }
      fill_logits(row, vocab_size, negative_infinity);
      for (int32_t i = 0; i < num_allowed; ++i) {
        row[row_allowed[i]] = saved_logits[i];
      }
    }

    const int32_t num_bias = num_bias_ptr[req_idx];
    const int32_t* row_bias_ids =
        bias_token_ids_ptr + req_idx * bias_ids_stride;
    const float* row_bias = bias_ptr + req_idx * bias_stride;
    for (int32_t i = 0; i < num_bias; ++i) {
      row[row_bias_ids[i]] += row_bias[i];
    }

    const int32_t num_stop = num_stop_ptr[req_idx];
    if (num_stop == 0 ||
        positions_ptr[token_idx] + 1 >= min_lens_ptr[req_idx]) {
      continue;
    }

    const int32_t* row_stop_ids = stop_token_ids_ptr + req_idx * stop_stride;
    if (!check_all_masked_rows || restore_ptr[req_idx] == 0) {
      for (int32_t i = 0; i < num_stop; ++i) {
        row[row_stop_ids[i]] = negative_infinity;
      }
      continue;
    }

    std::array<float, MAX_STOP_TOKEN_IDS> saved_stop_logits;
    for (int32_t i = 0; i < num_stop; ++i) {
      saved_stop_logits[i] = row[row_stop_ids[i]];
      row[row_stop_ids[i]] = negative_infinity;
    }

    const bool all_masked =
        std::all_of(row, row + vocab_size, [negative_infinity](float value) {
          return value == negative_infinity;
        });
    if (all_masked) {
      for (int32_t i = 0; i < num_stop; ++i) {
        const float value = saved_stop_logits[i];
        if (std::isfinite(value)) {
          row[row_stop_ids[i]] = value;
        }
      }
    }
  }
}
