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

// The no-Triton MRV2 host ops (vllm#61995), under vLLM's own `_C` namespace so
// vllm/_custom_ops.py resolves them unchanged. A fragment, not TORCH_LIBRARY: a
// VLLM_TARGET_DEVICE=empty vLLM has no `_C`, and a CPU build would register the
// same schemas, so this extension must not be loaded alongside one.

#include <torch/all.h>
#include <torch/library.h>

torch::Tensor gumbel_sample_cpu(const torch::Tensor& logits,
                                const torch::Tensor& expanded_idx_mapping,
                                const torch::Tensor& temperature,
                                const torch::Tensor& seeds,
                                const torch::Tensor& positions,
                                bool apply_temperature, bool is_drafting,
                                bool use_fp64);

void apply_logit_bias_cpu(
    torch::Tensor& logits, const torch::Tensor& expanded_idx_mapping,
    const torch::Tensor& positions, const torch::Tensor& num_allowed_token_ids,
    const torch::Tensor& allowed_token_ids, const torch::Tensor& num_logit_bias,
    const torch::Tensor& logit_bias_token_ids, const torch::Tensor& logit_bias,
    const torch::Tensor& min_lens, const torch::Tensor& num_stop_token_ids,
    const torch::Tensor& restore_when_all_masked,
    const torch::Tensor& stop_token_ids, bool check_all_masked_rows);

namespace cpu_utils {
void gather_block_table_cpu(const torch::Tensor& block_table,
                            torch::Tensor& output,
                            const torch::Tensor& idx_mapping,
                            const torch::Tensor& num_blocks,
                            const int64_t num_reqs_padded);
void compute_slot_mapping_cpu(
    const torch::Tensor& block_table, torch::Tensor& slot_mapping,
    const torch::Tensor& idx_mapping, const torch::Tensor& query_start_loc,
    const torch::Tensor& positions, const int64_t block_size,
    const bool mapping_enabled, const int64_t pad_slot_id);
void prepare_prefill_inputs_cpu(torch::Tensor& input_ids,
                                torch::Tensor& next_prefill_tokens,
                                const torch::Tensor& idx_mapping,
                                const torch::Tensor& query_start_loc,
                                const torch::Tensor& all_token_ids,
                                const torch::Tensor& prefill_len,
                                const torch::Tensor& num_computed_tokens);
torch::Tensor get_prompt_logprobs_token_ids_cpu(
    int64_t num_tokens, const torch::Tensor& query_start_loc,
    const torch::Tensor& idx_mapping, const torch::Tensor& num_computed_tokens,
    const torch::Tensor& all_token_ids);
void prepare_pos_seq_lens_cpu(const torch::Tensor& idx_mapping,
                              const torch::Tensor& query_start_loc,
                              const torch::Tensor& num_computed_tokens,
                              torch::Tensor& pos, torch::Tensor& seq_lens);
void prepare_mrope_positions_cpu(torch::Tensor& positions,
                                 const torch::Tensor& prefill_positions,
                                 const torch::Tensor& prefill_delta,
                                 const torch::Tensor& idx_mapping,
                                 const torch::Tensor& query_start_loc,
                                 const torch::Tensor& prefill_lens,
                                 const torch::Tensor& num_computed_tokens);
torch::Tensor combine_sampled_and_draft_tokens_cpu(
    torch::Tensor& input_ids, const torch::Tensor& idx_mapping,
    const torch::Tensor& last_sampled_tokens,
    const torch::Tensor& query_start_loc, const torch::Tensor& seq_lens,
    const torch::Tensor& prefill_len, const torch::Tensor& draft_tokens,
    const torch::Tensor& cu_num_logits, const int64_t num_logits,
    const int64_t num_new_sampled_tokens);
void post_update_cpu(const torch::Tensor& idx_mapping,
                     torch::Tensor& num_computed_tokens,
                     torch::Tensor& last_sampled_tokens,
                     const c10::optional<torch::Tensor>& output_bin_counts,
                     const torch::Tensor& sampled_tokens,
                     const torch::Tensor& num_sampled,
                     const torch::Tensor& num_rejected,
                     const c10::optional<torch::Tensor>& query_start_loc,
                     torch::Tensor& all_token_ids, torch::Tensor& total_len);
}  // namespace cpu_utils

TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "gather_block_table_cpu(Tensor block_table, Tensor(a1!) output, "
      "Tensor idx_mapping, Tensor num_blocks, SymInt num_reqs_padded) -> ()",
      &cpu_utils::gather_block_table_cpu);

  ops.def(
      "compute_slot_mapping_cpu("
      "Tensor block_table, Tensor(a1!) slot_mapping, Tensor idx_mapping, "
      "Tensor query_start_loc, Tensor positions, SymInt block_size, "
      "bool mapping_enabled, SymInt pad_slot_id) -> ()",
      &cpu_utils::compute_slot_mapping_cpu);

  ops.def(
      "prepare_prefill_inputs_cpu(Tensor(a0!) input_ids, "
      "Tensor(a1!) next_prefill_tokens, Tensor idx_mapping, "
      "Tensor query_start_loc, Tensor all_token_ids, Tensor prefill_len, "
      "Tensor num_computed_tokens) -> ()",
      &cpu_utils::prepare_prefill_inputs_cpu);

  ops.def(
      "get_prompt_logprobs_token_ids_cpu("
      "SymInt num_tokens, Tensor query_start_loc, Tensor idx_mapping, "
      "Tensor num_computed_tokens, Tensor all_token_ids) -> Tensor",
      &cpu_utils::get_prompt_logprobs_token_ids_cpu);

  ops.def(
      "prepare_pos_seq_lens_cpu(Tensor idx_mapping, Tensor query_start_loc, "
      "Tensor num_computed_tokens, Tensor(a3!) pos, "
      "Tensor(a4!) seq_lens) -> ()",
      &cpu_utils::prepare_pos_seq_lens_cpu);

  ops.def(
      "prepare_mrope_positions_cpu("
      "Tensor(a0!) positions, Tensor prefill_positions, Tensor prefill_delta, "
      "Tensor idx_mapping, Tensor query_start_loc, Tensor prefill_lens, "
      "Tensor num_computed_tokens) -> ()",
      &cpu_utils::prepare_mrope_positions_cpu);

  ops.def(
      "combine_sampled_and_draft_tokens_cpu("
      "Tensor(a0!) input_ids, Tensor idx_mapping, Tensor last_sampled_tokens, "
      "Tensor query_start_loc, Tensor seq_lens, Tensor prefill_len, "
      "Tensor draft_tokens, Tensor cu_num_logits, SymInt num_logits, "
      "SymInt num_new_sampled_tokens) -> Tensor",
      &cpu_utils::combine_sampled_and_draft_tokens_cpu);

  ops.def(
      "post_update_cpu(Tensor idx_mapping, "
      "Tensor(a1!) num_computed_tokens, Tensor(a2!) last_sampled_tokens, "
      "Tensor(a3!)? output_bin_counts, Tensor sampled_tokens, "
      "Tensor num_sampled, Tensor num_rejected, Tensor? query_start_loc, "
      "Tensor(a8!) all_token_ids, Tensor(a9!) total_len) -> ()",
      &cpu_utils::post_update_cpu);

  ops.def(
      "gumbel_sample_cpu(Tensor logits, Tensor expanded_idx_mapping, "
      "Tensor temperature, Tensor seeds, Tensor positions, "
      "bool apply_temperature, bool is_drafting, bool use_fp64) -> Tensor");
  ops.impl("gumbel_sample_cpu", torch::kCPU, &gumbel_sample_cpu);

  ops.def(
      "apply_logit_bias_cpu(Tensor(a0!) logits, "
      "Tensor expanded_idx_mapping, Tensor positions, "
      "Tensor num_allowed_token_ids, Tensor allowed_token_ids, "
      "Tensor num_logit_bias, Tensor logit_bias_token_ids, "
      "Tensor logit_bias, Tensor min_lens, Tensor num_stop_token_ids, "
      "Tensor restore_when_all_masked, Tensor stop_token_ids, "
      "bool check_all_masked_rows) -> ()",
      &apply_logit_bias_cpu);
}
