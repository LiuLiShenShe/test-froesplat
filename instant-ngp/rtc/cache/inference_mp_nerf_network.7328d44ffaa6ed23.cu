			/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
			 */

			/** @file   inference_mp_nerf_network.cu
			 *  @author Thomas Müller, NVIDIA
			 *  @brief  Automatically generated kernel inference_mp_nerf_network
			 */

			/* Compiler options
			--gpu-architecture=compute_86
-DTCNN_HALF_PRECISION=1
-DTCNN_MIN_GPU_ARCH=86
--std=c++14
--use_fast_math
--extra-device-vectorization
			*/

			// NVRTC does not come with the C++ standard library out of the box and
			// it would be troublesome to bundle it or require users to have it installed
			// in readily available paths. So we instead include a minimal custom
			// implementation of just those function of std:: that we require.
			#include <tiny-cuda-nn/ministd.h>
			#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/mma.h>

using namespace tcnn;

			__device__ auto eval_model_density_network(const tvec<__half, 32>& input, const __half* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<__half, 16> {

	mma_vec<32> in{input};
	if (fwd_ctx) {
		in.into_native_memory((__half*)fwd_ctx);
		fwd_ctx += 32 * sizeof(__half) * 32;
	}

	auto first_mat = mma_mat<32, 64>::from_native_memory(params);
	params += 32 * 64;

	auto hidden = in * first_mat;
	hidden.activate<Activation::ReLU>();

	if (fwd_ctx) {
		hidden.into_native_memory((__half*)fwd_ctx);
		fwd_ctx += 32 * sizeof(__half) * 64;
	}

	TCNN_PRAGMA_UNROLL
	for (uint32_t i = 0; i < 0; ++i) {
		auto hidden_mat = mma_mat<64, 64>::from_native_memory(params);
		params += 64 * 64;
		hidden = hidden * hidden_mat;
		hidden.activate<Activation::ReLU>();

		if (fwd_ctx) {
			hidden.into_native_memory((__half*)fwd_ctx);
			fwd_ctx += 32 * sizeof(__half) * 64;
		}
	}

	auto last_mat = mma_mat<64, 16>::from_native_memory(params);
	auto out = hidden * last_mat;
	

	return out.vec<16>();
}

__device__ auto eval_model_rgb_network(const tvec<__half, 32>& input, const __half* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<__half, 3> {

	mma_vec<32> in{input};
	if (fwd_ctx) {
		in.into_native_memory((__half*)fwd_ctx);
		fwd_ctx += 32 * sizeof(__half) * 32;
	}

	auto first_mat = mma_mat<32, 64>::from_native_memory(params);
	params += 32 * 64;

	auto hidden = in * first_mat;
	hidden.activate<Activation::ReLU>();

	if (fwd_ctx) {
		hidden.into_native_memory((__half*)fwd_ctx);
		fwd_ctx += 32 * sizeof(__half) * 64;
	}

	TCNN_PRAGMA_UNROLL
	for (uint32_t i = 0; i < 1; ++i) {
		auto hidden_mat = mma_mat<64, 64>::from_native_memory(params);
		params += 64 * 64;
		hidden = hidden * hidden_mat;
		hidden.activate<Activation::ReLU>();

		if (fwd_ctx) {
			hidden.into_native_memory((__half*)fwd_ctx);
			fwd_ctx += 32 * sizeof(__half) * 64;
		}
	}

	auto last_mat = mma_mat<64, 16>::from_native_memory(params);
	auto out = hidden * last_mat;
	

	return out.vec<3>();
}

__device__ auto eval_model_pos_encoding_lookup(
	const float scale,
	vec<3> pos,
	const __half* __restrict__ grid,
	const uint32_t hashmap_size
) -> tvec<__half, 4, 8> {
	const uint32_t resolution = grid_resolution(scale);

	auto grid_val = [&](const uvec<3>& local_pos) {
		const uint32_t index = grid_index<3, HashType::CoherentPrime>(GridType::Hash, hashmap_size, resolution, local_pos) * 4;
		return *(tvec<__half, 4, 8>*)&grid[index];
	};

	uvec<3> pos_grid;
	if (false) {
		uint32_t cell_size_fixed = max((uint32_t)((float)0xFFFFFFFF / scale), 1);
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < 3; ++i) {
			uint32_t pos_fixed = __float_as_uint(pos[i]) + cell_size_fixed / 2;
			pos_grid[i] = pos_fixed / cell_size_fixed + ((pos_fixed < cell_size_fixed / 2) ? (uint32_t)scale : 0);
			pos[i] = scale / (float)0xFFFFFFFF * (pos_fixed - pos_grid[i] * cell_size_fixed);
		}
	} else {
		pos = fma(scale, pos, 0.5f);
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < 3; ++i) {
			float tmp = floor(pos[i]);
			pos[i] -= tmp;
			pos_grid[i] = (uint32_t)(int)tmp;
		}
	}

	if (InterpolationType::Linear == InterpolationType::Nearest) {
		return grid_val(pos_grid);
	}

	if (InterpolationType::Linear == InterpolationType::Smoothstep) {
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < 3; ++i) {
			pos[i] = smoothstep(pos[i]);
		}
	}

	tvec<__half, 4, 8> result((__half)0.0f);

	TCNN_PRAGMA_UNROLL
	for (uint32_t idx = 0; idx < (1 << 3); ++idx) {
		float weight = 1.0f;
		uvec<3> pos_grid_local = pos_grid;

		TCNN_PRAGMA_UNROLL
		for (uint32_t dim = 0; dim < 3; ++dim) {
			weight *= ((idx >> dim) & 1) ? pos[dim] : (1.0f - pos[dim]);
			pos_grid_local[dim] += (idx >> dim) & 1;
		}

		result = fma((__half)weight, grid_val(pos_grid_local), result);
	}

	return result;
}

__device__ auto eval_model_pos_encoding(const tvec<float, 3>& input, const __half* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<__half, 32> {
	if (fwd_ctx) fwd_ctx += lane_id() * 12;
	if (fwd_ctx) { input.to_array((float*)fwd_ctx); }
	tvec<__half, 32> result;
	result.slice<0, 4>() = eval_model_pos_encoding_lookup((float)15.0000000000, input, params + 0, 4096);
	result.slice<4, 4>() = eval_model_pos_encoding_lookup((float)38.0084381104, input, params + 16384, 64000);
	result.slice<8, 4>() = eval_model_pos_encoding_lookup((float)94.1036453247, input, params + 272384, 524288);
	result.slice<12, 4>() = eval_model_pos_encoding_lookup((float)230.8653106689, input, params + 2369536, 524288);
	result.slice<16, 4>() = eval_model_pos_encoding_lookup((float)564.2940063477, input, params + 4466688, 524288);
	result.slice<20, 4>() = eval_model_pos_encoding_lookup((float)1377.2020263672, input, params + 6563840, 524288);
	result.slice<24, 4>() = eval_model_pos_encoding_lookup((float)3359.0949707031, input, params + 8660992, 524288);
	result.slice<28, 4>() = eval_model_pos_encoding_lookup((float)8191.0058593750, input, params + 10758144, 524288);
	for (uint32_t i = 32; i < 32; ++i) {
		result[i] = (__half)0.0f;
	}
	return result;
}

__device__ auto eval_model_dir_encoding_0_spherical_harmonics(const tvec<float, 3>& input, const __half* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<__half, 16> {
	if (fwd_ctx) fwd_ctx += lane_id() * 12;
	vec3 d = 2.0f * input - 1.0f;
	if (fwd_ctx) {
		d.to_array((float*)fwd_ctx);
	}

	tvec<__half, 16> result;
	sh_enc<__half, tvec<__half, 16>>(4, d[0], d[1], d[2], result);
	TCNN_PRAGMA_UNROLL
	for (uint32_t i = 16; i < 16; ++i) {
		result[i] = (__half)1.0f;
	}
	return result;
}

__device__ auto eval_model_dir_encoding(const tvec<float, 3>& input, const __half* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<__half, 16> {

	tvec<__half, 16> result;
	result.slice<0, 16>() = eval_model_dir_encoding_0_spherical_harmonics(input.slice<0, 3>(), params + 0, fwd_ctx ? fwd_ctx + WARP_SIZE * 0 : nullptr);
	return result;
}

__device__ auto eval_model(const tvec<float, 7>& input, const __half* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<__half, 4> {

	auto pos_enc_out = eval_model_pos_encoding(input.slice<0, 3>(), params + 10240, fwd_ctx ? fwd_ctx + WARP_SIZE * 512 : nullptr);

	tvec<__half, 32> rgb_mlp_in;
	rgb_mlp_in.slice<0, 16>() = eval_model_density_network(pos_enc_out, params, fwd_ctx);
	rgb_mlp_in.slice<16, 16>() = eval_model_dir_encoding(input.slice<4, 3>(), params + 12865536, fwd_ctx ? fwd_ctx + WARP_SIZE * 524 : nullptr);

	auto rgb_mlp_out = eval_model_rgb_network(rgb_mlp_in, params + 3072, fwd_ctx ? fwd_ctx + WARP_SIZE * 192 : nullptr);

	return {rgb_mlp_out[0], rgb_mlp_out[1], rgb_mlp_out[2], rgb_mlp_in[0]};
}

__global__ void inference_mp_nerf_network(const uint32_t num_elements, MatrixView<const float> data_in, MatrixView<__half> data_out, const __half* __restrict__ params) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

	auto input = data_in.col<7>(i);
	auto output = eval_model(input, params, nullptr);
	if (data_out) {
		data_out.set_col(i, output);
	}
}