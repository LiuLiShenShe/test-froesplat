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

			/** @file   training_step_nerf_network.cu
			 *  @author Thomas Müller, NVIDIA
			 *  @brief  Automatically generated kernel training_step_nerf_network
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

			__device__ auto training_step_nerf_network_forward_density_network(const tvec<__half, 32>& input, const __half* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<__half, 16> {

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

__device__ auto training_step_nerf_network_forward_rgb_network(const tvec<__half, 32>& input, const __half* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<__half, 3> {

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

__device__ auto training_step_nerf_network_forward_pos_encoding_lookup(
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

__device__ auto training_step_nerf_network_forward_pos_encoding(const tvec<float, 3>& input, const __half* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<__half, 32> {
	if (fwd_ctx) fwd_ctx += lane_id() * 12;
	if (fwd_ctx) { input.to_array((float*)fwd_ctx); }
	tvec<__half, 32> result;
	result.slice<0, 4>() = training_step_nerf_network_forward_pos_encoding_lookup((float)15.0000000000, input, params + 0, 4096);
	result.slice<4, 4>() = training_step_nerf_network_forward_pos_encoding_lookup((float)38.0084381104, input, params + 16384, 64000);
	result.slice<8, 4>() = training_step_nerf_network_forward_pos_encoding_lookup((float)94.1036453247, input, params + 272384, 524288);
	result.slice<12, 4>() = training_step_nerf_network_forward_pos_encoding_lookup((float)230.8653106689, input, params + 2369536, 524288);
	result.slice<16, 4>() = training_step_nerf_network_forward_pos_encoding_lookup((float)564.2940063477, input, params + 4466688, 524288);
	result.slice<20, 4>() = training_step_nerf_network_forward_pos_encoding_lookup((float)1377.2020263672, input, params + 6563840, 524288);
	result.slice<24, 4>() = training_step_nerf_network_forward_pos_encoding_lookup((float)3359.0949707031, input, params + 8660992, 524288);
	result.slice<28, 4>() = training_step_nerf_network_forward_pos_encoding_lookup((float)8191.0058593750, input, params + 10758144, 524288);
	for (uint32_t i = 32; i < 32; ++i) {
		result[i] = (__half)0.0f;
	}
	return result;
}

__device__ auto training_step_nerf_network_forward_dir_encoding_0_spherical_harmonics(const tvec<float, 3>& input, const __half* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<__half, 16> {
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

__device__ auto training_step_nerf_network_forward_dir_encoding(const tvec<float, 3>& input, const __half* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<__half, 16> {

	tvec<__half, 16> result;
	result.slice<0, 16>() = training_step_nerf_network_forward_dir_encoding_0_spherical_harmonics(input.slice<0, 3>(), params + 0, fwd_ctx ? fwd_ctx + WARP_SIZE * 0 : nullptr);
	return result;
}

__device__ auto training_step_nerf_network_forward(const tvec<float, 7>& input, const __half* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<__half, 4> {

	auto pos_enc_out = training_step_nerf_network_forward_pos_encoding(input.slice<0, 3>(), params + 10240, fwd_ctx ? fwd_ctx + WARP_SIZE * 512 : nullptr);

	tvec<__half, 32> rgb_mlp_in;
	rgb_mlp_in.slice<0, 16>() = training_step_nerf_network_forward_density_network(pos_enc_out, params, fwd_ctx);
	rgb_mlp_in.slice<16, 16>() = training_step_nerf_network_forward_dir_encoding(input.slice<4, 3>(), params + 12865536, fwd_ctx ? fwd_ctx + WARP_SIZE * 524 : nullptr);

	auto rgb_mlp_out = training_step_nerf_network_forward_rgb_network(rgb_mlp_in, params + 3072, fwd_ctx ? fwd_ctx + WARP_SIZE * 192 : nullptr);

	return {rgb_mlp_out[0], rgb_mlp_out[1], rgb_mlp_out[2], rgb_mlp_in[0]};
}

__device__ void training_step_nerf_network_backward_density_network(const tvec<__half, 16>& dL_dy, const __half* __restrict__ params, const uint8_t* __restrict__ fwd_ctx, __half* __restrict__ dL_dparams = nullptr, tvec<__half, 32>* __restrict__ dL_dx = nullptr) {

	mma_vec<16> out_grad{dL_dy};
	

	auto hidden = mma_vec<64>::from_native_memory((__half*)fwd_ctx + 32 * (32 + 0 * 64));
	if (dL_dparams) {
		outer_product(out_grad, hidden).sum_into_linear_global_memory_hierarchical<128>(dL_dparams + 64 * (32 + 0 * 64));
	}

	auto out_mat = mma_mat<64, 16>::from_native_memory(params + 64 * (32 + 0 * 64));
	auto hidden_grad = out_grad * out_mat.transpose();
	hidden_grad.activate_bwd<Activation::ReLU>(hidden);

	TCNN_PRAGMA_UNROLL
	for (int i = 0-1; i >= 0; --i) {
		hidden = mma_vec<64>::from_native_memory((__half*)fwd_ctx + 32 * (32 + i * 64));
		if (dL_dparams) {
			outer_product(hidden_grad, hidden).sum_into_linear_global_memory_hierarchical<128>(dL_dparams + 64 * (32 + i * 64));
		}

		auto hidden_mat = mma_mat<64, 64>::from_native_memory(params + 64 * (32 + i * 64));
		hidden_grad = hidden_grad * hidden_mat.transpose();
		hidden_grad.activate_bwd<Activation::ReLU>(hidden);
	}

	auto in = mma_vec<32>::from_native_memory((__half*)fwd_ctx);
	if (dL_dparams) {
		outer_product(hidden_grad, in).sum_into_linear_global_memory_hierarchical<128>(dL_dparams);
	}

	if (!dL_dx) {
		return;
	}

	auto in_mat = mma_mat<32, 64>::from_native_memory(params);
	auto in_grad = hidden_grad * in_mat.transpose();
	*dL_dx = in_grad.vec<32>();
}

__device__ void training_step_nerf_network_backward_rgb_network(const tvec<__half, 3>& dL_dy, const __half* __restrict__ params, const uint8_t* __restrict__ fwd_ctx, __half* __restrict__ dL_dparams = nullptr, tvec<__half, 32>* __restrict__ dL_dx = nullptr) {

	mma_vec<16> out_grad{dL_dy};
	

	auto hidden = mma_vec<64>::from_native_memory((__half*)fwd_ctx + 32 * (32 + 1 * 64));
	if (dL_dparams) {
		outer_product(out_grad, hidden).sum_into_linear_global_memory_hierarchical<128>(dL_dparams + 64 * (32 + 1 * 64));
	}

	auto out_mat = mma_mat<64, 16>::from_native_memory(params + 64 * (32 + 1 * 64));
	auto hidden_grad = out_grad * out_mat.transpose();
	hidden_grad.activate_bwd<Activation::ReLU>(hidden);

	TCNN_PRAGMA_UNROLL
	for (int i = 1-1; i >= 0; --i) {
		hidden = mma_vec<64>::from_native_memory((__half*)fwd_ctx + 32 * (32 + i * 64));
		if (dL_dparams) {
			outer_product(hidden_grad, hidden).sum_into_linear_global_memory_hierarchical<128>(dL_dparams + 64 * (32 + i * 64));
		}

		auto hidden_mat = mma_mat<64, 64>::from_native_memory(params + 64 * (32 + i * 64));
		hidden_grad = hidden_grad * hidden_mat.transpose();
		hidden_grad.activate_bwd<Activation::ReLU>(hidden);
	}

	auto in = mma_vec<32>::from_native_memory((__half*)fwd_ctx);
	if (dL_dparams) {
		outer_product(hidden_grad, in).sum_into_linear_global_memory_hierarchical<128>(dL_dparams);
	}

	if (!dL_dx) {
		return;
	}

	auto in_mat = mma_mat<32, 64>::from_native_memory(params);
	auto in_grad = hidden_grad * in_mat.transpose();
	*dL_dx = in_grad.vec<32>();
}

__device__ void training_step_nerf_network_backward_pos_encoding_lookup(
	const float scale,
	vec<3> pos,
	const __half* __restrict__ grid,
	const uint32_t hashmap_size,
	const tvec<__half, 4>& dL_dy,
	__half* __restrict__ dL_dparams,
	vec<3>* __restrict__ dL_dpos
) {
	const uint32_t resolution = grid_resolution(scale);

	auto grid_val = [&](const uvec<3>& local_pos) {
		const uint32_t index = grid_index<3, HashType::CoherentPrime>(GridType::Hash, hashmap_size, resolution, local_pos) * 4;
		return *(tvec<__half, 4, 8>*)&grid[index];
	};

	auto add_grid_gradient = [&](const uvec<3>& local_pos, const float weight) {
		const uint32_t index = grid_index<3, HashType::CoherentPrime>(GridType::Hash, hashmap_size, resolution, local_pos) * 4;
		atomic_add_gmem(dL_dparams + index, (__half)weight * dL_dy);
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

	vec<3> pos_derivative = vec<3>::ones();

	if (InterpolationType::Linear == InterpolationType::Nearest) {
		if (dL_dparams) {
			add_grid_gradient(pos_grid, 1.0f);
		}
		return; // Can return early, because dL_dpos will be zero in any case.
	}

	if (InterpolationType::Linear == InterpolationType::Smoothstep) {
		TCNN_PRAGMA_UNROLL
		for (uint32_t i = 0; i < 3; ++i) {
			pos_derivative[i] = smoothstep_derivative(pos[i]);
			pos[i] = smoothstep(pos[i]);
		}
	}

	if (dL_dparams) {
		TCNN_PRAGMA_UNROLL
		for (uint32_t idx = 0; idx < (1 << 3); ++idx) {
			float weight = 1.0f;
			uvec<3> pos_grid_local = pos_grid;

			TCNN_PRAGMA_UNROLL
			for (uint32_t dim = 0; dim < 3; ++dim) {
				weight *= ((idx >> dim) & 1) ? pos[dim] : (1.0f - pos[dim]);
				pos_grid_local[dim] += (idx >> dim) & 1;
			}

			add_grid_gradient(pos_grid_local, weight);
		}
	}

	if (!dL_dpos) {
		return;
	}

	vec<3> grad = {0.0f};

	TCNN_PRAGMA_UNROLL
	for (uint32_t grad_dim = 0; grad_dim < 3; ++grad_dim) {
		TCNN_PRAGMA_UNROLL
		for (uint32_t idx = 0; idx < (1 << (3-1)); ++idx) {
			float weight = scale;
			uvec<3> pos_grid_local;

			TCNN_PRAGMA_UNROLL
			for (uint32_t non_grad_dim = 0; non_grad_dim < 3-1; ++non_grad_dim) {
				const uint32_t dim = non_grad_dim >= grad_dim ? (non_grad_dim+1) : non_grad_dim;

				if ((idx & (1<<non_grad_dim)) == 0) {
					weight *= 1 - pos[dim];
					pos_grid_local[dim] = pos_grid[dim];
				} else {
					weight *= pos[dim];
					pos_grid_local[dim] = pos_grid[dim] + 1;
				}
			}

			pos_grid_local[grad_dim] = pos_grid[grad_dim];
			auto val_left = grid_val(pos_grid_local);
			pos_grid_local[grad_dim] = pos_grid[grad_dim] + 1;
			auto val_right = grid_val(pos_grid_local);

			TCNN_PRAGMA_UNROLL
			for (uint32_t feature = 0; feature < 4; ++feature) {
				grad[grad_dim] += weight * ((float)val_right[feature] - (float)val_left[feature]) * (float)dL_dy[feature];
			}
		}
	}

	*dL_dpos += grad * pos_derivative;
}

__device__ void training_step_nerf_network_backward_pos_encoding(const tvec<__half, 32>& dL_dy, const __half* __restrict__ params, const uint8_t* __restrict__ fwd_ctx, __half* __restrict__ dL_dparams = nullptr, tvec<float, 3>* __restrict__ dL_dx = nullptr) {
	fwd_ctx += lane_id() * 12;
	if (!dL_dx && !dL_dparams) {
		return;
	}

	tvec<float, 3> input((float*)fwd_ctx);
	training_step_nerf_network_backward_pos_encoding_lookup((float)15.0000000000, input, params + 0, 4096, dL_dy.slice<0, 4>(), dL_dparams ? dL_dparams + 0 : nullptr, dL_dx);
	training_step_nerf_network_backward_pos_encoding_lookup((float)38.0084381104, input, params + 16384, 64000, dL_dy.slice<4, 4>(), dL_dparams ? dL_dparams + 16384 : nullptr, dL_dx);
	training_step_nerf_network_backward_pos_encoding_lookup((float)94.1036453247, input, params + 272384, 524288, dL_dy.slice<8, 4>(), dL_dparams ? dL_dparams + 272384 : nullptr, dL_dx);
	training_step_nerf_network_backward_pos_encoding_lookup((float)230.8653106689, input, params + 2369536, 524288, dL_dy.slice<12, 4>(), dL_dparams ? dL_dparams + 2369536 : nullptr, dL_dx);
	training_step_nerf_network_backward_pos_encoding_lookup((float)564.2940063477, input, params + 4466688, 524288, dL_dy.slice<16, 4>(), dL_dparams ? dL_dparams + 4466688 : nullptr, dL_dx);
	training_step_nerf_network_backward_pos_encoding_lookup((float)1377.2020263672, input, params + 6563840, 524288, dL_dy.slice<20, 4>(), dL_dparams ? dL_dparams + 6563840 : nullptr, dL_dx);
	training_step_nerf_network_backward_pos_encoding_lookup((float)3359.0949707031, input, params + 8660992, 524288, dL_dy.slice<24, 4>(), dL_dparams ? dL_dparams + 8660992 : nullptr, dL_dx);
	training_step_nerf_network_backward_pos_encoding_lookup((float)8191.0058593750, input, params + 10758144, 524288, dL_dy.slice<28, 4>(), dL_dparams ? dL_dparams + 10758144 : nullptr, dL_dx);

}

__device__ void training_step_nerf_network_backward_dir_encoding_0_spherical_harmonics(const tvec<__half, 16>& dL_dy, const __half* __restrict__ params, const uint8_t* __restrict__ fwd_ctx, __half* __restrict__ dL_dparams = nullptr, tvec<float, 3>* __restrict__ dL_dx = nullptr) {
	fwd_ctx += lane_id() * 12;
	if (dL_dx) {
		vec3 d((float*)fwd_ctx);
		*dL_dx = 2.0f * sh_enc_grad<__half, tvec<__half, 16>>(4, d[0], d[1], d[2], dL_dy);
	}
}

__device__ void training_step_nerf_network_backward_dir_encoding(const tvec<__half, 16>& dL_dy, const __half* __restrict__ params, const uint8_t* __restrict__ fwd_ctx, __half* __restrict__ dL_dparams = nullptr, tvec<float, 3>* __restrict__ dL_dx = nullptr) {

	if (!dL_dx && !dL_dparams) { return; }
	training_step_nerf_network_backward_dir_encoding_0_spherical_harmonics(dL_dy.slice<0, 16>(), params + 0, fwd_ctx + WARP_SIZE * 0, dL_dparams ? dL_dparams + 0 : nullptr, dL_dx ? &dL_dx->slice<0, 3>() : nullptr);

}

__device__ void training_step_nerf_network_backward(const tvec<__half, 4>& dL_dy, const __half* __restrict__ params, const uint8_t* __restrict__ fwd_ctx, __half* __restrict__ dL_dparams = nullptr, tvec<float, 7>* __restrict__ dL_dx = nullptr) {

	bool requires_pos_encoding_bwd = 12855296 != 0 || dL_dx;
	bool requires_dir_encoding_bwd = 0 != 0 || dL_dx;

	tvec<__half, 32> dL_drgb_mlp_in;
	training_step_nerf_network_backward_rgb_network(
		tvec<__half, 3>(dL_dy.rgb()),
		params + 3072,
		fwd_ctx + WARP_SIZE * 192,
		dL_dparams ? dL_dparams + 3072 : nullptr,
		&dL_drgb_mlp_in
	);
	dL_drgb_mlp_in[0] = dL_drgb_mlp_in[0] + dL_dy[3];

	if (requires_dir_encoding_bwd) {
		training_step_nerf_network_backward_dir_encoding(
			dL_drgb_mlp_in.slice<16, 16>(),
			params + 12865536,
			fwd_ctx + WARP_SIZE * 524,
			dL_dparams ? dL_dparams + 12865536 : nullptr,
			dL_dx ? &dL_dx->slice<4, 3>() : nullptr
		);
	}

	tvec<__half, 32> dL_dpos_enc_out;
	training_step_nerf_network_backward_density_network(
		dL_drgb_mlp_in.slice<0, 16>(),
		params,
		fwd_ctx,
		dL_dparams,
		requires_pos_encoding_bwd ? &dL_dpos_enc_out : nullptr
	);

	if (requires_pos_encoding_bwd) {
		training_step_nerf_network_backward_pos_encoding(
			dL_dpos_enc_out,
			params + 10240,
			fwd_ctx + WARP_SIZE * 512,
			dL_dparams ? dL_dparams + 10240 : nullptr,
			dL_dx ? &dL_dx->slice<0, 3>() : nullptr
		);
	}
}



__global__ void training_step_nerf_network(
	const uint32_t n_elements,
	const float loss_scale,
	MatrixView<const float> data_in,
	MatrixView<const float> data_target,
	MatrixView<const float> data_pdf,
	MatrixView<const __half> external_dL_dy,
	MatrixView<float> data_dL_dx,
	MatrixView<float> data_loss,
	const __half* __restrict__ params,
	uint8_t* __restrict__ fwd_ctx_gmem,
	__half* __restrict__ dL_dparams
) {
	extern __shared__ uint8_t fwd_ctx_shmem[]; uint8_t* fwd_ctx = (uint8_t*)fwd_ctx_shmem + 16384;
	fwd_ctx += previous_multiple(threadIdx.x, WARP_SIZE) * 536;

	// Here, fwd_ctx is aligned to each _warp_, i.e. every warp
	// has 536 bytes of context memory at its
	// disposal, starting at `fwd_ctx`, and it can order its
	// accesses of this memory however it wishes.

	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

	auto out = training_step_nerf_network_forward(data_in.col<7>(i), params, fwd_ctx);

	auto dL_dy = external_dL_dy.col<4>(i);

	auto dL_dx = tvec<float, 7>::zero();
	training_step_nerf_network_backward(dL_dy, params, fwd_ctx, dL_dparams, data_dL_dx ? &dL_dx : nullptr);
	if (data_dL_dx) {
		data_dL_dx.set_col(i, dL_dx);
	}
}