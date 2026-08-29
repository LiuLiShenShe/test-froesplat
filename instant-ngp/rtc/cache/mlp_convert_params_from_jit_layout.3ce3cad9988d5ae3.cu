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

			/** @file   mlp_convert_params_from_jit_layout.cu
			 *  @author Thomas Müller, NVIDIA
			 *  @brief  Automatically generated kernel mlp_convert_params_from_jit_layout
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

			__global__ void mlp_convert_params_from_jit_layout(__half* __restrict__ params) {
	if (2 == 0) {
		auto mat = mma_mat<32, 16>::from_native_memory(params);
		mat.into_linear_memory(params);
		return;
	}

	if (blockIdx.x == 0) {
		auto first_mat = mma_mat<32, 64>::from_native_memory(params);
		first_mat.into_linear_memory(params);
	} else if (blockIdx.x == 1) {
		params += 32 * 64 + (2 - 1) * 64 * 64;
		auto last_mat = mma_mat<64, 16>::from_native_memory(params);
		last_mat.into_linear_memory(params);
	} else {
		params += 32 * 64 + (blockIdx.x - 2) * 64 * 64;
		auto hidden_mat = mma_mat<64, 64>::from_native_memory(params);
		hidden_mat.into_linear_memory(params);
	}
}