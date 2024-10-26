/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2024, Eyal Rozenberg. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
* This sample demonstrates how use texture fetches in CUDA
*
* This sample takes an input PGM image (image_filename) and generates
* an output PGM image (image_filename_out).  This CUDA kernel performs
* a simple 2D transform (rotation) on the texture coordinates (u,v).
* The results between simpleTexture and simpleTextureDrv are identical.
* The main difference is the implementation.  simpleTextureDrv makes calls
* to the CUDA driver API and demonstrates how to use cuModuleLoad to load
* the CUDA ptx (*.ptx) kernel just prior to kernel launch.
*
*/

// includes, system
#include <iostream>
#include <string>
#include <string.h>

// includes, CUDA
#include "../../common.hpp"

char * sdkFindFilePath(char const * ref_file, char const *)
{
	return strdup(ref_file);
}

#include "helper_image.h"

const char *image_filename = "data/teapot512.pgm";
const char *reference_filename = "data/ref_rotated.pgm";
float const angle = 0.5f;  // angle to rotate image by (in radians)
float const error_threshold { 5e-3f };
float const max_allowed_mismatch_rate {0.0f};

namespace kernel {

constexpr const char *fatbin_filename = "simpleTexture_kernel.fatbin";
constexpr const char *name = "transformKernel";

} // namespace kernel

std::string get_file_contents(const char *path)
{
	std::ios::openmode open_mode = std::ios::in | std::ios::binary;
	std::ifstream ifs(path, open_mode);
	if (ifs.bad() or ifs.fail()) {
		throw std::system_error(errno, std::system_category(), std::string("opening ") + path + " in binary read mode");
	}
	std::ostringstream oss;
	oss << ifs.rdbuf();
	return oss.str();
}

struct pgm {
	std::unique_ptr<float> data;
	unsigned height;
	unsigned width;

	size_t size() const { return (size_t) width * height; }
};

template <class T>
inline pgm sdkLoadPGM_(char const* path)
{
	float* data = nullptr;
	pgm loaded;
	auto result = sdkLoadPGM(path, &data, &loaded.width, &loaded.height);
	if (not result) { throw std::runtime_error("Failed loading a PGM from " + std::string{path}); }
	loaded.data.reset(data);
	return loaded;
}

std::string output_filename_for(std::string filename)
{
	auto suffix = std::string{".pgm"};
	auto input_filename_without_suffix = filename.substr(0, filename.length() - suffix.length());
	return input_filename_without_suffix.append("_out").append(suffix);
}

bool runTest(int device_id) {

  auto device = cuda::device::get(device_id);
  auto context = device.create_context();
  auto stream = context.create_stream(cuda::stream::async);
  auto fatbin = get_file_contents(kernel::fatbin_filename);
  auto module = context.create_module(fatbin);
  auto kernel = module.get_kernel(kernel::name);

  auto image = sdkLoadPGM_<float>(image_filename);
  auto image_data = cuda::span<float> {image.data.get(), image.size()};
  std::cout << "Loaded '" << image_filename << "', " << image.width << " x " << image.height << " pixels\n";

  auto arr = cuda::array::create<float, 2>(context, { image.width, image.height });
  cuda::memory::copy_2(arr, image_data);

  auto texture_descriptor = cuda::texture::descriptor_t{};
  texture_descriptor.filterMode = CU_TR_FILTER_MODE_LINEAR;
  texture_descriptor.addressMode[0] = CU_TR_ADDRESS_MODE_WRAP;
  texture_descriptor.addressMode[1] = CU_TR_ADDRESS_MODE_WRAP;
  texture_descriptor.addressMode[2] = CU_TR_ADDRESS_MODE_WRAP;
  texture_descriptor.flags = CU_TRSF_NORMALIZED_COORDINATES;
  auto texture_view = cuda::texture_view(arr, texture_descriptor);

  auto d_output_image = cuda::memory::device::make_unique_span<float>(device, image.size());

  constexpr int const block_dim { 8 };
  auto config = cuda::launch_config_builder()
	  .overall_dimensions(image.width, image.height)
	  .block_dimensions(block_dim, block_dim)
	  .no_dynamic_shared_memory()
	  .build();

  context.launch(kernel, config, d_output_image.data(), image.width, image.height, angle,  texture_view.raw_handle());
  stream.synchronize();

  // allocate mem for the result on host side
  auto output_image = cuda::memory::host::make_unique_span<float>(image.size());
  // copy result from device to host
  cuda::memory::copy_2(output_image, d_output_image);

  auto output_filename = output_filename_for(image_filename);
  sdkSavePGM(output_filename.c_str(), output_image.data(), image.width, image.height);
  std::cout << "Wrote transformed image to '" << output_filename << "'\n";

  // We need to reload the data from disk, because it is inverted upon output
  auto reference_image = sdkLoadPGM_<float>(reference_filename);

  std::cout
  	<< "Comparing files\n"
  	<< "\toutput:    " << output_filename << "\n"
  	<< "\treference: " << reference_filename << "\n";

  return compareData(
	  reference_image.data.get(), output_image.data(), image.size(),
	  error_threshold, max_allowed_mismatch_rate);
}

int main(int argc, char **argv) {
	auto device_id = choose_device(argc, argv);
	auto result = runTest(device_id);
	std::cout << (result ? "SUCCESS" : "FAILURE") << "\n";
	return (result ? EXIT_SUCCESS : EXIT_FAILURE);
}

