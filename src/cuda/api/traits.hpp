/**
 * @file
 *
 * @brief Non-CUDA-specific type-related utility code, especially type traits.
 */

 #ifndef CUDA_API_WRAPPERS_TRAITS_HPP
 #define CUDA_API_WRAPPERS_TRAITS_HPP
 
 #include <type_traits>
 
 namespace cuda
 {
 namespace traits
 {

/**
 * @brief Trait to determine whether a type can be used as a kernel argument.
 * 
 * In CUDA programming, kernel arguments must be trivially copyable, as they are
 * passed by value to the device. This trait is an extension of std::is_trivially_copy_constructible
 * which serves to identify types that can be safely passed to CUDA kernels.
 * 
 * @tparam T The type to check for kernel argument validity
 * @note This trait is used internally in the CUDA API wrappers to enforce
 *        the constraints on kernel arguments at compile time.
 * @note The trait is based on the CUDA C++ Programming Guide, which states that
 *       kernel arguments must be trivially copyable. See the following link for more details:
 *       https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-function-argument-processing
 */
 template <typename T>
 struct is_valid_kernel_argument : ::std::is_trivially_copy_constructible<T>
 {
 };
 
 }  // namespace traits
 
 } // namespace cuda
 
 
 #endif //CUDA_API_WRAPPERS_TRAITS_HPP
 