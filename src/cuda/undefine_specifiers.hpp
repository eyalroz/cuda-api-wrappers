/**
 * @file undefine_specifiers.hpp
 *
 * @brief A "preprocessor utility" header for undefining the CUDA function declaration
 * specifier shorthands, defined in @ref define_specifiers.hpp .
 */

#ifdef CUDA_FD
#undef CUDA_FD
#endif

#ifdef CUDA_FH
#undef CUDA_FH
#endif

#ifdef CUDA_FHD
#undef CUDA_FHD
#endif

#ifdef CUDA_HD
#undef CUDA_HD
#endif

#ifdef CUDA_D
#undef CUDA_D
#endif

#ifdef CUDA_H
#undef CUDA_H
#endif
