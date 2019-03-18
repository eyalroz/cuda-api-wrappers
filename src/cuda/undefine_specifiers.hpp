/**
 * @file undefine_specifiers.hpp
 *
 * @brief A "preprocessor utility" header for undefining the CUDA function declaration
 * specifier shorthands, defined in @ref define_specifiers.hpp .
 */

#ifdef __fd__
#undef __fd__
#endif

#ifdef __fh__
#undef __fh__
#endif

#ifdef __fhd__
#undef __fhd__
#endif

#ifdef __hd__
#undef __hd__
#endif
