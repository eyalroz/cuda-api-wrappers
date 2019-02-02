# Copyright (c) 2014, Ruslan Baratov
# All rights reserved.

include(sugar_add_this_to_sourcelist)
sugar_add_this_to_sourcelist()

include(CMakeParseArguments) # cmake_parse_arguments
include(sugar_fatal_error)
include(sugar_generate_warning_flag_by_name)
include(sugar_generate_warning_xcode_attr_by_name)
include(sugar_get_all_xcode_warning_attrs)
include(sugar_status_debug)
include(sugar_warning_unpack_one)

# MS Visual Studio: http://msdn.microsoft.com/en-us/library/thxezb7y.aspx
# Clang: http://clang.llvm.org/docs/UsersManual.html
# GCC: https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
function(sugar_generate_warning_flags)
  ### Detect compilers: is_clang, is_msvc, is_gcc
  string(COMPARE EQUAL "${CMAKE_CXX_COMPILER_ID}" "Clang" is_clang)
  string(COMPARE EQUAL "${CMAKE_CXX_COMPILER_ID}" "AppleClang" is_apple_clang)
  if(is_clang OR is_apple_clang)
    set(is_clang TRUE)
  else()
    set(is_clang FALSE)
  endif()
  set(is_msvc ${MSVC})
  set(is_gcc ${CMAKE_COMPILER_IS_GNUCXX})

  if(is_clang OR is_msvc OR is_gcc)
    # Supported compilers
  else()
    sugar_fatal_error("Compiler (${CMAKE_CXX_COMPILER_ID}) is not supported")
  endif()

  set(multi DISABLE ENABLE TREAT_AS_ERROR)
  set(opts CLEAR_GLOBAL)
  cmake_parse_arguments(x "${opts}" "" "${multi}" ${ARGV})

  ### Remove warning flags from global variable
  set(new_cmake_cxx_flags "${CMAKE_CXX_FLAGS}")
  string(REPLACE "/W3" "" new_cmake_cxx_flags "${new_cmake_cxx_flags}")
  string(COMPARE NOTEQUAL "${new_cmake_cxx_flags}" "${CMAKE_CXX_FLAGS}" x)

  if(x)
    if(x_CLEAR_GLOBAL)
      set(CMAKE_CXX_FLAGS "${new_cmake_cxx_flags}" PARENT_SCOPE)
    else()
      message(
          WARNING
          "CMAKE_CXX_FLAGS variable contains warning flag"
          " that may cause a conflict."
          " Consider using CLEAR_GLOBAL suboption to remove warning"
          " flags from CMAKE_CXX_FLAGS."
      )
    endif()
  endif()

  ### Unpack warning groups
  set(new_list "")
  foreach(warning ${x_DISABLE})
    sugar_warning_unpack_one(warning)
    list(APPEND new_list ${warning})
  endforeach()
  set(x_DISABLE ${new_list})

  set(new_list "")
  foreach(warning ${x_ENABLE})
    sugar_warning_unpack_one(warning)
    list(APPEND new_list ${warning})
  endforeach()
  set(x_ENABLE ${new_list})

  set(new_list "")
  foreach(warning ${x_TREAT_AS_ERROR})
    sugar_warning_unpack_one(warning)
    list(APPEND new_list ${warning})
  endforeach()
  set(x_TREAT_AS_ERROR ${new_list})

  ### Length
  list(LENGTH x_UNPARSED_ARGUMENTS unparsed_length)
  list(LENGTH x_DISABLE disable_length)
  list(LENGTH x_ENABLE enable_length)
  list(LENGTH x_TREAT_AS_ERROR treat_as_error_length)

  ### Find special warning `ALL`
  list(FIND x_DISABLE "ALL" disable_all)
  list(FIND x_ENABLE "ALL" enable_all)
  list(FIND x_TREAT_AS_ERROR "ALL" treat_as_error_all)

  ### Convert to BOOL
  if(disable_all EQUAL -1)
    set(disable_all NO)
  else()
    set(disable_all YES)
  endif()

  if(enable_all EQUAL -1)
    set(enable_all NO)
  else()
    set(enable_all YES)
  endif()

  if(treat_as_error_all EQUAL -1)
    set(treat_as_error_all NO)
  else()
    set(treat_as_error_all YES)
  endif()

  ### If special option ALL present, check there is no others
  if(disable_all AND NOT disable_length EQUAL 1)
    sugar_fatal_error("If ALL present there must be no other warnings")
  endif()

  if(enable_all AND NOT enable_length EQUAL 1)
    sugar_fatal_error("If ALL present there must be no other warnings")
  endif()

  if(treat_as_error_all AND NOT treat_as_error_length EQUAL 1)
    sugar_fatal_error("If ALL present there must be no other warnings")
  endif()

  ### Verify result variable ###
  if(unparsed_length EQUAL 0)
    sugar_fatal_error("Expected 2 result variables")
  endif()
  if(NOT unparsed_length EQUAL 2)
    sugar_fatal_error("Unparsed: ${x_UNPARSED_ARGUMENTS}")
  endif()
  list(GET x_UNPARSED_ARGUMENTS 0 result_opts)
  list(GET x_UNPARSED_ARGUMENTS 1 result_props)
  sugar_status_debug(
      "Generate warnings (COMPILE_OPTIONS) for variable `${result_opts}`"
  )
  sugar_status_debug(
      "Generate warnings (PROPERTIES) for variable `${result_props}`"
  )
  set(${result_opts} "")
  set(${result_props} "")

  ### Clear default Xcode flags
  if(XCODE_VERSION)
    list(APPEND ${result_props} XCODE_ATTRIBUTE_WARNING_CFLAGS)
    list(APPEND ${result_props} " ")
  endif()

  ### Disable all
  if(disable_all)
    # Set all Xcode attributes to NO;
    # Note that some of them may be rewritten further (so resulting list
    # may contain several values for attributes with same name, last used)
    sugar_get_all_xcode_warning_attrs(attr_list)
    foreach(attr ${attr_list})
      list(APPEND ${result_props} ${attr})
      list(APPEND ${result_props} NO)
    endforeach()
    if(is_msvc)
      list(APPEND ${result_opts} "/w" "/W0")
    elseif(is_clang OR is_gcc)
      list(APPEND ${result_opts} "-w")
    else()
      sugar_fatal_error("")
    endif()
  endif()

  ### Enable all
  if(enable_all)
    # Set all Xcode attributes to YES (See note above)
    sugar_get_all_xcode_warning_attrs(attr_list)
    foreach(attr ${attr_list})
      list(APPEND ${result_props} ${attr})
      list(APPEND ${result_props} YES)
    endforeach()
    if(is_msvc)
      list(APPEND ${result_opts} "/Wall")
    elseif(is_gcc)
      list(APPEND ${result_opts} "-Wall" "-Wextra" "-Wpedantic")
    elseif(is_clang)
      list(APPEND ${result_opts} "-Wall" "-Weverything" "-pedantic")
    else()
      sugar_fatal_error("")
    endif()
  endif()

  ### All treat as error
  if(treat_as_error_all)
    if(is_msvc)
      list(APPEND ${result_opts} "/WX")
    elseif(is_gcc OR is_clang)
      list(APPEND ${result_opts} "-Werror")
    else()
      sugar_fatal_error("")
    endif()
  endif()

  ### DISABLE and ENABLE must not intersects
  foreach(warning ${x_DISABLE})
    list(FIND x_ENABLE "${warning}" x)
    if(NOT x EQUAL -1)
      sugar_fatal_error(
          "Warning `${warning}` in both DISABLE and ENABLE sections"
      )
    endif()
  endforeach()

  ### DISABLE and TREAT_AS_ERROR must not intersects
  foreach(warning ${x_DISABLE})
    list(FIND x_TREAT_AS_ERROR "${warning}" x)
    if(NOT x EQUAL -1)
      sugar_fatal_error(
          "Warning `${warning}` in both DISABLE and TREAT_AS_ERROR sections"
      )
    endif()
  endforeach()

  ### Generate ENABLE
  foreach(warning ${x_ENABLE})
    sugar_generate_warning_xcode_attr_by_name(warning_flag ${warning})
    if(warning_flag)
      list(APPEND ${result_props} ${warning_flag})
      list(APPEND ${result_props} YES)
    else()
      sugar_generate_warning_flag_by_name(warning_flag ${warning})
      foreach(x ${warning_flag})
        if(is_msvc)
          list(APPEND ${result_opts} "/w1${x}")
        elseif(is_gcc OR is_clang)
          list(APPEND ${result_opts} "-W${x}")
        else()
          sugar_fatal_error("")
        endif()
      endforeach()
    endif()
  endforeach()

  ### Generate DISABLE
  foreach(warning ${x_DISABLE})
    sugar_generate_warning_xcode_attr_by_name(warning_flag ${warning})
    if(warning_flag)
      list(APPEND ${result_props} ${warning_flag})
      list(APPEND ${result_props} NO)
    endif()
    # If xcode attribute set to NO then no flags will be generated, so
    # generate '-Wno-' flag explicitly
    sugar_generate_warning_flag_by_name(warning_flag ${warning})
    foreach(x ${warning_flag})
      if(is_msvc)
        list(APPEND ${result_opts} "/wd${x}")
      elseif(is_gcc OR is_clang)
        list(APPEND ${result_opts} "-Wno-${x}")
      else()
        sugar_fatal_error("")
      endif()
    endforeach()
  endforeach()

  ### Generate TREAT_AS_ERROR
  foreach(warning ${x_TREAT_AS_ERROR})
    sugar_generate_warning_flag_by_name(warning_flags ${warning})
    foreach(x ${warning_flags})
      if(is_msvc)
        list(APPEND ${result_opts} "/we${x}")
      elseif(is_gcc OR is_clang)
        list(APPEND ${result_opts} "-Werror=${x}")
      else()
        sugar_fatal_error("")
      endif()
    endforeach()
  endforeach()

  sugar_status_debug("Generated from:")
  sugar_status_debug("  DISABLE: ${x_DISABLE}")
  sugar_status_debug("  ENABLE: ${x_ENABLE}")
  sugar_status_debug("  TREAT_AS_ERROR: ${x_TREAT_AS_ERROR}")
  sugar_status_debug("Generated (COMPILE_OPTIONS): ${${result_opts}}")
  sugar_status_debug("Generated (PROPERTIES): ${${result_props}}")

  set(${result_opts} "${${result_opts}}" PARENT_SCOPE)
  set(${result_props} "${${result_props}}" PARENT_SCOPE)
endfunction()
