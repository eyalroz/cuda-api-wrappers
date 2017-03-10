if(CMAKE_BUILD_TYPE)
	return()
endif()

if("$ENV{RELEASE}" STREQUAL ""  AND  "$ENV{DEBUG}" STREQUAL "" )
message("Neither DEBUG nor RELEASE environment variables set, defaulting to DEBUG.")
set(CMAKE_BUILD_TYPE "DEBUG")
endif( "$ENV{RELEASE}" STREQUAL ""  AND  "$ENV{DEBUG}" STREQUAL "" )

if (NOT "$ENV{DEBUG}" STREQUAL "")
set(CMAKE_BUILD_TYPE "DEBUG")
endif (NOT "$ENV{DEBUG}" STREQUAL "")

if (NOT "$ENV{RELEASE}" STREQUAL "")
set(CMAKE_BUILD_TYPE "RELEASE")
endif (NOT "$ENV{RELEASE}" STREQUAL "")

if ("${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
	message("\
  ____           _                               _ 
 |  _ \\    ___  | |   ___    __ _   ___    ___  | |
 | |_) |  / _ \\ | |  / _ \\  / _` | / __|  / _ \\ | |
 |  _ <  |  __/ | | |  __/ | (_| | \\__ \\ |  __/ |_|
 |_| \\_\\  \\___| |_|  \\___|  \\__,_| |___/  \\___| (_)

")
else ("${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
if("${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
	message("\
  ____           _                     
 |  _ \\    ___  | |__    _   _    __ _ 
 | | | |  / _ \\ | '_ \\  | | | |  / _` |
 | |_| | |  __/ | |_) | | |_| | | (_| |
 |____/   \\___| |_.__/   \\__,_|  \\__, |
                                 |___/ 
")
	endif("${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
endif("${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")

