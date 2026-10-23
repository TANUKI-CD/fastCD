cmake_minimum_required(VERSION 3.18)

# The version is defined once, in src/fastcd.c; pyproject.toml reads the same
# define through scikit-build-core, so the three cannot drift apart.  CMake
# uses the numeric MAJOR.MINOR.PATCH part; a PEP 440 pre-release suffix such
# as "rc1" is kept only in the library's version string.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/fastcd.c")
file(STRINGS src/fastcd.c FASTCD_VERSION_LINE REGEX "^#define FASTCD_VERSION \"")
string(REGEX REPLACE "^#define FASTCD_VERSION \"([0-9]+\\.[0-9]+\\.[0-9]+)[^\"]*\".*$" "\\1"
    FASTCD_VERSION_STRING "${FASTCD_VERSION_LINE}")
if(NOT FASTCD_VERSION_STRING MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR "could not read FASTCD_VERSION from src/fastcd.c")
endif()

project(fastCD
    VERSION ${FASTCD_VERSION_STRING}
    DESCRIPTION "Fast real Cayley--Dickson arithmetic"
    LANGUAGES C)

include(CTest)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

option(BUILD_SHARED_LIBS "Build fastCD as a shared library" OFF)
option(FASTCD_BUILD_BENCHMARK "Build the multiplication benchmark runner" ON)
option(FASTCD_BUILD_PYTHON "Build the pyfastcd extension module" OFF)
option(FASTCD_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(FASTCD_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# Internal tuning macros (for example FASTCD_DIRECT_LEAF=32) used
# while measuring kernel thresholds.  Production builds leave this empty.
set(FASTCD_TUNING_DEFINITIONS "" CACHE STRING
    "Semicolon-separated preprocessor definitions applied to the library")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED YES)
set(CMAKE_C_EXTENSIONS NO)

# Apply the project's diagnostic and floating-point policy to first-party
# targets without exporting those flags to users of the installed library.
function(fastcd_configure_target target)
    if(MINGW)
        # Use printf formats compatible with MinGW's extended long double.
        target_compile_definitions(${target} PRIVATE __USE_MINGW_ANSI_STDIO=1)
    endif()
    if(MSVC)
        target_compile_options(${target} PRIVATE /fp:precise /W4)
        if(FASTCD_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            -fno-fast-math -ffp-contract=off -Wall -Wextra -Wpedantic
            -Wconversion -Wshadow -Wstrict-prototypes)
        if(FASTCD_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
    if(FASTCD_ENABLE_SANITIZERS)
        if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target} PRIVATE
                -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${target} PRIVATE
                -fsanitize=address,undefined -fno-omit-frame-pointer)
        else()
            message(FATAL_ERROR "FASTCD_ENABLE_SANITIZERS requires GCC or Clang")
        endif()
    endif()
endfunction()

add_library(fastcd
    src/fastcd.c
    src/quasilinear.c
    src/cariow_cariowa.c
    src/direct.c)
add_library(fastcd::fastcd ALIAS fastcd)
target_include_directories(fastcd
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src)
set_target_properties(fastcd PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(fastcd
        PRIVATE FASTCD_BUILDING_LIBRARY
        PUBLIC FASTCD_SHARED)
endif()
if(FASTCD_TUNING_DEFINITIONS)
    target_compile_definitions(fastcd PRIVATE ${FASTCD_TUNING_DEFINITIONS})
endif()
fastcd_configure_target(fastcd)
if(NOT MSVC)
    target_link_libraries(fastcd PUBLIC m)
endif()

if(BUILD_TESTING OR FASTCD_BUILD_BENCHMARK)
    add_library(fastcd_reference STATIC tests/reference_cd.c)
    target_include_directories(fastcd_reference PUBLIC
        ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests)
    set_target_properties(fastcd_reference PROPERTIES POSITION_INDEPENDENT_CODE ON)
    fastcd_configure_target(fastcd_reference)
endif()

if(BUILD_TESTING)
    add_executable(fastcd_public_link_test tests/test_public_api.c)
    target_link_libraries(fastcd_public_link_test PRIVATE fastcd::fastcd)
    fastcd_configure_target(fastcd_public_link_test)
    add_test(NAME fastcd.public_api_link COMMAND fastcd_public_link_test)

    add_executable(fastcd_tests tests/test_fastcd.c)
    target_link_libraries(fastcd_tests PRIVATE fastcd::fastcd fastcd_reference)
    fastcd_configure_target(fastcd_tests)
    add_test(NAME fastcd.tests COMMAND fastcd_tests)
    set_tests_properties(fastcd.tests PROPERTIES TIMEOUT 600)

    add_executable(fastcd_leaf_tests tests/test_quasilinear_leaf.c)
    target_include_directories(fastcd_leaf_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(fastcd_leaf_tests PRIVATE fastcd_reference)
    fastcd_configure_target(fastcd_leaf_tests)
    add_test(NAME fastcd.quasilinear_leaf COMMAND fastcd_leaf_tests)
endif()

if(FASTCD_BUILD_BENCHMARK)
    add_executable(fastcd_benchmark benchmarks/benchmark.c)
    target_link_libraries(fastcd_benchmark PRIVATE fastcd::fastcd fastcd_reference)
    fastcd_configure_target(fastcd_benchmark)
    # Record the source contents and effective library options beside the runner.
    file(GLOB_RECURSE fastcd_benchmark_sources CONFIGURE_DEPENDS
        src/*.c src/*.h include/*.h tests/reference_cd.* benchmarks/*.c
        benchmarks/*.py tools/*.py)
    list(APPEND fastcd_benchmark_sources "${PROJECT_SOURCE_DIR}/CMakeLists.txt")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${fastcd_benchmark_sources})
    set(fastcd_source_hashes "")
    foreach(source IN LISTS fastcd_benchmark_sources)
        file(SHA256 "${source}" source_hash)
        file(RELATIVE_PATH relative_source "${PROJECT_SOURCE_DIR}" "${source}")
        string(APPEND fastcd_source_hashes "${source_hash}  ${relative_source}\n")
    endforeach()
    get_target_property(fastcd_build_options fastcd COMPILE_OPTIONS)
    get_target_property(fastcd_build_definitions fastcd COMPILE_DEFINITIONS)
    string(CONCAT fastcd_build_record
        "compiler=${CMAKE_C_COMPILER}\nversion=${CMAKE_C_COMPILER_VERSION}\n"
        "configuration=$<CONFIG>\nflags=${CMAKE_C_FLAGS}\n"
        "configuration_flags=$<$<CONFIG:Release>:${CMAKE_C_FLAGS_RELEASE}>"
        "$<$<CONFIG:Debug>:${CMAKE_C_FLAGS_DEBUG}>"
        "$<$<CONFIG:RelWithDebInfo>:${CMAKE_C_FLAGS_RELWITHDEBINFO}>"
        "$<$<CONFIG:MinSizeRel>:${CMAKE_C_FLAGS_MINSIZEREL}>\n"
        "options=$<JOIN:$<TARGET_PROPERTY:fastcd,COMPILE_OPTIONS>, >\n"
        "definitions=$<JOIN:$<TARGET_PROPERTY:fastcd,COMPILE_DEFINITIONS>, >\n"
        "source_sha256:\n${fastcd_source_hashes}")
    string(SHA256 fastcd_build_id
        "${fastcd_build_record};${fastcd_build_options};${fastcd_build_definitions}")
    target_compile_definitions(fastcd_benchmark PRIVATE
        FASTCD_BENCHMARK_BUILD_ID="${fastcd_build_id}-$<CONFIG>")
    file(GENERATE OUTPUT "${PROJECT_BINARY_DIR}/fastcd-build-$<CONFIG>.txt"
        CONTENT "build_id=${fastcd_build_id}-$<CONFIG>\n${fastcd_build_record}")
    if(BUILD_TESTING)
        add_test(NAME fastcd.benchmark_metadata COMMAND fastcd_benchmark --metadata)
    endif()
endif()

if(FASTCD_BUILD_PYTHON)
    if(BUILD_SHARED_LIBS)
        message(FATAL_ERROR "pyfastcd must statically link fastcd; set BUILD_SHARED_LIBS=OFF")
    endif()
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
    Python_add_library(pyfastcd_core MODULE WITH_SOABI python/src/pyfastcd/_core.c)
    set_target_properties(pyfastcd_core PROPERTIES
        OUTPUT_NAME "_core"
        C_VISIBILITY_PRESET hidden)
    target_link_libraries(pyfastcd_core PRIVATE fastcd::fastcd)
    target_include_directories(pyfastcd_core PRIVATE ${PROJECT_SOURCE_DIR}/include)
    fastcd_configure_target(pyfastcd_core)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        # PyMethodDef uses one generic function-pointer type for several
        # CPython calling conventions.
        target_compile_options(pyfastcd_core PRIVATE -Wno-cast-function-type)
    endif()
    install(TARGETS pyfastcd_core
        LIBRARY DESTINATION pyfastcd COMPONENT python
        RUNTIME DESTINATION pyfastcd COMPONENT python)
endif()

install(TARGETS fastcd
    EXPORT fastcdTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/fastcd
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h")
set(FASTCD_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/fastcd")
configure_package_config_file(
    cmake/fastcdConfig.cmake.in
    "${PROJECT_BINARY_DIR}/fastcdConfig.cmake"
    INSTALL_DESTINATION "${FASTCD_CMAKE_INSTALL_DIR}")
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/fastcdConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion)
install(EXPORT fastcdTargets
    FILE fastcdTargets.cmake
    NAMESPACE fastcd::
    DESTINATION "${FASTCD_CMAKE_INSTALL_DIR}")
install(FILES
    "${PROJECT_BINARY_DIR}/fastcdConfig.cmake"
    "${PROJECT_BINARY_DIR}/fastcdConfigVersion.cmake"
    DESTINATION "${FASTCD_CMAKE_INSTALL_DIR}")
install(FILES LICENSE
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/fastCD")
