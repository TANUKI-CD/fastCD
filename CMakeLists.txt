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
    DESCRIPTION "Cayley--Dickson arithmetic over real, complex and finite fields"
    LANGUAGES C)

include(CTest)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

option(BUILD_SHARED_LIBS "Build fastCD as a shared library" OFF)
option(FASTCD_BUILD_BENCHMARK "Build the multiplication benchmark runner" ON)
option(FASTCD_BUILD_FLINT_BENCHMARK "Build the optional FLINT comparison runner" OFF)
option(FASTCD_BUILD_PYTHON "Build the pyfastcd extension module" OFF)
option(FASTCD_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(FASTCD_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
if(FASTCD_BUILD_FLINT_BENCHMARK AND NOT FASTCD_BUILD_BENCHMARK)
    message(FATAL_ERROR "FASTCD_BUILD_FLINT_BENCHMARK requires FASTCD_BUILD_BENCHMARK=ON")
endif()

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
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
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
    src/direct.c
    src/arithmetic.c
    src/complex.c
    src/finite/field.c
    src/finite/finite.c
    src/finite/plan.c
    src/finite/operations.c
    src/finite/finite_backend.c
    src/finite/finite_prime.c)
add_library(fastcd::fastcd ALIAS fastcd)
target_include_directories(fastcd
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/src/finite)
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

# Only these translation units contain optional ISA instructions. Portable
# builds and the existing real kernels retain their original compiler flags.
option(FASTCD_ENABLE_FINITE_SIMD "Runtime-dispatched finite x86 kernels" ON)
set(FASTCD_HAVE_FINITE_SIMD OFF)
if(FASTCD_ENABLE_FINITE_SIMD AND NOT FASTCD_TUNING_DEFINITIONS MATCHES "FASTCD_DISABLE_SIMD=1"
        AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86)$"
        AND CMAKE_C_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    target_compile_definitions(fastcd PRIVATE FASTCD_FF_FINITE_SIMD=1)
    set(FASTCD_HAVE_FINITE_SIMD ON)
    foreach(lanes 4 8 16)
        target_sources(fastcd PRIVATE src/finite/generated/finite_mont_${lanes}.c)
    endforeach()
    foreach(lanes 8 16 32)
        target_sources(fastcd PRIVATE src/finite/generated/finite_short_${lanes}.c
            src/finite/generated/finite_shuffle_${lanes}.c)
    endforeach()
    target_sources(fastcd PRIVATE src/finite/generated/finite_maps.c
        src/finite/generated/finite_medium_bar.c src/finite/generated/finite_medium_mont.c
        src/finite/generated/finite_byte.c)
    foreach(bits 16 32 64)
        target_sources(fastcd PRIVATE src/finite/generated/finite_ternary_${bits}.c)
        set_source_files_properties(src/finite/generated/finite_ternary_${bits}.c
            PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw")
    endforeach()
    set_source_files_properties(src/finite/generated/finite_mont_4.c
        src/finite/generated/finite_short_8.c src/finite/generated/finite_shuffle_8.c
        PROPERTIES COMPILE_OPTIONS "-msse4.1")
    set_source_files_properties(src/finite/generated/finite_mont_8.c
        src/finite/generated/finite_short_16.c src/finite/generated/finite_shuffle_16.c
        PROPERTIES COMPILE_OPTIONS "-mavx2")
    set_source_files_properties(src/finite/generated/finite_mont_16.c
        PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512dq")
    set_source_files_properties(src/finite/generated/finite_short_32.c
        src/finite/generated/finite_shuffle_32.c src/finite/generated/finite_medium_bar.c
        src/finite/generated/finite_medium_mont.c src/finite/generated/finite_byte.c
        PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw")
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

    add_executable(fastcd_stability_tests tests/test_stability.c)
    target_include_directories(fastcd_stability_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(fastcd_stability_tests PRIVATE fastcd::fastcd)
    fastcd_configure_target(fastcd_stability_tests)
    add_test(NAME fastcd.stability COMMAND fastcd_stability_tests)
    set_tests_properties(fastcd.stability PROPERTIES TIMEOUT 600)

    add_executable(fastcd_domain_tests tests/test_domains.c)
    if(FASTCD_HAVE_FINITE_SIMD)
        target_compile_definitions(fastcd_domain_tests PRIVATE FASTCD_TEST_FINITE_SIMD=1)
    endif()
    target_link_libraries(fastcd_domain_tests PRIVATE fastcd::fastcd)
    fastcd_configure_target(fastcd_domain_tests)
    add_test(NAME fastcd.domains COMMAND fastcd_domain_tests)
    set_tests_properties(fastcd.domains PROPERTIES TIMEOUT 600)

    add_executable(fastcd_finite_direct_tests tests/test_finite_direct.c)
    target_include_directories(fastcd_finite_direct_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
    fastcd_configure_target(fastcd_finite_direct_tests)
    add_test(NAME fastcd.finite_direct_schedule COMMAND fastcd_finite_direct_tests)

    add_executable(fastcd_portable_reduce_tests tests/test_field_reduce.c)
    target_link_libraries(fastcd_portable_reduce_tests PRIVATE fastcd::fastcd)
    fastcd_configure_target(fastcd_portable_reduce_tests)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(fastcd_portable_reduce_tests PRIVATE -U__SIZEOF_INT128__)
    endif()
    add_test(NAME fastcd.portable_reduce COMMAND fastcd_portable_reduce_tests)

    add_executable(fastcd_domain_public_link_test tests/test_domain_public_api.c)
    target_link_libraries(fastcd_domain_public_link_test PRIVATE fastcd::fastcd)
    fastcd_configure_target(fastcd_domain_public_link_test)
    add_test(NAME fastcd.domains_public_link COMMAND fastcd_domain_public_link_test)
endif()

if(FASTCD_BUILD_BENCHMARK)
    add_executable(fastcd_benchmark benchmarks/benchmark.c)
    target_link_libraries(fastcd_benchmark PRIVATE fastcd::fastcd fastcd_reference)
    fastcd_configure_target(fastcd_benchmark)
    add_executable(fastcd_domains_benchmark benchmarks/domains.c)
    target_include_directories(fastcd_domains_benchmark PRIVATE ${PROJECT_SOURCE_DIR}/src/finite)
    target_link_libraries(fastcd_domains_benchmark PRIVATE fastcd::fastcd)
    fastcd_configure_target(fastcd_domains_benchmark)
    # FLINT is confined to this executable. The library, installed CMake
    # target, Python package and default benchmark have no new dependency.
    if(FASTCD_BUILD_FLINT_BENCHMARK)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(FLINT QUIET IMPORTED_TARGET flint)
        endif()
        find_path(FASTCD_FLINT_INCLUDE_DIR flint/flint.h HINTS ${FLINT_INCLUDE_DIRS})
        find_library(FASTCD_FLINT_LIBRARY NAMES flint HINTS ${FLINT_LIBRARY_DIRS})
        if(NOT FASTCD_FLINT_INCLUDE_DIR OR NOT FASTCD_FLINT_LIBRARY)
            message(FATAL_ERROR
                "FLINT development files are required for FASTCD_BUILD_FLINT_BENCHMARK. "
                "Install FLINT and set CMAKE_PREFIX_PATH or PKG_CONFIG_PATH to its prefix.")
        endif()
        add_executable(fastcd_flint_benchmark benchmarks/flint.c)
        target_include_directories(fastcd_flint_benchmark PRIVATE
            ${PROJECT_SOURCE_DIR}/src/finite)
        target_include_directories(fastcd_flint_benchmark SYSTEM PRIVATE
            ${FASTCD_FLINT_INCLUDE_DIR})
        target_link_libraries(fastcd_flint_benchmark PRIVATE fastcd::fastcd)
        if(TARGET PkgConfig::FLINT)
            target_link_libraries(fastcd_flint_benchmark PRIVATE PkgConfig::FLINT)
        else()
            target_link_libraries(fastcd_flint_benchmark PRIVATE ${FASTCD_FLINT_LIBRARY})
        endif()
        fastcd_configure_target(fastcd_flint_benchmark)
    endif()
    # Record the source contents and effective library options beside the runner.
    file(GLOB_RECURSE fastcd_benchmark_sources CONFIGURE_DEPENDS
        src/*.c src/*.h include/*.h tests/reference_cd.* benchmarks/*.c benchmarks/*.h
        benchmarks/*.py tools/*.py tools/*.h)
    list(APPEND fastcd_benchmark_sources "${PROJECT_SOURCE_DIR}/CMakeLists.txt")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${fastcd_benchmark_sources})
    set(fastcd_source_hashes "")
    foreach(source IN LISTS fastcd_benchmark_sources)
        file(SHA256 "${source}" source_hash)
        file(RELATIVE_PATH relative_source "${PROJECT_SOURCE_DIR}" "${source}")
        string(APPEND fastcd_source_hashes "${source_hash}  ${relative_source}\n")
    endforeach()
    set(fastcd_source_options "")
    get_target_property(fastcd_library_sources fastcd SOURCES)
    foreach(source IN LISTS fastcd_library_sources)
        get_source_file_property(options "${source}" COMPILE_OPTIONS)
        if(options)
            string(APPEND fastcd_source_options "${source}=${options}\n")
        endif()
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
        "source_options:\n${fastcd_source_options}"
        "source_sha256:\n${fastcd_source_hashes}")
    string(SHA256 fastcd_build_id
        "${fastcd_build_record};${fastcd_build_options};${fastcd_build_definitions}")
    target_compile_definitions(fastcd_benchmark PRIVATE
        FASTCD_BENCHMARK_BUILD_ID="${fastcd_build_id}-$<CONFIG>")
    target_compile_definitions(fastcd_domains_benchmark PRIVATE
        FASTCD_BENCHMARK_BUILD_ID="${fastcd_build_id}-$<CONFIG>")
    file(GENERATE OUTPUT "${PROJECT_BINARY_DIR}/fastcd-build-$<CONFIG>.txt"
        CONTENT "build_id=${fastcd_build_id}-$<CONFIG>\n${fastcd_build_record}")
    if(FASTCD_BUILD_FLINT_BENCHMARK)
        file(SHA256 "${FASTCD_FLINT_LIBRARY}" fastcd_flint_library_hash)
        file(SHA256 "${FASTCD_FLINT_INCLUDE_DIR}/flint/flint.h" fastcd_flint_header_hash)
        file(GLOB fastcd_flint_headers CONFIGURE_DEPENDS "${FASTCD_FLINT_INCLUDE_DIR}/flint/*.h")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${FASTCD_FLINT_LIBRARY}" ${fastcd_flint_headers})
        set(fastcd_flint_header_hashes "")
        foreach(header IN LISTS fastcd_flint_headers)
            file(SHA256 "${header}" header_hash)
            get_filename_component(header_name "${header}" NAME)
            string(APPEND fastcd_flint_header_hashes "${header_hash}  ${header_name}\n")
        endforeach()
        string(SHA256 fastcd_flint_headers_hash "${fastcd_flint_header_hashes}")
        string(CONCAT fastcd_flint_record
            "flint_library=${FASTCD_FLINT_LIBRARY}\n"
            "flint_library_sha256=${fastcd_flint_library_hash}\n"
            "flint_header_sha256=${fastcd_flint_header_hash}\n"
            "flint_headers_sha256=${fastcd_flint_headers_hash}\n"
            "flint_include=${FASTCD_FLINT_INCLUDE_DIR}\n"
            "flint_pkgconfig_version=${FLINT_VERSION}\n"
            "flint_pkgconfig_flags=${FLINT_CFLAGS_OTHER};${FLINT_LDFLAGS}\n"
            "${fastcd_build_record}")
        string(SHA256 fastcd_flint_build_id "${fastcd_flint_record}")
        target_compile_definitions(fastcd_flint_benchmark PRIVATE
            FASTCD_BENCHMARK_BUILD_ID="${fastcd_flint_build_id}-$<CONFIG>")
        file(GENERATE OUTPUT "${PROJECT_BINARY_DIR}/fastcd-flint-build-$<CONFIG>.txt"
            CONTENT "build_id=${fastcd_flint_build_id}-$<CONFIG>\n${fastcd_flint_record}")
    endif()
    if(BUILD_TESTING)
        add_test(NAME fastcd.benchmark_metadata COMMAND fastcd_benchmark --metadata)
        add_test(NAME fastcd.domains_benchmark_metadata COMMAND fastcd_domains_benchmark --metadata)
        if(FASTCD_BUILD_FLINT_BENCHMARK)
            add_test(NAME fastcd.flint_benchmark_metadata COMMAND fastcd_flint_benchmark --metadata)
            add_test(NAME fastcd.flint_agreement COMMAND fastcd_flint_benchmark --self-test)
        endif()
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
    Python_add_library(pyfastcd_domains MODULE WITH_SOABI python/src/pyfastcd/_domains_core.c)
    set_target_properties(pyfastcd_domains PROPERTIES OUTPUT_NAME "_domains_core" C_VISIBILITY_PRESET hidden)
    target_link_libraries(pyfastcd_domains PRIVATE fastcd::fastcd)
    target_include_directories(pyfastcd_domains PRIVATE ${PROJECT_SOURCE_DIR}/src/finite)
    fastcd_configure_target(pyfastcd_domains)
    install(TARGETS pyfastcd_domains LIBRARY DESTINATION pyfastcd COMPONENT python
        RUNTIME DESTINATION pyfastcd COMPONENT python)
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
