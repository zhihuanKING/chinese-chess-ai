cmake_minimum_required(VERSION 3.18)
project(xqcore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Optimization flags requested by the project (-O3 -march=native).
# -march=native is gated so cross-compiles / unusual CPUs can disable it.
option(XQ_NATIVE "Build with -march=native" ON)
add_library(xq_flags INTERFACE)
target_compile_options(xq_flags INTERFACE
  $<$<CONFIG:Release>:-O3>
  $<$<AND:$<BOOL:${XQ_NATIVE}>,$<CONFIG:Release>>:-march=native>
  -Wall -Wextra)

# ---------------------------------------------------------------------------
# Core rules + engine static library (one source of truth, reused everywhere).
# ---------------------------------------------------------------------------
add_library(xqcore STATIC
  src/position.cpp
  src/eval.cpp
  src/search.cpp)
target_include_directories(xqcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(xqcore PUBLIC xq_flags)
# Built into the _xqcore shared module, so the static lib must be PIC.
set_target_properties(xqcore PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Python extension module: _xqcore  (imported as xqai._xqcore)
# pybind11 comes from pip; locate it via Python or pybind11-config.
# ---------------------------------------------------------------------------
find_package(Python COMPONENTS Interpreter Development.Module QUIET)

# Ask the active Python for pybind11's CMake dir if not already on the path.
if(NOT pybind11_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pybind11_cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_rc)
  if(_pybind11_rc EQUAL 0)
    set(pybind11_DIR "${_pybind11_cmakedir}")
  endif()
endif()

find_package(pybind11 CONFIG QUIET)

if(pybind11_FOUND)
  pybind11_add_module(_xqcore bindings/module.cpp)
  target_link_libraries(_xqcore PRIVATE xqcore)
  # When built under scikit-build-core the module is installed into the wheel
  # so that `import xqai._xqcore` works.
  install(TARGETS _xqcore DESTINATION xqai)
else()
  message(STATUS
    "pybind11 not found -> skipping _xqcore module; perft/xqab/tests still build")
endif()

# ---------------------------------------------------------------------------
# Executables: perft and xqab
# ---------------------------------------------------------------------------
add_executable(perft src/perft_main.cpp)
target_link_libraries(perft PRIVATE xqcore)

add_executable(xqab src/xqab_main.cpp)
target_link_libraries(xqab PRIVATE xqcore)

# ---------------------------------------------------------------------------
# Tests (header-only mini-doctest in third_party/).
# ---------------------------------------------------------------------------
enable_testing()

add_executable(test_perft tests/test_perft.cpp)
target_link_libraries(test_perft PRIVATE xqcore)
target_include_directories(test_perft PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

add_executable(test_rules tests/test_rules.cpp)
target_link_libraries(test_rules PRIVATE xqcore)
target_include_directories(test_rules PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

add_test(NAME perft COMMAND test_perft)
add_test(NAME rules COMMAND test_rules)
