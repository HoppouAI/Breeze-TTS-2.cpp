cmake_minimum_required(VERSION 3.19)
project(breeze_tts_cpp LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BREEZE_VULKAN "Build ggml with the Vulkan backend" ON)
option(BREEZE_CUDA "Build ggml with the CUDA backend" OFF)
option(BREEZE_BUILD_CLI "Build the command line app" ON)
option(BREEZE_BUILD_SERVER "Build the HTTP server" ON)
option(BREEZE_MP3 "Let the server encode mp3 with the shine submodule" ON)
option(BREEZE_BUILD_SHARED "Build the shared C library" ON)
option(BREEZE_BUILD_TESTS "Build inference regression tests" OFF)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/CMakeLists.txt")
    message(FATAL_ERROR "third_party/ggml is missing. Run: git submodule update --init --recursive")
endif()

set(GGML_VULKAN ${BREEZE_VULKAN} CACHE BOOL "" FORCE)
set(GGML_CUDA ${BREEZE_CUDA} CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
if(MINGW)
    # libgomp has no static build here, and it would leak into the shared library
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
endif()
add_subdirectory(third_party/ggml)

set(BREEZE_SOURCES
    src/common.cpp
    src/audio.cpp
    src/config.cpp
    src/gguf_loader.cpp
    src/tokenizer.cpp
    src/text_encoder.cpp
    src/backbone.cpp
    src/depth_decoder.cpp
    src/codec.cpp
    src/codec_conv.cpp
    src/codec_decoder.cpp
    src/codec_transformer.cpp
    src/sampling.cpp
    src/generation.cpp
    src/text_split.cpp
    src/voice.cpp
    src/model.cpp
)

add_library(breeze_core STATIC ${BREEZE_SOURCES})
target_include_directories(breeze_core PUBLIC include)
target_link_libraries(breeze_core PUBLIC ggml)

# make mingw binaries self contained so they do not pick up stray runtime DLLs
if(MINGW)
    add_link_options(-static-libgcc -static-libstdc++ -static)
endif()

if(BREEZE_BUILD_TESTS)
    enable_testing()
    add_executable(breeze-test-depth tests/depth_replay.cpp)
    target_link_libraries(breeze-test-depth PRIVATE breeze_core)
    add_test(NAME depth-replay-cpu COMMAND breeze-test-depth)
    add_test(NAME depth-replay-gpu COMMAND breeze-test-depth --gpu)
    set_tests_properties(depth-replay-gpu PROPERTIES SKIP_RETURN_CODE 77)
endif()

set(BREEZE_WEBUI_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/breeze_webui_assets.h)
add_custom_command(
    OUTPUT ${BREEZE_WEBUI_HEADER}
    COMMAND ${CMAKE_COMMAND}
        -DWEBUI_DIR=${CMAKE_CURRENT_SOURCE_DIR}/webui
        -DOUT_HEADER=${BREEZE_WEBUI_HEADER}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_webui.cmake
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/webui/index.html
        ${CMAKE_CURRENT_SOURCE_DIR}/webui/style.css
        ${CMAKE_CURRENT_SOURCE_DIR}/webui/app.js
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_webui.cmake
    COMMENT "Embedding WebUI assets"
    VERBATIM
)
add_custom_target(breeze_webui DEPENDS ${BREEZE_WEBUI_HEADER})

if(BREEZE_BUILD_SHARED)
    add_library(breeze SHARED src/c_api.cpp)
    target_link_libraries(breeze PRIVATE breeze_core)
    target_compile_definitions(breeze PRIVATE BREEZE_BUILD_SHARED)
    set_target_properties(breeze PROPERTIES OUTPUT_NAME breeze)
endif()

if(BREEZE_BUILD_CLI)
    add_executable(breeze-cli apps/cli/main.cpp)
    target_link_libraries(breeze-cli PRIVATE breeze_core)

    add_executable(breeze-convert apps/convert/main.cpp)
    target_link_libraries(breeze-convert PRIVATE breeze_core)
endif()

add_executable(breeze-quantize apps/quantize/main.cpp)
target_link_libraries(breeze-quantize PRIVATE breeze_core)

if(BREEZE_BUILD_SERVER)
    add_executable(breeze-server apps/server/main.cpp apps/server/server.cpp apps/server/voices.cpp
                                 apps/server/ws.cpp apps/server/ws_api.cpp apps/server/openai.cpp
                                 apps/server/encode.cpp)
    add_dependencies(breeze-server breeze_webui)
    target_include_directories(breeze-server PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated third_party)
    target_link_libraries(breeze-server PRIVATE breeze_core)
    if(WIN32)
        target_link_libraries(breeze-server PRIVATE ws2_32)
    endif()
    find_package(Threads REQUIRED)
    target_link_libraries(breeze-server PRIVATE Threads::Threads)
    if(BREEZE_MP3 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/shine/src/lib/layer3.c")
        # shine is lgpl, it stays in its own submodule and its own static lib
        file(GLOB SHINE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/third_party/shine/src/lib/*.c)
        add_library(shine STATIC ${SHINE_SOURCES})
        target_include_directories(shine PRIVATE third_party/shine/src/lib)
        if(MSVC)
            target_compile_options(shine PRIVATE /FI${CMAKE_CURRENT_SOURCE_DIR}/cmake/shine_msvc.h)
        endif()
        if(UNIX)
            target_link_libraries(shine PRIVATE m)
        endif()
        target_link_libraries(breeze-server PRIVATE shine)
        target_compile_definitions(breeze-server PRIVATE BREEZE_HAVE_MP3)
    elseif(BREEZE_MP3)
        message(WARNING "third_party/shine is missing so mp3 requests will get wav. Run: git submodule update --init --recursive")
    endif()
endif()
