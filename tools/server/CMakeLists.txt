include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# tree-sitter AST library for prompt compaction

set(TARGET treesitter)

# Tree-sitter core
set(TS_SRC
    treesitter/treesitter.c
    treesitter/wasm_store.c
)

# AST compaction wrapper
set(COMPACT_SRC
    treesitter/compact.cpp
)

# Language grammar files (parser + scanner pairs)
set(TS_GRAMMARS
    treesitter/grammars/python.c
    treesitter/grammars/python-scanner.c
    treesitter/grammars/javascript.c
    treesitter/grammars/javascript-scanner.c
    treesitter/grammars/typescript.c
    treesitter/grammars/typescript-scanner.c
    treesitter/grammars/rust.c
    treesitter/grammars/rust-scanner.c
    treesitter/grammars/go.c
    treesitter/grammars/c.c
    treesitter/grammars/cpp.c
    treesitter/grammars/cpp-scanner.c
    treesitter/grammars/java.c
    treesitter/grammars/bash.c
    treesitter/grammars/bash-scanner.c
)

add_library(${TARGET} STATIC ${TS_SRC} ${COMPACT_SRC} ${TS_GRAMMARS})

target_include_directories(${TARGET} PUBLIC
    treesitter/include          # for tree_sitter/api.h / parser.h
    treesitter                  # for local headers (alloc.h, etc.)
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Grammar C files often have large tables and warnings - suppress them
target_compile_options(${TARGET} PRIVATE -w)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_link_libraries(${TARGET} PUBLIC ${CMAKE_THREAD_LIBS_INIT})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    server-tools.cpp
    server-tools.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd treesitter ${CMAKE_THREAD_LIBS_INIT})


# llama-server executable

set(TARGET llama-server)

set(TARGET_SRCS
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)

option(LLAMA_BUILD_WEBUI "Build the embedded Web UI" ON)

if (LLAMA_BUILD_WEBUI)
    set(PUBLIC_ASSETS
        index.html
        bundle.js
        bundle.css
        loading.html
    )

    foreach(asset ${PUBLIC_ASSETS})
        set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
        set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
        list(APPEND TARGET_SRCS ${output})
        add_custom_command(
            DEPENDS "${input}"
            OUTPUT "${output}"
            COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
        )
        set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
    endforeach()
    add_definitions(-DLLAMA_BUILD_WEBUI)
else()
endif()

add_executable(${TARGET} ${TARGET_SRCS})
install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE server-context PUBLIC llama-common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

target_compile_features(${TARGET} PRIVATE cxx_std_17)
