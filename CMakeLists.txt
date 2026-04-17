cmake_minimum_required(VERSION 3.14)
project(magic_renderer_merged)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 基础编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static -fopenmp")
# 使用控制台子系统以便看到输出
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mconsole")

# 发布版本优化选项
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -flto -funroll-loops -fopenmp -DNDEBUG")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-flto")

# 链接时优化
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

# 设置本地依赖路径
set(LOCAL_DEPS_DIR ${CMAKE_SOURCE_DIR}/../2d-renderer/deps)
set(INSTALL_DIR ${LOCAL_DEPS_DIR}/install)

# 检查依赖是否已构建
if(NOT EXISTS ${INSTALL_DIR})
    message(WARNING "本地依赖未构建。请先运行以下命令：")
    message(WARNING "  PowerShell -ExecutionPolicy Bypass -File ${CMAKE_SOURCE_DIR}/../2d-renderer/download_deps.ps1")
    message(WARNING "  PowerShell -ExecutionPolicy Bypass -File ${CMAKE_SOURCE_DIR}/../2d-renderer/build_deps.ps1")
    message(FATAL_ERROR "依赖库未找到。请按照上述说明构建依赖库。")
endif()

# 查找GLEW
set(GLEW_INCLUDE ${INSTALL_DIR}/include)
set(GLEW_LIB_DIR ${INSTALL_DIR}/lib)

if(NOT EXISTS ${GLEW_INCLUDE}/GL/glew.h)
    # 尝试在glew子目录中查找
    if(EXISTS ${LOCAL_DEPS_DIR}/glew/include/GL/glew.h)
        set(GLEW_INCLUDE ${LOCAL_DEPS_DIR}/glew/include)
        if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(GLEW_LIB_DIR ${LOCAL_DEPS_DIR}/glew/lib/Release/x64)
        else()
            set(GLEW_LIB_DIR ${LOCAL_DEPS_DIR}/glew/lib)
        endif()
    else()
        message(FATAL_ERROR "GLEW头文件未找到。请确保已正确构建依赖库。")
    endif()
endif()

# 查找GLFW
set(GLFW_INCLUDE ${INSTALL_DIR}/include)
set(GLFW_LIB_DIR ${INSTALL_DIR}/lib)

if(NOT EXISTS ${GLFW_INCLUDE}/GLFW/glfw3.h)
    # 尝试在glfw子目录中查找
    if(EXISTS ${LOCAL_DEPS_DIR}/glfw/include/GLFW/glfw3.h)
        set(GLFW_INCLUDE ${LOCAL_DEPS_DIR}/glfw/include)
        if(WIN32)
            set(GLFW_LIB_DIR ${LOCAL_DEPS_DIR}/glfw/src)
        else()
            set(GLFW_LIB_DIR ${LOCAL_DEPS_DIR}/glfw/src)
        endif()
    else()
        message(FATAL_ERROR "GLFW头文件未找到。请确保已正确构建依赖库。")
    endif()
endif()

# 设置包含目录
include_directories(${GLEW_INCLUDE} ${GLFW_INCLUDE})
include_directories(${CMAKE_SOURCE_DIR}/include)

# 创建可执行文件
add_executable(magic_renderer_merged
    src/main.cpp
    # 其他源文件将在创建后添加
)

# 查找库文件
if(WIN32)
    # Windows库文件
    find_library(GLEW_LIB 
        NAMES libglew32.a libglew32s.a glew32s.lib glew32.lib
        PATHS ${GLEW_LIB_DIR}
        NO_DEFAULT_PATH
    )
    
    find_library(GLFW_LIB 
        NAMES glfw3.lib libglfw3.a glfw3dll.lib
        PATHS ${GLFW_LIB_DIR}
        NO_DEFAULT_PATH
    )
    
    if(NOT GLEW_LIB)
        message(FATAL_ERROR "未找到GLEW库文件。请在 ${GLEW_LIB_DIR} 中查找。")
    endif()
    
    if(NOT GLFW_LIB)
        message(FATAL_ERROR "未找到GLFW库文件。请在 ${GLFW_LIB_DIR} 中查找。")
    endif()
    
    # 链接库
    target_link_libraries(magic_renderer_merged 
        ${GLFW_LIB}
        ${GLEW_LIB}
        opengl32
        gdi32
        user32
    )
    
    # 对于静态链接GLEW，需要定义GLEW_STATIC
    if(GLEW_LIB MATCHES "glew32s|libglew32s|libglew32\\.a")
        target_compile_definitions(magic_renderer_merged PRIVATE GLEW_STATIC)
    endif()
    
    # 复制GLEW DLL到输出目录（如果使用动态链接）
    if(GLEW_LIB MATCHES "glew32[^s]")
        # 查找DLL
        find_file(GLEW_DLL 
            NAMES glew32.dll
            PATHS ${GLEW_LIB_DIR} ${LOCAL_DEPS_DIR}/glew/lib
            NO_DEFAULT_PATH
        )
        
        if(GLEW_DLL)
            add_custom_command(TARGET magic_renderer_merged POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${GLEW_DLL}
                    $<TARGET_FILE_DIR:magic_renderer_merged>
                COMMENT "Copying GLEW DLL to output directory"
            )
        else()
            message(WARNING "未找到glew32.dll。如果使用动态链接，运行时可能需要此文件。")
        endif()
    endif()
    
else()
    # Linux/macOS库文件
    find_library(GLEW_LIB 
        NAMES GLEW glew
        PATHS ${GLEW_LIB_DIR}
        NO_DEFAULT_PATH
    )
    
    find_library(GLFW_LIB 
        NAMES glfw3 glfw
        PATHS ${GLFW_LIB_DIR}
        NO_DEFAULT_PATH
    )
    
    target_link_libraries(magic_renderer_merged 
        ${GLFW_LIB}
        ${GLEW_LIB}
        GL
    )
endif()

# 打印配置信息
message(STATUS "==========================================")
message(STATUS "Magic Renderer Merged 配置信息")
message(STATUS "==========================================")
message(STATUS "GLEW头文件目录: ${GLEW_INCLUDE}")
message(STATUS "GLEW库目录: ${GLEW_LIB_DIR}")
message(STATUS "GLEW库文件: ${GLEW_LIB}")
message(STATUS "GLFW头文件目录: ${GLFW_INCLUDE}")
message(STATUS "GLFW库目录: ${GLFW_LIB_DIR}")
message(STATUS "GLFW库文件: ${GLFW_LIB}")
message(STATUS "==========================================")

# 创建构建目录结构
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)