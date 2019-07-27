macro(add_ispc_library ISPC_TARGET ISPC_SOURCE)
    set(oneValueArgs HEADER # -h
                     HEADER_DIR # -h {HEADER_DIR}/{HEADER}
                     OBJECT # -o {OBJECT}
                     INCLUDE_DIR # -I {INCLUDE_DIR}/*.ispc
                     OPT_LEVEL # -O{0, 1, 2, 3}
                     OBJECT_DIR # -o ${OBJECT_DIR}/{OBJECT}
                     ARCH # --target=${ARCH}
    )

    set(options VEROBSE PIC)

    cmake_parse_arguments(ISPC "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(ISPC_EXE "ispc")

    if(ISPC_ARCH)
        set(${ISPC_TARGET}_ARCH ${ISPC_ARCH})
    else()
        set(${ISPC_TARGET}_ARCH "avx2-i32x8")
    endif()
    
    if(ISPC_HEADER)
        set(${ISPC_TARGET}_HEADER ${ISPC_HEADER})
    else()
        set(${ISPC_TARGET}_HEADER ${ISPC_TARGET}.h)
    endif()

    if(ISPC_OBJECT)
        set(${ISPC_TARGET}_OBJECT ${ISPC_OBJECT})
    else()
        set(${ISPC_TARGET}_OBJECT ${ISPC_TARGET}.o)
    endif()

    if(ISPC_HEADER_DIR)
        set(${ISPC_TARGET}_HEADER_DIR ${ISPC_HEADER_DIR})
    else()
        set(${ISPC_TARGET}_HEADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/${ISPC_TARGET}/include/)
    endif()

    if(ISPC_OBJECT_DIR)
        set(${ISPC_TARGET}_OBJECT_DIR ${ISPC_OBJECT_DIR})
    else()
        set(${ISPC_TARGET}_OBJECT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    if(PIC)
        set(${PIC} "--pic")
    endif()

    if(ISPC_INCLUDE_DIR)
        set(${ISPC_TARGET}_INCLUDE_DIR ${ISPC_INCLUDE_DIR})
    else()
        set(${ISPC_TARGET}_INCLUDE_DIR "./")
    endif()

    file(GLOB ISPC_INCLUDE_FILES
        "${${ISPC_TARGET}_INCLUDE_DIR}/*.ispc"
    )

    # message("${ISPC_TARGET}")
    # message("${ISPC_SOURCE}")
    # message("${${ISPC_TARGET}_HEADER_DIR}")
    # message("${${ISPC_TARGET}_HEADER}")
    # message("${${ISPC_TARGET}_OBJECT_DIR}")
    # message("${${ISPC_TARGET}_OBJECT}")
    # message("${${ISPC_TARGET}_INCLUDE_DIR}")
    # message("${${ISPC_TARGET}_ARCH}")
    # message("${ISPC_INCLUDE_FILES}")

    set(${ISPC_TARGET}_OUTPUT ${${ISPC_TARGET}_HEADER_DIR}/${${ISPC_TARGET}_HEADER} 
                              ${${ISPC_TARGET}_OBJECT_DIR}/${${ISPC_TARGET}_OBJECT})

    # message("${${ISPC_TARGET}_OBJECT_DIR}/${${ISPC_TARGET}_OBJECT}")
    add_custom_command(OUTPUT ${${ISPC_TARGET}_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${${ISPC_TARGET}_HEADER_DIR}
        COMMAND ${ISPC_EXE} -h ${${ISPC_TARGET}_HEADER_DIR}/${${ISPC_TARGET}_HEADER} 
                     ${PIC}
                     -I ${${ISPC_TARGET}_INCLUDE_DIR}
                     -o ${${ISPC_TARGET}_OBJECT_DIR}/${${ISPC_TARGET}_OBJECT}
                     --target ${${ISPC_TARGET}_ARCH} ${ISPC_SOURCE}
        MAIN_DEPENDENCY ${ISPC_SOURCE}
        DEPENDS ${ISPC_INCLUDE_FILES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Building ISPC library"
        VERBATIM)

    add_custom_target(${ISPC_TARGET}_ DEPENDS ${${ISPC_TARGET}_OUTPUT})

    add_library(${ISPC_TARGET} STATIC IMPORTED GLOBAL)
    set_property(TARGET ${ISPC_TARGET} PROPERTY IMPORTED_LOCATION ${${ISPC_TARGET}_OBJECT_DIR}/${${ISPC_TARGET}_OBJECT})
    add_dependencies(${ISPC_TARGET} ${ISPC_TARGET}_)
endmacro()
