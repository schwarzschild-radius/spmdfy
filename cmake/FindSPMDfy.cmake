function(add_spmdfy_source ISPC_SOURCE_TARGET SPMDFY_CUDA_SOURCE SPMDFY_ISPC_SOURCE)
    set(oneValueArgs HINTS ISPC_DIR)

    cmake_parse_arguments(SPMDFY "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(SPMDFY_EXE ${SPMDFY_HINTS}/spmdfy)

    if(SPMDFY_VERBOSE)
        set(${SPMDFY_ISPC_SOURCE}_VERBOSE "-v")
    endif()
    
    if(SPMDFY_ISPC_DIR)
        set(${SPMDFY_ISPC_SOURCE}_DIR ${SPMDFY_ISPC_DIR})
    else()
        set(${SPMDFY_ISPC_SOURCE}_DIR ${CMAKE_CURRENT_BINARY_DIR})
    endif()

    add_custom_command(
        OUTPUT ${${SPMDFY_ISPC_SOURCE}_DIR}/${SPMDFY_ISPC_SOURCE}
        COMMAND ${SPMDFY_EXE} -o ${${SPMDFY_ISPC_SOURCE}_DIR}/${SPMDFY_ISPC_SOURCE} 
                              ${${SPMDFY_ISPC_SOURCE}_VERBOSE} 
                              ${CMAKE_CURRENT_SOURCE_DIR}/${SPMDFY_CUDA_SOURCE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SPMDFY_CUDA_SOURCE} spmdfy
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Building ISPC source"
        VERBATIM
    )
    add_custom_target(${ISPC_SOURCE_TARGET} DEPENDS ${${SPMDFY_ISPC_SOURCE}_DIR}/${SPMDFY_ISPC_SOURCE})
endfunction()