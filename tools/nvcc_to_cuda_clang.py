'''
Currently Supported Options:
nvcc to cuda clang:
    --arch={value}           - --cuda-gpu-arch={value}
    --use_fast_math          - not supported
    --expt-relaxed-constexpr - not supported
    --maxrregcount={value}   - -Xcuda-ptxas --maxrregcount={value}
    --generate-line-info     - -Xcuda-ptxas --generate-line-info
    -Xcompiler=-fPIC         - -fPIC
    -x {c++ | cu}            - -x {c++ | cuda}
    -dc {path}               - not supported

'''

test_arguments = '''
/usr/bin/clang++-9  -DAllenLib_EXPORTS -I/usr/local/cuda-10.1/targets/x86_64-linux/include -I../main/include -I../x86/muon/decoding/include -I../x86/velo/clustering/include -I../x86/global_event_cut/include -I../x86/utils/prefix_sum/include -I../cuda/global_event_cut/include -I../cuda/UT/common/include -I../cuda/UT/PrVeloUT/include -I../cuda/UT/compassUT/include -I../cuda/UT/UTDecoding/include -I../cuda/UT/consolidate/include -I../cuda/velo/common/include -I../cuda/velo/calculate_phi_and_sort/include -I../cuda/velo/consolidate_tracks/include -I../cuda/velo/mask_clustering/include -I../cuda/velo/search_by_triplet/include -I../cuda/velo/simplified_kalman_filter/include -I../cuda/SciFi/common/include -I../cuda/SciFi/looking_forward/common/include -I../cuda/SciFi/looking_forward/calculate_first_layer_window/include -I../cuda/SciFi/looking_forward/calculate_second_layer_window/include -I../cuda/SciFi/looking_forward/form_seeds_from_candidates/include -I../cuda/SciFi/looking_forward/calculate_candidate_extrapolation_window/include -I../cuda/SciFi/looking_forward/promote_candidates/include -I../cuda/SciFi/looking_forward/calculate_track_extrapolation_window/include -I../cuda/SciFi/looking_forward/extend_tracks/include -I../cuda/SciFi/looking_forward_sbt/search_initial_windows/include -I../cuda/SciFi/looking_forward_sbt/collect_candidates/include -I../cuda/SciFi/looking_forward_sbt/triplet_seeding/include -I../cuda/SciFi/looking_forward_sbt/triplet_keep_best/include -I../cuda/SciFi/looking_forward_sbt/extend_tracks_x/include -I../cuda/SciFi/looking_forward_sbt/composite_algorithms/include -I../cuda/SciFi/looking_forward_sbt/extend_tracks_uv/include -I../cuda/SciFi/looking_forward_sbt/quality_filter/include -I../cuda/SciFi/looking_forward_sbt/quality_filter_x/include -I../cuda/SciFi/looking_forward_sbt/search_uv_windows/include -I../cuda/SciFi/PrForward/include -I../cuda/SciFi/consolidate/include -I../cuda/muon/common/include -I../cuda/utils/prefix_sum/include -I../cuda/event_model/velo/include -I../cuda/event_model/UT/include -I../cuda/event_model/SciFi/include -I../cuda/event_model/common/include -I../checker/tracking/include -I../checker/pv/include -I../stream/sequence/include -I../x86/SciFi/PrForward/include -I../x86/SciFi/LookingForward/include -I../x86/SciFi/MomentumForward/include -I../cuda/kalman/ParKalman/include -I../mdf/include -I../integration/non_event_data/include -I../x86/spmd_handlers/velo/include   -march=native -O3 -g -DNDEBUG -fPIC   -std=gnu++17 -o CMakeFiles/AllenLib.dir/main/src/Allen.cpp.o -c /home/pradeep/Experiments/Allen_SPMD/main/src/Allen.cpp
'''


def nvcc_argument_parser():
    import argparse as argp
    parser = argp.ArgumentParser()
    parser.add_argument("files", nargs="+", help="Files to be complied")
    parser.add_argument("-I", action="append",
                        help="Add directory to include search path")
    parser.add_argument("-L", action="append",
                        help="Add directory to library search path")
    parser.add_argument("-std", help="Language standard to compile for")
    parser.add_argument("-O", help="Optimization levels")
    parser.add_argument("-arch", "--gpu-architecture",
                        help="CUDA architecture")
    parser.add_argument("--use_fast_math", action="store_true",
                        default=False, help="turn on fast math")
    parser.add_argument("--expt-relaxed-constexpr", action="store_true",
                        default=False, help="constexpr constraints")
    parser.add_argument("--maxrregcount", type=int,
                        help="ptx maxrregcount value")
    parser.add_argument(
        "-Xcompiler", help="Passing arugments to host compiler")
    parser.add_argument(
        "-x", choices={"c++", "cu"}, help="Passing arugments to host compiler")
    parser.add_argument("-dc", action="store_true", default=False,
                        help="compile file for separate compilation")
    parser.add_argument("-g", action="store_true",
                        default=False, help="Debug mode")
    parser.add_argument("--generate-line-info",
                        action="store_true", default=False, help="Debug mode")
    parser.add_argument("-D", help="Macro arguments")
    parser.add_argument("-o", help="Output arugments")
    parser.add_argument("-M", action="store_true",
                        default=False, help="Generate dependency info")
    parser.add_argument("-MT", help="Generate dependecy target name")
    parser.add_argument("-march", help="CPU architecture")
    parser.add_argument("-c", action="store_true",
                        default=False, help="only compile")
    parser.add_argument("-m64", action="store_true",
                        default=False, help="enable only 64 bit")
    return parser


def generate_cuda_clang_command(nvcc_arguments):
    cuda_clang_command = "/usr/bin/clang++-9 --cuda-path=/usr/local/cuda-9.0/ --cuda-device-only"
    if nvcc_arguments.files:
        cuda_clang_command += " {}".format(" ".join(nvcc_arguments.files))
    if nvcc_arguments.I:
        cuda_clang_command += " -I {}".format(" -I".join(nvcc_arguments.I))
    if nvcc_arguments.L:
        cuda_clang_command += "-L {}".format("-L".join(nvcc_arguments.L))
    if nvcc_arguments.std:
        cuda_clang_command += " -std={}".format(nvcc_arguments.std)
    if nvcc_arguments.O:
        cuda_clang_command += " -O{}".format(nvcc_arguments.O)
    if nvcc_arguments.gpu_architecture:
        cuda_clang_command += " --cuda-gpu-arch={}".format(
            nvcc_arguments.gpu_architecture)
    if nvcc_arguments.use_fast_math:
        pass
    if nvcc_arguments.expt_relaxed_constexpr:
        pass
    if nvcc_arguments.maxrregcount:
        cuda_clang_command += " -Xcuda-ptxas maxrregcount={}".format(
            nvcc_arguments.maxrregcount)
    if nvcc_arguments.Xcompiler:
        cuda_clang_command += " {}".format(nvcc_arguments.Xcompiler)
    if nvcc_arguments.x:
        cuda_clang_command += " -x {}".format(
            {"cu": "cuda", "c++": "c++"}[nvcc_arguments.x])
    if nvcc_arguments.dc:
        # cuda_clang_command += " {}".format(nvcc_arguments.dc)
        pass
    if nvcc_arguments.g:
        cuda_clang_command += " -g"
    if nvcc_arguments.generate_line_info:
        cuda_clang_command += " -Xcuda-ptxas --generate-line-info"
    if nvcc_arguments.D:
        cuda_clang_command += " -D{}".format(nvcc_arguments.D)
    if nvcc_arguments.o:
        cuda_clang_command += " -o {}".format(nvcc_arguments.o)
    if nvcc_arguments.M:
        cuda_clang_command += " -M"
    if nvcc_arguments.MT:
        cuda_clang_command += " -MT {}".format(nvcc_arguments.MT)
    if nvcc_arguments.march:
        cuda_clang_command += " -march={}".format(nvcc_arguments.march)
    if nvcc_arguments.c:
        cuda_clang_command += " -c"
    if nvcc_arguments.m64:
        cuda_clang_command += " -m64"
    return cuda_clang_command


def parse_compile_commands(compile_commands):
    import json
    compilations = json.loads(compile_commands)
    commands = []
    for compilation in compilations:
        commands.append(compilation["command"])
    return commands


def test_nvcc_parser(parser, command):
    cmds = command.split("&&")
    cmd_out = []
    for cmd in cmds:
        _cmd = cmd
        if(_cmd.find("nvcc") != -1):
            _cmd = generate_cuda_clang_command(
                parser.parse_args(cmd.split()[1:]))
        cmd_out.append(_cmd)
    return " && ".join(cmd_out)


def modify_compile_commands(parser, compile_commands):
    import json
    compilations = json.loads(compile_commands)
    commands = []
    for compilation in compilations:
        cmds = compilation["command"].split(" && ")
        cmd_out = []
        for cmd in cmds:
            _cmd = cmd
            if(_cmd.find("nvcc") != -1):
                _cmd = generate_cuda_clang_command(
                    parser.parse_args(cmd.split()[1:]))
            cmd_out.append(_cmd)
            compilation["command"] = _cmd  # " && ".join(cmd_out)
    return compilations


def test_nvcc_parser():
    with open("../examples/Allen/compile_commands.json") as compile_commands:
        parser = nvcc_argument_parser()
        commands = parse_compile_commands(compile_commands.read())
        for command in commands:
            test_nvcc_parser(parser, command)


if __name__ == "__main__":
    import argparse as argp
    cmd_parser = argp.ArgumentParser()
    cmd_parser.add_argument("compilation_database", help = "CUDA compilation database")
    cmd_parser.add_argument("-o", default = "./compile_commands.json", help = "filename of the new compilation database")
    cmd_args = cmd_parser.parse_args()
    compilation_database = cmd_args.compilation_database
    new_compile_commands = {}
    with open(compilation_database) as compile_commands:
        parser = nvcc_argument_parser()
        new_compile_commands = modify_compile_commands(
            parser, compile_commands.read())

    with open(cmd_args.o, "w") as compile_commands:
        import json
        compile_commands.write(json.dumps(new_compile_commands))
