import os, subprocess, re, argparse, time

from tuner import FlagInfo, Evaluator, FLOAT_MAX
from tuner import SRTuner

# Define GCC flags
class GCCFlagInfo(FlagInfo):
    def __init__(self, name, configs):
        super().__init__(name, configs)

def read_gcc_opts(path):
    search_space = {}
    with open(path, "r") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            search_space[line] = GCCFlagInfo(name=line, configs=[False, True])
    return search_space


def convert_to_str(opt_setting, search_space):
    str_opt_setting = ""
    for flag_name, config in opt_setting.items():
        assert flag_name in search_space
        assert isinstance(config, bool), f"{flag_name} must be boolean"
        if config:
            str_opt_setting += f" {flag_name}"
        else:
            negated_flag_name = flag_name.replace("-f", "-fno-", 1)
            str_opt_setting += f" {negated_flag_name}"
    return str_opt_setting.strip()


def write_log(ss, file):
    """ Write to log """
    with open(file, 'a') as log:
        log.write(ss + '\n')

def execute_terminal_command(command):
    """ Execute command """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            if result.stdout:
                print("命令输出：")
                print(result.stdout)
        else:
            if result.stderr:
                print("错误输出：")
                print(result.stderr)
    except Exception as e:
        print("执行命令时出现错误：", str(e))


class cBenchEvaluator(Evaluator):
    def __init__(self, path, search_space, exec_param, include_path):
        super().__init__(path)
        self.search_space = search_space
        self.include_path = include_path
        self.exec_param = exec_param

    def evaluate(self, opt_setting):
        opt = convert_to_str(opt_setting, self.search_space)
        command = f"gcc -O2 {opt} -c {self.include_path} {self.path}/*.c"
        execute_terminal_command(command)
        command2 = f"gcc -o a.out -O2 {opt} *.o -lm"
        execute_terminal_command(command2)
        time_start = time.time()
        command3 = f"./a.out {self.exec_param}"
        execute_terminal_command(command3)
        time_end = time.time()  
        cmd4 = 'rm -rf *.o *.I *.s a.out'
        execute_terminal_command(cmd4)
        perf = time_end - time_start 
        return perf

    def evaluate_default(self):
        command = f"gcc -O3 -c {self.include_path} {self.path}/*.c"
        execute_terminal_command(command)
        command2 = f"gcc -o a.out -O3 *.o -lm"
        execute_terminal_command(command2)
        time_o3 = time.time()
        command3 = f"./a.out {self.exec_param}"
        execute_terminal_command(command3)
        time_o3_end = time.time()  
        cmd4 = 'rm -rf *.o *.I *.s a.out'
        execute_terminal_command(cmd4)
        perf_o3 = time_o3_end - time_o3
        return perf_o3

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Effective Compiler Optimization Customization by Exposing Synergistic Relations")
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to the source program for tuning")

    parser.add_argument("--exec_param", type=str, default='',
                        help="Execution parameter for the output executable (can be empty)")
    
    parser.add_argument("--flag_path", type=str, required=True,
                        help="Tuning flags file")

    parser.add_argument("--log_file", type=str, required=True,
                        help="File to save log")
    args = parser.parse_args()

    # 读取 flag 搜索空间
    search_space = read_gcc_opts(args.flag_path)
    include_path = '-I /data/mingxuanzhu/PDCAT-master/Benchmarks/polyBench/utilities /data/mingxuanzhu/PDCAT-master/Benchmarks/polyBench/utilities/polybench.c'
    evaluator = cBenchEvaluator(path=args.source_path, search_space=search_space, exec_param=args.exec_param, include_path=include_path)
    ts = []   # time consumption
    ts.append(0)
    time_zero = time.time()
    budget = 1
    srtuner = SRTuner(search_space, evaluator, log_file = args.log_file)
    best_opt_setting, best_perf = srtuner.tune(5000)

        