import argparse
import subprocess
import random
import time
import os

def write_log(ss, file):
    """ Write to log """
    with open(file, 'a') as log:
        log.write(ss + '\n')

def execute_terminal_command(command):
    """ Execution command """
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

def generate_random_conf(x, all_flags):
    """ Generation 0-1 mapping for disable-enable options """
    comb = bin(x).replace('0b', '')
    comb = '0' * (len(all_flags) - len(comb)) + comb
    conf = [int(s) for s in comb]
    return conf


def read_flags_from_file(file_path):
    """
    obtain all flags
    """
    with open(file_path) as f:
        flags = [line.strip() for line in f if line.strip()]
    return flags

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Iterative Optimization")
    parser.add_argument("--log_file", type=str, required=True,
                        help="File to save log")
    
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to the source program for tuning")
    
    parser.add_argument("--gcc_path", type=str, required=True,
                        help="Path of compiler")

    parser.add_argument("--exec_param", type=str, default=None,
                        help="Execution parameter for the output executable (can be empty)")
    
    parser.add_argument("--flag_path", type=str, required=True,
                        help="Tuning flags file")
    
    args = parser.parse_args()

    if args.exec_param:
        EXEC_PARAM = args.exec_param
    else:
        EXEC_PARAM = '' 
    
    LOG_DIR = 'log' + os.sep
    LOG_FILE = LOG_DIR +  args.log_file
    ERROR_FILE = LOG_DIR + 'err.log'
    SOURCE_PATH = args.source_path
    GCC_PATH = args.gcc_path
    INCLUDE_PATH = ''
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if args.flag_path:
        all_flags = read_flags_from_file(args.flag_path)
    else:
        all_flags = ['-O2']
        print('No flags')
    
    ts = []   # time consumption
    res = []  # speedup for different flag combinations
    seqs = [] # different flag combinations
    print(all_flags)
    ts.append(0)
    time_zero = time.time()
    last_log_time = 0.0 
    while ts[-1] < 5000:
        x = random.randint(0, 2 ** len(all_flags) - 1)
        seq = generate_random_conf(x, all_flags)
        opt = ' '.join(flag if bit else flag.replace("-f", "-fno-", 1) for flag, bit in zip(all_flags, seq))
        command = f"{GCC_PATH} -O2 {opt} -c {INCLUDE_PATH} {SOURCE_PATH}/*.c"
        execute_terminal_command(command)
        command2 = f"{GCC_PATH} -o a.out -O2 {opt} *.o -lm"
        execute_terminal_command(command2)
        time_start = time.time()
        command3 = f"./a.out {EXEC_PARAM}"
        execute_terminal_command(command3)
        time_end = time.time()
        cmd4 = 'rm -rf *.o *.I *.s a.out'
        execute_terminal_command(cmd4)
        time_c = time_end - time_start   #time_opt

        command = f"{GCC_PATH} -O3 -c {INCLUDE_PATH} {SOURCE_PATH}/*.c"
        execute_terminal_command(command)
        command2 = f"{GCC_PATH} -o a.out -O3 *.o -lm"
        execute_terminal_command(command2)
        time_o3 = time.time()
        command3 = f"./a.out {EXEC_PARAM}"
        execute_terminal_command(command3)
        time_o3_end = time.time()
        cmd4 = 'rm -rf *.o *.I *.s a.out'
        execute_terminal_command(cmd4)
        time_o3_c = time_o3_end - time_o3  #time_o3
        res.append(time_o3_c /time_c)
        ts.append(time.time()-time_zero)
        seqs.append(seq)
        best_per = max(res)
        best_seq = seqs[res.index(max(res))] 
        if ts[-1] - last_log_time >= 20:
            ss = f'{round(ts[-1])}: best-per {max(res)}, best-seq {seqs[res.index(max(res))]}'
            write_log(ss, LOG_FILE)
            last_log_time = ts[-1]
        