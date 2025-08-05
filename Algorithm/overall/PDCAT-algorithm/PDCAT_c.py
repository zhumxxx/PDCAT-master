import os,sys
import random, time, copy,subprocess, argparse
import math
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.stats import norm


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

def get_objective_score(independent, k_iter, SOURCE_PATH, GCC_PATH, INCLUDE_PATH, EXEC_PARAM, LOG_FILE, all_flags):
    """ Obtain the speedup """
    opt = ''
    for i in range(len(independent)):
        if independent[i]:
            opt = opt + all_flags[i] + ' '
        else:
            negated_flag_name = all_flags[i].replace("-f", "-fno-", 1)
            opt = opt + negated_flag_name + ' '
    command = f"{GCC_PATH} -O2 {opt} -c {INCLUDE_PATH} {SOURCE_PATH}/*.c"
    execute_terminal_command(command)
    command2 = f"{GCC_PATH} -o a.out -O2 {opt} *.o -lm "
    execute_terminal_command(command2)
    time_start = time.time()
    command3 = f"./a.out {EXEC_PARAM}"
    execute_terminal_command(command3)
    time_end = time.time()  
    cmd4 = 'rm -rf *.o *.I *.s a.out'
    execute_terminal_command(cmd4)
    time_c = time_end - time_start   #time opt
    time_o3 = time.time()
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
    time_o3_c = time_o3_end - time_o3   #time o3
    return (time_o3_c /time_c)

def extract_sequences_from_file(file_path):
    """
    Obtain sequences with speedup > 1.0
    data: 0,1,1,0,...,1
    return: list[list[int]]
    """
    extracted_sequences = []
    with open(file_path, 'r') as file:
        for line in file:
            sequence = [int(x) for x in line.split(',')]
            extracted_sequences.append(sequence)
    return extracted_sequences


def read_flags_from_file(file_path):
    """
    obtain all flags
    """
    
    with open(file_path) as f:
        flags = [line.strip() for line in f if line.strip()]
    return flags

def parse_constraints(file_path):
    strong_dependency = []
    weak_dependency = []
    synergistic_relationship = []
    current_category = None
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("Strong dependency:"):
            current_category = "strong_dependency"
        elif line.startswith("Weak dependency:"):
            current_category = "weak_dependency"
        elif line.startswith("Synergistic relationship:"):
            current_category = "synergistic_relationship"
        elif line:
            if "->" in line:
                constraints = [item.strip() for item in line.replace("->", ",").split(",")]
            elif "and" in line:
                constraints = [item.strip() for item in line.replace("and", ",").split(",")]
            else:
                continue
            if current_category == "strong_dependency":
                strong_dependency.append(constraints)
            elif current_category == "weak_dependency":
                weak_dependency.append(constraints)
            elif current_category == "synergistic_relationship":
                synergistic_relationship.append(constraints)
    return {
        "strong_dependency": strong_dependency,
        "weak_dependency": weak_dependency,
        "synergistic_relationship": synergistic_relationship
    }


class PDCAT:
    def __init__(self, a, b, c, get_objective_score, source_path, gcc_path, include_path, exec_param, log_file, flags, seqs, constraints, permax, permin):
        """
        :param a: parameter of initial process
        :param b: parameter of initial process
        :param c: parameter of tuning process
        :param get_objective_score: obtain true speedup
        :param source_path: program's path
        :param gcc_path: gcc's path
        :param include_path: header file for program
        :param exec_param: exection paramter
        :param log_file: record results
        :param flags: all flags
        :param seqs: all initial tuning sequences
        :param constraints: three type constraints
        :param permax: best performance
        :param permax: lowest performance
        """
        self.a = a
        self.b = b
        self.c = c
        self.proini = []
        self.get_objective_score = get_objective_score 
        self.SOURCE_PATH = source_path
        self.GCC_PATH = gcc_path
        self.INCLUDE_PATH = include_path
        self.EXEC_PARAM = exec_param
        self.LOG_FILE = log_file
        self.all_flags = flags
        self.initial_seqs = seqs
        self.initial_pro = self.Obtain_initial_pro()
        self.constraints = constraints
        self.permax = permax
        self.permin = permin

    def constraints_check(self, seq):
        """
        检查 seq 是否违反强依赖、弱依赖或协同关系约束
        若违反任何一种，返回 True（表示存在问题）
        """
        flag_index = {flag: idx for idx, flag in enumerate(self.all_flags)}

        # Strong dependcy
        for a, b in self.constraints['strong_dependency']:
            if a in flag_index and b in flag_index:
                if seq[flag_index[a]] == 0 and seq[flag_index[b]] == 1:
                    return True

        # Weak dependcy
        for a, b in self.constraints['weak_dependency']:
            if a in flag_index and b in flag_index:
                if seq[flag_index[a]] == 1 and seq[flag_index[b]] == 0:
                    return True

        # Synergistic relationship
        for a, b in self.constraints['synergistic_relationship']:
            if a in flag_index and b in flag_index:
                if seq[flag_index[a]] + seq[flag_index[b]] == 1:
                    return True

        return False
    
    def Obtain_initial_pro(self):
        cal = np.array(self.initial_seqs)
        x = cal.sum(axis=0)
        D = cal.shape[0]
        alpha = 1 
        beta = 1 
        alpha_post = alpha + x
        beta_post = beta + D - x
        posterior_means = alpha_post / (alpha_post + beta_post)
        expored_flag_pro = posterior_means[47:] #common flags are 47
        return expored_flag_pro

    def transProbtoflags(self, prob):
        """
        Enable flags as enable probabilities
        """
        enable_state = []
        for p in prob:
            if random.random() < p:
                enable_state.append(1)
            else:
                enable_state.append(0)
        return enable_state

    def run(self):
        ts = []   # time consumption
        res = []  # speedup for different flag combinations
        seqs = [] # different flag combinations
        ts.append(0)
        time_zero = time.time()
        Es = []
        common_flags =  [1] * 47 # -O1 flag number 
        current_pro = copy.deepcopy(self.initial_pro)
        min_thresh = 0.1
        last_log_time = 0.0 
        while ts[-1] < 5000:
            explored_flags = self.transProbtoflags(current_pro)
            seq = common_flags + explored_flags
            while(self.constraints_check(seq)):
                explored_flags = self.transProbtoflags(current_pro)
                seq = common_flags + explored_flags
            E = 0.0
            seqs.append(seq)
            temp = self.get_objective_score(seq, len(ts), SOURCE_PATH=self.SOURCE_PATH, GCC_PATH=self.GCC_PATH, INCLUDE_PATH=self.INCLUDE_PATH, EXEC_PARAM=self.EXEC_PARAM, LOG_FILE=self.LOG_FILE, all_flags=self.all_flags)
            res.append(temp)
            E = (temp - self.permin) / (self.permax - self.permin)
            Es.append(E)
            avg = sum(Es)/len(Es)
            diff = abs(E - avg)
            old_pro = current_pro.copy()
            if E > avg:
                for i in range(len(current_pro)):
                    if (explored_flags[i] == 1):
                        current_pro[i] = old_pro[i] + self.c * (1 - old_pro[i]) * diff
                    else:
                        current_pro[i] = old_pro[i] - self.c * old_pro[i] * diff
                    if current_pro[i] < min_thresh:
                        current_pro[i] = self.initial_pro[i]
            else:
                for i in range(len(current_pro)):
                    if(explored_flags[i] == 1):
                        current_pro[i] = old_pro[i] - self.c * old_pro[i] * diff
                    else:
                        current_pro[i] = old_pro[i] + self.c * (1 - old_pro[i]) * diff
                    if current_pro[i] < min_thresh:
                        current_pro[i] = self.initial_pro[i]
            time_now = time.time()
            ts.append(time_now-time_zero)
            best_result = max(res)
            best_seq = seqs[res.index(best_result)]
            if ts[-1] - last_log_time >= 20:
                ss = '{}: cur-best {}, cur-best-seq {}'.format(str(round(ts[-1])), str(best_result), str(best_seq))
                write_log(ss, self.LOG_FILE)
                last_log_time = ts[-1]
       


if __name__ == "__main__":
    LOG_DIR = 'log' + os.sep

    if not os.path.exists(LOG_DIR):
        os.system('mkdir '+LOG_DIR)

    parser = argparse.ArgumentParser(description="Preference-Driven Compiler Auto-Tuning")
    
    parser.add_argument("--log_file", type=str, required=True,
                        help="File to save log")
    
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to the source program for tuning")
    
    parser.add_argument("--gcc_path", type=str, required=True,
                        help="Path of compiler")
    
    parser.add_argument("--exec_param", type=str, default='',
                        help="Execution parameter for the output executable (can be empty)")
    
    parser.add_argument("--flag_path", type=str, required=True,
                        help="Tuning flags file")
    
    parser.add_argument("--sequences_path", type=str, required=True,
                        help="Initial tuning data")
    
    parser.add_argument("--constraints_path", type=str, required=True,
                        help="Three type constraints")
    
    parser.add_argument("--permax", type=float, required=True,
                        help="Best performance of generated sequences")
    
    parser.add_argument("--permin", type=float, required=True,
                        help="Lowest performance of generated sequences")
    
    args = parser.parse_args()
    if args.exec_param:
        EXEC_PARAM = args.exec_param
    else:
        EXEC_PARAM = '' 

    LOG_FILE = LOG_DIR +  args.log_file

    if args.flag_path:
        all_flags = read_flags_from_file(args.flag_path)
    else:
        all_flags = ['-O2']
        print('No flags')

    all_constraints = parse_constraints(args.constraints_path)
    good_sequence = extract_sequences_from_file(args.sequences_path)
    pdcat_params = {}
    pdcat_params['get_objective_score'] = get_objective_score
    pdcat_params['a'] = 1
    pdcat_params['b'] = 1
    pdcat_params['c'] = 0.5
    pdcat_params['source_path'] = args.source_path
    pdcat_params['gcc_path'] = args.gcc_path
    pdcat_params['include_path'] = ''
    pdcat_params['exec_param'] = args.exec_param
    LOG_DIR = 'log' + os.sep
    LOG_FILE = LOG_DIR +  args.log_file
    pdcat_params['log_file'] = LOG_FILE
    pdcat_params['flags'] = all_flags
    pdcat_params['seqs'] = good_sequence
    pdcat_params['constraints'] = all_constraints
    pdcat_params['permax'] = args.permax
    pdcat_params['permin'] = args.permin
    pd = PDCAT(**pdcat_params)
    pd.run()
    