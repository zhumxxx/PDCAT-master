import os, sys, random, time, copy, subprocess, itertools, math, argparse
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm

def write_log(ss, file):
    with open(file, 'a') as log:
        log.write(ss + '\n')
    
def execute_terminal_command(command):
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
    """
    Obtain the speedup
    """
    opt = ''
    for i in range(len(independent)):
        if independent[i]:
            opt = opt + all_flags[i] + ' '
        else:
            negated_flag_name = all_flags[i].replace("-f", "-fno-", 1)
            opt = opt + negated_flag_name + ' '
    
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
    time_c = time_end - time_start   #time opt

    time_o3 = time.time()
    command = f"{GCC_PATH} -O3 -c {INCLUDE_PATH} {SOURCE_PATH}/*.c"
    execute_terminal_command(command)
    command2 = f"{GCC_PATH} -o a.out -O3 *.o -lm"
    execute_terminal_command(command2)
    time_o3 = time.time()
    command3 = "./a.out {EXEC_PARAM}"
    execute_terminal_command(command3)
    time_o3_end = time.time()  
    cmd4 = 'rm -rf *.o *.I *.s a.out'
    execute_terminal_command(cmd4)
    time_o3_c = time_o3_end - time_o3   #time o3

    op_str = "iteration:{} speedup:{}".format(str(k_iter), str(time_o3_c /time_c))
    write_log(op_str, LOG_FILE)
    return (time_o3_c /time_c)


      
time_tem = []
class CFSCA:
    def __init__(self, dim, get_objective_score, seed, related_flags, source_path, gcc_path, include_path, exec_param, log_file, flags):
        """
        :param dim: number of compiler flags
        :param get_objective_score: obtain true speedup
        :param random: random parameter
        :param related_flags: program related flags for the target program
        :param source_path: program's path
        :param gcc_path: gcc's path
        :param include_path: header file for program
        :param exec_param: exection paramter
        :param log_file: record results
        :param flags: all flags
        """
        self.dim = dim
        self.get_objective_score = get_objective_score
        self.seed = seed
        self.related = related_flags
        self.critical = []
        self.global_best_per = 0.0
        self.global_best_seq = []
        self.random = random
        self.SOURCE_PATH = source_path
        self.GCC_PATH = gcc_path
        self.INCLUDE_PATH = include_path
        self.EXEC_PARAM = exec_param
        self.LOG_FILE = log_file
        self.all_flags = flags
    def generate_random_conf(self, x):
        """
        :param x: random generate number
        :return: the binary sequence for x
        """
        comb = bin(x).replace('0b', '')
        comb = '0' * (self.dim - len(comb)) + comb
        conf = []
        for k, s in enumerate(comb):
            if s == '1':
                conf.append(1)
            else:
                conf.append(0)
        return conf

    def get_ei(self, preds, eta):
        """
        :param preds: sequences' speedup for EI
        :param eta: global best speedup
        :return: the EI of a sequence
        """
        preds = np.array(preds).transpose(1, 0)
        m = np.mean(preds, axis=1)
        s = np.std(preds, axis=1)

        def calculate_f(eta, m, s):
            z = (eta - m) / s
            return (eta - m) * norm.cdf(z) + s * norm.pdf(z)

        if np.any(s == 0.0):
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f(eta, m, s)
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f(eta, m, s)
        return f

    def get_ei_predict(self, model, now_best, wait_for_train):
        """
        :param model: RandomForest Model
        :param now_best: global best speedup
        :param wait_for_train: sequences Set
        :return: the sequences' EI
        """
        preds = []
        estimators = model.estimators_
        for e in estimators:
            preds.append(e.predict(np.array(wait_for_train)))
        acq_val_incumbent = self.get_ei(preds, now_best)
        return [[i, a] for a, i in zip(acq_val_incumbent, wait_for_train)]
    
    def runtime_predict(self, model, wait_for_train):
        """
        :param model: RandomForest Model
        :param wait_for_train: sequences set
        :return: the speedup of sequences set
        """
        estimators = model.estimators_
        sum_of_predictions = np.zeros(len(wait_for_train))
        for tree in estimators:
            predictions = tree.predict(wait_for_train)
            sum_of_predictions += predictions
        a = []
        average_prediction = sum_of_predictions / len(estimators)
        for i in range(len(wait_for_train)):
            x = [wait_for_train[i], average_prediction[i]]
            a.append(x)
        return a
    
    def getPrecision(self, model, seq):
        """
        :param model: RandomForest Model
        :param seq: sequence for prediction
        :return: the precision of a sequence and true speedup
        """
        true_running = self.get_objective_score(seq, k_iter=100086, SOURCE_PATH=self.SOURCE_PATH, GCC_PATH=self.GCC_PATH, INCLUDE_PATH=self.INCLUDE_PATH, EXEC_PARAM=self.EXEC_PARAM, LOG_FILE=self.LOG_FILE, all_flags=self.all_flags)
        estimators = model.estimators_
        res = []
        for e in estimators:
            tmp = e.predict(np.array(seq).reshape(1, -1))
            res.append(tmp)
        acc_predict = np.mean(res)
        return abs(true_running - acc_predict) / true_running, true_running
    
    def selectByDistribution(self, merged_predicted_objectives):
        """
        :param merged_predicted_objectives: the sequences' EI and the sequences
        :return: the selected sequence
        """
        # sequences = [seq for seq, per in merged_predicted_objectives]
        diffs = [abs(perf - merged_predicted_objectives[0][1]) for seq, perf in merged_predicted_objectives]
        diffs_sum = sum(diffs)
        probabilities = [diff / diffs_sum for diff in diffs]
        index = list(range(len(diffs)))
        idx = np.random.choice(index, p=probabilities)
        return idx
    
    def build_RF_by_CompTuner(self):
        """
        :return: model, inital_indep, inital_dep
        """
        inital_indep = []
        time_begin = time.time()
        # randomly sample initial training instances
        while len(inital_indep) < 2:
            x = random.randint(0, 2 ** self.dim - 1)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance not in inital_indep:
                inital_indep.append(initial_training_instance)
        inital_dep = [self.get_objective_score(indep, k_iter=0, SOURCE_PATH=self.SOURCE_PATH, GCC_PATH=self.GCC_PATH, INCLUDE_PATH=self.INCLUDE_PATH, EXEC_PARAM=self.EXEC_PARAM, LOG_FILE=self.LOG_FILE, all_flags=self.all_flags) for indep in inital_indep]
        
        all_acc = []
        time_tem.append(time.time() - time_begin)
        model = RandomForestRegressor(random_state=self.seed)
        model.fit(np.array(inital_indep), np.array(inital_dep))
        rec_size = 2
        
        while rec_size < 11:
            model = RandomForestRegressor(random_state=self.seed)
            model.fit(np.array(inital_indep), np.array(inital_dep))
            global_best = max(inital_dep)
            estimators = model.estimators_
            if all_acc:
                all_acc = sorted(all_acc)
            neighbors = []
            for i in range(30000):
                x = random.randint(0, 2 ** self.dim - 1)
                x = self.generate_random_conf(x)
                if x not in neighbors:
                    neighbors.append(x)
            pred = []
            for e in estimators:
                pred.append(e.predict(np.array(neighbors)))
            acq_val_incumbent = self.get_ei(pred, global_best)
            ei_for_current = [[i, a] for a, i in zip(acq_val_incumbent, neighbors)]
            merged_predicted_objectives = sorted(ei_for_current, key=lambda x: x[1], reverse=True)
            acc = 0
            flag = False
            for x in merged_predicted_objectives:
                if flag:
                    break
                if x[0] not in inital_indep:
                    inital_indep.append(x[0])
                    acc, lable = self.getPrecision(model, x[0])
                    inital_dep.append(lable)
                    all_acc.append(acc)
                    flag = True
            rec_size += 1

            if acc > 0.05:
                indx = self.selectByDistribution(merged_predicted_objectives)
                while merged_predicted_objectives[int(indx)][0] in inital_indep:
                    indx = self.selectByDistribution(merged_predicted_objectives)
                inital_indep.append(merged_predicted_objectives[int(indx)][0])
                acc, label = self.getPrecision(model, merged_predicted_objectives[int(indx)][0])
                inital_dep.append(label)
                all_acc.append(acc)
                rec_size += 1
            time_tem.append(time.time() - time_begin)
            ss = '{}: best_seq {}, best_per {}'.format(str(round(time_tem[-1])), str(max(inital_dep)), str(inital_indep[inital_dep.index(max(inital_dep))]))
            write_log(ss, self.LOG_FILE)
        self.global_best_per = max(inital_dep)
        self.global_best_seq = inital_indep[inital_dep.index(max(inital_dep))]
        return model, inital_indep, inital_dep
    
    def get_critical_flags(self, model, inital_indep, inital_dep):
        """
        :param: model: RandomForest Model
        :param: inital_indep: selected sequences
        :param: inital_dep: selected sequences' performance
        :return: critical_flags_idx, new_model
        """
        candidate_seq = []
        candidate_per = []
        inital_indep_temp = copy.deepcopy(inital_indep)
        inital_dep_temp = copy.deepcopy(inital_dep)
        while len(candidate_seq) < 30000:
            x = random.randint(0, 2 ** self.dim - 1)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance not in candidate_seq:
                candidate_seq.append(initial_training_instance)
        begin = time.time()
        all_per = self.runtime_predict(model,candidate_seq)
        candidate_per = [all[1] for all in all_per]
        pos_seq = [0] * len(self.related)    
        now_best = max(candidate_per)
        now_best_seq = candidate_seq[candidate_per.index(now_best)]
        now_best = self.get_objective_score(now_best_seq, k_iter=100086, SOURCE_PATH=self.SOURCE_PATH, GCC_PATH=self.GCC_PATH, INCLUDE_PATH=self.INCLUDE_PATH, EXEC_PARAM=self.EXEC_PARAM, LOG_FILE=self.LOG_FILE, all_flags=self.all_flags)
        inital_indep_temp.append(now_best_seq)
        inital_dep_temp.append(now_best)
        model_new = RandomForestRegressor(random_state=self.seed)
        model_new.fit(np.array(inital_indep_temp), np.array(inital_dep_temp))
        before_time = time_tem[-1]
        time_tem.append(time.time() - begin + before_time)
        if self.global_best_per < now_best:
            self.global_best_per = now_best
            self.global_best_seq = now_best_seq
        ss = '{}: best_seq {}, best_per {}'.format(str(round(time_tem[-1])), str(self.global_best_per), str(self.global_best_seq))
        write_log(ss, self.LOG_FILE)

        for idx in range(len(self.related)):
            new_candidate = []
            for j in range(len(candidate_seq)):
                seq = copy.deepcopy(candidate_seq[j])
                seq[self.related[idx]] = 1 - seq[self.related[idx]]
                new_candidate.append(seq)
            new_per = [all[1] for all in self.runtime_predict(model_new,new_candidate)]
            new_seq = [all[0] for all in self.runtime_predict(model_new,new_candidate)]
            new_best_seq = new_seq[new_per.index(max(new_per))]
            new_best = self.get_objective_score(new_best_seq, k_iter=100086, SOURCE_PATH=self.SOURCE_PATH, GCC_PATH=self.GCC_PATH, INCLUDE_PATH=self.INCLUDE_PATH, EXEC_PARAM=self.EXEC_PARAM, LOG_FILE=self.LOG_FILE, all_flags=self.all_flags)
            if new_best > self.global_best_per:
                self.global_best_per = new_best
                self.global_best_seq = new_best_seq

            for l in range(len(new_candidate)):
                if (candidate_per[l] > new_per[l] and new_candidate[l][self.related[idx]] == 1) or (candidate_per[l] < new_per[l] and new_candidate[l][self.related[idx]] == 0):
                    pos_seq[idx] -= 1
                else:
                    pos_seq[idx] += 1
            inital_indep_temp.append(new_best_seq)
            inital_dep_temp.append(new_best)
            model_new = RandomForestRegressor(random_state=self.seed)
            model_new.fit(np.array(inital_indep_temp), np.array(inital_dep_temp))
            time_tem.append(time.time() - begin + before_time)
            
            ss = '{}: best_seq {}, best_per {}'.format(str(round(time_tem[-1])), str(self.global_best_per), str(self.global_best_seq))
            write_log(ss, self.LOG_FILE)

        sort_pos = sorted(enumerate(pos_seq), key=lambda x: x[1], reverse=True)
        critical_flag_idx = []
        for i in range(10):
            critical_flag_idx.append(self.related[sort_pos[i][0]])
        return critical_flag_idx, model_new
    
    def searchBycritical(self, critical_flag):
        """
        :param: critical_flag: idx of critical flag
        :return: the bias generation sequences
        """
        permutations = list(itertools.product([0, 1], repeat=10))
        seqs = []
        while len(seqs) < 1024 * 40:
            x = random.randint(0, 2 ** self.dim - 1)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance not in seqs:
                seqs.append(initial_training_instance)
        for i in range(len(permutations)):
            for idx in range(len(critical_flag)):
                for offset in range(0, 1024 * 40, 1024):
                    seqs[i + offset][critical_flag[idx]] = permutations[i][idx]
        return seqs
    
    def run(self):
        begin_all = time.time()
        """
        build model and get data set
        """
        model, inital_indep, inital_dep = self.build_RF_by_CompTuner()
        critical_flag, model_new = self.get_critical_flags(model, inital_indep, inital_dep)
        all_before = time_tem[-1]
        begin_all = time.time()
        while (time_tem[-1] < 5000):
            seq = self.searchBycritical(critical_flag)
            result = self.runtime_predict(model_new, seq)
            sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
            true_reslut = self.get_objective_score(sorted_result[0][0], k_iter=0, SOURCE_PATH=self.SOURCE_PATH, GCC_PATH=self.GCC_PATH, INCLUDE_PATH=self.INCLUDE_PATH, EXEC_PARAM=self.EXEC_PARAM, LOG_FILE=self.LOG_FILE, all_flags=self.all_flags)
            if true_reslut > self.global_best_per:
                self.global_best_per = true_reslut
                self.global_best_seq = sorted_result[0][0]
            time_tem.append(time.time() - begin_all + all_before)
            ss = '{}: cur-best {}, cur-best-seq {}'.format(str(round(time_tem[-1])), str(self.global_best_per), str(self.global_best_seq))
            write_log(ss, self.LOG_FILE)
        best_result = self.get_objective_score(self.global_best_seq, k_iter=0, SOURCE_PATH=self.SOURCE_PATH, GCC_PATH=self.GCC_PATH, INCLUDE_PATH=self.INCLUDE_PATH, EXEC_PARAM=self.EXEC_PARAM, LOG_FILE=self.LOG_FILE, all_flags=self.all_flags)
        ss = '{}: cur-best {}, cur-best-seq {}'.format(str(round(time_tem[-1])), str(best_result), str(self.global_best_seq))
    
def read_flags_from_file(file_path):
    """
    obtain all flags
    """ 
    with open(file_path) as f:
        flags = [line.strip() for line in f if line.strip()]
    return flags

if __name__ == '__main__':
    LOG_DIR = 'log' + os.sep
    if not os.path.exists(LOG_DIR):
        os.system('mkdir '+LOG_DIR)
    parser = argparse.ArgumentParser(description="CompTuner")
    
    parser.add_argument("--log_file", type=str, required=True,
                        help="File to save log")
    
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to the source program for tuning")
    
    parser.add_argument("--gcc_path", type=str, required=True,
                        help="Path of compiler")
    
    parser.add_argument("--exec_param", type=str, default=None,
                        help="Execution parameter for the output executable (can be empty)")
    
    parser.add_argument("--related_flags", type=str, default=None,
                        help="Related flags for tuning (can be a comma-separated string).")
    
    parser.add_argument("--flag_path", type=str, required=True,
                        help="Tuning flags file")
    
    args = parser.parse_args()
    if args.exec_param:
        EXEC_PARAM = args.exec_param
    else:
        EXEC_PARAM = '' 

    LOG_FILE = LOG_DIR +  args.log_file

    if args.related_flags is not None:
        related_flags_list = [int(x) for x in args.related_flags.split(',')]
    else:
        related_flags_list = []

    if args.flag_path:
        all_flags = read_flags_from_file(args.flag_path)
    else:
        all_flags = ['-O2']
        print('No flags')
    cfsca_params = {}
    cfsca_params['dim'] = len(all_flags)
    cfsca_params['get_objective_score'] = get_objective_score
    cfsca_params['seed'] = 456
    cfsca_params['related_flags'] = related_flags_list
    cfsca_params['source_path'] = args.source_path
    cfsca_params['gcc_path'] = args.gcc_path
    cfsca_params['include_path'] = ''
    cfsca_params['exec_param'] = args.exec_param
    cfsca_params['log_file'] = LOG_FILE
    cfsca_params['flags'] = all_flags
    cfsca = CFSCA(**cfsca_params)
    cfsca.run()