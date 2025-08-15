import time
# Define constant
FLOAT_MAX = float('inf')

class FlagInfo:
    def __init__(self, name, configs):
        self.name = name
        self.configs = configs

class Evaluator:
    def __init__(self, path):
        self.path = path

    def evaluate(self):
        assert 0, "Undefined"

    def evaluate_default(self):
        assert 0, "Undefined"


def write_log(ss, file):
    """ Write to log """
    with open(file, 'a') as log:
        log.write(ss + '\n')

class Tuner:
    def __init__(self, search_space, evaluator, log_file):
        self.search_space = search_space
        self.evaluator = evaluator
        self.LOG_FILE = log_file
        self.default_perf = evaluator.evaluate_default()
        self.visited = set()

        print(f"default_perf : {self.default_perf:.3f}")

    
    def generate_candidates(self, batch_size=1):
        assert 0, "Undefined"
    
    def evaluate_candidates(self, candidates):
        assert 0, "Undefined"

    def reflect_feedback(perfs):
        assert 0, "Undefined"

    def tune(self, budget, batch_size=1):
        best_opt_setting, best_perf = None, FLOAT_MAX
        ts = [0] 
        time_zero = time.time()
        while ts[-1] < budget:
            self.default_perf = self.evaluator.evaluate_default()
            candidates = self.generate_candidates(batch_size=batch_size)
            perfs = self.evaluate_candidates(candidates)
            for opt_setting, perf in zip(candidates, perfs):
                if perf < best_perf:
                    best_perf = perf
                    best_opt_setting = opt_setting
            
            print(f" current trial: {perf:.3f}s, best performance so far: {best_perf:.3f}s")
            
            time_now = time.time()
            ts.append(time_now - time_zero)
            ss = '{}: cur-best {}, cur-best-seq {}'.format(str(round(ts[-1])), str(best_perf), str(best_opt_setting))
            write_log(ss, self.LOG_FILE)
            self.reflect_feedback(perfs)
        return best_opt_setting, best_perf