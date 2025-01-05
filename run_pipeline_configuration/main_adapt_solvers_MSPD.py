import json
import os

from main_base_adaptation import run, preprocess
import matplotlib.pyplot as plt
import time
import copy


def plot(x, y):
    plt.plot(x, y, label="Cost-Accuracy")
    plt.xlabel("Cost")
    plt.ylabel("Accuracy")
    plt.show()


class PromptSwitcher:
    def __init__(self, llm_model_pool, prompt_pool, solver2res_dict):
        self.solver2WrgQuesSet_dict = None
        self.solver2CrtQuesSet_dict = None
        self.solver2crtRatio_dict = None
        self.solverQues2ans_dict = None
        self.wrong_questions = None
        self.left_candidates = None
        self.initial_solver = None
        self.candidates = []
        self.llm_model_pool = llm_model_pool
        self.prompt_pool = prompt_pool
        self.solver2res_dict = solver2res_dict

    def update(self, llm_model, num_sample):
        self.candidates = []
        for p in self.prompt_pool:
            solver = (llm_model, p)
            if solver in self.solver2res_dict.keys():
                self.candidates.append(solver)
        thresh = numSample2thresh_dict[num_sample]
        self.solverQues2ans_dict, self.solver2crtRatio_dict, self.solver2CrtQuesSet_dict, self.solver2WrgQuesSet_dict \
            = preprocess(self.candidates, self.solver2res_dict, num_sample=num_sample, thresh=thresh, num_run=num_run)
        print(self.solver2crtRatio_dict)
        # 1) Designate the solver with the highest accuracy as the first solver.
        # 2) Determine subsequent solvers: Construct a new dataset from the questions answered incorrectly
        # by previous solvers, and select the solver with the highest accuracy on this dataset
        # that can enhance performance when added.
        sorted_solver_crtRatio = sorted(self.solver2crtRatio_dict.items(), key=lambda x: x[1], reverse=True)
        print(sorted_solver_crtRatio)
        self.initial_solver = sorted_solver_crtRatio[0][0]
        if self.wrong_questions is not None:
            biggest_num = 0
            for solver in set(self.candidates):
                temp_num = len(self.wrong_questions & self.solver2CrtQuesSet_dict[solver])
                if temp_num > biggest_num:
                    biggest_num = temp_num
                    self.initial_solver = solver

        self.left_candidates = set(self.candidates)
        self.left_candidates.discard(self.initial_solver)
        self.wrong_questions = self.solver2WrgQuesSet_dict[self.initial_solver]
        return self.initial_solver

    def get_next_prompt(self):
        biggest_num = 0
        the_solver = None
        for solver in self.left_candidates:
            temp_num = len(self.wrong_questions & self.solver2CrtQuesSet_dict[solver])
            if temp_num > biggest_num:
                biggest_num = temp_num
                the_solver = solver
        if the_solver is None:
            return None
        else:
            self.left_candidates.discard(the_solver)
            return the_solver[1]


class Pipeline:
    def __init__(self, solver2res_dict, llm_model_pool, sample_size_pool, prompt_pool, max_prompt_last,
                 max_sample_size_last):
        self.llm_model_pool = llm_model_pool
        self.sample_size_pool = sample_size_pool
        self.prompt_pool = prompt_pool
        # self.decompose_granularity_pool = ["L1", "L2", "L3"]

        self.prompt_switcher = PromptSwitcher(self.llm_model_pool, self.prompt_pool, solver2res_dict)

        self.pipeline = []
        self.pipeline.append(dict())
        # self.num_solver = 1
        self.initialize_pipeline()

        self.prompt_num_last = 1
        self.sample_size_num_last = 1
        self.prompt_max_last = min(max_prompt_last, len(self.prompt_pool))
        self.sample_size_max_last = min(max_sample_size_last, len(self.sample_size_pool))

        self.terminate_flag = False
        self.llm_model_index = 0
        self.sample_size_index = 0

    def initialize_pipeline(self):
        self.pipeline[0]["llm_model"] = self.llm_model_pool[0]
        self.pipeline[0]["sample_size"] = self.sample_size_pool[0]
        self.pipeline[0]["prompt"] = self.prompt_switcher.update(self.llm_model_pool[0], self.sample_size_pool[0])[1]

    def switch_llm_model(self):
        self.llm_model_index = self.llm_model_index + 1
        current_model = self.llm_model_pool[self.llm_model_index]
        self.pipeline[-1]["llm_model"] = current_model

        # if self.llm_model_index == len(self.llm_model_pool) - 1:
        #     self.terminate_flag = True

        # a manual constraint: update sample size
        if current_model == "GPT4":
            self.pipeline[-1]["sample_size"] = 1
        else:
            self.pipeline[-1]["sample_size"] = self.sample_size_pool[0]
        self.sample_size_index = 0
        # self.sample_size_num_last = 1

        # update prompt
        initial_solver = self.prompt_switcher.update(self.llm_model_pool[self.llm_model_index],
                                                     self.sample_size_pool[self.sample_size_index])
        self.pipeline[-1]["prompt"] = initial_solver[1]
        # self.prompt_num_last = 1

    def switch_sample_size(self):
        self.sample_size_index = self.sample_size_index + 1
        self.pipeline[-1]["sample_size"] = self.sample_size_pool[self.sample_size_index]
        self.sample_size_num_last += 1

        # # update prompt
        initial_solver = self.prompt_switcher.update(self.llm_model_pool[self.llm_model_index],
                                                     self.sample_size_pool[self.sample_size_index])
        self.pipeline[-1]["prompt"] = initial_solver[1]
        # self.prompt_num_last = 1

    def switch_prompt(self):
        temp_prompt = self.prompt_switcher.get_next_prompt()
        if temp_prompt is None:
            return False
        else:
            self.pipeline[-1]["prompt"] = temp_prompt
            self.prompt_num_last += 1
            return True

    def adapt_solver(self):
        # The core idea is:
        # 1) Start with low-cost solvers;
        # 2) Tune the solvers to enhance performance as much as possible;
        # 3) Meanwhile, ensure the cost does not increase too rapidly.
        # Therefore, the typical tuning order is set as: prompt, sample size, llm model.
        temp_dict = copy.deepcopy(self.pipeline[-1])
        self.pipeline.append(temp_dict)
        adapt_flag = False
        if self.prompt_num_last < self.prompt_max_last and self.switch_prompt():
            adapt_flag = True
        elif self.sample_size_num_last < self.sample_size_max_last:
            self.switch_sample_size()
            adapt_flag = True
        else:
            if self.llm_model_index + 1 >= len(self.llm_model_pool):
                self.terminate_flag = True
            else:
                self.switch_llm_model()
                adapt_flag = True
        print("adapt_flag", adapt_flag)
        if not adapt_flag:
            self.pipeline.pop()


if __name__ == '__main__':
    data_set = "gsm8k"  # gsm8k, SVAMP, MultiArith, AddSub, SingleEq, AQuA, CommonsenseQA
    llm_models = ["GPT3.5", "GPT4"]  # ["GPT3.5", "GPT4"]
    sample_sizes = [3, 5, 10]  # [3, 5, 10]
    prompts = ["ZeroCoT", "PS", "CoT", "L2M", "L2M_L1", "L2M_L2", "L2M_L3"]
    prompt_last = 3
    sample_size_last = 3
    is_check = True
    remark = "%s_AS-MSPD" % is_check
    num_run = 10

    solver2resFile_dict = {"train": dict()}
    for train_or_test in ["train"]:
        with open("./solver2result/%s/%s/%s_%s_solver2resFile_dict.json" % (
                data_set, train_or_test, data_set, train_or_test)) as f:
            tmp_dict = json.load(f)
        for k, v in tmp_dict.items():
            solver2resFile_dict[train_or_test][tuple(k.split("#"))] = v
    numSample2thresh_dict = {1: 1.0, 3: 1.0, 5: 0.8, 10: 0.6}
    # solver = (LLM model, sample size/thresh, prompt, decomposition granularity)
    write_file = "train_results/%s_train_results_%s.csv" % (data_set, remark)
    while os.path.exists(write_file):
        write_file = write_file.replace(".csv", "(1).csv")
    writer = open(write_file, 'w')
    writer.write("Train_Acc,Train_Cost,Pipeline\n")
    for epoch in range(5):
        initial_time = time.time()
        best_acc = 0
        final_cost = None
        best_solver_list = None
        configurator = Pipeline(solver2resFile_dict["train"], llm_models, sample_sizes, prompts, prompt_last,
                                sample_size_last)
        x_data, y_data = [], []
        while True:
            print("#" * 40)
            print("pipeline: ", configurator.pipeline)
            temp_acc, num_call_list, temp_cost, _ = run(num_run, solver2resFile_dict["train"], configurator.pipeline,
                                                        numSample2thresh_dict,
                                                        select_strategy="last_solver")

            if is_check and len(configurator.pipeline) >= 2:
                temp_pipeline = copy.deepcopy(configurator.pipeline)
                temp_pipeline = temp_pipeline[:-2] + [temp_pipeline[-1]]
                temp_acc_2, num_call_list_2, temp_cost_2, _ = run(num_run, solver2resFile_dict["train"],
                                                                  temp_pipeline,
                                                                  numSample2thresh_dict,
                                                                  select_strategy="last_solver")
                if temp_acc_2 >= temp_acc and temp_cost_2 <= temp_cost:
                    configurator.pipeline.pop(-2)
                    print("Condense pipeline to: ", configurator.pipeline)
                    temp_acc = temp_acc_2
                    num_call_list = num_call_list_2
                    temp_cost = temp_cost_2

            # temp_cost = cal_cost_per_pipeline(num_call_list, configurator.pipeline)
            print("temp_cost=", temp_cost)
            print("temp_acc=", temp_acc)
            if temp_acc - best_acc >= 0.002:
                best_acc = temp_acc
                final_cost = temp_cost
                # best_solver_list = configurator.pipeline
                last_solver = (configurator.pipeline[-1]["llm_model"], configurator.pipeline[-1]["prompt"])
                configurator.prompt_switcher.wrong_questions = (configurator.prompt_switcher.wrong_questions
                                                                - configurator.prompt_switcher.solver2CrtQuesSet_dict[
                                                                    last_solver])

                x_data.append(temp_cost)
                y_data.append(temp_acc)
            else:
                configurator.pipeline.pop()

            configurator.adapt_solver()
            if configurator.terminate_flag:
                break
        best_solver_list = configurator.pipeline
        print("======== final train result ========")
        print("best_acc=", best_acc)
        print("final_cost=", final_cost)
        print("best_solver_list: ", best_solver_list)
        print("num_call_list: ", num_call_list)
        print("x_data", x_data)
        print("y_data", y_data)
        # plot(x_data, y_data)

        writer.write("%.4f,%.4f,%s\n" % (best_acc, final_cost, best_solver_list))

        print("spend %d seconds" % (time.time() - initial_time))
