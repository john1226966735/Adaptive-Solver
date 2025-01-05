import copy
import json
from collections import Counter
import matplotlib.pyplot as plt
import random
import statistics
from train_utils import *

NUM_CORRECT = None


class BaseEnsemble:
    def __init__(self, solver2resFile_dict, solver_list, numSample2thresh_dict, select_strategy):
        self.solver2resFile_dict = solver2resFile_dict
        self.solver_list = solver_list
        self.select_strategy = select_strategy
        assert self.select_strategy in ["highest_consistency", "last_solver"]

        self.thresh_list = [numSample2thresh_dict[e["sample_size"]] for e in solver_list]

        self.num_solver = len(self.solver_list)
        self.ques_info_list = []
        self.data_list = []
        self.solverQues2answers_dict = dict()
        self.ques2solverIds_dict = dict()

        self.num_problem = 0
        self.numCrt_list = []
        self.numCall_lists = []
        self.crtRatio_list = []
        self.ApiCost_list = []

        self.AvgCrtRatio = None
        self.AvgCost = []
        self.CrtRatioStd = None
        self.AvgNumCallList = [0] * self.num_solver

        self._prepare_data()

    def _prepare_data(self):
        FLAG = True
        for solver in self.solver_list:
            res_file = self.solver2resFile_dict[(solver["llm_model"], solver["prompt"])]
            tmp_results = read_result(res_file)
            if FLAG:
                FLAG = False
                for res in tmp_results:
                    self.num_problem += 1
                    ques = res["question"]
                    self.ques_info_list.append({"question": ques, "true_answer": res["answer"]})
            for res in tmp_results:
                ques = res["question"]
                if "temp_CoTs" not in res.keys():
                    temp_CoTs = [[res["chain-of-thought"]] * len(res["temp_answers"][0])]
                else:
                    temp_CoTs = res["temp_CoTs"]
                self.solverQues2answers_dict[((solver["llm_model"], solver["prompt"]), ques)] = {
                    "temp_answers": res["temp_answers"][0],
                    "temp_CoTs": temp_CoTs[0]
                }

    def update_data(self):
        self.data_list = []
        for data in self.ques_info_list:
            ques = data["question"]
            true_answer = data["true_answer"]
            answer_batches, cot_batches, model_list, prompt_list = [], [], [], []
            for solver in self.solver_list:
                num_sample = solver["sample_size"]
                temp_dict = self.solverQues2answers_dict[((solver["llm_model"], solver["prompt"]), ques)]
                index_batch = random.choices(range(len(temp_dict["temp_answers"])), k=num_sample)
                answer_batch = [temp_dict["temp_answers"][i] for i in index_batch]
                answer_batches.append(answer_batch)
                cot_batch = [temp_dict["temp_CoTs"][i] for i in index_batch]
                cot_batches.append(cot_batch)
                model_list.append({"GPT3.5": "gpt-3.5-turbo-0301", "GPT4": "gpt-4-0613"}[solver["llm_model"]])
                prompt_list.append(solver["prompt"])

            self.data_list.append({
                "answer_batches": answer_batches,
                "true_answer": true_answer,
                "cot_batches": cot_batches,
                "problem": ques,
                "model_list": model_list,
                "prompt_list": prompt_list
            })

    def adaptive_solver(self, data):
        answer_batches = data["answer_batches"]
        cot_batches = data["cot_batches"]
        problem = data["problem"]
        model_list = data["model_list"]
        prompt_list = data["prompt_list"]

        final_answer = None
        final_occur_count = 0
        num_call_list = [0] * len(answer_batches)
        solver_id = 0
        api_cost = 0
        for i, thresh in enumerate(self.thresh_list):
            answer_batch = answer_batches[i]
            # answer_batch += answer_batches[i]
            answer, occur_count = Counter(answer_batch).most_common(1)[0]
            num_call_list[i] = 1

            inputs, outputs = construct_input_output(problem, cot_batches[i], answer_batch, prompt_list[i])
            api_cost += calculate_cost_with_input_output(inputs, outputs, model_list[i])

            # way 1. global consideration
            if self.select_strategy == "highest_consistency":
                if occur_count >= final_occur_count:
                    final_occur_count = occur_count
                    final_answer = answer
                    solver_id = i
            else:
                # way 2. local consideration
                final_answer = answer

            if occur_count / len(answer_batch) >= thresh:
                break

        return final_answer, num_call_list, solver_id, api_cost

    def process(self):
        num_correct = 0
        total_cost = 0
        num_call_list = [0] * self.num_solver
        for q_id, data in enumerate(self.data_list):
            true_answer = data["true_answer"]

            final_answer, tmp_num_call_list, solver_id, api_cost = self.adaptive_solver(data)
            total_cost += api_cost
            # solver_ids.append(solver_id)
            if q_id not in self.ques2solverIds_dict.items():
                self.ques2solverIds_dict[q_id] = []
            self.ques2solverIds_dict[q_id].append(solver_id)
            for i in range(len(num_call_list)):
                num_call_list[i] += tmp_num_call_list[i]

            res = evaluate_answer(final_answer, true_answer)
            if res:
                num_correct += 1

        self.numCrt_list.append(num_correct)
        self.crtRatio_list.append(num_correct / self.num_problem)
        self.ApiCost_list.append(total_cost)
        self.numCall_lists.append(num_call_list)
        self.AvgNumCallList = [a + b for a, b in zip(self.AvgNumCallList, num_call_list)]

    def post_process(self):
        print("num_problem: %d" % self.num_problem)
        self.AvgCrtRatio = (sum(self.numCrt_list) / len(self.numCrt_list)) / self.num_problem
        # print("average correct ratio: ", self.AvgCrtRatio)
        self.AvgCost = sum(self.ApiCost_list) / len(self.ApiCost_list)
        self.CrtRatioStd = statistics.stdev(self.crtRatio_list)
        # print("std of correct ratio: ", self.CrtRatioStd)
        self.AvgNumCallList = [e / len(self.numCall_lists) for e in self.AvgNumCallList]
        # print("average num call list: ", self.AvgNumCallList)


def run(num_run, solver2resFile_dict, pipeline, numSample2thresh_dict, select_strategy):
    copy_pipeline = copy.deepcopy(pipeline)
    ensemble_solver = BaseEnsemble(solver2resFile_dict, copy_pipeline, numSample2thresh_dict, select_strategy)
    for i in range(num_run):
        ensemble_solver.update_data()
        ensemble_solver.process()
    ensemble_solver.post_process()
    return ensemble_solver.AvgCrtRatio, ensemble_solver.AvgNumCallList, ensemble_solver.AvgCost, ensemble_solver.CrtRatioStd


def read_result(file):
    try:
        with open(file) as f:
            data_list = json.load(f)
    except Exception as e:
        # print("Error: ", e)
        with open(file, encoding="utf-8") as f:
            data = f.read()
        data = data.replace("}{", "},{")

        data_list = json.loads(f'[{data}]')
    return data_list


def evaluate_answer(pred_answer, true_answer):
    result = False
    if pred_answer is not None:
        if isinstance(pred_answer, float):
            if abs(pred_answer - true_answer) < 1e-3:
                result = True
        else:
            if pred_answer == true_answer:
                result = True
    return result


def preprocess(candidate_solvers, solver2resFile_dict, num_sample, thresh, num_run):
    solverQues2ans_dict = dict()
    solver2CrtQuesSet_dict, solver2WrgQuesSet_dict = dict(), dict()
    solver2crtRatio_dict = dict()
    for solver in candidate_solvers:
        num_problem = 0
        num_crt = 0
        res_file = solver2resFile_dict[solver]
        tmp_results = read_result(res_file)
        solver2CrtQuesSet_dict[solver] = set()
        solver2WrgQuesSet_dict[solver] = set()
        for res in tmp_results:
            ques = res["question"]
            num_problem += 1
            crt_count = 0
            for i in range(num_run):
                temp_answers = random.choices(res["temp_answers"][0], k=num_sample)
                answer, occur_count = Counter(temp_answers).most_common(1)[0]
                ans = evaluate_answer(answer, res["answer"])
                if ans and occur_count / len(temp_answers) >= thresh:
                    crt_count += 1
            if crt_count / num_run >= 0.5:
                num_crt += 1
                solver2CrtQuesSet_dict[solver].add(ques)
                solverQues2ans_dict[(solver, ques)] = True
            else:
                solver2WrgQuesSet_dict[solver].add(ques)
                solverQues2ans_dict[(solver, ques)] = False
        solver2crtRatio_dict[solver] = num_crt / num_problem
    return solverQues2ans_dict, solver2crtRatio_dict, solver2CrtQuesSet_dict, solver2WrgQuesSet_dict
