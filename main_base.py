import json
from prompt import get_prompts_dict
from loader import load_data
from utils import write_process, write_json, write_result, find_answer
from prompt_controller import PromptController
import random
from collections import Counter
import copy
import time


class QuestionNode:
    def __init__(self, question, self_id):
        self.question = question
        self.final_answer = None
        self.final_COT = None
        self.temp_token_dict = dict()

        self.temp_COTs = []
        self.temp_answers = []

        self.self_id = self_id


class BaseSolver:
    def __init__(self, args, Decoder_Error_File, Predict_File, Process_File, select_strategy,
                 model_list=None, num_sample_list=None, thresh_list=None):
        self.params = args
        self.decoder_error_file = Decoder_Error_File
        self.predict_file = Predict_File
        self.process_file = Process_File
        self.model_list = model_list
        self.num_sample_list = num_sample_list
        self.thresh_list = thresh_list
        self.select_strategy = select_strategy
        assert self.select_strategy in ["highest_consistency", "last_solver"]

        self.problem_infos, self.answers, self.ids = load_data(args)
        self.total_question = len(self.problem_infos)
        print("number of question=%d" % self.total_question)

        self.prompts_dict = get_prompts_dict()
        print(self.prompts_dict.keys())
        self.prompt_controller = PromptController(Process_File, Decoder_Error_File)

        self.current_problem = None

        # for adaptive solving
        self.default_thresh = args.SC_THRESH
        self.num_call_list = [0] * len(args.prompt_id["solve_main"])
        self.node_list = []
        self.ques2id_dict = dict()
        self.num_node = 0

        # for calculating ACC
        self.num_correct = 0
        self.num_problem = 0

        self.start_time = time.time()
        self.elapsed_time = 0
        self.pred_dict = dict()

    def init_one_problem(self):
        self.node_list = []
        self.ques2id_dict = dict()
        self.num_node = 0
        self.prompt_controller.num_call = 0

    def build_update_node(self, ques):
        if ques not in self.ques2id_dict.keys():
            self.ques2id_dict[ques] = self.num_node
            self.node_list.append(QuestionNode(ques, self.num_node))
            self.num_node = self.num_node + 1

    def solve_main(self, param_dict, step):
        problem_info = {"question": self.current_problem}
        _, pred_dict = self.prompt_controller.solve_one_problem(self.prompts_dict["solve_main"][step], problem_info,
                                                                param_dict)
        return pred_dict

    @staticmethod
    def sum_two_dicts(dict1, dict2):
        for k, v in dict2.items():
            dict1[k] = dict1.get(k, 0) + v
        return dict1

    def run_cell(self, _param_dict):  # solve one problem（can solve a problem for multiple rounds）
        param_dict = copy.deepcopy(_param_dict)
        # most_pred, most_answer = None, None
        final_occur_count = 0
        final_answer, final_pred = None, None
        for step in range(len(self.prompts_dict["solve_main"])):
            print("---------Round %d of solving-----------" % (step + 1))
            if self.model_list is not None:
                param_dict.update({'model': self.model_list[step]})
                if step == len(self.prompts_dict["solve_main"]) - 1:
                    param_dict['SC'] = False
            if self.num_sample_list is not None:
                param_dict.update({'num_sample': self.num_sample_list[step]})
            pred_dict = self.solve_main(param_dict, step)

            self.node_list[0].temp_token_dict = self.sum_two_dicts(self.node_list[0].temp_token_dict, pred_dict["token_dict"])
            self.pred_dict = pred_dict
            # print("pred_dict: ", pred_dict)
            self.num_call_list[step] = self.num_call_list[step] + 1
            if self.thresh_list is not None:
                thresh = self.thresh_list[step]
            else:
                thresh = self.default_thresh
            if self.model_list is not None:
                is_finish, most_pred, most_answer, occur_count, pred_dict1 = self.solution_evaluation(pred_dict['pred'],
                                                                                                      thresh,
                                                                                                      self.model_list[
                                                                                                          step])
            else:
                is_finish, most_pred, most_answer, occur_count, pred_dict1 = self.solution_evaluation(pred_dict['pred'],
                                                                                                      thresh)
            if pred_dict1 is not None:
                self.node_list[0].temp_token_dict = self.sum_two_dicts(self.node_list[0].temp_token_dict,
                                                                       pred_dict1["token_dict"])

            # global consideration
            if self.select_strategy == "highest_consistency":
                if occur_count >= final_occur_count:
                    final_occur_count = occur_count
                    final_answer = most_answer
                    final_pred = most_pred
            else:
                # local consideration
                final_answer = most_answer

            if is_finish:
                print("----------no need of successive solving--------------")
                break
        self.node_list[0].final_COT = final_pred
        self.node_list[0].final_answer = final_answer
        print("num_call_list: ", self.num_call_list)
        return

    def solution_evaluation(self, pred_list, thresh, model=None):
        answer2preds_dict, answer_list, pred_dict = find_answer(self.current_problem, pred_list, self.params,
                                                                self.process_file, model)
        print("answer_list: ", answer_list)
        print("current thresh: ", thresh)
        collection_words = Counter(answer_list)
        most_answer, occur_count = collection_words.most_common(1)[0]
        most_pred = random.choice(answer2preds_dict[most_answer])

        self.node_list[0].temp_COTs.append(pred_list)
        self.node_list[0].temp_answers.append(answer_list)

        if occur_count / len(pred_list) >= thresh:
            return True, most_pred, most_answer, occur_count, pred_dict
        else:
            return False, most_pred, most_answer, occur_count, pred_dict

    def run(self, solve_param_dict):
        for idx, proInfo in enumerate(self.problem_infos):
            time_start = time.time()
            print("------------%d-th question------------" % (idx + 1))
            write_process("\n------------%d-th question------------\n" % (idx + 1), self.process_file)

            self.current_problem = proInfo['question']

            self.init_one_problem()

            self.build_update_node(self.current_problem)

            # self.run_cell(solve_param_dict)
            # self.post_process_problem(idx, time_start)
            try:
                self.run_cell(solve_param_dict)
                self.post_process_problem(idx, time_start)
            except:
                decode_error_data = {'question': self.current_problem}
                write_json(decode_error_data, self.decoder_error_file)
                continue

        elapsed_time = time.time() - self.start_time
        write_process("spend %.3f seconds in total, spend %.3f second per problem\n" % (
            elapsed_time, elapsed_time / self.num_problem), self.process_file)
        write_process("token: %s, price: %s" % (self.pred_dict["token"], self.pred_dict["price"]), self.process_file)

    def post_process_problem(self, idx, t1):
        duration = "%.4f" % (time.time() - t1)
        pred_answer = self.node_list[0].final_answer
        ans = write_result(self.current_problem, self.answers[idx], self.ids[idx], self.node_list[0].final_COT,
                           pred_answer, self.params, self.predict_file,
                           self.node_list[0].temp_answers, self.node_list[0].temp_COTs,
                           self.node_list[0].temp_token_dict, duration)
        if ans:
            self.num_correct += 1

        self.num_problem += 1
        print("pred_answer: %s, true_answer: %s, correctness: %s" % (pred_answer, self.answers[idx], ans))
        print("total question: %d, num correct: %d, num question: %d, correct ratio: %.4f, duration: %s" % (
            self.total_question, self.num_correct, self.num_problem, self.num_correct / self.num_problem, duration))
        # write_process("pred_answer: %s, true_answer: %s, correctness: %s" % (pred_answer, self.answers[idx], ans), self.process_file)
        # write_process("total question: %d, num correct: %d, num question: %d, correct ratio: %.4f" % (
        #     self.total_question, self.num_correct, self.num_problem, self.num_correct / self.num_problem), self.process_file)

        elapsed_time = time.time() - self.start_time
        print("spend %.3f seconds in total, spend %.3f second per problem" % (
        elapsed_time, elapsed_time / self.num_problem))
