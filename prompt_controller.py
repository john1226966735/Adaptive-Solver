from config import args
# from prediction_runner_azure import basic_runner
from prediction_runner import basic_runner
from utils import write_json, write_process
from build_question_chain import MultiStepPrompt


class PromptController:
    def __init__(self, Process_File, Decoder_Error_File):
        self.Process_File = Process_File
        self.Decoder_Error_File = Decoder_Error_File

        self.prompter = None
        self.num_call = 0

    def _call_llm_api(self, inputs, param_dict):
        self.num_call += 1
        print("------------%d-th API Calling------------" % self.num_call)
        write_process("------------%d-th API Calling------------\n" % self.num_call, self.Process_File)

        pred_dict = basic_runner(args, inputs, args.max_length_cot, param_dict)
        write_process("INPUT: %s\nPREDICTION: %s\n" % (inputs, pred_dict["pred"]), self.Process_File)
        return pred_dict

    def _build_prompter_each_problem(self, prompts, problemInfo):
        if args.prompt_mode == "multi_step":
            self.prompter = MultiStepPrompt(prompts, problemInfo)
        else:
            raise ValueError('Not support prompt mode')

    def solve_one_problem(self, prompts, problemInfo, param_dict):
        self._build_prompter_each_problem(prompts, problemInfo)
        num_step = self.prompter.chain_length
        inputs = None
        pred_dict = {"pred": ''}
        for step in range(num_step):
            inputs = self.prompter.construct_input_step(step)

            pred_dict = self._call_llm_api(inputs, param_dict)
            self.prompter.process_pred_step(step, pred_dict["pred"][0])
        return inputs, pred_dict
