from main_base import BaseSolver
from config import args
from utils import pre_setting, print_exp


def main():
    # set some active parameters
    args.SC_THRESH = 1
    args.SC_N = 3
    args.SC = True

    # AS-MSPD
    # [{'llm_model': 'GPT3.5', 'sample_size': 3, 'prompt': 'ZeroCoT'},
    # {'llm_model': 'GPT3.5', 'sample_size': 3, 'prompt': 'L2M_L1'},
    # {'llm_model': 'GPT3.5', 'sample_size': 3, 'prompt': 'CoT'},
    # {'llm_model': 'GPT3.5', 'sample_size': 5, 'prompt': 'ZeroCoT'},
    # {'llm_model': 'GPT3.5', 'sample_size': 10, 'prompt': 'ZeroCoT'},
    # {'llm_model': 'GPT4', 'sample_size': 1, 'prompt': 'ZeroCoT'}]
    model_list = ['gpt-3.5-turbo-0301'] * 5 + ['gpt-4-0613']
    num_sample_list = [3, 3, 3, 5, 10, 1]
    prompt_list = ['ZeroCOT', 'L2M_L1', 'CoT', 'ZeroCoT', 'ZeroCoT', 'ZeroCoT']
    thresh_list = [{1: 1.0, 3: 1.0, 5: 0.8, 10: 0.6}[i] for i in num_sample_list]

    args.prompt_id = {"solve_main": prompt_list}
    args.dataset = 'gsm8k'
    args.data_file = "gsm8k_1119"
    args.engine = 'gpt-3.5-turbo-0301'
    args.remark = ''
    print_exp(args)
    Decoder_Error_File, Predict_File, Process_File = pre_setting(args)

    reasoner = BaseSolver(args, Decoder_Error_File, Predict_File, Process_File, select_strategy="last_solver",
                          model_list=model_list, num_sample_list=num_sample_list, thresh_list=thresh_list)
    reasoner.run({"SC": args.SC})


if __name__ == '__main__':
    main()
