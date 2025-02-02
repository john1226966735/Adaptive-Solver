import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_length_cot", type=int, default=1024,
        help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32,
        help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0,
        help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=2.0, help=""
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help=""
    )
    parser.add_argument(
        '--dataset', default='gsm8k',
        help="dataset",
        # choices=["SVAMP", "gsm8k", "AQuA", "MultiArith", "AddSub", "SingleEq", "CommonsenseQA", "coin_flip",
        #          "last_letters", "FinQA", "TATQA", "ConvFinQA", "StrategyQA", "geometryQA"]
    )
    parser.add_argument(
        '--data_file', default='gsm8k'
    )
    parser.add_argument(
        "--prompt_id", default={"solve_step": "solve_101"}, type=dict
    )
    parser.add_argument(
        "--engine", default='gpt-3.5-turbo-0301',
        help="text-davinci-002, text-davinci-003, code-davinci-002, gpt-3.5-turbo, gpt-4, glm2",
        choices=["text-davinci-002", "text-davinci-003", "code-davinci-002", "gpt-3.5-turbo", "gpt-4",
                 "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "glm2", "gpt-35-turbo", "gpt-4-1106-preview"]
    )
    parser.add_argument(
        "--test_start", default='0', help='string, number'
    )
    parser.add_argument(
        "--test_end", default='full', help='string, number'
    )
    parser.add_argument(
        "--datapath", default=None, type=str, help='file path'
    )
    # parser.add_argument(
    #     "--learning_type", default='few_shot', type=str, help='zero shot or few shot',
    #     choices=['zero_shot', 'few_shot']
    # )
    # parser.add_argument(
    #     "--few_shot_num", default=1, type=int, help='sample number of few shot learning'
    # )
    parser.add_argument(
        "--domain", default='numeral', type=str, choices=['financial', 'numeral']
    )
    parser.add_argument(
        "--SC", default=False, type=bool, help="self consistency"
    )
    parser.add_argument(
        '--answer_extracting_prompt', default='Therefore,the answer is', type=str
    )
    parser.add_argument(
        '--prompt_mode', default='multi_step', choices=['multi_step']
    )
    parser.add_argument(
        '--remark', default='', type=str
    )

    parsed_args = parser.parse_args()
    parsed_args.direct_answer_trigger_for_zeroshot = "Let's think step by step."
    parsed_args.direct_answer_trigger_for_direct = "Therefore, the answer is"
    return parsed_args


args = parse_arguments()
