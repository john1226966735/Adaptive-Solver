import json
import datetime
import os
import copy
from extracter import get_precision, extract_answer
from prediction_runner import basic_runner


def remove_duplicates(lst):
    result = []
    seen = set()
    for item in lst:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def mkpath(path):
    if not os.path.exists(path):
        os.mkdir(path)


def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    now = now.strftime('%Y/%m/%d %H:%M:%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass


def print_exp(args, return_flag=0):
    info = ''
    for k, v in vars(args).items():
        info += '{}:{}\n'.format(k, v)
    print('---------------experiment args---------------')
    print(info)
    print('---------------------------------------------')
    if return_flag == 0:
        return
    elif return_flag == 1:
        return info
    else:
        pass


def write_json(data, path):
    f = open(path, mode='a', encoding='utf-8')
    json.dump(data, f, indent=4, ensure_ascii=False)
    f.close()


def write_process(data, path):
    f = open(path, mode='a', encoding='utf-8')
    f.write(data)
    f.close()


def pre_setting(args):
    now = print_now(1).split(' ')[0].replace('/', '-')

    Result_Folder = 'result/{}'.format(now)
    mkpath('result')
    mkpath(Result_Folder)
    mkpath(f'{Result_Folder}/{args.dataset}')

    Log_Folder = 'log/{}'.format(now)
    mkpath('log')
    mkpath(Log_Folder)
    mkpath(f'{Log_Folder}/{args.dataset}')

    prompt = list(args.prompt_id.values())
    t = print_now(1).split(' ')[1].replace(':', '-')
    Decoder_Error_File = f'{Result_Folder}/{t}-{args.data_file}-{prompt}-{args.engine}-{args.remark}_deco.json'
    Predict_File = f'{Result_Folder}/{args.dataset}/{t}-{args.data_file}-{prompt}-{args.engine}-{args.remark}.json'
    Process_File = f'{Log_Folder}/{args.dataset}/{t}-{args.data_file}-{prompt}-{args.engine}-{args.remark}.process'

    return Decoder_Error_File, Predict_File, Process_File


def find_answer(ques, pred_list, args, Process_File, model):
    input_1 = "Q: %s\nA: " % ques
    answer2preds_dict = dict()
    answer_list = []
    pred_dict = None
    for pred in pred_list:
        if 'answer is ' in pred:
            pred2 = pred.split('answer is ')[-1]
        else:
            print("------------Call the API again to get the final answer------------")
            write_process("------------Call the API again to get the final answer------------\n", Process_File)

            inputs2 = input_1 + pred + ' ' + args.direct_answer_trigger_for_direct

            # pred_dict = basic_runner(args, inputs2, 32, {'SC': False, 'model': 'gpt-3.5-turbo-0301'})
            if model is not None:
                pred_dict = basic_runner(args, inputs2, 32, {'SC': False, 'model': model})
            else:
                pred_dict = basic_runner(args, inputs2, 32, {'SC': False})

            pred2 = pred_dict["pred"][0]
            # print("SUB-INPUT: %s\nSUB-PREDICTION: %s\n" % (inputs2, pred2))
            write_process("SUB-INPUT: %s\nSUB-PREDICTION: %s\n" % (inputs2, pred2), Process_File)

        try:
            pred_answer = extract_answer(args, copy.deepcopy(pred2))
        except:
            pred_answer = None

        answer_list.append(pred_answer)
        if pred_answer not in answer2preds_dict.keys():
            answer2preds_dict[pred_answer] = []
        answer2preds_dict[pred_answer].append(pred)

    return answer2preds_dict, answer_list, pred_dict


def write_result(question, answer, index, cot, pred_answer, args, Predict_File, temp_answers=None, temp_cots=None,
                 temp_token_dict=None, duration=None):
    json_data = {
        "ID": index,
        "question": question,
        "chain-of-thought": cot,
        "pred": pred_answer,
        "answer": answer,
        "temp_answers": temp_answers,
        "temp_CoTs": temp_cots,
        "temp_token_dict": temp_token_dict,
        "duration": duration
    }
    ans = False
    if pred_answer is not None:
        if args.dataset.lower() in ["svamp", "gsm8k", "multiarith", "addsub", "singleeq"]:
            if abs(pred_answer - answer) < 1e-3:
                ans = True
        else:
            if isinstance(pred_answer, float) and isinstance(answer, float):
                precision = min(get_precision(pred_answer), get_precision(answer))
                if round(pred_answer, precision) == round(answer, precision):
                    ans = True
            else:
                if pred_answer == answer:
                    ans = True
    json_data["ans"] = ans
    write_json(json_data, Predict_File)
    return ans


def get_result(args, pred_answer, answer):
    ans = False
    if pred_answer is not None:
        if args.dataset.lower() in ["svamp", "gsm8k", "multiarith", "addsub", "singleeq"]:
            if abs(pred_answer - answer) < 1e-3:
                ans = True
        else:
            if isinstance(pred_answer, float) and isinstance(answer, float):
                precision = min(get_precision(pred_answer), get_precision(answer))
                if round(pred_answer, precision) == round(answer, precision):
                    ans = True
            else:
                if pred_answer == answer:
                    ans = True

    return ans


def read_jsonl(path: str):
    with open(path, "r", encoding='utf-8') as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def find_formula(step):
    assert step.count("<<") == step.count(">>") == 1
    left, right = step.find("<<") + 2, step.find(">>")
    return step[left: right]


def delete_extra_zero(n):
    try:
        n = float(n)
    except:
        print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)
        n = str(n)
        return n


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string
