import time
import openai

FLAG = True
PROMPT_TOKEN_DICT = dict()
COMPLETION_TOKEN_DICT = dict()
NUM_GPT4 = 0
NUM_GPT35 = 0
openai.api_key = "sk-xxx"


def decoder_for_gpt3_chat(args, inputs, max_length, param_dict):
    global FLAG, NUM_GPT4, NUM_GPT35
    engine = param_dict.get('model', args.engine)
    SC = param_dict.get('SC', False)

    # top_p = 1
    # frequency_penalty = 0
    # presence_penalty = 0
    # temperature = 0.7 if (args.SC or SC) and max_length != 32 else 0.0
    temperature = param_dict.get('temperature', 0.7 if (SC and max_length != 32) else 0.0)
    # n = args.SC_N if (args.SC or SC) and max_length != 32 else 1
    n = param_dict.get('num_sample', args.SC_N if (SC and max_length != 32) else 1)
    stop = ["\n\n"] if max_length == 32 else None

    system_content = "Follow the instruction or given examples to solve problems."

    final_input = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": inputs},
    ]
    print("INPUT:\n", inputs)

    response = openai.ChatCompletion.create(
        model=engine,
        messages=final_input,
        temperature=temperature,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=n,
        stop=stop
    )
    NUM_GPT35 += 1
    print("number of calling GPT3.5=%d" % NUM_GPT35)
    if FLAG:
        print("temperature=%.2f, n=%d" % (temperature, n))
        print(response)
        FLAG = False

    global PROMPT_TOKEN_DICT, COMPLETION_TOKEN_DICT
    temp_token_dict = {"%s-%s" % (engine, "prompt_tokens"): response['usage']['prompt_tokens'],
                       "%s-%s" % (engine, "completion_tokens"): response['usage']['completion_tokens']
                       }
    PROMPT_TOKEN_DICT[engine] = PROMPT_TOKEN_DICT.get(engine, 0) + response['usage']['prompt_tokens']
    COMPLETION_TOKEN_DICT[engine] = COMPLETION_TOKEN_DICT.get(engine, 0) + response['usage']['completion_tokens']
    token_dict_str = "prompt_tokens=%s, completion_tokens=%s" % (PROMPT_TOKEN_DICT, COMPLETION_TOKEN_DICT)
    print(token_dict_str)

    gpt35_prompt_token = PROMPT_TOKEN_DICT.get('gpt-35-turbo', 0) + PROMPT_TOKEN_DICT.get('gpt-3.5-turbo-0301', 0)
    gpt35_completion_token = COMPLETION_TOKEN_DICT.get('gpt-35-turbo', 0) + COMPLETION_TOKEN_DICT.get(
        'gpt-3.5-turbo-0301', 0)
    gpt35_price = gpt35_prompt_token / 1000 * 0.0015 + gpt35_completion_token / 1000 * 0.002

    gpt4_prompt_token = PROMPT_TOKEN_DICT.get('gpt-4', 0) + PROMPT_TOKEN_DICT.get('gpt-4-0613', 0)
    gpt4_completion_token = COMPLETION_TOKEN_DICT.get('gpt-4', 0) + COMPLETION_TOKEN_DICT.get('gpt-4-0613', 0)
    gpt4_price = gpt4_prompt_token / 1000 * 0.03 + gpt4_completion_token / 1000 * 0.06

    cost_str = "API Cost of GPT3.5=%.4f, API Cost of GPT4=%.4f, Total API Cost=%.4f" % (gpt35_price, gpt4_price, gpt35_price + gpt4_price)
    print(cost_str)

    if n == 1:
        pred = response['choices'][0]['message']["content"].strip()
        print("PREDICTION:\n", pred)
        return {"pred": [pred], "token": token_dict_str, "price": cost_str, "token_dict": temp_token_dict}
    else:
        text = response["choices"]
        tem_rational = []
        for i in range(len(text)):
            tem_rational.append(text[i]['message']["content"].strip())
        print("PREDICTION:\n", tem_rational)
        return {"pred": tem_rational, "token": token_dict_str, "price": cost_str, "token_dict": temp_token_dict}


def basic_runner(args, inputs, max_length, param_dict, max_retry=3):
    retry = 0
    get_result = False
    pred_dict = None
    while not get_result:
        try:
            pred_dict = decoder_for_gpt3_chat(args, inputs, max_length, param_dict)
            get_result = True
        except openai.error.RateLimitError as e:
            if e.user_message == 'You exceeded your current quota, please check your plan and billing details.':
                raise e
            elif retry < max_retry:
                time.sleep(2)
                retry += 1
            else:
                raise e.user_message
        except Exception as e:
            raise e
    if not get_result:
        raise ValueError('do not get result!!!')
    return pred_dict
