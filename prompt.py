from config import args


Prompt_Folder = '../Prompt/'


def get_prompts_dict():
    prompts_dict = dict()
    for key, prompt_id in args.prompt_id.items():
        prompts_dict[key] = []
        for p_id in prompt_id:
            with open(Prompt_Folder + f'Prompt_{p_id}.txt', encoding="utf-8") as file:
                prompt_str = file.read()
            prompts = prompt_str.split("======")
            prompts_dict[key].append(prompts)
    return prompts_dict


def get_prompts(prompt_id):
    with open(Prompt_Folder + f'Prompt_{prompt_id}.txt', encoding="utf-8") as file:
        prompt_str = file.read()
    prompts = prompt_str.split("======")
    return prompts

