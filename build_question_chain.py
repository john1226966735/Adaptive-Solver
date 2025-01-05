class QuestionNode:
    def __init__(self, question):
        self.question = question

        self.answer = None

    def fill_answer(self, answer):
        self.answer = answer


class BasePrompt:
    def __init__(self, prompts, problemInfo):
        self.prompts = prompts
        self.problemInfo = problemInfo

        self.problem = self.problemInfo["question"]
        self.question_chain = None
        self.chain_length = None

    def _build_question_chain(self):
        pass

    def construct_input_step(self, step):
        pass

    def process_pred_step(self, step, pred):
        curr_node = self.question_chain[step]
        curr_node.fill_answer(pred)


class MultiStepPrompt(BasePrompt):
    # Multiple solving steps for the same problem (or sub-problem), such as the first step generating known conditions
    # and the second step solving, each step targets the same issue.
    def __init__(self, prompts, problemInfo):
        super().__init__(prompts, problemInfo)
        self._build_question_chain()

    def _build_question_chain(self):
        self.question_chain = []
        for _ in self.prompts:
            self.question_chain.append(QuestionNode(self.problem))
        self.chain_length = len(self.question_chain)

    def construct_input_step(self, step):
        prompt = self.prompts[step]
        if step >= 1 and "[PRED]" in prompt:
            pre_node = self.question_chain[step - 1]
            pred = pre_node.answer
        else:
            pred = None

        inputs = prompt
        if "[QUES]" in prompt:
            inputs = inputs.replace("[QUES]", self.problem)
        if "[PRED]" in prompt:
            inputs = inputs.replace("[PRED]", pred)
        return inputs

