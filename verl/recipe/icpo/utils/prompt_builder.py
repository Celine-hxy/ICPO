instruction = """ Let's think step by step and output the final answer within \\boxed{{}}"""
ref_template = """
Before we get to the main problem, here are a few examples with their solution approaches for your reference.
{demo_prompt}
"""

class ICLPromptBuilder:
    def __init__(self, w_inst: bool, w_prefix: bool = True, innner_splitter: str = "\n", splitter: str = "\n\n\n"):
        self.w_inst = w_inst
        self.w_prefix = w_prefix
        self.innner_splitter = innner_splitter
        self.splitter = splitter

        if w_prefix:
            question_template, ans_template = "Question: {question}", "Answer: {answer}"
        else:
            question_template, ans_template = "{question}", "{answer}"

        if w_inst:
            question_template += instruction

        self.icl_template = question_template + innner_splitter + ans_template
        self.question_prefix = "Question: " if w_prefix else ""
        self.ans_prefix = innner_splitter + "Answer: " if w_prefix else innner_splitter

    def construct_n_shot_prompt(self, demonstrations, system_prompt, raw_question, tokenizer):
        if demonstrations:
            n_shot_examples = []
            for demo_question, demo_answer in demonstrations:
                formatted_icl_template = self.icl_template.format(question=demo_question, answer=demo_answer)
                n_shot_examples.append(formatted_icl_template)
            n_shot_prompt = self.splitter.join(n_shot_examples)

            if system_prompt != "":
                # 使用全局定义的ref_template变量
                formatted_ref_prompt = ref_template.format(demo_prompt=n_shot_prompt)
                # [Qwen2.5] The system_prompt does not take effect after apply_chat_template, so manual string processing is required here.
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": raw_question}]
                final_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                split_marker = "User: This is the problem:"
                if split_marker in final_prompt:
                    parts = final_prompt.split(split_marker)
                    assert len(parts) == 2, f"final_prompt 按 '{split_marker}' 分割后长度不是2，实际为{len(parts)}，内容为：{final_prompt}"
                    final_prompt = parts[0] + formatted_ref_prompt + split_marker + parts[1]
                else:
                    final_input = n_shot_prompt + self.splitter + self.question_prefix + raw_question + self.ans_prefix
                    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": final_input}]
                    final_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                raw_question = raw_question + instruction if self.w_inst else raw_question
                final_prompt = n_shot_prompt + self.splitter + self.question_prefix + raw_question + self.ans_prefix

            return final_prompt[:-1]
        else:
            return self.question_prefix + raw_question + self.ans_prefix

    

        
    