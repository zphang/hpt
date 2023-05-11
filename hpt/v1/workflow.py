import string
import tqdm.auto as tqdm
import numpy as np
import pandas as pd


def format_input(example, input_keys):
    alphabet = string.ascii_letters[26:]
    formatted_list = []
    if len(input_keys) == 1:
        formatted_list.append("  <input>{}</input>".format(example[input_keys[0]].strip()))
    else:
        for i, input_key in enumerate(input_keys):
            formatted_list.append("  <input-{}>{}</input-{}>".format(alphabet[i], example[input_key].strip(), alphabet[i]))
    return "\n".join(formatted_list)


def format_initial_instructions(instruction_examples, input_keys, target_key):
    if isinstance(input_keys, str):
        input_keys = [input_keys]
    formatted_list = []
    for example in instruction_examples:
        formatted_list.append("<example>")
        formatted_list.append(format_input(example, input_keys))
        formatted_list.append("  <label>{}</label>".format(example[target_key]))
        formatted_list.append("</example>\n")
    s = "The following is a dataset for a task that we are trying to train a worker for. The worker will be provided the inputs and is required to output the corresponding label."
    s += "\n\n"
    s += "\n".join(formatted_list)
    s += "\n\n"
    s += """Based on the above, think step-by-step to determine what the task is. Describe what is consistent across examples, highlight what seems unclear, and make your best guess. Then, write a set of instructions for how the worker should perform the task. Be as detailed as necessary. The instructions should follow the following format:
    <thinking>...</thinking>
    <li>...</li>
    <li>...</li>
    <li>...</li>
    """
    return s


def format_tackle_task(example, instruction_list, input_keys="input"):
    if isinstance(input_keys, str):
        input_keys = [input_keys]
    # noinspection PyListCreation
    line_list = []
    line_list.append("Follow these instructions and complete the task:")
    for inst in instruction_list:
        line_list.append(f"- {inst}")
    line_list.append("")
    line_list.append("Your answer should consist of a 'working' and an 'answer' component, in the following format:")
    line_list.append("First, think step-by-step through the problem in <working>...</working>")
    line_list.append("Finally, provide the answer in <answer>...</answer>")
    line_list.append("")
    line_list.append(format_input(example, input_keys))
    s = "\n".join(line_list)
    return s


def format_improve_instructions(examples,
                                instruction_list,
                                answer_list,
                                working_list=None,
                                eval_list=None,
                                input_keys="input",
                                target_key="target"):
    assert len(examples) == len(answer_list)
    if isinstance(input_keys, str):
        input_keys = [input_keys]
    formatted_list = []
    for i, example in enumerate(examples):
        formatted_list.append("<example>")
        formatted_list.append(format_input(example, input_keys))
        if working_list is not None:
            formatted_list.append("  <submitted-working>{}</submitted-working>".format(working_list[i]))
        formatted_list.append("  <answer>{}</answer>".format(answer_list[i]))
        formatted_list.append("  <true-answer>{}</true>".format(example[target_key]))
        if eval_list is not None:
            formatted_list.append("  <evaluation>{}</evaluation>".format(eval_list[i]))
        formatted_list.append("</example>\n")

    s = "The following is a dataset for a task that we are trying to train a worker for. The worker was provided with the following instructions.\n"
    s += "\n" + "\n".join([f"- {x}" for x in instruction_list])
    s += "\n\nThe following are the inputs and answers they submitted, along with the true answers for each example. The task inputs are shown in <input> or <input-*> tags, the worker's answers are shown in <answer> tags, and the true answers are shown in <true-answer> tags (which are not shown to workers). We want the worker's answers to match the true answers."
    if eval_list is not None:
        s += "\nThe worker's answers are scored in <evaluation> tags."
    s += "\n\n"
    s += "\n".join(formatted_list)
    s += "\n\n"
    s += "\nThe instructions provided above may be partially correct, or completely incorrect and from a different task (in which case the previous set of task instructions should be discarded and new ones written from scratch)."
    s += "\nBased on the above, think step-by-step about what the task is, why each answer was marked correct or wrong (using at least one example explicitly), and how to modify the instructions accordingly."
    s += "\nThen, describe in detail the exact formatting of the answers in <true-answer> (e.g. parenthesis, capitalization, spacing)."
    s += "\nThen, rewrite the instructions to let future workers perform better on the above task. Be specific in the instructions, particularly on matching the exact format of the answer."
    s += """\nThe instructions should follow the following format:
    <thinking>...</thinking>
    <li>...</li>
    <li>...</li>
    <li>...</li>
    """
    return s


def evolution(model,
              data,
              input_keys="input",
              target_key="target",
              num_initial: int = 5,
              num_eval_examples: int = 10,
              num_survive: int = 3,
              num_children: int = 3,
              num_context_examples: int = 5,
              num_generations: int = 5,
              keep_parents: bool = False):
    # Population members are instruction-lists (lists of strings, corresponding to a single set of instructions)

    instructions_prompt = format_initial_instructions(
        data,
        input_keys=input_keys, target_key=target_key,
    )
    current_popn = []
    raw_instruction_list = []
    for _ in tqdm.trange(num_initial, desc="Initial population"):
        raw_instruction = model.query(instructions_prompt)
        raw_instruction_list.append(raw_instruction)
        current_popn.append(parse_instructions(raw_instruction))
    initial_data = {
        "current_popn": current_popn,
        "raw_instruction_list": raw_instruction_list,
        "instructions_prompt": instructions_prompt,
    }

    # Evolution
    generation_data = []
    for _ in tqdm.trange(num_generations, desc="Generation"):

        popn_score_data = []
        eval_examples = [data[int(i)] for i in np.random.choice(len(data), size=num_eval_examples)]
        for i, instruction_list in enumerate(tqdm.tqdm(current_popn, desc="Apply prompt")):
            raw_response_list = []
            working_list = []
            answer_list = []
            for example in tqdm.tqdm(eval_examples, desc="Example"):
                raw_response = model.query(format_tackle_task(
                    example,
                    instruction_list=instruction_list, input_keys=input_keys)
                )
                raw_response_list.append(raw_response)
                working_list.append(parse_single(raw_response, tag="working"))
                answer_list.append(parse_single(raw_response, tag="answer"))
            correct_list = [answer_list[i] == eval_examples[i][target_key] for i in range(len(eval_examples))]
            eval_list = ["CORRECT" if x else "WRONG" for x in correct_list]  # Todo: replace w/ scoring function
            num_correct = np.mean(correct_list)
            popn_score_data.append({
                "raw_response_list": raw_response_list,
                "working_list": working_list,
                "answer_list": answer_list,
                "correct_list": correct_list,
                "eval_list": eval_list,
                "num_correct": num_correct
            })

        # Selection
        df = pd.DataFrame(popn_score_data)
        df["rand"] = np.random.randn(len(df))
        df["instruction_list"] = current_popn
        survive_popn_full = list(df.sort_values(
            ["num_correct", "rand"], ascending=False)[:num_survive].T.to_dict().values())

        # Mutation
        expanded_popn = []
        improve_prompt_list = []
        improve_raw_response_list = []
        for elem in tqdm.tqdm(survive_popn_full, desc="Mutate prompt"):
            improve_instructions_prompt = format_improve_instructions(
                examples=eval_examples[num_context_examples:],
                instruction_list=elem["instruction_list"],
                answer_list=elem["answer_list"][num_context_examples:],
                working_list=None,
                eval_list=elem["eval_list"][num_context_examples:],
            )
            for _ in tqdm.trange(num_children, desc="Child"):
                improve_raw_response = model.query(improve_instructions_prompt)
                improved_instruction_list = parse_instructions(improve_raw_response)
                improve_prompt_list.append(improve_instructions_prompt)
                improve_raw_response_list.append(improve_raw_response)
                expanded_popn.append(improved_instruction_list)

        if keep_parents:
            current_popn = expanded_popn
        else:
            current_popn = expanded_popn + [x["instruction_list"] for x in survive_popn_full]
        generation_data.append({
            "popn_score_data": popn_score_data,
            "survive_popn": survive_popn_full,
            "expanded_popn": expanded_popn,
            "improve_raw_response_list": improve_raw_response_list,
            "improve_prompt_list": improve_prompt_list,
        })

    # noinspection PyUnboundLocalVariable
    return {
        "initial_data": initial_data,
        "generation_data": generation_data,
        "best": survive_popn_full[0]["instruction_list"],
        "best_score": survive_popn_full[0]["num_correct"],
    }


def parse_instructions(s):
    instruction_list = []
    for part in s.split("<li>")[1:]:
        if "</li>" in part:
            instruction_list.append(part.split("</li>")[0].strip())
    return instruction_list


def parse_single(s, tag):
    if f"<{tag}>" not in s:
        return ""
    return s.split(f"<{tag}>")[1].split(f"</{tag}>")[0].strip()


def format_instruction_list(instruction_list):
    return "\n".join([f"- {x}" for x in instruction_list])
