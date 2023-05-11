NON_OBVIOUS_TASKS = [
    "disambiguation_qa",
    "dyck_languages",
    "hyperbaton",
    "movie_recommendation",
    "navigate",
    "ruin_names",
    "snarks",
    "sports_understanding",
    "word_sorting",
]
SEMI_OBVIOUS_TASKS = [
    "geometric_shapes",
]
OBVIOUS_OR_EXAMPLE_CONTAINS_INSTRUCTIONS_SET = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "formal_fallacies",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "multistep_arithmetic_two",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "salient_translation_error_detection",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
]


def format_input(task_name, input_str):
    if task_name == "disambiguation_qa":
        return remove_before(input_str, "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\n")
    elif task_name == "dyck_languages":
        return remove_before(input_str, "Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: ")
    elif task_name == "geometric_shapes":
        return just_remove_all(input_str, [
            "This SVG path element ",
            " draws a",
        ])
    elif task_name == "hyperbaton":
        return remove_before(input_str, "Which sentence has the correct adjective order:\n")
    elif task_name == "movie_recommendation":
        return remove_before(input_str, "Find a movie similar to \n")
    elif task_name == "navigate":
        return remove_before(input_str, "If you follow these instructions, do you return to the starting point? ")
    elif task_name == "ruin_names":
        return input_str.replace("Which of the following is a humorous edit of this artist or movie name: ", "").replace("?", "")
    elif task_name == "snarks":
        return remove_before(input_str, "Which statement is sarcastic?\n")
    elif task_name == "sports_understanding":
        return remove_before(input_str, "Is the following sentence plausible? ")
    elif task_name == "word_sorting":
        return remove_before(input_str, "Sort the following words alphabetically: List: ")
    else:
        return input_str


def remove_before(string, substr):
    assert string[:len(substr)] == substr
    return string[len(substr):]


def remove_after(string, substr):
    assert string[-len(substr):] == substr
    return string[:-len(substr)]


def just_remove(string, substr):
    return string.replace(substr, "")


def just_remove_all(string, substr_list):
    for substr in substr_list:
        string = string.replace(substr, "")
    return string
