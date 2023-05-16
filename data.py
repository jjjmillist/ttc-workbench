import json


def mbpp_sanitized():
    with open("/home/ICTDOMAIN/d20126116/Datasets/MBPP/sanitized.json") as file:
        rows = json.load(file)

    return rows


def few_shot_mbpp(test_instance, training_instances):
    prefix_and_prompt = ""
    for instance in training_instances:
        for line in instance["prompt"].splitlines():
            prefix_and_prompt += "# " + line + "\n"
        prefix_and_prompt += instance["code"] + "\n\n"

    prefix_and_prompt += "# " + test_instance["prompt"] + "\n"
    for line in test_instance["code"].splitlines():
        if line.startswith("def "):
            prefix_and_prompt += line.strip() + "\n"
            break
    return prefix_and_prompt


def mbpp_zero_shot():
    for row in mbpp_sanitized():
        prompt = ""
        for line in row["prompt"].splitlines():
            prompt += "# " + line + "\n"
        
        for line in row["code"].splitlines():
            if line.startswith("def "):
                prompt += line.strip() + "\n"
                break
        yield prompt