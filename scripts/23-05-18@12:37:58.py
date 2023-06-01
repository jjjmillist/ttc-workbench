import transformers
import torch


def shape(structure):
    try:
        return structure.shape
    except AttributeError:
        return (f"list[{len(structure)}]", *shape(structure[0]))


short_prompt = """\
# This is a short prompt
def hello_world():
"""

long_prompt = """\
# This is a slightly longer prompt with more tokens in it
def hello_world():
"""
    

if __name__ == "__main__":
    print("Started")

    model_uri = "Daoguang/PyCodeGPT"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_uri)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_uri)
    model.to("cuda:0")

    last_token_ids = []
    past_kvs = []
    for prompt in [short_prompt, long_prompt]:
        
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda:0")
        prompt_length = inputs["input_ids"].shape[1]
        print(prompt_length)
        
        with torch.no_grad():
            outputs = model(**inputs)

        argmax = outputs.logits[:, -1].argmax()
        
        last_token_ids.append(argmax)
        past_kvs.append(outputs.past_key_values)

        print(shape(past_kvs[-1]))

    # new_layers = []
    # for layer1, layer2 in zip(*past_kvs):
    #     new_layers.append(
    #         (
    #             [layer1[0][0], layer2[0][0]],
    #             [layer1[1][0], layer2[1][0]]
    #         )
    #     )

    # inputs = {
    #     "input_ids": torch.tensor(last_token_ids).to("cuda:0"),
    #     "past_key_values": new_layers
    # }

    # model(**inputs)

    batch_size = 2
    batched_past_kvs = []
    for i in range(len(past_kvs[0])):
        batched_past_kvs.append(torch.cat([past_kvs[0][i], past_kvs[1][i]]))

    print(shape(batched_past_kvs))