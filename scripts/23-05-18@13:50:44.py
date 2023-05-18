import transformers
import torch


def shape(structure):
    try:
        return structure.shape
    except AttributeError:
        return (f"list[{len(structure)}]", *shape(structure[0]))


short_prompt = """To be or not to"""

long_prompt = """It was the best of times, it was the worst"""
    

if __name__ == "__main__":
    print("Started")

    model_uri = "gpt2"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_uri)
    tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(model_uri)
    model.to("cuda:0")

    inputs = tokenizer([short_prompt, long_prompt], padding=True, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        outputs = model(**inputs)

    next_tokens = outputs.logits[:, -1, :].max(axis=-1).indices
    
    insertion_points = inputs.attention_mask.sum(axis=1)
    # new_column = torch.tensor(tokenizer.pad_token_id).repeat(2).to("cuda:0")    
    # new_inputs = torch.cat((inputs.input_ids, new_column[:, None]), dim=1)
    # new_inputs.scatter_(1, insertion_points[:, None], next_tokens[:, None])

    mask = inputs.attention_mask
    new_mask_column = torch.zeros((len(inputs.input_ids), 1)).to("cuda:0")
    new_mask = torch.cat((mask, new_mask_column), dim=1)
    new_mask.scatter_(1, insertion_points[:, None], torch.ones(2, 1).to("cuda:0"))

    print(next_tokens[:, None].shape)
    new_inputs = {
        "input_ids": next_tokens[:, None],
        "attention_mask": new_mask,
        "past_key_values": outputs.past_key_values
    }

    with torch.no_grad():
        outputs = model(**new_inputs)

    next_tokens = outputs.logits[:, -1, :].max(axis=-1).indices

    print(next_tokens)

    exit()

    last_token_ids = []
    past_kvs = []
    for prompt in [short_prompt, long_prompt]:
        
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