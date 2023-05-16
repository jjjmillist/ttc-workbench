import transformers
import torch
import pickle

from workshop import *
from data import *

root = output_directory()

model = transformers.BertModel.from_pretrained("bert-base-cased")
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

mbpp = mbpp_sanitized()
sentences = [row["prompt"] for row in mbpp]

inputs = tokenizer(sentences, padding=True, return_tensors="pt")

outputs = model(**inputs)

print(inputs.attention_mask.shape)
print(outputs.last_hidden_state.shape)

masked_states = inputs.attention_mask[:, :, None] * outputs.last_hidden_state
print(masked_states.shape)

sentence_embeddings = masked_states.mean(axis=1)
print(sentence_embeddings.shape)

tagged_sentence_embeddings = [
    {
        **mbpp_instance,
        "bert_embedding": embedding
    }
    for (embedding, mbpp_instance) in zip(sentence_embeddings, mbpp)
]

output_file = root / "mbpp_sanitized_with_bert_encodings.pickle"
with open(output_file, "wb") as file:
    pickle.dump(tagged_sentence_embeddings, file)