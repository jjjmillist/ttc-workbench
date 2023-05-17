import pickle
import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib.ticker import MultipleLocator


if __name__ == "__main__":
    with open("results/16-05-23@10:47:49/timing.pickle", "rb") as file:
        timing = pickle.load(file)
    
    token_context_size = []
    slice_sizes = []
    token_duration = []
    forward_duration = []

    n_tokens_generated = 0

    total_time_in_forward_pass = 0
    total_time_in_prediction = 0
    total_time_in_token = 0
    for predict in timing.subintervals:
        assert predict.name == "predict"

        total_time_in_prediction += predict.duration

        for token in predict.subintervals:
            assert token.name == "token"
            assert len(token.subintervals) == 1

            forward = token.subintervals[0]
            assert forward.name == "forward"

            n_tokens_generated += token.data["batch_size"]

            token_context_size.append(token.data["context_size"])
            slice_sizes.append(token.data["batch_size"])
            token_duration.append(token.duration)
            forward_duration.append(forward.duration)

            total_time_in_token += token.duration
            total_time_in_forward_pass += forward.duration
    
    token_context_size = np.array(token_context_size)
    slice_sizes = np.array(slice_sizes)
    token_duration = np.array(token_duration)
    forward_duration = np.array(forward_duration)

    sorted_indices = np.argsort(token_duration)
    m = sorted_indices[:-500] # outlier mask

    ms_per_token = 1000 * token_duration / slice_sizes

    pyplot.plot(
        ms_per_token[m],
        slice_sizes[m],
        "o"
    )
    
    pyplot.ylabel("Batch size")
    pyplot.xlabel("Milliseconds per token")

    pyplot.savefig("output.png")

    locus = np.where(slice_sizes[m] == 100)
    max_time_per_token_ms = np.max(ms_per_token[m][locus])
    best_possible_total_token_duration = max_time_per_token_ms * n_tokens_generated / 1000

    print("forward:", total_time_in_forward_pass)
    print("predict:", total_time_in_prediction)
    print("token:", np.sum(token_duration))
    print("best possible token:", best_possible_total_token_duration)
    print("total:", timing.duration)