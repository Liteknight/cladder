import pickle
import pprint

def load_and_print_pickle(path, skip_keys=None):
    with open(path, "rb") as f:
        data = pickle.load(f)

    pp = pprint.PrettyPrinter(indent=2, width=160)
    print(f"\nLoaded {len(data)} samples from {path}\n")

    for i, item in enumerate(data):
        print(f"--- Sample {i} ---")

        # Filter out unwanted keys if specified
        if skip_keys:
            item = {
                k: v for k, v in item.items()
                if not any(k.startswith(skip) for skip in skip_keys)
            }

        pp.pprint(item)
        print("\n")

    return data

# Example usage
if __name__ == "__main__":
    path = "ollama_debate_round_2.pkl"  # adjust as needed
    skip_keys = ["magistral_output_"]  # omit magistral entries if needed
    load_and_print_pickle(path, skip_keys=skip_keys)
