import json


# Function to sort the jsonl data
def sort_jsonl(input_file, output_file):
    with open(input_file, "r") as file:
        lines = file.readlines()

    # Parse JSONL lines
    data = [json.loads(line) for line in lines]

    # Sort data first by model, then by question_id
    sorted_data = sorted(data, key=lambda x: (x["model"], x["question_id"]))

    # Write sorted data to output file
    with open(output_file, "w") as file:
        for entry in sorted_data:
            file.write(json.dumps(entry) + "\n")


# Usage
input_file = "data/mt_bench/model_judgment/gpt-4-turbo_pair.jsonl"
output_file = "data/mt_bench/model_judgment/gpt-4-turbo_pair.jsonl"
sort_jsonl(input_file, output_file)
