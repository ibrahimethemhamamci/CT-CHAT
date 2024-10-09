import transformers
import torch
import pandas as pd
import argparse
import re
import json
import tqdm


def parse_conversation(conversation_str):
    all_conversations = {}
    conversation_sets = re.findall(r'<(\d+)>(.*?)<\1>', conversation_str, re.DOTALL)

    for set_number, conversation in conversation_sets:
        conversation_dict = {"conversations": []}
        qas = re.findall(r'<q>(.*?)<a>(.*?)<', conversation, re.DOTALL)
        for q, a in qas:
            q_dict = {
                "from": "human",
                "value": q.strip().replace("<q>", "").replace("</q>", "")
            }
            a_dict = {
                "from": "gpt",
                "value": a.strip().replace("<a>", "").replace("</a>", "")
            }
            conversation_dict["conversations"].append(q_dict)
            conversation_dict["conversations"].append(a_dict)
        all_conversations[set_number] = conversation_dict

    return all_conversations

def has_empty_qa(parsed_output):
    for set_number, conversation_dict in parsed_output.items():
        for pair in conversation_dict["conversations"]:
            if pair["value"].strip() == "":
                return True
    return False

def main(part_number):
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    #df = pd.read_csv(f"new_train_reports_{part_number}.csv")
    df = pd.read_csv("validation_reports.csv")
    all_outputs = {}
    for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        AccessionNo = row["AccessionNo"]
        report = "Findings: " + str(row["Findings_EN"]) + " Impression: " + str(row["Impressions_EN"])
        messages = [
            {"role": "system", "content": "You are a radiologist specializing in creating VQA (Visual Question Answering) datasets for Chest CT volumes. Generate three sets of conversation-style Q&A pairs based on the CT volume information, using the provided report as background without referring directly to it. Enclose each set within tags <1>...<1>, <2>...<2>, and <3>...<3>. Use <q>...</q> for questions and <a>...</a> for answers, ensuring most answers are derived from the images. Example: <1><q>Question 1</q><a>Answer 1</a><q>Question 2</q><a>Answer 2</a>...<1><2><q>Question 1</q><a>Answer 1</a><q>Question 2</q><a>Answer 2</a>...<2><3><q>Question 1</q><a>Answer 1</a><q>Question 2</q><a>Answer 2</a>...<3>. You can include as many question and answer couples as you find appropriate. Here is the report for you to get information about CT volume:\n"},
            {"role": "user", "content": f"{report}"}
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=2500,
        )

        conversation_str = outputs[0]["generated_text"][-1]["content"]

        parsed_output = parse_conversation(conversation_str)
        # Check for empty questions or answers and rerun if necessary
        while has_empty_qa(parsed_output):
            print("broken output, rerunning inference.")
            outputs = pipeline(
                messages,
                max_new_tokens=2500,
            )
            conversation_str = outputs[0]["generated_text"][-1]["content"]
            parsed_output = parse_conversation(conversation_str)

        all_outputs[AccessionNo] = parsed_output
        print(AccessionNo, flush=True)
        print(parsed_output,flush=True)


    # Open the file in write mode and use json.dump to write the dictionary to the file
    #with open(f"{part_number}_conversations_new.json", 'w') as json_file:
    with open("conversations_new_valid.json","w") as json_file:
        json.dump(all_outputs, json_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--part_number', type=str, required=True, help='a string for the part number')
    args = parser.parse_args()
    main(args.part_number)