import transformers
import torch
import pandas as pd
import argparse
import re
import json
import tqdm

def parse_conversation(conversation_str):
    all_conversations = {}
    free_response = re.search(r'<free_response>(.*?)</free_response>', conversation_str, re.DOTALL)
    description = re.search(r'<description>(.*?)</description>', conversation_str, re.DOTALL)
    multiple_choice = re.search(r'<multiple_choice>(.*?)</multiple_choice>', conversation_str, re.DOTALL)

    def extract_qas(section):
        conversation_dict = {"conversations": []}
        qas = re.findall(r'<q>(.*?)<a>(.*?)<', section, re.DOTALL)
        for q, a in qas:
            q_dict = {
                "from": "human",
                "value": q.strip().replace("<q>", "")
            }
            a_dict = {
                "from": "gpt",
                "value": a.strip().replace("<a>", "")
            }
            conversation_dict["conversations"].append(q_dict)
            conversation_dict["conversations"].append(a_dict)
        return conversation_dict

    if free_response:
        all_conversations["free_response"] = extract_qas(free_response.group(1))
    if description:
        all_conversations["description"] = extract_qas(description.group(1))
    if multiple_choice:
        all_conversations["multiple_choice"] = extract_qas(multiple_choice.group(1))

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

    #df = pd.read_csv(f"train_reports_{part_number}.csv")
    df = pd.read_csv("validation_reports.csv")
    all_outputs = {}
    for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        AccessionNo = row["AccessionNo"]
        report = "Findings: " + str(row["Findings_EN"]) + " Impression: " + str(row["Impressions_EN"])
        messages = [
            {"role": "system", "content": """You are a radiologist VQA dataset creator specializing in Chest CT reports. 
                                              I need you to generate three sets of questions based on the given radiology report: free response, description, and multiple-choice questions. 
                                              Each set should contain 3 Q/A pairs.
                                              The free response questions should be enclosed within <free_response>...</free_response>.
                                              The description questions should be enclosed within <description>...</description>.
                                              The multiple-choice questions should be enclosed within <multiple_choice>...</multiple_choice>, with each question having 4 choices labeled as (a), (b), (c), and (d).
                                              Ensure that the questions are specific to the Chest CT image described in the report.
                                              Example for free response: <q>What abnormalities are present in the lung fields of this Chest CT image?<q>
                                              Example for description: <q>Describe the findings shown in this Chest CT image.<q>
                                              Example for multiple-choice: <q>Which of the following abnormalities are present in this Chest CT image? (a) Choice 1 (b) Choice 2 (c) Choice 3 (d) Choice 4<q><a>Choice 2<a>
                                              The structure should look like this:
                                              <free_response><q>Question 1<q><a>Answer 1<a><q>Question 2<q><a>Answer 2<a><q>Question 3<q><a>Answer 3<a></free_response>
                                              <description><q>Question 1<q><a>Answer 1<a><q>Question 2<q><a>Answer 2<a><q>Question 3<q><a>Answer 3<a></description>
                                              <multiple_choice><q>Question 1 (a) Choice 1 (b) Choice 2 (c) Choice 3 (d) Choice 4<q><a>(b) Choice 2<a><q>Question 2 (a) Choice 1 (b) Choice 2 (c) Choice 3 (d) Choice 4<q><a>(a) Choice 1<a><q>Question 3 (a) Choice 1 (b) Choice 2 (c) Choice 3 (d) Choice 4<q><a>(c) Choice 3<a></multiple_choice>
                                              Please always follow this format to create the questions and answers."""},
            {"role": "user", "content": f"{report}"}
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=2500,
        )

        conversation_str = outputs[0]["generated_text"][-1]["content"]
        print(conversation_str)
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

    # Open the file in write mode and use json.dump to write the dictionary to the file
    #with open(f"{part_number}_freeresponse_description_multiplechoice_validation.json", 'w') as json_file:
    with open("freeresponse_description_multiplechoice_validation.json","w") as json_file:
        json.dump(all_outputs, json_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--part_number', type=str, required=True, help='a string for the part number')
    args = parser.parse_args()
    main(args.part_number)