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

    df = pd.read_excel(f"articles_{part_number}.xlsx")

    all_outputs = {}
    for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        article_title = row["Title"]
        article_content = row["Full Text"]
        messages = [
                {
                    "role": "system",
                    "content": """
                    You are a chest CT dataset creator specializing in generating question-and-answer sets based on radiology articles focusing on chest CTs. 
                    I need you to create 5 question-and-answer pairs for each article. Each question-and-answer pair should be enclosed within tags like <q>...<q> for questions and <a>...<a> for answers. Also include all output in <1>...<1> (remember to put one at the beginning one at the end).
                    For example, the structure should look like this: <1><q>Question 1<q><a>Answer 1<a><q>Question 2<q><a>Answer 2<a><q>Question 3<q><a>Answer 3<a><q>Question 4<q><a>Answer 4<a><q>Question 5<q><a>Answer 5<a><1>.
                    Please follow this format to create the conversation questions and answers.
                    Make sure the questions are variable, educative, and the answers are long and instructive enough. 
            
                    The questions should be those that a student or radiologist might ask about chest CT images in general. Do not generate questions specific to the content or research idea of the provided article. Focus solely on general knowledge about chest CT imaging.
            
                    Examples of such questions include:
                    - What are the typical imaging findings of interstitial lung disease on a chest CT?
                    - How is ground-glass opacity on a chest CT typically interpreted?
                    - What are the common causes of lymphadenopathy seen in chest CT scans?
                    - How can chest CT help in the diagnosis of pulmonary embolism?
                    - What are the distinguishing features of lung nodules on a chest CT?
                    - How are lung nodules found in chest CT scans managed?
                    
                    The answers must be factually accurate and should be retrieved from the given article (but not about the case just the given factual knowledge). Do not reference specific details, cases, or research findings from the provided article. Ensure that the information in the answers is universally applicable to chest CT imaging. 
                """
                },
                {"role": "user", "content": f"{article_title}\n\n{article_content}"}
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

        all_outputs[article_title] = parsed_output

    # Open the file in write mode and use json.dump to write the dictionary to the file
    with open(f"{part_number}_questions.json", 'w') as json_file:
        json.dump(all_outputs, json_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--part_number', type=str, required=True, help='a string for the part number')
    args = parser.parse_args()
    main(args.part_number)