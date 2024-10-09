import argparse
import transformers
import torch
import json
import tqdm

def evaluate( file1_path, file2_path, output_file_path):
    pipeline = transformers.pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    # Read the first JSON file into a list
    with open(file1_path, 'r') as file1:
        outputs = json.load(file1)

    # Read the second JSON file into a list
    with open(file2_path, 'r') as file2:
        ground_truth = json.load(file2)

    outputs_all = []
    for i in tqdm.tqdm(range(len(ground_truth))):
        conv_ground = ground_truth[i]["conversations"]
        conv_out = outputs[i]["conversations_out"]
        outputs_list = []
        for k in range(len(conv_ground)):

            if conv_ground[k]["from"] == "human":
                try:
                    if conv_ground[k+1]["from"] == "gpt":
                        question = conv_ground[k]["value"]
                        type = conv_ground[k]["type"]
                        if type == "free_response" or type == "description" or type == "conversation":
                            type_print = "Long answer"
                        elif type == "multiple_choice":
                            type_print = "Multiple choice"
                        elif type == "report_generation":
                            type_print = "Report generation"
                        else:
                            type_print = "Short answer"

                        real_answer = conv_ground[k+1]["value"]
                        output = conv_out[k//2]["answer"]
                        question = question.replace("<image>\n","").replace("<report_generation>","").replace("<short_answer>","").replace("<long_answer>","").replace("<multiple_choice>","")
                        output = output.replace("<s>","").replace("</s>","").replace("","")

                        messages = [
                            {"role": "system", "content": """
                                Role: You are an expert medical evaluator specializing in the assessment of clinical accuracy for AI-generated reports and answers derived from 3D chest CT volumes. Your task is to compare the AI-generated answers with the provided ground truth answers and assign a clinical accuracy score based on specific evaluation criteria. Your assessment should focus on precision, relevance, and clinical soundness.

                                Scoring:

                                    •	10/10: The generated response is fully aligned with the ground truth—completely accurate, relevant, comprehensive, clear, and clinically sound.
                                    •	7-9/10: The generated response is mostly accurate, with minor discrepancies or omissions that do not significantly affect clinical interpretation.
                                    •	4-6/10: The generated response contains noticeable errors or omissions that could impact clinical understanding, though some correct information is present.
                                    •	1-3/10: The generated response has significant inaccuracies, irrelevant content, or lacks essential information, potentially leading to clinical misunderstanding.
                                    •	0/10: The generated response is completely incorrect or irrelevant, offering no clinical value.

                                Question Types:

                                For each evaluation, you will be provided with the question type, question, ground truth answer, and generated answer.

                                    •	Report Generation: Requires a detailed narrative report summarizing findings from the 3D chest CT volume, covering all relevant clinical observations, including pathology, location, and significant findings. The report should follow the CT report format. Consider the formatting of the output for this question in addition to clinical accuracy. Non-existent pathologies are not necessarily required to be mentioned in the generated report. Do not deduct points if this is the case (e.g., “no pleural effusion” might not be in the output, but if “pleural effusion” is present, it must be included). Deduct points if the model hallucinates pathologies that are not in the ground truth.
                                    •	Long Answer Questions: Require comprehensive, detailed responses that thoroughly address the question, including necessary explanations or justifications.
                                    •	Short Answer Questions: Require brief, direct answers addressing the question with essential information, such as identifying pathology or confirming a diagnosis.
                                    •	Multiple Choice Questions: Require the selection of the most clinically accurate option from predefined choices, matching the ground truth answer. There should be only one correct answer for this question type. If more than one answer is provided, assign a very low score. If the answer is incorrect, assign a very low score as well.

                                Instructions:

                                Please review the question type, question, ground truth answer, and generated answer. Assign a score out of 10 based on the evaluation criteria above. Provide only the score without additional commentary or explanation.

                            """},

                            {"role": "user", "content":
                                f"""
                            Question type: {type_print}

                            Question: {question}

                            Ground truth answer: {real_answer}

                            Generated answer: {output}

                            """
                             },
                        ]

                        model_out = pipeline(
                            messages,
                            max_new_tokens=256,
                            temperature=0,
                            do_sample=False
                        )
                        outputs_list.append({"question": question, "type": type, "score": model_out[0]["generated_text"][-1]})
                except Exception as e:
                    print(e)
        outputs_all.append({"id":  ground_truth[i]["id"], "image": ground_truth[i]["image"], "scores": outputs_list})

    # Save the outputs_all to a JSON file
    with open(output_file_path, 'w') as output_file:
        json.dump(outputs_all, output_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate AI-generated answers against ground truth.')
    parser.add_argument('file1_path', type=str, help='Path to the first JSON file (output).')
    parser.add_argument('file2_path', type=str, help='Path to the second JSON file (ground truth).')
    parser.add_argument('output_file_path', type=str, help='Path where the output JSON file should be saved.')

    args = parser.parse_args()

    evaluate(args.file1_path, args.file2_path, args.output_file_path)