import argparse
import json
import os
import re
import time
import json_repair
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from tqdm import tqdm

def parse_response(response):
    domain_pattern = r'(?<=\[START_DOMAIN\])(.*?)(?=\[END_DOMAIN\])'
    scenario_pattern = r'(?<=\[START_SCENARIO\])(.*?)(?=\[END_SCENARIO\])'
    schema_pattern = r'(?<=\[START_DATABASE_SCHEMA\])(.*?)(?=\[END_DATABASE_SCHEMA\])'

    try:
        domain_match = re.search(domain_pattern, response, re.DOTALL)
        domain = domain_match.group(0).strip() if domain_match else None

        scenario_match = re.search(scenario_pattern, response, re.DOTALL)
        scenario = scenario_match.group(0).strip() if scenario_match else None

        schema_match = re.search(schema_pattern, response, re.DOTALL)
        schema = schema_match.group(0).strip() if schema_match else None
        schema_dict = json_repair.loads(schema)
        schema = json.dumps(schema_dict, indent=2, ensure_ascii=False)

        return domain, scenario, schema
    except Exception as e:
        print(response)
        print("Parsing Exception:", str(e))
        return None, None, None

def llm_inference(model, prompts):
    '''
    This function leverages a large language model (LLM) to generate responses for a given list of prompts.
    You can integrate your preferred LLM within this function.

    Args:
        model: The LLM to be used for inference.
        prompts: A list of prompts for which the LLM will generate responses.

    Returns:
        A list of dictionaries containing the prompt, the generated response, and extracted components 
        (domain, scenario, schema) from the response. Invalid responses are filtered out.
    '''

    # Generate responses using the LLM (each prompt corresponds to one response)
    responses = []

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to("cuda:0")
    model.eval()
    model = torch.compile(model)

    batch_size = 8 

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        print(f"Generating schemas {i+1} to {i+len(batch_prompts)} of {len(prompts)}")


        # Tokenize the batch
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, padding_side ='left', truncation=True).to(model.device)

        # Generate output in batch
        output_ids = model.generate(
            **inputs,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=5000, temperature=0.3, repetition_penalty=1.1
        )

        # Decode and remove input portion from each sample
        for j in range(len(batch_prompts)):
            input_len = len(inputs.input_ids[j])
            output = output_ids[j][input_len:]
            response = tokenizer.decode(output, skip_special_tokens=True)
            responses.append(response)

    # Initialize a list to store the processed results
    results = []

    # Iterate over prompts and their corresponding responses
    for prompt, response in zip(prompts, responses):
        # Parse the response to extract domain, scenario, and schema
        domain, scenario, schema = parse_response(response)

        # Filter out invalid responses where any component is missing
        if domain is None or scenario is None or schema is None:
            continue

        # Append valid results to the list
        results.append({
            "prompt": prompt,
            "generated_content": {
                "response": response,
                "domain": domain,
                "scenario": scenario,
                "schema": schema
            }
        })

    return results

if __name__ == '__main__':
    torch.cuda.set_device(3)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str)
    args = parser.parse_args()
    
    print(args)

    prompts = json.load(open("./prompts/prompts_test_two.json"))

    output_file = "./results/schema_synthesis.json"
    results = llm_inference(args.model, prompts)

    with open(output_file, "w", encoding = "utf-8") as f:
        f.write(json.dumps(results, indent = 2, ensure_ascii = False))
