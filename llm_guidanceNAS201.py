import torch
import torch.nn as nn
import json
import requests
import openai
import genotypesNAS201
import utils
from chromosomesNAS201 import chromosome
import logging
import time
import random
import populationNAS201
from cell_operationsNAS201 import NAS_BENCH_201


def architecture_to_text(alphas_normal, model=None):
    alphas_normal = alphas_normal.to(model.get_alphas()[0].device)
    model.update_alphas(alphas_normal)
    assert model.check_alphas(alphas_normal)
    arch_text = model.genotype().tostr()
    return arch_text


def text_to_architecture(architecture_text, device=torch.device("cpu"), current_population=None, search_space=NAS_BENCH_201):
    try:
        # Calculate the average mutate_factor
        if current_population is not None:
            num_edges = current_population.get_num_edges()
            new_chromosome = chromosome(num_edges, device, search_space)
            pop_list = current_population.get_population()
            avg_mutate_factor = sum(indv.get_mutate_factor() for indv in pop_list) / len(pop_list)
            new_chromosome.set_mutate_factor(avg_mutate_factor)
        else:
            num_edges = 6
            new_chromosome = chromosome(num_edges, device, search_space)
            new_chromosome.set_mutate_factor(0.5)

        clean_text = architecture_text.strip()
        parsed_successfully = False
        if "|" in clean_text and "+" in clean_text and "~" in clean_text:
            node_parts = clean_text.split("+")

            cleaned_parts = []
            for part in node_parts:
                part = part.strip()
                if part.startswith("|"):
                    part = part[1:]
                if part.endswith("|"):
                    part = part[:-1]
                if part:
                    cleaned_parts.append(part)

            connections = []
            for part in cleaned_parts:
                if part:
                    edges = part.split("|")
                    for edge in edges:
                        edge = edge.strip()
                        if "~" in edge:
                            parts = edge.split("~")
                            if len(parts) == 2:
                                op_name = parts[0].strip()
                                node_idx = int(parts[1].strip())
                                connections.append((op_name, node_idx))

            if connections:
                new_chromosome.alphas_normal = torch.ones(num_edges, len(search_space)) * (1.0 / len(search_space))
                new_chromosome.alphas_normal = new_chromosome.alphas_normal.to(device)

                # Set the high probability of the specified operation
                for edge_idx, (op_name, node_idx) in enumerate(connections):
                    if edge_idx < num_edges and op_name in search_space:
                        op_idx = search_space.index(op_name)
                        new_chromosome.alphas_normal[edge_idx].fill_(0.1)
                        new_chromosome.alphas_normal[edge_idx][op_idx] = 0.9

                new_chromosome.arch_parameters = [new_chromosome.alphas_normal]
                parsed_successfully = True

        if not parsed_successfully:
            # Failed
            logging.warning(
                f"Could not parse Genotype format from LLM response: {architecture_text[:200]}...")
            return None

        return new_chromosome
    except Exception as e:
        logging.error(f"Error parsing architecture from LLM: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_genotype_format(genotype_text):
    import re

    if "Genotype" not in genotype_text or "=" not in genotype_text:
        return False, "Missing 'Genotype =' format"

    try:
        genotype_part = genotype_text.split("=")[1].strip()
    except:
        return False, "Invalid format: missing '=' symbol"

    # Check if the format is correct: "|op~node|+|op~node|op~node|+|op~node|op~node|op~node|"
    pattern = r'^\|([a-z_0-9]+~\d{1})\|\+\|([a-z_0-9]+~\d{1})\|([a-z_0-9]+~\d{1})\|\+\|([a-z_0-9]+~\d{1})\|([a-z_0-9]+~\d{1})\|([a-z_0-9]+~\d{1})\|$'
    match = re.match(pattern, genotype_part)
    if not match:
        return False, "Format doesn't match required pattern: '|op~node|+|op~node|op~node|+|op~node|op~node|op~node|'"

    operations = []
    for i in range(1, 7):
        op_node = match.group(i)
        op, node = op_node.split('~')
        operations.append((op, int(node)))

    valid_ops = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    for op, node in operations:
        if op not in valid_ops:
            return False, f"Invalid operation '{op}' not in NAS-Bench-201 search space"

    expected_connections = [
        (operations[0][1], 0),
        (operations[1][1], 0),
        (operations[2][1], 1),
        (operations[3][1], 0),
        (operations[4][1], 1),
        (operations[5][1], 2),
    ]

    for i, (actual_node, expected_node) in enumerate(expected_connections):
        if actual_node != expected_node:
            return False, f"Invalid connection: operation {i + 1} connects to node {actual_node}, expected node {expected_node}"

    return True, "Valid Genotype format"


def call_llm_api(architecture_descriptions):
    # LLM API
    client = openai.OpenAI(
        api_key="YOUR_API_KEY_HERE",
        base_url="YOUR_API_BASE_URL_HERE"
    )

    #You are an expert in neural architecture search (NAS). Please conduct a comparative analysis of a set of high-performance and low-performance network architectures in the NAS-Bench-201 search space for the classification task of the CIFAR-10 dataset. From this, summarize the common operation combination patterns, node connection rules and local topological features of the high-performance architectures, and identify the invalid structures and inefficient configurations that lead to performance degradation. Based on this, output an architecture generation strategy.
    prompt = f"""
    You are an expert in Neural Architecture Search (NAS).  Please analyze the following group of high-performance neural network architectures within the NAS-Bench-201 search space for the CIFAR-10 dataset classification task, summarize their structural features, and combining your relevant expertise in this field to generate a new network architecture which performance could be better.
    
    {architecture_descriptions}

    IMPORTANT: Your response should only include the genotype code within the NAS-Bench-201 search space, and must be presented strictly in the following format without adding any additional text, explanations, or comments:

    Genotype = |op~node|+|op~node|op~node|+|op~node|op~node|op~node|

    CRITICAL REQUIREMENTS:
    1. Output ONLY the Genotype structure, nothing else
    2. Keep it on a single line OR use proper indentation
    3. Include ALL three fields:  the word "Genotype", the "=" symbol, and the complete genotype structure.And the genotype structure must follow the aforementioned format; no edge can be omitted in the operation
    4. Use operations from this exact set: ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    5. Each operation follows format |operation~node|
    6. Make sure that all the "|" symbols have been correctly closed
    7. Do NOT add any explanatory text, comments, or markdown formatting
    """

    max_retries = 3
    retry_count = 0

    while retry_count <= max_retries:
        try:
            response = client.chat.completions.create(
                model="YOUR_MODEL_NAME_HERE",
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                max_tokens=1024
            )
            
            llm_response = response.choices[0].message.content
            logging.info(f"[LLM Guidance] Received response from LLM API (attempt {retry_count + 1})")

            is_valid, message = validate_genotype_format(llm_response)
            if is_valid:
                logging.info(f"[LLM Guidance] Genotype format validation passed")
                return llm_response
            else:
                logging.warning(f"[LLM Guidance] Genotype format validation failed: {message}")

                if retry_count < max_retries:
                    retry_prompt = f"""
                    Your previous response was not in the correct format. Please provide ONLY the Genotype code in the exact format specified.

                    The error was: {message}

                    Previous response: {llm_response}

                    Please respond with ONLY the Genotype code in EXACTLY this format, nothing else:

                    Genotype = |op~node|+|op~node|op~node|+|op~node|op~node|op~node|

                    CRITICAL REQUIREMENTS:
                    1. Output ONLY the Genotype structure, nothing else
                    2. Keep it on a single line OR use proper indentation
                    3. Include ALL three fields:  the word "Genotype", the "=" symbol, and the complete genotype structure.And the genotype structure must follow the aforementioned format; no edge can be omitted in the operation
                    4. Use operations from this exact set: ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
                    5. Each operation follows format |operation~node|
                    6. Make sure that all the "|" symbols have been correctly closed
                    7. Do NOT add any explanatory text, comments, or markdown formatting
                    """

                    prompt = retry_prompt
                    retry_count += 1
                else:
                    logging.error(
                        f"[LLM Guidance] Max retries reached, returning last response despite validation failure")
                    return llm_response
        except Exception as e:
            logging.error(f"Error calling DeepSeek API: {e}")
            if retry_count < max_retries:
                retry_count += 1
                continue
            else:
                return None
    return None


def llm_guided_search(model, population, device, epoch, checkpoint_list):
    #LLM-guided search
    if epoch < 15:
        return population

    logging.info(f"[LLM Guidance] Starting LLM-guided search at epoch {epoch}")

    llm_generated_architectures = []

    generated_count = 0
    max_attempts = 20
    population.pop_sort()

    while generated_count < 10 and len(llm_generated_architectures) < max_attempts:

        pop_size = population.get_population_size()
        top_k = min(10, pop_size)
        #bottom_k = min(10, pop_size)

        top_individuals = population.get_population()[:top_k]
        architecture_descriptions = []

        for i, individual in enumerate(top_individuals):
            arch_text = architecture_to_text(individual.arch_parameters[0], model)
            architecture_descriptions.append(
                f"Top {i + 1} Architecture:\n{arch_text}\nFitness: {individual.get_fitness()}")

        # bottom_individuals = population.get_population()[-bottom_k:] if pop_size > bottom_k else population.get_population()
        # for i, individual in enumerate(bottom_individuals):
        #     arch_text = architecture_to_text(individual.arch_parameters[0], model)
        #     architecture_descriptions.append(
        #         f"Bottom {i + 1} Architecture:\n{arch_text}\nFitness: {individual.get_fitness()}")

        # History
        historical_arch_descriptions = []
        for i, arch_info in enumerate(llm_generated_architectures):
            if arch_info['chromosome'] is not None and arch_info['success']:
                arch_text = architecture_to_text(
                    arch_info['chromosome'].alphas_normal,
                    model
                )
                objs_avg = arch_info['chromosome'].objs.avg
                top1_avg = arch_info['chromosome'].top1.avg
                top5_avg = arch_info['chromosome'].top5.avg

                if torch.is_tensor(objs_avg):
                    objs_avg = objs_avg.item()
                if torch.is_tensor(top1_avg):
                    top1_avg = top1_avg.item()
                if torch.is_tensor(top5_avg):
                    top5_avg = top5_avg.item()

                historical_arch_descriptions.append(
                    f"Previously Generated Architecture {i + 1}:\n{arch_text}\n"
                    f"Loss: {objs_avg:.4f}, "
                    f"Top1 Acc: {top1_avg:.2f}%, "
                    f"Top5 Acc: {top5_avg:.2f}%\n"
                    f"Result: {'SUCCESS' if arch_info['success'] else 'FAILED'}"
                )
            else:
                historical_arch_descriptions.append(
                    f"Previously Generated Architecture {i + 1}: Parsing failed\n"
                    f"Response: {str(arch_info['response'][:100]) if arch_info['response'] else 'None'}\n"
                    f"Result: FAILED"
                )

        all_descriptions = "\n".join(architecture_descriptions)
        if historical_arch_descriptions:
            all_descriptions += "\n\nPreviously Generated Architectures (avoid generating similar ones):\n" + "\n".join(
                historical_arch_descriptions)

        logging.info(
            f"[LLM Guidance] Sending top {top_k} architectures to LLM, attempt {len(llm_generated_architectures) + 1}, generated architectures: {generated_count}/10")

        # Call the LLM API
        llm_response = call_llm_api(all_descriptions)
        #llm_response = call_llm_api_generate(llm_response)
       
        if llm_response:
            logging.info("[LLM Guidance] Received response from LLM")

            try:
                new_chromosome = text_to_architecture(llm_response, device=device, current_population=population)

                if new_chromosome is not None:
                    llm_generated_architectures.append({
                        'chromosome': new_chromosome,
                        'response': llm_response,
                        'success': True
                    })

                    generated_count += 1
                    logging.info(f"[LLM Guidance] Successfully generated new architecture #{generated_count}")
                else:
                    logging.warning(f"[LLM Guidance] Failed to parse chromosome from LLM response")

                    # Failed
                    llm_generated_architectures.append({
                        'chromosome': None,
                        'response': llm_response,
                        'success': False
                    })
            except Exception as e:
                logging.error(f"[LLM Guidance] Error processing LLM response: {e}")

                llm_generated_architectures.append({
                    'chromosome': None,
                    'response': llm_response,
                    'success': False
                })
        else:
            logging.warning("[LLM Guidance] Failed to get response from LLM")

            llm_generated_architectures.append({
                'chromosome': None,
                'response': None,
                'success': False
            })

    # Add the newly generated architectures to the end of the population
    for arch_info in llm_generated_architectures:
        if arch_info['chromosome'] is not None and arch_info['success']:
            population.get_population().append(arch_info['chromosome'])

    logging.info(
        f"[LLM Guidance] Completed. Added {generated_count} new architectures to population. New population size: {population.get_population_size()}")

    return population