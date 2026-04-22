import torch
import torch.nn as nn
import json
import requests
import openai
import genotypes
import utils
from chromosomes import chromosome
import logging
import time
import random
import population



def architecture_to_text(alphas_normal, alphas_reduce, steps=4):
    """
    Convert the architectural parameters into a text description in the Genotype format for the LLM to understand.
    """
    genotype = genotypes.PRIMITIVES

    # Analysis of normal cell connection
    normal = []
    offset = 0
    for i in range(steps):
        for j in range(2 + i):
            weights = torch.softmax(alphas_normal[offset + j], dim=-1)
            top_op_idx = torch.argmax(weights).item()
            top_op = genotype[top_op_idx]
            normal.append((top_op, j))
        offset += j + 1
    normal_concat = list(range(2, 2 + steps))
    
    # Analysis of reduction cell connection
    reduce = []
    offset = 0
    for i in range(steps):
        for j in range(2 + i):
            weights = torch.softmax(alphas_reduce[offset + j], dim=-1)
            top_op_idx = torch.argmax(weights).item()
            top_op = genotype[top_op_idx]
            reduce.append((top_op, j))
        offset += j + 1
    reduce_concat = list(range(2, 2 + steps))
    
    # Build Genotype format text description
    normal_str = '[' + ', '.join(f"('{op}', {j})" for op, j in normal) + ']'
    reduce_str = '[' + ', '.join(f"('{op}', {j})" for op, j in reduce) + ']'
    normal_concat_str = str(normal_concat)
    reduce_concat_str = str(reduce_concat)
    
    # Use multiline f-string to ensure correct format
    return f"Genotype(normal={normal_str}, normal_concat={normal_concat_str},\n" \
           f"       reduce={reduce_str}, reduce_concat={reduce_concat_str})"

def text_to_architecture(architecture_text, steps=4, device=torch.device("cpu"), current_population=None):
    """
    Parse the architecture text returned by LLM into architecture parameters, and set the mutate_factor of the new individual to the average value of the population.
    """
    try:
        new_chromosome = chromosome(steps, device)
        k = new_chromosome.k

        if current_population is not None and hasattr(current_population, 'get_population') and current_population.get_population_size() > 0:
            pop_list = current_population.get_population()
            avg_mutate_factor = sum(indv.get_mutate_factor() for indv in pop_list) / len(pop_list)
            new_chromosome.set_mutate_factor(avg_mutate_factor)

        # otype format from the architecture text using multiple patterns sorted by priority level
        import re
        
        patterns = [

            r"Genotype\(normal=\[([^\]]+)\],\s*normal_concat=([^\]]+)\],\s*reduce=\[([^\]]+)\],\s*reduce_concat=([^\]]+)\]\)",

            r"Genotype\s*\(\s*normal\s*=\s*\[([^\]]+)\]\s*,[\s\n\r]*normal_concat\s*=\s*([^\]]+)\s*\][\s\n\r,]*reduce\s*=\s*\[([^\]]+)\]\s*,[\s\n\r]*reduce_concat\s*=\s*([^\]]+)\s*\)\s*",

            r"Genotype\s*\(\s*[\s\n\r]*normal\s*=\s*\[\s*([^\]]+)\s*\]\s*,[\s\n\r]*normal_concat\s*=\s*\s*([^\]]+)\s*\]\s*,[\s\n\r]*[\s\n\r]*reduce\s*=\s*\[\s*([^\]]+)\s*\]\s*,[\s\n\r]*reduce_concat\s*=\s*\s*([^\]]+)\s*\]\s*\)",

            r"Genotype\s*\(\s*[^n]*normal\s*=\s*\[([^\]]+)\][^r]*normal_concat\s*=\s*\[([^\]]+)\][^r]*reduce\s*=\s*\[([^\]]+)\][^r]*reduce_concat\s*=\s*\[([^\]]+)\][^\)]*\)",
        ]
        
        match = None
        for pattern in patterns:
            match = re.search(pattern, architecture_text, re.DOTALL)
            if match:
                break
        

        if not match:
            normalized_text = re.sub(r'\s+', ' ', architecture_text.strip())
            for pattern in patterns:
                temp_match = re.search(pattern, normalized_text)
                if temp_match:
                    match = temp_match
                    break
        
        if match:
            normal_str = match.group(1)
            normal_concat_str = match.group(2)
            reduce_str = match.group(3)
            reduce_concat_str = match.group(4)
            

            normal_pairs = parse_cell_operations(normal_str)
            reduce_pairs = parse_cell_operations(reduce_str)
            
            # Map the operation to the parameters
            new_chromosome.alphas_normal = torch.ones(k, new_chromosome.num_ops) * 0.1
            new_chromosome.alphas_reduce = torch.ones(k, new_chromosome.num_ops) * 0.1
            
            # Set the high probability of the specified operations
            set_operations_in_chromosome(new_chromosome.alphas_normal, normal_pairs, genotypes.PRIMITIVES)
            set_operations_in_chromosome(new_chromosome.alphas_reduce, reduce_pairs, genotypes.PRIMITIVES)
            
        else:
            logging.warning(f"Could not parse Genotype format from LLM response: {architecture_text[:200]}...")
            return None
        
        return new_chromosome
    except Exception as e:
        logging.error(f"Error parsing architecture from LLM: {e}")
        return None


def parse_cell_operations(cell_str):
    #Analysis of operations within the cell
    import re
    ops = re.findall(r"\('([^']*)',\s*(\d+)\)", cell_str)
    return [(op, int(node)) for op, node in ops]


def validate_genotype_format(genotype_text):
    import re
    
    if "Genotype" not in genotype_text:
        return False, "Missing 'Genotype' keyword"
    
    patterns = [

        r"Genotype\(normal=\[([^\]]+)\],\s*normal_concat=([^\]]+)\],\s*reduce=\[([^\]]+)\],\s*reduce_concat=([^\]]+)\]\)",

        r"Genotype\s*\(\s*normal\s*=\s*\[([^\]]+)\]\s*,[\s\n\r]*normal_concat\s*=\s*([^\]]+)\s*\][\s\n\r,]*reduce\s*=\s*\[([^\]]+)\]\s*,[\s\n\r]*reduce_concat\s*=\s*([^\]]+)\s*\)\s*",

        r"Genotype\s*\(\s*[\s\n\r]*normal\s*=\s*\[\s*([^\]]+)\s*\]\s*,[\s\n\r]*normal_concat\s*=\s*\s*([^\]]+)\s*\]\s*,[\s\n\r]*[\s\n\r]*reduce\s*=\s*\[\s*([^\]]+)\s*\]\s*,[\s\n\r]*reduce_concat\s*=\s*\s*([^\]]+)\s*\]\s*\)",

        r"Genotype\s*\(\s*[^n]*normal\s*=\s*\[([^\]]+)\][^r]*normal_concat\s*=\s*\[([^\]]+)\][^r]*reduce\s*=\s*\[([^\]]+)\][^r]*reduce_concat\s*=\s*\[([^\]]+)\][^\)]*\)",
    ]
    
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, genotype_text, re.DOTALL)
        if match:
            try:
                normal_str = match.group(1)
                normal_concat_str = match.group(2)
                reduce_str = match.group(3)
                reduce_concat_str = match.group(4)
                
                normal_pairs = parse_cell_operations(normal_str)
                reduce_pairs = parse_cell_operations(reduce_str)
                
                valid_primitives = set(genotypes.PRIMITIVES)
                for op, node in normal_pairs + reduce_pairs:
                    if op not in valid_primitives:
                        continue
                        
                try:
                    normal_concat_str_clean = normal_concat_str.strip().strip('[]')
                    reduce_concat_str_clean = reduce_concat_str.strip().strip('[]')
                    
                    if normal_concat_str_clean:
                        normal_concat = [int(x.strip()) for x in normal_concat_str_clean.split(',') if x.strip()]
                    else:
                        normal_concat = []
                        
                    if reduce_concat_str_clean:
                        reduce_concat = [int(x.strip()) for x in reduce_concat_str_clean.split(',') if x.strip()]
                    else:
                        reduce_concat = []
                except:
                    continue
                
                if not all(isinstance(x, int) for x in normal_concat + reduce_concat):
                    continue
                
                return True, "Valid Genotype format"
            except:
                continue
    
    normalized_text = re.sub(r'\s+', ' ', genotype_text.strip())
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, normalized_text)
        if match:
            try:
                normal_str = match.group(1)
                normal_concat_str = match.group(2)
                reduce_str = match.group(3)
                reduce_concat_str = match.group(4)
                
                normal_pairs = parse_cell_operations(normal_str)
                reduce_pairs = parse_cell_operations(reduce_str)
                
                valid_primitives = set(genotypes.PRIMITIVES)
                for op, node in normal_pairs + reduce_pairs:
                    if op not in valid_primitives:
                        continue
                        
                try:
                    normal_concat_str_clean = normal_concat_str.strip().strip('[]')
                    reduce_concat_str_clean = reduce_concat_str.strip().strip('[]')
                    
                    if normal_concat_str_clean:
                        normal_concat = [int(x.strip()) for x in normal_concat_str_clean.split(',') if x.strip()]
                    else:
                        normal_concat = []
                        
                    if reduce_concat_str_clean:
                        reduce_concat = [int(x.strip()) for x in reduce_concat_str_clean.split(',') if x.strip()]
                    else:
                        reduce_concat = []
                except:
                    continue
                
                if not all(isinstance(x, int) for x in normal_concat + reduce_concat):
                    continue
                
                return True, "Valid Genotype format"
            except:
                continue
    
    return False, "No valid Genotype pattern matched"


def set_operations_in_chromosome(alphas, operations, primitives):
    """
    Set the specified operation to the parameters of the chromosome.
    """
    for edge_idx, (op_name, node_idx) in enumerate(operations):
        if edge_idx >= alphas.shape[0]:
            break
            
        if op_name in primitives:
            op_idx = primitives.index(op_name)
            # Set the high parameter value for the specified operation, and set the other operations to lower values. The specified operation has already been set to 0.1 during initialization. Just set the specified operation to 0.9.
            alphas[edge_idx][op_idx] = 0.9

def call_llm_api(architecture_descriptions):
    """
    Call the LLM API
    """
    
    client = openai.OpenAI(
        api_key="your_api_key",
        base_url="your_api_url"
    )
    
    # First request
    #You are an expert in neural architecture search (NAS). Please conduct a comparative analysis of a set of high-performance and low-performance network architectures in the DARTS search space for the classification task of the CIFAR-10 dataset. From this, summarize the common operation combination patterns, node connection rules and local topological features of the high-performance architectures, and identify the invalid structures and inefficient configurations that lead to performance degradation. Based on this, output an architecture generation strategy.
    prompt = f"""
    You are an expert in Neural Architecture Search (NAS).  Please analyze the following group of high-performance neural network architectures within the DARTS search space for the CIFAR-10 dataset classification task, summarize their structural features, and combining your relevant expertise in this field to generate a new network architecture which performance could be better.

    {architecture_descriptions}

    IMPORTANT: Your response should contain ONLY the Genotype code in EXACTLY this format, with no additional text, explanations, or comments before or after:

    Genotype(normal=[('op', node), ('op', node), ...], normal_concat=[2, 3, 4, 5], reduce=[('op', node), ('op', node), ...], reduce_concat=[2, 3, 4, 5])

    CRITICAL REQUIREMENTS:
    1. Output ONLY the Genotype structure, nothing else
    2. Keep it on a single line OR use proper indentation
    3. Include ALL four fields: normal, normal_concat, reduce, and reduce_concat
    4. Use operations from this exact set: ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']
    5. Each operation follows format ('operation_name', node_number)
    6. The normal_concat and reduce_concat should always be [2, 3, 4, 5]
    7. Make sure the parentheses and brackets are properly closed
    8. Do NOT add any explanatory text, comments, or markdown formatting
    """
    
    max_retries = 3
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            response = client.chat.completions.create(
                model="your_model",
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                max_tokens=1024
            )
            
            llm_response = response.choices[0].message.content
            logging.info(f"[LLM Guidance] Received response from your_model API (attempt {retry_count + 1})")
            
            # validate the response format
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
                    
                    Genotype(normal=[('op', node), ('op', node), ...], normal_concat=[2, 3, 4, 5], reduce=[('op', node), ('op', node), ...], reduce_concat=[2, 3, 4, 5])
                    
                    CRITICAL REQUIREMENTS:
                    1. Output ONLY the Genotype structure, nothing else
                    2. Keep it on a single line OR use proper indentation
                    3. Include ALL four fields: normal, normal_concat, reduce, and reduce_concat
                    4. Use operations from this exact set: ['none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']
                    5. Each operation follows format ('operation_name', node_number)
                    6. The normal_concat and reduce_concat should always be [2, 3, 4, 5]
                    7. Make sure the parentheses and brackets are properly closed
                    8. Do NOT add any explanatory text, comments, or markdown formatting
                    """
                    
                    prompt = retry_prompt
                    retry_count += 1
                else:
                    logging.error(f"[LLM Guidance] Max retries reached, returning last response despite validation failure")
                    return llm_response
        except Exception as e:
            logging.error(f"Error calling your_model API: {e}")
            if retry_count < max_retries:
                retry_count += 1
                continue
            else:
                return None
    
    return None


def llm_guided_search(model, population, train_queue, valid_queue, criterion, device, args, epoch, checkpoint_list):
    # LLM-guided architecture search
    if epoch <= 29:
        return population

    logging.info(f"[LLM Guidance] Starting LLM-guided search at epoch {epoch}")
    
    llm_generated_architectures = []
    
    generated_count = 0
    max_attempts = 20
    
    while generated_count < 10 and len(llm_generated_architectures) < max_attempts:
        population.pop_sort()
        
        pop_size = population.get_population_size()
        top_k = min(10, pop_size)
        
        #bottom_k = min(10, pop_size)
        
        # Extract the architectural description of outstanding individuals
        top_individuals = population.get_population()[:top_k]
        architecture_descriptions = []
        
        for i, individual in enumerate(top_individuals):
            arch_text = architecture_to_text(
                individual.alphas_normal, 
                individual.alphas_reduce,
                steps=individual._steps
            )
            architecture_descriptions.append(f"Top {i+1} Architecture:\n{arch_text}\nFitness: {individual.get_fitness()}")
        
        # bottom_individuals = population.get_population()[-bottom_k:] if pop_size > bottom_k else population.get_population()
        # for i, individual in enumerate(bottom_individuals):
        #     arch_text = architecture_to_text(
        #         individual.alphas_normal, 
        #         individual.alphas_reduce,
        #         steps=individual._steps
        #     )
        #     architecture_descriptions.append(f"Bottom {i+1} Architecture:\n{arch_text}\nFitness: {individual.get_fitness()}")
        
        # Historical generated architectures
        historical_arch_descriptions = []
        for i, arch_info in enumerate(llm_generated_architectures):
            if arch_info['chromosome'] is not None:
                arch_text = architecture_to_text(
                    arch_info['chromosome'].alphas_normal,
                    arch_info['chromosome'].alphas_reduce,
                    steps=arch_info['chromosome']._steps
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
                    f"Previously Generated Architecture {i+1}:\n{arch_text}\n"
                    f"Loss: {objs_avg:.4f}, "
                    f"Top1 Acc: {top1_avg:.2f}%, "
                    f"Top5 Acc: {top5_avg:.2f}%\n"
                    f"Result: {'SUCCESS' if arch_info['success'] else 'FAILED'}"
                )
            else:
                # Failed
                historical_arch_descriptions.append(
                    f"Previously Generated Architecture {i+1}: Parsing failed\n"
                    f"Response: {str(arch_info['response'][:100]) if arch_info['response'] else 'None'}\n"
                    f"Result: FAILED"
                )
        
        all_descriptions = "\n".join(architecture_descriptions)
        if historical_arch_descriptions:
            all_descriptions += "\n\nPreviously Generated Architectures (avoid generating similar ones):\n" + "\n".join(historical_arch_descriptions)
        
        logging.info(f"[LLM Guidance] Sending top {top_k} architectures to LLM, attempt {len(llm_generated_architectures)+1}, generated architectures: {generated_count}/10")
        
        # LLM generate new architectures
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
                    
                    llm_generated_architectures.append({
                        'chromosome': None,
                        'response': llm_response,
                        'success': False
                    })
            except Exception as e:
                logging.error(f"[LLM Guidance] Error processing LLM response: {e}")
                # Failed
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
        if arch_info['chromosome'] is not None:
            population.get_population().append(arch_info['chromosome'])
    
    logging.info(f"[LLM Guidance] Completed. Added {generated_count} new architectures to population. New population size: {population.get_population_size()}")
    
    return population
