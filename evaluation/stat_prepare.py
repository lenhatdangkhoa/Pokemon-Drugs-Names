import json
import os

def extract_scores(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return {
        'BLEU4': [item['BLEU-4'] for item in data],
        'ROUGEL': [item['ROUGE-L'] for item in data],
        'Bertscore': [item['BertScore'] for item in data],
        'F1-cheXbert': [item['F1-cheXbert'] for item in data],
        'F1-RadGraph': [item['F1-RadGraph'] for item in data]
    }

def main():
    # Paths to the JSON files
    # chexbert_file = './llama3/hidden_test_chexbert_output_32_llama3_instance_scores.json'
    # laymen_file = './llama3/hidden_test_laymen_output_32_llama3_instance_scores.json'
    # chexbert_file = f'./llama3/test_chexbert_output_32_llama3_instance_scores.json'
    # laymen_file = f'./llama3/test_laymen_output_32_llama3_instance_scores.json'
    # chexbert_file = f'./llama3/mimic/test_chexbert_output_16_llama3_instance_scores.json'
    # laymen_file = f'./llama3/mimic/test_laymen_output_12_llama3_instance_scores.json'

    # Configuration parameters
    model_name = 'ministral' # ministral, llama3, gemma 
    dataset_name = 'third'  # Options: hidden_test, test, mimic, third
    
    # Create output directory
    output_folder = f'stat/{model_name}_{dataset_name}'
    os.makedirs(output_folder, exist_ok=True)
    
    # Define file paths based on model and dataset
    if dataset_name == 'test':
        chexbert_file = f'./{model_name}/test_multimodal_32_instance_scores.json'
        laymen_file = f'./{model_name}/test_laymen_32_instance_scores.json'
    elif dataset_name == 'hidden_test':
        chexbert_file = f'./{model_name}/hidden_test_multimodal_32_instance_scores.json'
        laymen_file = f'./{model_name}/hidden_test_laymen_32_instance_scores.json'
    elif dataset_name == 'mimic':
        chexbert_file = f'./mimic/{model_name}/test_multimodal_18_instance_scores.json'
        laymen_file = f'./mimic/{model_name}/test_laymen_18_instance_scores.json'
    elif dataset_name == 'third':
        chexbert_file = f'./third/{model_name}/test_multimodal_24_instance_scores.json'
        laymen_file = f'./third/{model_name}/test_laymen_24_instance_scores.json'


    # Extract scores
    chexbert_scores = extract_scores(chexbert_file)
    laymen_scores = extract_scores(laymen_file)

    # Ensure the stat folder exists
    os.makedirs('stat', exist_ok=True)

    # Save to separate text files in the stat folder
    metrics = ['BLEU4', 'ROUGEL', 'Bertscore', 'F1-cheXbert', 'F1-RadGraph']
    
    for metric in metrics:
        chexbert_output_file = f'{output_folder}/{metric.lower()}_scores_chexbert.txt'
        laymen_output_file = f'{output_folder}/{metric.lower()}_scores_laymen.txt'

        with open(chexbert_output_file, 'w') as f:
            for score in chexbert_scores[metric]:
                f.write(f"{score}\n")

        with open(laymen_output_file, 'w') as f:
            for score in laymen_scores[metric]:
                f.write(f"{score}\n")

        print(f"{metric} scores have been saved to {chexbert_output_file} and {laymen_output_file}")

if __name__ == "__main__":
    main()
