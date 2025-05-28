# This is an example of data processing for webshop environment
# You can change the data path according to different environments (e.g., alfworld, virtualhome)

import json
with open('prm/exploration_inference_results_webshop.json', 'r') as f:
    prm_annotations = json.load(f)
    
# Reconstruct turn values
import numpy as np
import random

# Create dictionary to store prediction values for each id's sample_ids
id_predictions = {}
for annotation in prm_annotations:
    id = annotation['id']
    if id not in id_predictions:
        id_predictions[id] = []
    id_predictions[id].append(annotation['agent_final_reward'])

# Update turn_values for each sample
for annotation in prm_annotations:
    id = annotation['id']
    
    # Add adjustment value to each turn value
    each_signal = annotation['agent_final_reward']
    
    each_id_all_values = id_predictions[id]
    # Remove only one current each_signal value from the list
    filtered_values = each_id_all_values.copy()
    filtered_values.remove(each_signal)
    # If filtered list is empty, use original value
    if len(filtered_values) == 0:
        each_value = each_signal
    else:
        # Recalculate mean and standard deviation
        new_mean = np.mean(filtered_values)
        new_std = np.std(filtered_values) if len(filtered_values) > 1 else 1.0
        if new_std == 0:
            new_std = 1.0
        # Calculate standardized value using new mean and std    
        each_value = each_signal - new_mean

    annotation['turn_values'] = [each_value for v in annotation['turn_values']]

    
template_human = "<|start_header_id|>user<|end_header_id|>\n\n{system_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
template_gpt = "{system_message}<|eot_id|>"


all_prompts = []
all_responses = []
all_rewards = []

# Just check the IDs
all_ids = []
all_final_reward = []

for i in range(len(prm_annotations)):
    each_prompt = []
    each_response = []
    each_reward = []
    
    each_conversations = prm_annotations[i]['conversations']
    assert len(each_conversations) % 2 == 0
    
    if len(each_conversations)//2 != (len(prm_annotations[i]['turn_values'])):
        # print('continue')
        continue
    
    for j in range(len(each_conversations)//2):
        each_prompt.append(template_human.format(system_message=each_conversations[2*j]['value']))
        each_response.append(template_gpt.format(system_message=each_conversations[2*j+1]['value']))
        # each_reward.append(prm_annotations[i]['turn_values'][j])
        
    each_reward = prm_annotations[i]['turn_values']
    
    # Add grounding values
    grounding_values = []
    for j in range(1, len(each_conversations)//2):
        if 'Nothing happens' not in each_conversations[2*j]['value']:
            grounding_values.append(0.5)
            # print('nothing happens')
        else:
            grounding_values.append(0.0)
            # print('nothing happens')
            # print(each_conversations[2*j+1]['value'])
    grounding_values.append(0.5)
            
    print('grounding_values:', grounding_values)
    each_reward = [each_reward[j] + grounding_values[j] for j in range(len(each_reward))]
    
    if len(each_reward) != len(grounding_values):
        print('length not match')
        print(len(each_reward), len(grounding_values))
        continue
    
    all_prompts.append(each_prompt)
    all_responses.append(each_response)
    all_rewards.append(each_reward)
    
    all_ids.append(prm_annotations[i]['id'])
    all_final_reward.append(prm_annotations[i]['agent_final_reward'])
    

# Set random seed for reproducibility
random.seed(42)

# Get total number of samples
total_samples = len(all_prompts)

# Select indices for 8000 samples
if total_samples > 8000:
    selected_indices = random.sample(range(total_samples), 4200)
else:
    selected_indices = list(range(total_samples))

# Get data for selected indices
sampled_prompts = [all_prompts[i] for i in selected_indices]
sampled_responses = [all_responses[i] for i in selected_indices]
sampled_rewards = [all_rewards[i] for i in selected_indices]
# sampled_ids = [all_ids[i] for i in selected_indices]
print(f"Number of samples after sampling: {len(sampled_prompts)}")


# Filter out samples longer than 25
filtered_indices = [i for i in selected_indices if len(all_rewards[i]) < 35]

# Get data for filtered indices
sampled_prompts = [all_prompts[i] for i in filtered_indices]
sampled_responses = [all_responses[i] for i in filtered_indices] 
sampled_rewards = [all_rewards[i] for i in filtered_indices]
# sampled_ids = [all_ids[i] for i in filtered_indices]
print(f"Number of samples after filtering: {len(sampled_prompts)}")


import json

# Create dictionary to store data
data_dict = {
    'prompt': sampled_prompts[:1596],
    'response': sampled_responses[:1596], 
    'reward': sampled_rewards[:1596]
}

# Save data to json file
with open('prm/sampled_data_rl_training_webshop.json', 'w', encoding='utf-8') as f:
    json.dump(data_dict, f, ensure_ascii=False, indent=2)