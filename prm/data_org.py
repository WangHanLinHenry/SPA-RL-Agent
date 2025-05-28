# This is an example of data processing for webshop environment
# You can change the data path according to different environments (e.g., alfworld, virtualhome)
import os
import json
import glob

# Define target directory
explore_dir = "exploration/webshop/exploration_outputs/explore"

# Use glob to collect all json files
json_files = glob.glob(os.path.join(explore_dir, "*.json"))

print(f"Found {len(json_files)} json files")

# Read contents of all json files
all_data = []
for json_file in json_files:
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.append(data)
        print(f"Successfully read: {json_file}")
    except Exception as e:
        print(f"Error reading {json_file}: {str(e)}")

print(f"\nCollected data from {len(all_data)} json files")

# Create new data structure to store truncated data
truncated_data = []

# Iterate through all data
for data_list in all_data:
    truncated_list = []
    for item in data_list:
        # Copy original data item
        new_item = item.copy()
        # Truncate conversations, keep first 2 and after 12th
        new_item['agent_conversations'] = (
            item['agent_conversations'][:2] + 
            item['agent_conversations'][10:]
        )
        truncated_list.append(new_item)
    truncated_data.append(truncated_list)

our_data = []
for data_list in truncated_data:
    our_data.extend(data_list)

# Save processed data: 1. Keep original items (especially agent_conversations and agent_final_reward) 2.
saved_data = []
for each_item in our_data:
    conversation = each_item['agent_conversations']
    new_conversation = []
    
    for i in range(len(conversation)):
        if conversation[i]['role'] == 'user':
            new_conversation.append({'from': 'human', 'value': conversation[i]['content'].strip()})
        else:
            new_conversation.append({'from': 'gpt', 'value': conversation[i]['content'].strip()})
    
    if new_conversation[-1]['from'] == 'human':
        new_conversation = new_conversation[:-1]
        
    saved_data.append({
        'conversations': new_conversation,
        'agent_final_reward': each_item['agent_final_reward'],
        'id': each_item['id'],
        'iteration': each_item['iteration'],
        # 'game_file': each_item['game_file'],
        'success': each_item['success'],
    })
    
# Save data to json file
with open("exploration/webshop/exploration_outputs/exploration.json", "w") as f:
    json.dump(saved_data, f, indent=4)
print("Data has been saved to json file")


with open("exploration/webshop/exploration_outputs/exploration_tiny.json", "w") as f:
    json.dump(saved_data[:100], f, indent=4)
print("Data has been saved to json file")