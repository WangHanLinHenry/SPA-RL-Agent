import os

os.environ["OPENAI_API_KEY"] = ""

import json
import logging
import pathlib
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import Fore
from copy import deepcopy

from glob import glob

import eval_agent.tasks as tasks
import eval_agent.agents as agents
import eval_agent.envs as envs
from eval_agent.utils.datatypes import State
import time

import numpy as np


prompt_for_environment ={
    "webshop": [{
            "from": "human",
            "value": "You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nAn action should be of the following structure:\nsearch[keywords]\nclick[value]\nIf the action is not valid, perform nothing.\nKeywords in search are up to you, but the value in click must be a value in the list of available actions.\nRemember that your keywords in search should be carefully designed.\nYour response should use the following format:\n\nThought: I think ...\nAction: click[something]"
        },
        {
            "from": "gpt",
            "value": "OK"
        },
        {
            "from": "human",
            "value": "Observation:\nWebShop [SEP] Instruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Search"
        },
        {
            "from": "gpt",
            "value": "Thought: I think I should use the search bar to look for the product I need.\nAction: search[l'eau d'issey 6.76 fl oz bottle price < 100.00]"
        },
        {
            "from": "human",
            "value": "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] Page 1 (Total results: 50) [SEP] Next > [SEP] B000VOHH8I [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] $64.98 [SEP] B000MJZOPK [SEP] L'eau d'Issey by Issey Miyake for Women 3.3 oz Eau de Toilette Spray [SEP] $49.98 [SEP] B0012S249E [SEP] L'eau D'issey By Issey Miyake For Women. Shower Cream 6.7-Ounces [SEP] $31.36 [SEP] B01H8PGKZS [SEP] L'eau D'Issey FOR MEN by Issey Miyake - 6.7 oz EDT Spray [SEP] $67.97 [SEP] B00G3C8FHE [SEP] L'Eau d'Issey pour Homme - Eau de Toilette 4.2 fl oz [SEP] $51.25 [SEP] B000R94HRG [SEP] Issey Miyake L'Eau D'Issey Pour Homme Eau De Toilette Natural Spray [SEP] $44.99 [SEP] B000C214CO [SEP] Issey Miyake L'eau D'issey Eau de Toilette Spray for Men, 4.2 Fl Oz [SEP] $53.99 [SEP] B0018SBRDC [SEP] Issey Miyake L'eau d'Issey for Women EDT, White, 0.84 Fl Oz [SEP] $27.04 [SEP] B000XEAZ9Y [SEP] L'eau De Issey By Issey Miyake For Men. Eau De Toilette Spray 6.7 Fl Oz [SEP] $67.08 [SEP] B079HZR2RX [SEP] L'eau d'Issey Pure by Issey Miyake for Women 3.0 oz Nectar de Parfum Spray [SEP] $71.49"
        },
        {
            "from": "gpt",
            "value": "Thought: I think I should click on the product I need, which is B000VOHH8I.\nAction: click[B000VOHH8I]"
        },
        {
            "from": "human",
            "value": "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
        },
        {
            "from": "gpt",
            "value": "Thought: I think I should click on the '6.76 fl oz (pack of 1)' option to select the size I need.\nAction: click[6.76 fl oz (pack of 1)]"
        },
        {
            "from": "human",
            "value": "Observation:\nInstruction: [SEP] i need a long lasting 6.76 fl oz bottle of l'eau d'issey, and price lower than 100.00 dollars [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] 2.5 fl oz [SEP] 6.76 fl oz (pack of 1) [SEP] L'eau D'issey By Issey Miyake for MenEau De Toilette Spray, 6.7 Fl Oz Bottle [SEP] Price: $64.98 [SEP] Rating: N.A. [SEP] Description [SEP] Features [SEP] Reviews [SEP] Buy Now"
        },
        {
            "from": "gpt",
            "value": "Thought: I think I should click on the 'Buy Now' button to purchase the product.\nAction: click[Buy Now]"
        }],
    "alfworld": [{
            "from": "human",
            "value": "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"Thought: your thoughts.\\nAction: your next action\".\n\nThe available actions are:\n1. go to {recep}\n2. task {obj} from {recep}\n3. put {obj} in/on {recep}\n4. open {recep}\n5. close {recep}\n6. toggle {obj} {recep}\n7. clean {obj} with {recep}\n8. heat {obj} with {recep}\n9. cool {obj} with {recep}\nwhere {obj} and {recep} correspond to objects and receptacles.\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>"
        },
        {
            "from": "gpt",
            "value": "OK"
        },
        {
            "from": "human",
            "value": "You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.\n\nYour task is to: put a toiletpaper in toiletpaperhanger."
        },
        {
            "from": "gpt",
            "value": "Thought: I need to place a toilet paper roll on the toilet paper hanger. First, I should check if there's already a roll on the hanger or if it's empty.\nAction: go to toiletpaperhanger 1"
        },
        {
            "from": "human",
            "value": "Observation: On the toiletpaperhanger 1, you see nothing."
        },
        {
            "from": "gpt",
            "value": "Thought: The toilet paper hanger is empty, so I need to find a new roll of toilet paper to put on it.\nAction: go to toilet 1"
        },
        {
            "from": "human",
            "value": "Observation: On the toilet 1, you see a soapbottle 1, and a toiletpaper 1."
        },
        {
            "from": "gpt",
            "value": "Thought: I've found a toilet paper roll on the toilet. I should pick it up so I can place it on the hanger.\nAction: take toiletpaper 1 from toilet 1"
        },
        {
            "from": "human",
            "value": "Observation: You pick up the toiletpaper 1 from the toilet 1."
        },
        {
            "from": "gpt",
            "value": "Thought: Now that I have the toilet paper roll, I need to return to the toilet paper hanger to install it.\nAction: go to toiletpaperhanger 1"
        },
        {
            "from": "human",
            "value": "Observation: On the toiletpaperhanger 1, you see nothing."
        },
        {
            "from": "gpt",
            "value": "Thought: I'm back at the toilet paper hanger with the roll in hand. It's time to put the toilet paper on the hanger.\nAction: put toiletpaper 1 in/on toiletpaperhanger 1"
        }
    ],
    "intercode_sql":[
        {
            "from": "human",
            "value": "You are a helpful assistant assigned with the task of problem-solving. To achieve this, you will interact with a MySQL Database system using SQL queries to answer a question.\nAt each turn, you should first provide your step-by-step thinking for solving the task. Your thought process should start with \"Thought: \", for example: Thought: I should write a SQL query that gets the average GNP and total population from nations whose government is US territory.\nAfter that, you have two options:\n1) Interact with a mysql programming environment and receive the corresponding output. Your code should start with \"Action: \" and should be surrounded with ```sql ``` tag, for example: Action: \\n```sql\\nSELECT AVG(GNP), SUM(population) FROM nations WHERE government = 'US Territory'\\n```\n2) Directly submit the result, for example: Action: submit.\nYou should use this format: \"Thought: your thought\\nAction: \\n```sql\\n<the mysql command>\\n```\". You will receive the corresponding output for your sql command.\nYour output should contain only one \"Action\" part.\nThe \"Action\" part should be executed with a mysql interpreter or propose an answer. Any natural language in it should be commented out.\nThe SQL query and submit parts can not appear in your output simutaneously."
        },
        {
            "from": "gpt",
            "value": "OK"
        },
        {
            "from": "human",
            "value": "What is the average GNP and total population in all nations whose government is US territory?"
        },
        {
            "from": "gpt",
            "value": "Thought: I should write a SQL query that gets the average GNP and total population from nations whose government is US territory.\nAction: \n```sql\nSELECT AVG(GNP), SUM(population) FROM nations WHERE government = 'US Territory'\n```"
        },
        {
            "from": "human",
            "value": "Observation: Error executing query: Table 'world_1.nations' doesn't exist"
        },
        {
            "from": "gpt",
            "value": "Thought: The `nations` table I used in the SQL query does not exist. I should check what tables exist.\nAction: \n```sql\nSHOW TABLES\n```"
        },
        {
            "from": "human",
            "value": "Observation: [('city',), ('country',), ('countrylanguage',)]"
        },
        {
            "from": "gpt",
            "value": "Thought: There are three tables. I can try writing a query using the `country` table since the question is asking\nabout countries.\nAction: \n```sql\nSELECT AVG(GNP), SUM(population) FROM country WHERE governmentform = 'US Territory'\n```"
        },
        {
            "from": "human",
            "value": "Observation: [(510.333333, Decimal('329000'))]"
        },
        {
            "from": "gpt",
            "value": "Thought: The code executed successfully without any errors and produced the expected output. I can now submit the solution.\nAction: submit."
        }
    ],
    "virtualhome": [
        {
            "from": "human",
            "value": "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"Thought: your thoughts.\nAction: your next action\".\n\nThe available actions are:\nwalk to {obj}\nrun to {obj}\ngrab {obj}\nopen {obj}\nclose {obj}\nput {obj} on {recep}\nput {obj} in {recep}\nswitch on {obj}\nswitch off {obj}\ndrink {obj}\nlook at {obj}\nsit on {obj}\nstand up\nwatch {obj}\nwipe {obj}\ntype on {obj}\nwash {obj}\ncut {obj}\neat {obj}\nsleep\nwake up\nplug in {obj}\nplug out {obj}\npour {obj} into {recep}\nmove {obj}\nrelease\nturn to {obj}\n\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\n\nYour response should use the following format:\n\nThought: <your thoughts>\nAction: <your next action>"
        },
        {
            "from": "gpt",
            "value": "OK"
        },
        {
            "from": "human",
            "value": "The task is Drink (Drink water)."
        },
        {
            "from": "gpt",
            "value": "Thought: To drink water, I need to first locate a water source in the household, so moving towards the dining room might bring me closer to finding water or a suitable container for drinking.\nAction: Walk to dining room"
        },
        {
            "from": "human",
            "value": "Observation:"
        },
        {
            "from": "gpt",
            "value": "Thought: I chose to walk to the water glass as it is likely located in the dining room where I can access water easily, aligning with the goal of drinking water.\nAction: Walk to water glass"
        },
        {
            "from": "human",
            "value": "Observation: water_glass is CLEAN."
        },
        {
            "from": "gpt",
            "value": "Thought: The current observation indicates that I've located a clean water glass, which is essential to proceed with the task of drinking water. Now, I need to fill it with water to accomplish the task goal.\nAction: Find water glass"
        },
        {
            "from": "human",
            "value": "Observation: water_glass is CLEAN."
        },
        {
            "from": "gpt",
            "value": "Thought: The observation indicates that the water glass is clean, and since the task is to drink water, holding the glass will allow further steps to achieve this goal.\nAction: Grab water glass"
        },
        {
            "from": "human",
            "value": "Observation: character, water_glass is CLEAN. You have water_glass."
        },
        {
            "from": "gpt",
            "value": "Thought: Since I have already obtained the water glass, the next logical step is to drink from it to complete the task of drinking water.\nAction: Drink water glass"
        }
    ]}

def template_change(conversation):
    messages = []
    for item in conversation:
        message = {}
        if item['from'] == "gpt":
            message['role'] = "assistant"
            message['content'] = item['value']
        else:
            message['role'] = "user"
            message['content'] = item['value']
        messages.append(message)
    return messages


# def generate_llm_response(args: argparse.Namespace):
        

def construct_llm_data(args: argparse.Namespace):
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)
    
    if args.model_name is not None:
        agent_config['config']['model_name'] = args.model_name

    env_config = exp_config["env_config"]
    task_name = args.exp_config
    
    # initialize the agent
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"]
    )

    # initialize virtualhome all the tasks
    with open('data/virtualhome/new_train_conversation.jsonl', 'r') as file:
        vh_sft_data = []
        for line in file:
            json_object = json.loads(line)
            vh_sft_data.append(json_object)
    
    split_lists = np.array_split(vh_sft_data, args.part_num)
    splits_data = [list(arr) for arr in split_lists]
    sft_data = splits_data[args.part_idx]
    n_tasks = len(sft_data)

    initial_data = []
    if os.path.exists(f"{args.save_path}/{task_name}_traj_{args.part_idx}.json"):
        with open(f"{args.save_path}/{task_name}_traj_{args.part_idx}.json", "r") as f:
            initial_data = json.load(f)
        initial_data_path = []
        for each in initial_data:
            initial_data_path.append(each['path'])
        sft_data = [each for each in sft_data if each['path'] not in initial_data_path]
    else:
        with open(f"{args.save_path}/{task_name}_traj_{args.part_idx}.json", "w") as f:
            json.dump([], f, indent=4)

    print('n_tasks:', len(sft_data))

    final_data = initial_data
    
    print("Start generating pair_data")
    
    pbar = tqdm(total=n_tasks)

    for each_data in sft_data:

        env: envs.BaseEnv = getattr(envs, env_config["env_class"])(each_data, **env_config)
        
        conversations = each_data['conversations']
        # iteration_nums = len(conversations)//2-1
        iteration_nums = args.iteration_num
        
        for j in range(iteration_nums):
            if args.global_only and j != 0:
                continue
            start_time = time.time()
            
            new_conversations = conversations[2:3]
            # if task_name == "webshop":
            init_obs, state = env.reset()
            env.state.history = template_change(prompt_for_environment[task_name] + [new_conversations[0]])
            
            while not state.finished:
                try:
                    llm_output: str = agent(state.history)
                except Exception as e:
                    state.success = False
                    state.finished = True
                    state.terminate_reason = "exceeding maximum input length"
                    break
                # environment step
                observation, state = env.step(llm_output)
                # color the state in blue
                if state.finished:
                    break
            
            end_time = time.time()
            
            new_item = {
                "id": each_data['id'],
                "iteration": j,
                "agent_conversations": state.history,
                'agent_final_reward': state.reward if (task_name == "webshop" or task_name=="intercode_sql") else state.success,
                'time': end_time - start_time,
                'path': each_data['path']
            }
            
            final_data.append(new_item)
    
        pbar.update(1)
        json.dump(final_data, open(f"{args.save_path}/{task_name}_traj_{args.part_idx}.json", "w"), indent=4)
    pbar.close()
            
    json.dump(final_data, open(f"{args.save_path}/{task_name}_traj_{args.part_idx}.json", "w"), indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--exp_path",
        type=str,
        default="./eval_agent/configs/task",
        help="Config path of experiment.",
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default="alfworld",
        help="Config of experiment.",
    )
    parser.add_argument(
        "--agent_path",
        type=str,
        default="./eval_agent/configs/model",
        help="Config path of model.",
    )
    parser.add_argument(
        "--agent_config",
        type=str,
        default="fastchat",
        help="Config of model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="Llama-2-7b-hf-webshop-sft-explore-0",
        help="Model name. It will override the 'model_name' in agent_config"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Whether to ignore done tasks.",
    )
    parser.add_argument(
        "--interactive",
        action="store_false",
        help="Whether to run in interactive mode for demo purpose.",
    )
    parser.add_argument(
        "--part_num",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--part_idx",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--save_path",
        type=str,
    )
    parser.add_argument(
        "--global_only",
        action="store_true",
    )
    
    parser.add_argument(
        "--iteration_num",
        type=int,
        default=5,
    )

    args = parser.parse_args()
    construct_llm_data(args)