import os
import json
import logging
import pathlib
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import Fore

import eval_agent.tasks as tasks
import eval_agent.agents as agents
import eval_agent.envs as envs
from eval_agent.utils.datatypes import State

logger = logging.getLogger("agent_frame")

# 将virtualhome相关的环境进行配置
from enum import Enum
from collections import OrderedDict
import parse
import json

def jsonl_add_data(file_path, add_data):
    '''file_path是一个jsonl文件 add_data是一个列表'''
    ori_data = []
    with open(file_path, 'r') as file:
        for line in file:
            ori_data.append(json.loads(line))

    ori_data.extend(add_data)

    with open(file_path, 'w') as file:
        for item in ori_data:
            file.write(json.dumps(item) + '\n')

class EvolveGraphAction(Enum):
    """
    All supported actions, value of each enum is a pair (humanized name, required_number of parameters)
    """
    CLOSE = ("Close", 1, 'close {}')
    DRINK = ("Drink", 1, 'drink {}')
    FIND = ("Find", 1, 'find {}')
    WALK = ("Walk", 1, 'walk to {}')
    GRAB = ("Grab", 1, 'grab {}')
    LOOKAT = ("Look at", 1, 'look at {}')
    # LOOKAT_SHORT = ("Look at short", 1, 'look at {}')
    # LOOKAT_MEDIUM = LOOKAT
    # LOOKAT_LONG = ("Look at long", 1, 'look at {}')
    OPEN = ("Open", 1, 'open {}')
    POINTAT = ("Point at", 1, 'point at {}')
    PUTBACK = ("Put", 2, 'put {} on {}')
    #PUT = ("Put", 2, '')
    #PUTBACK = PU, ''T
    PUTIN = ("Put in", 2, 'put {} in {}')
    PUTOBJBACK = ("Put back", 1, 'put back {}')
    RUN = ("Run", 1, 'run to {}')
    SIT = ("Sit", 1, 'sit on {}')
    STANDUP = ("Stand up", 0, 'stand up')
    SWITCHOFF = ("Switch off", 1, 'switch off {}')
    SWITCHON = ("Switch on", 1, 'switch on {}')
    TOUCH = ("Touch", 1, 'touch {}')
    TURNTO = ("Turn to", 1, 'turn to {}')
    WATCH = ("Watch", 1, 'watch {}')
    WIPE = ("Wipe", 1, 'wipe {}')
    PUTON = ("PutOn", 1, 'put on {}')
    PUTOFF = ("PutOff", 1, 'take off {}')
    GREET = ("Greet", 1, 'greet {}')
    DROP = ("Drop", 1, 'drop {}')
    READ = ("Read", 1, 'read {}')
    LIE = ("Lie", 1, 'lie on {}')
    POUR = ("Pour", 2, 'pour {} into {}')
    TYPE = ("Type", 1, 'type on {}')
    PUSH = ("Push", 1, 'push {}')
    PULL = ("Pull", 1, 'pull {}')
    MOVE = ("Move", 1, 'move {}')
    WASH = ("Wash", 1, 'wash {}')
    RINSE = ("Rinse", 1, 'rinse {}')
    SCRUB = ("Scrub", 1, 'scrub {}')
    SQUEEZE = ("Squeeze", 1, 'squeeze {}')
    PLUGIN = ("PlugIn", 1, 'plug in {}')
    PLUGOUT = ("PlugOut", 1, 'plug out {}')
    CUT = ("Cut", 1, 'cut {}')
    EAT = ("Eat", 1, 'eat {}') 
    SLEEP = ("Sleep", 0, 'sleep')
    WAKEUP = ("WakeUp", 0, 'wake up')
    RELEASE = ("Release", 1, 'release')

def merge_add(d, k, v):
    if k == v:
        return
    # print(f'adding {k} --> {v}')
    if k in d:
        prev_v = d[k]
        # print(f'existing: {k} --> {prev_v}')
        merge_add(d, v, prev_v)
    else:
        d[k] = v

with open("/home/hanlin/hlwang_projects/FYP/class_name_equivalence.json", 'r') as f:
    abstract2detail = json.load(f)

detail2abstract = dict()
for abstract, details in abstract2detail.items():
    for detail in details:
        merge_add(detail2abstract, detail, abstract)

def process_format(arg):
    # don't use any underscore in args
    arg = arg.replace(' ', '_')
    return arg

def str2program_list(program_lines):
    '''
    这个代码的功能是将allowed action里面的动作给转化为virtualhome中的标准格式
    比如：
    input: ["close address_book","close address_book","close address_book"]
    output: ["[CLOSE] <address_book> (1)"]
    '''
    def _format_arg(arg):
        arg = arg.lower().strip().replace(' ', '_')
        if arg in detail2abstract:
            return detail2abstract[arg]
        return arg

    # start parsing ==============================
    # pl = program_str[program_str.index('Step 1:'):].split('\n')
    info = dict()
    info['parsing_error'] = []
    pl = program_lines
    parsed_lines = []
    success_count = 0
    for i, line in enumerate(pl):
        line = line.lower().strip()
        if len(line) == 0:
            continue
        if ':' in line:
            line = line[line.index(':') + 1:].strip()
        try:
            # try matching each possible action
            possible_parsed = OrderedDict()
            for action in EvolveGraphAction:
                action_template = action.value[2]
                expected_num_args = action.value[1]
                parsed = parse.parse(action_template, line)
                if parsed is not None:
                    assert action.name not in possible_parsed
                    if len(parsed.fixed) == expected_num_args:
                        # print(action_template, parsed, expected_num_args)
                        possible_parsed[action.name] = parsed
                    else:
                        # skip if number of parsed args does not match expected
                        pass
            assert len(possible_parsed) == 1, f'possible_parsed: {possible_parsed} does not equal to 1'
            parsed_action = list(possible_parsed.keys())[0]
            parsed_args = possible_parsed[parsed_action]
            if len(parsed_args.fixed) == 0:
                pl_str = '[{}]'
                pl_str = pl_str.format(parsed_action)
            elif len(parsed_args.fixed) == 1:
                pl_str = '[{}] <{}> (1)'
                # pl_str = pl_str.format(parsed_action, _format_arg(parsed_args[0]))                                  # 考虑了名字的问题
                pl_str = pl_str.format(parsed_action, process_format(parsed_args[0]))
            elif len(parsed_args.fixed) == 2:
                pl_str = '[{}] <{}> (1) <{}> (1)'
                # pl_str = pl_str.format(parsed_action, _format_arg(parsed_args[0]), _format_arg(parsed_args[1]))     # 考虑了名字的问题
                pl_str = pl_str.format(parsed_action, process_format(parsed_args[0]), process_format(parsed_args[1]))
            else:
                raise NotImplementedError
            parsed_lines.append(pl_str)
            success_count += 1
        except AssertionError as e:
            message = "| {} | {} | '{}'".format(e.__class__.__name__, e, line)
            info['parsing_error'].append(message)
            line = pl[i]
            if ':' in line:
                line = line[line.index(':') + 1:].strip()
            # none of these is likely going to work, but parse it this way to obey vh format
            if len(line) > 0:
                words = line.split(' ')
                if len(words) == 1:
                    pl_str = '[{}]'.format(words[0].upper())
                elif len(words) == 2:
                    pl_str = '[{}] <{}> (1)'.format(words[0].upper(), words[1])
                elif len(words) == 3:
                    pl_str = '[{}] <{}> (1) <{}> (1)'.format(words[0].upper(), words[1], words[2])
                else:
                    pl_str = '[{}] <{}> (1)'.format(words[0].upper(), '_'.join(words[1:]))
            else:
                pl_str = '[EMPTY]'
            parsed_lines.append(pl_str)
    info['num_parsed_lines'] = len(parsed_lines)
    info['num_total_lines'] = len(pl)
    if len(pl) != 0:
        info['parsibility'] = success_count / len(pl)
    else:
        info['parsibility'] = 0
    return parsed_lines, info

import sys
from tqdm import tqdm
import re
import random

sys.path.append('/data/hanlin/progprompt_vh_main/scripts/virtualhome_master')
from virtualhome.simulation.evolving_graph import utils
from virtualhome.simulation.evolving_graph.scripts import parse_script_line, Script
from virtualhome.simulation.evolving_graph.execution import ScriptExecutor
from virtualhome.simulation.evolving_graph.environment import EnvironmentGraph, EnvironmentState

def remove_duplicate_edge(input_dict):
    Edges = input_dict['edges']
    for edge in Edges:
        fgledge = {'from_id':edge['to_id'], 'relation_type': 'INSIDE', 'to_id': edge['from_id']}
        if fgledge in Edges:
            if edge == fgledge:
                Edges.remove(edge)
            else:
                Edges.remove(fgledge)
                Edges.remove(edge)
    input_dict['edges'] = Edges
    return input_dict

def change_obj_index(graph, program, id, specific_objects, last_obj_id):

    graph_dict = graph.to_dict()
    agent_has_objid = [n['to_id'] for n in graph_dict["edges"] if n['from_id'] == id and "HOLD" in n["relation_type"]]

    obj_id_dict = {}
    obj_ids_close = [n['to_id'] for n in graph_dict["edges"] if n['from_id'] == id and  n["relation_type"]=="CLOSE"]  # 离agent close的物品id
    obj_ids_close_two = [n['from_id'] for n in graph_dict["edges"] if n['to_id'] == id and  n["relation_type"]=="CLOSE"]
    obj_ids_close.extend(obj_ids_close_two)
    obj_ids_close = list(set(obj_ids_close))
    # obj = [node['class_name'] for node in graph_dict['nodes'] if node["id"] in obj_ids_close]  # 离agent close的物品名称
    obj = []
    for i in range(len(obj_ids_close)):
        obj.append([node['class_name'] for node in graph_dict['nodes'] if node['id']==obj_ids_close[i]][0])

    print('agent close to:', obj_ids_close)

    if last_obj_id != -1:
        last_obj_ids_close = [n['to_id'] for n in graph_dict["edges"] if n['from_id'] == last_obj_id and  n["relation_type"]=="CLOSE"]  # 离agent close的物品id
        last_obj_ids_close_two = [n['from_id'] for n in graph_dict["edges"] if n['to_id'] == last_obj_id and  n["relation_type"]=="CLOSE"]
        last_obj_ids_close.extend(last_obj_ids_close_two)
        last_obj_ids_close = list(set(last_obj_ids_close))
        # last_obj = [node['class_name'] for node in graph_dict['nodes'] if node["id"] in last_obj_ids_close]  # 离agent close的物品名称

        # 再加一个限制，不仅是在上一个物体的附近，也可能是在上一个物体的里面也是可以的
        last_obj_ids_inside = [n['to_id'] for n in graph_dict["edges"] if n['from_id'] == last_obj_id and  n["relation_type"]=="INSIDE"]  # 离agent close的物品id
        last_obj_ids_inside_two = [n['from_id'] for n in graph_dict["edges"] if n['to_id'] == last_obj_id and  n["relation_type"]=="INSIDE"]
        last_obj_ids_inside.extend(last_obj_ids_inside_two)
        last_obj_ids_inside = list(set(last_obj_ids_inside))   

        last_obj_ids_close.extend(last_obj_ids_inside)     

        last_obj = []
        for i in range(len(last_obj_ids_close)):
            last_obj.append([node['class_name'] for node in graph_dict['nodes'] if node['id']==last_obj_ids_close[i]][0])

        print('last obj id close:', last_obj_ids_close)
        print('last obj:', last_obj)

    else:
        last_obj_ids_close = []
        last_obj = []

    # 第一种格式 [ ]
    if program.count('<') == 0:
        return program, specific_objects, last_obj_id
    
    # 第二种格式 [ ] < > ( )
    if program.count('<') == 1:
        
        def extract_text(input_string):
            pattern = r'\[([^]]+)\]|\<([^>]+)\>|\(([^)]+)\)'  # 正则表达式模式，匹配方括号、尖括号和圆括号中的内容
            matches = re.findall(pattern, input_string)  # 查找所有匹配的内容
            extracted_text = [match[0] or match[1] or match[2] for match in matches]  # 提取匹配结果
            return extracted_text
        
        extracted_text = extract_text(program)

        for i in range(len(obj_ids_close)):
            if obj[i] == extracted_text[1]:
                obj_id_dict[obj[i]] = obj_ids_close[i]

        for i in range(len(last_obj_ids_close)):
            if last_obj[i] == extracted_text[1]:
                obj_id_dict[last_obj[i]] = last_obj_ids_close[i]

        if extracted_text[0] not in ['FIND', 'WALK']:
            obj_id1 = [node['id'] for node in graph_dict['nodes'] if node['class_name'] == extracted_text[1]]  # 环境中所有的名称相同的node

            # print('extracted text:', extracted_text[1])
            # print('obj_ids:', obj_id1)
            if extracted_text[1] in list(specific_objects.keys()):
                id1 = specific_objects[extracted_text[1]]
                print('specific objs')
            elif extracted_text[1] in list(obj_id_dict.keys()):
                id1 = obj_id_dict[extracted_text[1]]
                specific_objects[extracted_text[1]] = id1
                print('close objects')
            elif len(obj_id1) == 0:
                return extracted_text[1] + " isn't available in the environment.", specific_objects, last_obj_id
            else:
                id1 = random.choice(obj_id1)
                specific_objects[extracted_text[1]] = id1
                print('random objects')
            pattern = r'\d+'  # 正则表达式模式，匹配数字
            replaced_string = re.sub(pattern, str(id1), program)     
            return replaced_string, specific_objects, id1   
        else:
            obj_id1 = [node['id'] for node in graph_dict['nodes'] if node['class_name'] == extracted_text[1]]
            if len(obj_id1)==0:
                return extracted_text[1] + " isn't available in the environment.", specific_objects, last_obj_id
            
            # print('extracted text:', extracted_text[1])
            # print('obj_ids:', obj_id1)

            if extracted_text[1] in list(specific_objects.keys()):
                id1 = specific_objects[extracted_text[1]]
                print('specific objs')
            elif extracted_text[1] in list(obj_id_dict.keys()):
                id1 = obj_id_dict[extracted_text[1]]
                specific_objects[extracted_text[1]] = id1
                print('close objs')
            else:
                id1 = random.choice(obj_id1)
                specific_objects[extracted_text[1]] = id1
                print('random objs')
            
            pattern = r'\d+'  # 正则表达式模式，匹配数字
            replaced_string = re.sub(pattern, str(id1), program)
            return replaced_string, specific_objects, id1  
            
    # 第三种格式 [ ] < > ( ) < > ( )
    if program.count('<') == 2:
        
        ori_specific_objects = specific_objects

        def parse_content(input_string):
            pattern = r'\[(.*?)\]|\<(.*?)\>|\((.*?)\)'  # 正则表达式模式，匹配方括号、尖括号和圆括号中的内容
            matches = re.findall(pattern, input_string)  # 查找所有匹配的内容
            parsed_content = [group for match in matches for group in match if group]  # 解析匹配结果
            return parsed_content
        
        content = parse_content(program)
        obj_id1 = [node['id'] for node in graph_dict['nodes'] if node['class_name'] == content[1] and node['id'] in agent_has_objid]
        obj_id2 = [node['id'] for node in graph_dict['nodes'] if node['class_name'] == content[3]]
        
        for i in range(len(obj_ids_close)):
            if obj[i] == content[1]:
                obj_id_dict[obj[i]] = obj_ids_close[i]
            if obj[i] == content[3]:
                obj_id_dict[obj[i]] = obj_ids_close[i]

        for i in range(len(last_obj_ids_close)):
            if last_obj[i] == content[1]:
                obj_id_dict[last_obj[i]] = last_obj_ids_close[i]
            if last_obj[i] == content[3]:
                obj_id_dict[last_obj[i]] = last_obj_ids_close[i]

        if len(obj_id1) == 0:
            return content[1] + " not in hand. Robot agent should hold " + content[1] + " firstly.", specific_objects, last_obj_id

        id1 = random.choice(obj_id1)
        specific_objects[content[1]] = id1

        if len(obj_id2) == 0:
            return content[3] + " isn't available in the environment.", specific_objects, last_obj_id
        elif content[3] in list(specific_objects.keys()):
            id2 = specific_objects[content[3]]
        elif content[3] in list(obj_id_dict.keys()):
            id2 = obj_id_dict[content[3]]
            specific_objects[content[3]] = id2
        else:
            id2 = random.choice(obj_id2)
            specific_objects[content[3]] = id2

        # 防止两个物体时相同的，导致循环
        if id1 == id2:
            return content[1] + " can't be put or pour into itself.", ori_specific_objects, last_obj_id

        program_list = list(program)
        positions = [index for index, element in enumerate(program_list) if element == ')']
        qian_program = program[:positions[0]+1]
        hou_program = program[positions[0]+1:]
        qian_program = re.sub(r'\((\d+)\)', '('+str(id1)+')', qian_program, count=1)
        hou_program = re.sub(r'\((\d+)\)', '('+str(id2)+')', hou_program, count=1)
        program = qian_program + hou_program

        return program, specific_objects, id2

def check_action_format(program_text):
    action = re.findall(r'\[(.*?)\]', program_text)[0]
    num_para = EvolveGraphAction[action].value[1]
    action_para = program_text.count('<')
    if num_para == action_para:
        return program_text
    else:
        return action + " needs " + str(num_para) + " parameters. But there are " + str(action_para) + " parameters."


# 对于vitualhome环境的evaluation的过程
def interactive_loop(
    each_task: Dict[str, Any],
    agent: agents.LMAgent,
    env_config: Dict[str, Any],
) -> State:
    
    logger.info(f"Loading environment: {env_config['env_class']}")
    env: envs.BaseEnv = getattr(envs, env_config["env_class"])(each_task, **env_config)

    init_obs, state = env.reset()

    cur_step = 1
    while not state.finished:
        # print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
        logger.info(f"\n{Fore.RED}Step {cur_step}{Fore.RESET}\n")
        cur_step += 1
        # agent act
        try:
            llm_output: str = agent(state.history)
            # color the action in green
            # logger.info(f"\nLM Agent Action:\n\033[92m{action.value}\033[0m")
            logger.info(
                f"\n{Fore.GREEN}{llm_output}{Fore.RESET}\n"
            )
        except Exception as e:
            logger.info(f"Agent failed with error: {e}")
            state.success = False
            state.finished = True
            state.terminate_reason = "exceeding maximum input length"
            break
        # environment step
        observation, state = env.step(llm_output)
        # color the state in blue
        if not state.finished:
            # color the observation in blue
            logger.info(
                f"\n{Fore.BLUE}{observation}{Fore.RESET}\n"
            )

        if state.finished:
            break

    if state.reward is not None:
        logger.info(
            f"Task finished in {state.steps} steps. Success: {state.success}. Reward: {state.reward}"
        )
    else:
        logger.info(
            f"Task finished in {state.steps} steps. Success: {state.success}"
        )

    return state




# 后面的主函数
def main(args: argparse.Namespace):
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)
    
    
    if args.model_name is not None:
        agent_config['config']['model_name'] = args.model_name

    if args.output_path == "":
        output_path = os.path.join("/home/hanlin/hlwang_projects/IPR/outputs", agent_config['config']['model_name'].replace('/', '_'), args.exp_config+args.exp_name)
    else:
        output_path = args.output_path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(output_path, "log.txt"), mode='w')
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(), file_handler],
    )

    env_config = exp_config["env_config"]
    
    logger.info(f"Experiment config: \n{json.dumps(exp_config, indent=2)}")

    # initialize all the tasks
    # task_config: Dict[str, Any] = exp_config["task"]
    # task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    # all_tasks, n_tasks = task_class.load_tasks(args.split, args.part_num, args.part_idx)
    
    # initialize the agent
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"]
    )

    state_list = []

    done_task_id = []
    if os.path.exists(output_path) and not args.override:
        for file in os.listdir(output_path):
            if not file.endswith('json'):
                continue
            state = State.load_json(json.load(open(os.path.join(output_path, file))))
            state_list.append(state)
            done_task_id.append(file.split('.')[0])
        logger.info(f"Existing output file found. {len(done_task_id)} tasks done.")

    # 逐行读取jsonl文件,这里要采用已经处理完后的数据
    vh_test_data = []
    with open(args.test_path, "r") as file:
        for line in file:
            json_object = json.loads(line)
            vh_test_data.append(json_object)
    
    if len(done_task_id) == len(vh_test_data):
        logger.info("All tasks done. Exiting.")
        # calculate metrics
        reward_list = []
        success_list = []
        for state in state_list:
            if state.reward is not None:
                reward_list.append(state.reward)
            success_list.append(state.success)

        if len(reward_list) != 0:
            logger.warning(f"Average reward: {sum(reward_list)/len(success_list):.4f}")
        logger.warning(f"Success rate: {sum(success_list)/len(success_list):.4f}")
        return

    # run the loop for all tasks
    logging.info(f"Running interactive loop for {len(vh_test_data)} tasks.")
    n_todo_tasks = len(vh_test_data) - len(done_task_id)  # only run the remaining tasks

    with logging_redirect_tqdm():
        pbar = tqdm(total=n_todo_tasks)
        for i, task in enumerate(vh_test_data):
            # Only test 10 tasks in debug mode
            if args.debug and i == 5:
                break

            # skip done tasks
            if task['id'] in done_task_id:
                continue

            state = interactive_loop(
                task, agent, env_config
            )

            state_list.append(state)
            json.dump(state.to_dict(), open(os.path.join(output_path, task['id']+".json"), 'w'), indent=4)

            pbar.update(1)
        pbar.close()
    
    logger.warning("All tasks done.")
    logger.warning(f"Output saved to {output_path}")

    # calculate metrics
    reward_list = []
    success_list = []
    for state in state_list:
        if state.reward is not None:
            reward_list.append(state.reward)
        success_list.append(state.success)

    if len(reward_list) != 0:
        logger.warning(f"Average reward: {sum(reward_list)/len(success_list):.4f}")
    logger.warning(f"Success rate: {sum(success_list)/len(success_list):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="The name of the experiemnt.",
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        default="/home/hanlin/hlwang_projects/IPR/eval_agent/configs/task",
        help="Config path of experiment.",
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default="alfworld",
        help="Config of experiment.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Evaluation split.",
    )
    parser.add_argument(
        "--part_num",
        type=int,
        default=1,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--part_idx",
        type=int,
        default=-1,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--agent_path",
        type=str,
        default="/home/hanlin/hlwang_projects/IPR/eval_agent/configs/model",
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
        help="Model name. It will override the 'model_name' in agent_config"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
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
        action="store_true",
        help="Whether to run in interactive mode for demo purpose.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="",
    )
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    main(args)

