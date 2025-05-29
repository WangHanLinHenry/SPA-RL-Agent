import re
import json
import logging
from typing import Any, Dict, List, Tuple

from eval_agent.envs import BaseEnv
# from eval_agent.tasks import AlfWorldTask
from eval_agent.prompt import prompt_with_icl
from eval_agent.utils.datatypes import State

import sys
from tqdm import tqdm
import re
import random

from enum import Enum
from collections import OrderedDict
import parse
import json

from virtualhome_master.virtualhome.simulation.evolving_graph import utils
from virtualhome_master.virtualhome.simulation.evolving_graph.scripts import parse_script_line, Script
from virtualhome_master.virtualhome.simulation.evolving_graph.execution import ScriptExecutor
from virtualhome_master.virtualhome.simulation.evolving_graph.environment import EnvironmentGraph, EnvironmentState

logger = logging.getLogger("agent_frame")

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob

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

with open("virtualhome_master/class_name_equivalence.json", 'r') as f:
    abstract2detail = json.load(f)

detail2abstract = dict()
for abstract, details in abstract2detail.items():
    for detail in details:
        merge_add(detail2abstract, detail, abstract)

def process_format(arg):
    # don't use any underscore in args
    arg = arg.replace(' ', '_')
    return arg

def check_action_format(program_text):
    action = re.findall(r'\[(.*?)\]', program_text)[0]
    num_para = EvolveGraphAction[action].value[1]
    action_para = program_text.count('<')
    if num_para == action_para:
        return program_text
    else:
        return action + " needs " + str(num_para) + " parameters. But there are " + str(action_para) + " parameters."

def change_obj_index(graph, program, id, specific_objects, last_obj_id):
    '''graph: 表示的是environment state
       program: 表示的是动作执行的文本
       id: 表示的是agent的id
       specific_objects: 表示的是交互过的物体以及对应的id
       last_obj_id: 表示的是上一个物体的id'''

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

    # print('agent close to:', obj_ids_close)

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

        # print('last obj id close:', last_obj_ids_close)
        # print('last obj:', last_obj)

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
                # print('specific objs')
            elif extracted_text[1] in list(obj_id_dict.keys()):
                id1 = obj_id_dict[extracted_text[1]]
                specific_objects[extracted_text[1]] = id1
                # print('close objects')
            elif len(obj_id1) == 0:
                return extracted_text[1] + " isn't available in the environment.", specific_objects, last_obj_id
            else:
                id1 = random.choice(obj_id1)
                specific_objects[extracted_text[1]] = id1
                # print('random objects')
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
                # print('specific objs')
            elif extracted_text[1] in list(obj_id_dict.keys()):
                id1 = obj_id_dict[extracted_text[1]]
                specific_objects[extracted_text[1]] = id1
                # print('close objs')
            else:
                id1 = random.choice(obj_id1)
                specific_objects[extracted_text[1]] = id1
                # print('random objs')
            
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

class VirtualHomeEnv(BaseEnv):
    def __init__(
        self,
        task_path,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task_path = task_path['path']
        self.revised_graph = task_path['revised_graph']
        self.task_path = task_path
        # self.gt_id_record = task_path['object_id_dict']

        scene_path = "virtualhome_master/init_and_final_graphs/" + self.task_path['path'][151:-4] + ".json"
        with open(scene_path) as f:
            Tdata = json.load(f)
        Tdata = Tdata['init_graph']
        Tdata = remove_duplicate_edge(Tdata)
        env_graph = EnvironmentGraph(Tdata)
        self.agent_id = [n['id'] for n in Tdata["nodes"] if n['class_name'] == 'character'][0]
        name_equivalence = utils.load_name_equivalence()
        self.executor = ScriptExecutor(env_graph, name_equivalence)
        self.this_state = EnvironmentState(env_graph, name_equivalence, instance_selection=True)
        self.specific_objects = task_path['object_id_dict']
        self.last_obj_id = -1
        
        self.state = State()

    def parse_action(self, llm_output: str) -> str:
        llm_output = llm_output.strip()
        pattern = re.compile(r"Action:\s?(.*)", re.DOTALL)
        action = re.findall(pattern, llm_output)[0]
        assert action is not None
        return action
    
    def conduct_action(self, action: str):
        # observation, reward, done, info = self.env.step([action])
        # observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        # return observation, reward, done

        parsed_action = str2program_list([action])[0][0]
        parsed_action, self.specific_objects, self.last_obj_id = change_obj_index(self.this_state, parsed_action, self.agent_id, self.specific_objects, self.last_obj_id)

        # print('the agent is conducting action in this step and the action is...')
        # print(self.specific_objects)
        # print(parsed_action)

        mode = False

        # 检查物体是否存在
        if '[' in parsed_action:
            
            # 检验是否有这个动作
            matches_action = re.findall(r'\[(.*?)\]', parsed_action)[0]
            if matches_action in dir(EvolveGraphAction):
                
                # 检验参数数量是否一致
                parsed_action = check_action_format(parsed_action)
                if '[' in parsed_action:
                    script = parse_script_line(parsed_action, 0)
                    success, self.this_state = self.executor.execute_one_step(Script([script]), self.this_state) 
                    mode = success

        if mode:

            temp_total_graph = self.this_state.to_dict()

            partial_graph = utils.get_visible_nodes(temp_total_graph, agent_id=self.agent_id)

            # agent拿着什么东西
            agent_has_objid = [n['to_id'] for n in temp_total_graph["edges"] if n['from_id'] == self.agent_id and "HOLD" in n["relation_type"]]
            agent_has_obj = [n['class_name'] for n in temp_total_graph["nodes"] if n['id'] in agent_has_objid]
            # agent看到什么东西
            obj_ids_close = [n['to_id'] for n in temp_total_graph["edges"] if n['from_id'] == self.agent_id and  n["relation_type"]=="CLOSE"]
            obj = [node['class_name'] for node in partial_graph['nodes'] if node["id"] in obj_ids_close]
            obj_ids = dict([(node['id'], node['class_name']) for node in temp_total_graph['nodes'] if node["id"] in obj_ids_close and node['class_name'] in obj])
            relations = list(set([obj_ids[n['from_id']] +' '+ n["relation_type"] +' '+ obj_ids[n['to_id']] for n in temp_total_graph["edges"] if n['from_id'] in obj_ids and n['to_id'] in obj_ids and n["relation_type"] not in ["CLOSE","FACING", "INSIDE", "HOLDS_LH", "HOLDS_RH"]]))    
            obj_states = [(node['class_name'], node['states']) for node in temp_total_graph['nodes'] if node['class_name'] in obj]
            objs = ""
            
            for ob_states in obj_states:
                if len(ob_states[1])>0:
                    objs = objs + ob_states[0] + ' is ' + ' and '.join(ob_states[1]) + ', '
                else:
                    objs = objs + ob_states[0] + ', '
            objs = list(set(objs.split(', ')))
            objs = [ob for ob in objs if len(ob)>0]

            # objs = ', '.join(objs) + ', ' + ', '.join(relations)  + '. '
            if len(objs) == 0:
                if len(relations) != 0:
                    objs = ', '.join(relations) + '. '
                else:
                    objs = ""
            else:
                if len(relations) == 0:
                    objs = ', '.join(objs) + '. '
                else:
                    objs = ', '.join(objs) + ', ' + ', '.join(relations)  + '. '

            if len(agent_has_obj)>0:
                agent_has_obj = ', '.join(agent_has_obj)
                objs += f"You have {agent_has_obj}. "
        else:
            objs = "Nothing happens."

        state_dict = self.this_state.to_dict()

        # print('executed action mode:', mode)

        done = self.check_env(state_dict, self.revised_graph)
        # print('environment state:', done)

        return objs, done
        

    def step(self, llm_output: str) -> Tuple[str, State]:
        self.state.history.append({
            "role": "assistant",
            "content": llm_output
        })
        try:
            action = self.parse_action(llm_output)
            observation, done = self.conduct_action(action)
            # print('*****************succeed executing the action in this step.*********************')
            # print('*****************succeed executing the action in this step.*********************')
            # print('*****************succeed executing the action in this step.*********************')
        except Exception as e:
            # logger.debug(f"Agent failed with error: {e}")
            # print('*****************failed executing the action in this step.*********************')
            # print('*****************failed executing the action in this step.*********************')
            # print('*****************failed executing the action in this step.*********************')
            self.state.success = False
            self.state.finished = False
            self.state.reward=0
            observation = f"Observation: Error Input. Your input must contains 'Action: '"
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
            self.state.steps += 1
            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"
            return observation, self.state

        observation = f"Observation: {observation}"
        self.state.history.append({
            "role": "user",
            "content": observation,
        })

        self.state.steps += 1
        if self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"

        if done:
            self.state.finished = True
            self.state.success = True
            self.state.terminate_reason = "success"

        return observation, self.state

    def reset(self, game_files=None) -> Tuple[str, State]:
        self.state = State()

        # self.env.reset_states(game_files)
        # self.state.error = self.task.game_file

        # cur_task = self.task.observation
        # observation, messages = prompt_with_icl(self.instruction, self.raw_icl, cur_task, 1)

        task_name = self.task_path['task']
        task_description = self.task_path['description']
        cur_task = "The task is " + task_name + "(" + task_description + ")."
        observation, messages = prompt_with_icl(self.instruction, self.raw_icl, cur_task, 1)

        if self.icl_format == 'first':
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
        elif self.icl_format == 'conversation':
            self.state.history = messages
        return observation, self.state
    

    # 当给定一个state的时候，进行任务的检查。如果返回的是true表示的是任务完成了，否则的话就是任务没有完成
    def check_env(self, now_state, revised_graph):
        r_nodes = revised_graph['nodes']
        r_remove_edges = revised_graph['revised_nodes']
        r_add_edges = revised_graph['revised_add_edges']

        n_nodes = now_state['nodes']
        n_edges = now_state['edges']

        mode = True
        for each_node in r_nodes:

            each_node_id = each_node['id']
            for each_n in n_nodes:
                if each_n['id'] == each_node_id:
                    n_node = each_n

            for each in each_node.keys():
                if type(each_node[each]) == str:
                    if each_node[each] != n_node[each]:
                        mode = False
                elif type(each_node[each]) == list:
                    if set(each_node[each]) != set(n_node[each]):
                        mode = False
                else:
                    if each_node[each] != n_node[each]:
                        mode = False

        for each_edge in r_remove_edges:
            if each_edge in n_edges:
                mode = False
                # print(each_edge)

        for each_edge in r_add_edges:
            if each_edge not in n_edges:
                mode = False
                # print(each_edge)

        return mode