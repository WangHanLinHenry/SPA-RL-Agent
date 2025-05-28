save_path=eval/alfworld_eval/
logs_path=${save_path}logs
task=alfworld

# launch the FastChat controller
python -u -m fastchat.serve.controller >> ${logs_path}/model_worker.log 2>&1 &
fs_controller_pid=$!

# Part 2: Evaluate SFT agent
fs_worker_port=21012
CUDA_VISIBLE_DEVICES=0 python -u -m fastchat.serve.vllm_worker --model-path ckt/alfworld_llama3b_merged_model --port ${fs_worker_port} --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker.log 2>&1 &


fs_worker_pid=$!
sleep 90
# sleep 300

# evaluate on the test set
python -m eval_agent.main --agent_config fastchat --model_name alfworld_llama3b_merged_model --exp_config ${task} --split test --override --output_path eval/alfworld_eval

# if failed, exit
if [ $? -ne 0 ]; then
    echo "base agent evaluation failed"
    kill -9 $fs_worker_pid
    exit 1
fi

# kill the model worker
kill -9 $fs_worker_pid
# kill the controller
kill -9 $fs_controller_pid
