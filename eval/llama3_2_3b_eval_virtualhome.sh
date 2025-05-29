save_path=eval/virtualhome_eval/
logs_path=${save_path}logs
task=virtualhome

# 对于RFT的数据进行评估
python -u -m fastchat.serve.controller --host 0.0.0.0 --port 21001 >> ${logs_path}/model_worker.log 2>&1 & 
fs_controller_pid=$!

# Part 2: Evaluate SFT agent
fs_worker_port=21012
CUDA_VISIBLE_DEVICES=1 python -u -m fastchat.serve.vllm_worker --model-path ckt/virtualhome_llama3b_merged_model --host 0.0.0.0 --port ${fs_worker_port} --worker-address http://0.0.0.0:${fs_worker_port} >> ${logs_path}/model_worker.log 2>&1 &
fs_worker_pid=$!

sleep 180

echo "Start evaluation"

# evaluate on the test set
python -m eval_agent.main_vh --agent_config fastchat --model_name virtualhome_llama3b_merged_model --exp_config ${task} --split test --override --output_path eval/virtualhome_eval --test_path data/virtualhome/new_unseen_test.jsonl

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
