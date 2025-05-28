# This is an example of PPO training for webshop environment
# You can change the data path according to different environments (e.g., alfworld, virtualhome)

export PYTHONPATH=./
export TRAIN_PATH="data_train"
export TRAIN_SET="step_grained_for_ppo_example"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

export MODEL_TYPE="llama3-1"
export MODEL_PATH="ckt/llama3b_webshop_sft"

torchrun \
    --nproc_per_node 4 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6602 \
    ppo/step_ppo.py \
    --model_path ${MODEL_PATH} \
    --model_type ${MODEL_TYPE} \
    --config_path config/${MODEL_TYPE}/StepTool_ppo.json \
    --data_file prm/sampled_data_rl_training_webshop.json \
    --epochs 1 \
    
