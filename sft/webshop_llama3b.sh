node_num=4  # number of GPUs
batch_size=16
micro_batch_size=2
accumulation_step=$((${batch_size}/${node_num}/${micro_batch_size}))

# 采用2,3,4号来进行训练
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_P2P_LEVEL=NVL

model_path="./models/Llama-3.2-3B-Instruct"

python -m torch.distributed.run --nproc_per_node=${node_num} --master_port=20002 fastchat/train/train.py \
    --model_name_or_path ${model_path} \
    --data_path /data/webshop_sft.json \
    --bf16 True \
    --output_dir /ckt/llama3b_webshop_sft \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${micro_batch_size} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${accumulation_step} \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess False

# if failed, exit
if [ $? -ne 0 ]; then
    echo "SFT training failed"
    exit 1
fi
