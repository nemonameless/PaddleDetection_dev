export FLAGS_allocator_strategy=auto_growth
model_type=dino
job_name=distill_stu_dino_r18vd_12e
tea_job_name=distill_tea_dino_r50vd_to_r18vd_12e

config=configs/${model_type}/${job_name}.yml
tea_config=configs/${model_type}/${tea_job_name}.yml
log_dir=log_dir/${job_name}
weights=output/distill_stu_dino_r18vd_12e/0.pdparams

# 1. training
#CUDA_VISIBLE_DEVICES=4 python3.7 tools/train.py -c ${config} --slim_config ${tea_config} --eval #--amp
python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --slim_config ${tea_config} #--eval #--amp

# 2. eval
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
#CUDA_VISIBLE_DEVICES=5 python3.7 tools/eval.py -c ${config} -o weights=${weights} #--amp
