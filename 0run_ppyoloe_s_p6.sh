export FLAGS_allocator_strategy=auto_growth
name=s
model_type=ppyoloe
job_name=ppyoloe_plus_crn_${name}_p6_1280_300e_coco
config=configs/${model_type}/p6/${job_name}.yml
log_dir=log_dir/${job_name}
weights=.pdparams

# 1. training
#CUDA_VISIBLE_DEVICES=3 python3.7 tools/train.py -c ${config} #--amp #-r ${weights}
python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3 tools/train.py -c ${config} --eval # --amp

# 2. eval
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
#CUDA_VISIBLE_DEVICES=2 python3.7 tools/eval.py -c ${config} -o weights=${weights} #--amp
