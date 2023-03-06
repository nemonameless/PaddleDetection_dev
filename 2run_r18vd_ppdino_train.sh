export FLAGS_allocator_strategy=auto_growth
model_type=dino
job_name=ppdino_r18vd_pan_3_0_6_1x_coco
config=configs/${model_type}/${job_name}.yml
log_dir=log_dir/${job_name}
weights=output/dino_r18vd_pan_3_0_6_1x_coco/0.pdparams

# 1. training
#CUDA_VISIBLE_DEVICES=5 python3.7 tools/train.py -c ${config} --eval #--amp -r ${weights}
python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval #--amp

# 2. eval
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
#CUDA_VISIBLE_DEVICES=5 python3.7 tools/eval.py -c ${config} -o weights=${weights} #--amp
