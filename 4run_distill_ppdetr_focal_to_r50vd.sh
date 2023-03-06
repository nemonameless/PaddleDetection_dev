export FLAGS_allocator_strategy=auto_growth
job_name=distill_stu_ppdino_r50vd_12e

config=configs/dino/ppdino_distill/distill_stu_ppdino_r50vd_12e.yml
tea_config=configs/dino/ppdino_distill/distill_tea_ppdino_focal_to_r50vd_12e.yml

log_dir=log_dir/${job_name}
weights=output/distill_tea_ppdino_swin_to_r50vd_12e/0.pdparams

# 1. training
#CUDA_VISIBLE_DEVICES=2 python3.7 tools/train.py -c ${config} --slim_config ${tea_config} --eval #--amp
python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --slim_config ${tea_config} --eval #--amp

# 2. eval
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
#CUDA_VISIBLE_DEVICES=5 python3.7 tools/eval.py -c ${config} -o weights=${weights} #--amp
