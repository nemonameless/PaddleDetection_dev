export FLAGS_allocator_strategy=auto_growth
model_type=dino
job_name=dino_r50_4scale_1x_coco
tea_job_name=distill_dino_r50_to_r50

config=configs/${model_type}/${job_name}.yml
tea_config=configs/${model_type}/${tea_job_name}.yml
log_dir=log_dir/${job_name}
weights=../detrex_dino_swin_large_4scale.pdparams

# 1. training
#CUDA_VISIBLE_DEVICES=5 python3.7 tools/train.py -c ${config} --slim_config ${tea_config} --eval #--amp
python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 2,3,4,5,6,7 tools/train.py -c ${config} --slim_config ${tea_config} --eval #--amp

# 2. eval
#CUDA_VISIBLE_DEVICES=0 python3.7 tools/eval.py -c ${config} -o weights=https://paddledet.bj.bcebos.com/models/${job_name}.pdparams
#CUDA_VISIBLE_DEVICES=5 python3.7 tools/eval.py -c ${config} -o weights=${weights} #--amp

# 4.导出模型
#CUDA_VISIBLE_DEVICES=1 python3.7 tools/export_model.py -c ${config} -o weights=${weights} # exclude_nms=True trt=True

# 5.部署预测
#CUDA_VISIBLE_DEVICES=1 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU

# 6.部署测速
#CUDA_VISIBLE_DEVICES=1 python3.7 deploy/python/infer.py --model_dir=output_inference/${job_name} --image_file=demo/000000014439_640x640.jpg --device=GPU --run_benchmark=True # --run_mode=trt_fp16

# 7.onnx导出
#paddle2onnx --model_dir output_inference/${job_name} --model_filename model.pdmodel --params_filename model.pdiparams --opset_version 12 --save_file ${job_name}.onnx
