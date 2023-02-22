export FLAGS_allocator_strategy=auto_growth
model_type=dino
job_name=dino_swin_large_384_4scale_3x_coco ###
weights=../detrex_dino_swin_large_4scale.pdparams

# model_type=dino
# job_name=dino_r50_4scale_1x_coco
# weights=../dino_r50_4scale_1x_coco.pdparams

# job_name=dino_r50vd_pan_4_0_6_3x_coco ###
# weights=../xsl/dino_r50vd_pan_4_0_6_3x_coco.pdparams

# model_type=deformable_detr
# job_name=deformable_detr_r50_1x_coco ###
# weights=../deformable_detr_r50_1x_coco.pdparams

config=configs/${model_type}/${job_name}.yml
log_dir=log_dir/${job_name}

# 1. training
CUDA_VISIBLE_DEVICES=6 python3.7 tools/train.py -c ${config} -r ${weights} #--amp #-r ${weights}
#python3.7 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval #--amp

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


#CUDA_VISIBLE_DEVICES=5 python3.7 tools/eval.py -c configs/ppyoloe/ppyoloe_plus_crn_t_auxhead_300e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_t_auxhead_300e_coco.pdparams