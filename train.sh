CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=2333 \
    tools/train.py --cfg experiments/siam_r18_l4_dwxcorr_8gpu/config.yaml >nohup-yolo.log 2>&1 &
