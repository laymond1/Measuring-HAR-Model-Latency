# vision model
# for model in mobilenet_v2 mobilenet_v3_small mobilenet_v3_large mnasnet0_5 mnasnet0_75 mnasnet1_0 mnasnet1_3 shufflenet_v2_x0_5 shufflenet_v2_x1_0 shufflenet_v2_x1_5 shufflenet_v2_x2_0 resnet18 resnet34 resnet50 resnet101 resnext50_32x4d resnext101_32x8d resnext101_64x4d squeezenet1_0 squeezenet1_1 efficientnet_b0 efficientnet_b1 efficientnet_b2 efficientnet_b3 efficientnet_b4 efficientnet_b5 efficientnet_b6 efficientnet_b7 efficientnet_v2_s efficientnet_v2_m efficientnet_v2_l marnasnet_a marnasnet_b marnasnet_c marnasnet_d marnasnet_e; do
#     python train_visionmodel.py --dataset uci --arch $model --seed 0 --trial 0
# done

# vision model
for i in $(seq 1 10); do
    for model in mobilenet_v2 mobilenet_v3_small mobilenet_v3_large shufflenet_v2_x0_5 shufflenet_v2_x1_0 shufflenet_v2_x1_5 shufflenet_v2_x2_0; do
        CUDA_VISIBLE_DEVICES=0 python train_visionmodel.py --dataset opp --arch $model --seed $i --trial $i
    done
done
