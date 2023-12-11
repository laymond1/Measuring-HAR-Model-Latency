# har model
for i in $(seq 1 10); do
    for model in RTWCNN HARLSTM HARBiLSTM HARConvLSTM ResNetTSC FCNTSC GTSNet; do
        CUDA_VISIBLE_DEVICES=1 python train_harmodel.py --dataset uci --arch $model --seed $i --trial $i
    done
done