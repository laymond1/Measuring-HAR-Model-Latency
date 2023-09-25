# har model
for i in $(seq 1 10); do
    for model in RTWCNN HARLSTM HARBiLSTM HARConvLSTM ResNetTSC FCNTSC; do
        CUDA_VISIBLE_DEVICES=1 python train_harmodel.py --dataset opp --arch $model --seed $i --trial $i
    done
done
