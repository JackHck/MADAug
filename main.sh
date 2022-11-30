GPU=0

function run_reduced_svhn {
    DATASET=reduced_svhn
    MODEL=wresnet40_2
    EPOCH=160
    BATCH=128
    LR=0.05
    WD=0.01
    SLR=0.001
    CUTOUT=0
    SF=1
}

# cifar10
function run_reduced_cifar10 {
    DATASET=reduced_cifar10
    MODEL=wresnet40_2
    EPOCH=200
    BATCH=128
    LR=0.1
    WD=0.0005
    SLR=0.001
    CUTOUT=16
    SF=3
}

if [ $1 = "reduced_cifar10" ]; then
    run_reduced_cifar10
elif [ $1 = "reduced_svhn" ]; then
    run_reduced_svhn
fi

#SAVE=./${DATASET}_${MODEL}_${BATCH}_${EPOCH}_SLR${SLR}_SF${SF}_cutout_${CUTOUT}_lr${LR}_wd${WD}

python main.py --k_ops 1  --report_freq 10 --num_workers 4 --epochs ${EPOCH} --batch_size ${BATCH} --learning_rate ${LR} --dataset ${DATASET} --model_name ${MODEL}  --gpu ${GPU} --weight_decay ${WD} --proj_learning_rate ${SLR} --search_freq ${SF} --cutout --cutout_length ${CUTOUT} 
