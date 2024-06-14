#download ViT pretrain weights to initialization folder
#change dataset path for python files in dataloader folder

CUDA_VISIBLE_DEVICES=6 python main_mmd_train.py --gpu 0 --way 5 --test_way 5 --shot 1 \
  --dataset MiniImageNet \
  --init_weights ./initialization/miniimagenet/checkpoint1600.pth \
  --loss_gamma 0.1 \
  --temperature 1.0 \
  --max_epoch 1 \
  --exp mmd-fc100-main-5way-1shot-0.1-1.0 > mmd-fc100-main-5way-1shot-0.1-1.0.txt

CUDA_VISIBLE_DEVICES=6 python main_mmd_evaluate.py --gpu 0 --way 5 --test_way 5 --shot 1 \
  --dataset MiniImageNet \
  --init_weights ./initialization/miniimagenet/checkpoint1600.pth \
  --loss_gamma 0.1 \
  --temperature 1.0 \
  --max_epoch 10 \
  --exp mmd-fc100-main-5way-1shot-0.1-1.0 > mmd-fc100-main-5way-1shot-0.1-1.0-only_test.txt

