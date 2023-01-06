#python -m torch.distributed.launch --master_port 29501 --nproc_per_node=1 \
#train_vqvae.py \
# --do_pretrain --num_thread_reader=0 --epochs=50 \

python train_vqvae.py --epoch 3 --batch_size 32 --n_gpu 2 --num_workers 2 --path /home/share/chenaoxuan/laion2b_chinese_release/bench_1/00000/