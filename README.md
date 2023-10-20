# Llama-demo
llama model training from scratch and predict, offer a simple training type and demo

# train
python train.py

训练过程和结果

Epoch: 1/20,Micro Batches: 6:   0%|          | 0/20 [00:00<?, ?it/s, lr=3e-5, train average loss=4.3, train loss=3.23]

Epoch: 1/20,Micro Batches: 6:   5%|▌         | 1/20 [00:00<00:04,  4.53it/s, lr=3e-5, train average loss=4.3, train loss=3.23]

Epoch: 2/20,Micro Batches: 6:   5%|▌         | 1/20 [00:00<00:04,  4.68it/s, lr=3e-5, train average loss=7.44, train loss=3.31]

Epoch: 3/20,Micro Batches: 6:   0%|          | 0/20 [00:00<?, ?it/s, lr=3e-5, train average loss=9.89, train loss=3.07]

Epoch: 3/20,Micro Batches: 6:   5%|▌         | 1/20 [00:00<00:04,  4.61it/s, lr=3e-5, train average loss=9.89, train loss=3.07]

Epoch: 4/20,Micro Batches: 6:   5%|▌         | 1/20 [00:00<00:04,  4.64it/s, lr=3e-5, train average loss=11.4, train loss=0.501]

Epoch: 5/20,Micro Batches: 6:   0%|          | 0/20 [00:00<?, ?it/s, lr=3e-5, train average loss=12.1, train loss=0.315]

Epoch: 5/20,Micro Batches: 6:   5%|▌         | 1/20 [00:00<00:03,  4.83it/s, lr=3e-5, train average loss=12.1, train loss=0.315]

Epoch: 6/20,Micro Batches: 6:   5%|▌         | 1/20 [00:00<00:04,  4.49it/s, lr=3e-5, train average loss=12.5, train loss=0.125]

Epoch: 7/20,Micro Batches: 6:   0%|          | 0/20 [00:00<?, ?it/s, lr=3e-5, train average loss=12.7, train loss=0.268]

Epoch: 7/20,Micro Batches: 6:   5%|▌         | 1/20 [00:00<00:04,  3.87it/s, lr=3e-5, train average loss=12.7, train loss=0.268]

Epoch: 8/20,Micro Batches: 6:   5%|▌         | 1/20 [00:00<00:04,  4.56it/s, lr=3e-5, train average loss=12.8, train loss=0.0385]

Epoch: 9/20,Micro Batches: 6:   0%|          | 0/20 [00:00<?, ?it/s, lr=3e-5, train average loss=12.8, train loss=0.0135]

Epoch: 9/20,Micro Batches: 6:   5%|▌         | 1/20 [00:00<00:03,  4.87it/s, lr=3e-5, train average loss=12.8, train loss=0.0135]

Epoch: 10/20,Micro Batches: 6:   5%|▌         | 1/20 [00:00<00:03,  4.76it/s, lr=3e-5, train average loss=12.9, train loss=0.0471]

Epoch: 11/20,Micro Batches: 6:   0%|          | 0/20 [00:00<?, ?it/s, lr=3e-5, train average loss=12.9, train loss=0.0302]

Epoch: 11/20,Micro Batches: 6:   5%|▌         | 1/20 [00:00<00:04,  4.61it/s, lr=3e-5, train average loss=12.9, train loss=0.0302]

Epoch: 12/20,Micro Batches: 6:   5%|▌         | 1/20 [00:00<00:03,  4.85it/s, lr=3e-5, train average loss=12.9, train loss=0.0221]

Epoch: 13/20,Micro Batches: 6:   0%|          | 0/20 [00:00<?, ?it/s, lr=3e-5, train average loss=12.9, train loss=0.00628]

Epoch: 13/20,Micro Batches: 6:   5%|▌         | 1/20 [00:00<00:03,  4.91it/s, lr=3e-5, train average loss=12.9, train loss=0.00628]

Epoch: 14/20,Micro Batches: 6:   5%|▌         | 1/20 [00:00<00:03,  4.90it/s, lr=3e-5, train average loss=12.9, train loss=0.00568]

Epoch: 15/20,Micro Batches: 6:   0%|          | 0/20 [00:00<?, ?it/s, lr=3e-5, train average loss=12.9, train loss=0.0098]

Epoch: 15/20,Micro Batches: 6:   5%|▌         | 1/20 [00:00<00:03,  4.84it/s, lr=3e-5, train average loss=12.9, train loss=0.0098]

Epoch: 16/20,Micro Batches: 6:   5%|▌         | 1/20 [00:00<00:03,  4.87it/s, lr=3e-5, train average loss=12.9, train loss=0.00837]

Epoch: 17/20,Micro Batches: 6:   0%|          | 0/20 [00:00<?, ?it/s, lr=3e-5, train average loss=13, train loss=0.00427]

Epoch: 17/20,Micro Batches: 6:   5%|▌         | 1/20 [00:00<00:03,  4.97it/s, lr=3e-5, train average loss=13, train loss=0.00427]

Epoch: 18/20,Micro Batches: 6:   5%|▌         | 1/20 [00:00<00:03,  4.76it/s, lr=3e-5, train average loss=13, train loss=0.00396]

Epoch: 19/20,Micro Batches: 6:   0%|          | 0/20 [00:00<?, ?it/s, lr=3e-5, train average loss=13, train loss=0.00581]

Epoch: 19/20,Micro Batches: 6:   5%|▌         | 1/20 [00:00<00:03,  4.84it/s, lr=3e-5, train average loss=13, train loss=0.00581]

Epoch: 20/20,Micro Batches: 6:   5%|▌         | 1/20 [00:00<00:03,  4.85it/s, lr=3e-5, train average loss=13, train loss=0.00367]

# prdict 
python predict.py

>预测结果： "流畅度"
>实际结果： "流畅度"

