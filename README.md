# EET
Implementation code：Toward Efficient and Effective Vision Transformer for Large-Scale Fine-Grained Image Retrieval
## Training
```
python main.py --dataset cub_bird --model-name EET --code-length "$code_length" --epoch 90 --batch-size 64 --w 4 --lr 0.01 --gpu-ids 5
```