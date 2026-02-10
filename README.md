# Craftium（Minecraft風環境）における ChopTree タスク実験まとめ

## タスク概要
- Craftium による Minecraft 風環境で、**ChopTree（木を切る）**タスクを扱う。  
  - 参考: https://craftium.readthedocs.io/en/latest/

## 実験設定
実験は以下の 4 種類を実施する。

1. **PPO（ベースライン）**  
   - armfmppo の CLI を用いるが、**armfmppo を無効（enable=False）**にして実行する。
2. **LSTM-PPO**
3. **ARMFM-PPO（報酬密化モデル）**  
   - 参考: https://arxiv.org/abs/2510.14176
4. **Dreamer-V3**

## 実験の詳細資料
- 実験の詳しい内容は、指定の Speaker Deck を参照する。
 - （後日）

## 実験手順
### 1. Craftium 環境の構築
- Craftium 環境を構築する。

### 2. Dreamer-V3 実験の準備
- Dreamer-V3 を実験する際は、`dreamerv3-torch` をクローンする必要がある。  
  - https://github.com/NM512/dreamerv3-torch

## Dreamer-V3 実行例
### Training
```bash
$ python train.py   obs_size=64   num_envs=1   steps=4000000   action_repeat=1   frameskip=4   prefill=15000   pretrain=0   dataset_size=300000   train_ratio=32   batch_size=32   batch_length=48   precision=16   dyn_hidden=384   dyn_deter=384   units=384   encoder.cnn_depth=24   decoder.cnn_depth=24   actor.layers=3   critic.layers=3   reward_head.layers=2   cont_head.layers=2   imag_gradient=dynamics   expl_until=120000   eval_every=30000   log_every=1000   capture_video=false   video_pred_log=false   track=True
```

### Evaluation
```bash
$ python single_seed_dreamer_evaluate.py   --agent-path your-checkpoint-path   --num-episodes 5   --seed 77   --seed-num 1   --video-dir your-output-path   --mt-wd ./eval_runs   --mt-port 0   --prefer-ckpt-cfg
```

## 各実験結果（5 episode）
- **ベースライン（PPO）**: Mean Reward:  26.80 ± 6.18,  Min/Max: 16.0 / 34.0, Mean Length:  7570 ± 861, Mean Chops:   26.8 ± 6.2
- **LSTM**: Mean Return: 21.00 ± 12.88, Min/Max: 9.0 / 46.0, Mean Length: 7145 ± 1710, Mean Chops: 21.00 ± 12.88
- **報酬密モデル（ARMFM-PPO）**: Mean Reward: 30.00 ± 13.46, Min/Max: 13.0 / 46.0 , Mean Length: 8000 ± 0, Mean Chops: 30.0 ± 13.5
- **Dreamer-V3**: Mean Reward: 38.80 ± 24.51, Min/Max: 9.0 / 68.0 , Mean Length: 6795 ± 2410, Mean Chops: 38.80 ± 24.51
