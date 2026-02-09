craftiumによるマインクラフト環境(https://craftium.readthedocs.io/en/latest/)においてchoptreeという木をchopするタスク

実験は
ppo(armfmppoでcli上でarmfmppoなしに設定して実行)
lstmppo
armfmppo
dreamer-v3
の4実験

dreamer-v3を実験する際は、https://github.com/NM512/dreamerv3-torchをクローンする必要がある
実験についての詳しい内容は下記speaker deckを参照

実験やり方
craftium環境の構築
dreamer-v3を実験する際は、https://github.com/NM512/dreamerv3-torchをクローンする必要がある
e.g. dreamer-v3
training
```
$ python train.py   obs_size=64   num_envs=1   steps=4000000   action_repeat=1   frameskip=4   prefill=15000   pretrain=0   dataset_size=300000   train_ratio=32   batch_size=32   batch_length=48   precision=16   dyn_hidden=384   dyn_deter=384   units=384   encoder.cnn_depth=24   decoder.cnn_depth=24   actor.layers=3   critic.layers=3   reward_head.layers=2   cont_head.layers=2   imag_gradient=dynamics   expl_until=120000   eval_every=30000   log_every=1000   capture_video=false   video_pred_log=false   track=True
```
evaluation
```
$ python single_seed_dreamer_evaluate.py   --agent-path your-checkpoint-path   --num-episodes 5   --seed 77   --seed-num 1   --video-dir your-output-path   --mt-wd ./eval_runs   --mt-port 0   --prefer-ckpt-cfg
```

各実験結果


