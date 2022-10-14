# Instructions

This code is implemented based on [EPyMARL](https://github.com/uoe-agents/epymarl), and the running instructions are similar to that in the original project. We incorporate `Hallway`, `LBF` and `Traffic Junction` in this code, which you can experiment with config instructions like `env-config=hallway/lbf/traffic_junction`. Besides, we incorporte an implementation of `Full-Comm` which you can try by setting `config=full_comm` in your instruction.

For example, you can run `MASIA` on traffic_junction (medium) by using:
```sh
python3 src/main.py --config=masia --env-config=traffic_junction
```
and you can run `Full-Comm` on LBF (11x11-6p-4f-1s) by using:
```sh
python3 src/main.py --config=full_comm --env-config=lbf with env_args.sight=1 env_args.players=6 env_args.field_size=11 env_args.max_food=4 env_args.force_coop=False
```

This code will use tensorboard and save model by default, which will be saved in `./results/`
