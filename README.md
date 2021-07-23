# Template Pytorch-Lightning Hydra Mlflow Poetry


The code in this repository is based on [pytorch/examples](https://github.com/pytorch/examples/blob/2639cf050493df9d3cbf065d45e6025733add0f4/vae/main.py).


## Instllation

```
poetry install
```

## Run training

```bash
# single run with default settings
poetry run python main.py
# single run
poetry run python main.py gpus=[0,1,2,3] batch_size=128 trainer.accelerator=ddp trainer.precision=16 optimizer=sgd scheduler.step_size=1
# multi runs
poetry run python main.py -m optimizer=adam,rmsprop,sgd trainer.precision=16,32 scheduler.step_size=1
```

## Start Mlflow server

```
poetry run mlflow ui
```
