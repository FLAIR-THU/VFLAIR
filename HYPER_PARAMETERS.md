## Hyper-parameters

### One. Main Task ACC on MNIST, 30-epochs, Adam-optimizer
#### VFL - Main Task ACC
| main_lr |0.05|0.01|0.005|0.001|SGD 0.5|
|:-:|:-:|:-:|:-:|:-:|:-:|
|GolbalModel|0.9644||0.9705|-|?|
|No GlobalModel|0.8532|0.7637|0.8634|0.9527|0.96|

#### CAE-VFL - Main Task ACC
1. lr=0.01

| $\lambda$ | 1.0 |
|:-:|:-:|
|GolbalModel|0.9387|
|No GlobalModel|0.91|

#### MID-VFL - Main Task ACC
1. main_lr=0.05

| $\lambda$=0.01 | passive | active|
|:-:|:-:|:-:|
|GolbalModel mid_lr=0.1|0.9527|0.9280|
|No GlobalModel mid_lr=0.5|debug|0.4173|
|No GlobalModel mid_lr=0.1|0.8904|0.5797|
|No GlobalModel mid_lr=0.05|0.9343|0.8564|
|No GlobalModel mid_lr=0.01|0.9468|0.8659|
|No GlobalModel mid_lr=0.005|-|0.8887|
|No GlobalModel mid_lr=0.001|0.9550|0.9369|