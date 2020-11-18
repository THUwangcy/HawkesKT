# HawkesKT

![illustration](./data/_static/idea.png)

This is our implementation for the paper:

*Chenyang Wang, Weizhi Ma, Min Zhang, Chuancheng Lv, Fengyuan Wan, Huijie Lin, Taoran Tang, Yiqun Liu, and Shaoping Ma. [Temporal Cross-effects in Knowledge Tracing.]() In WSDM'21.*

**Please cite our paper if you use our codes. Thanks!**



## Usage		

1. Install [Anaconda](https://docs.conda.io/en/latest/miniconda.html) with Python >= 3.5
2. Clone the repository and install requirements

```bash
git clone https://github.com/THUwangcy/HawkesKT
```

3. Prepare datasets according to [README](https://github.com/THUwangcy/HawkesKT/tree/main/data/README.md) in data directory
4. Install requirements and step into the `src` folder

```bash
cd HawkesKT
pip install -r requirements.txt
cd src
```

5. Run model

```bash
python main.py --model_name HawkesKT --emb_size 64 --max_step 50 --lr 5e-3 --l2 1e-5 --time_log 5 --gpu 1 --dataset ASSISTments_09-10
```


Example training log can be found [here](https://github.com/THUwangcy/HawkesKT/blob/main/log/HawkesKT/HawkesKT__ASSISTments_12-13__2019__lr%3D0.001__l2%3D0.0__fold%3D0__time_log%3D5.0.txt).



## Arguments

The main arguments of HawkesKT are listed below.

| Args       | Default | Help                                                         |
| ---------- | ------- | ------------------------------------------------------------ |
| emb_size   | 64      | Size of embedding vectors                                    |
| time_log   | e       | Base of log transformation on time intervals                 |
| max_step   | 50      | Consider the first max_step interactions in each sequence    |
| fold       | 0       | Fold to run                                                  |
| lr         | 1e-3    | Learning rate                                                |
| l2         | 0       | Weight decay of the optimizer                                |
| batch_size | 100     | Batch size                                                   |
| regenerate | 0       | Whether to read data again and regenerate intermediate files |



## Performance

The table below lists the results of some representative models in `ASSISTments 12-13` dataset. 

| Model                                                        |  AUC   | Time/iter | Time-aware | Temporal cross |
| :----------------------------------------------------------- | :----: | :-------: | :--------: | :------------: |
| [DKT](https://github.com/THUwangcy/HawkesKT/blob/main/src/models/DKT.py) | 0.7308 |   3.8s    |            |                |
| [DKT-Forgetting](https://github.com/THUwangcy/HawkesKT/blob/main/src/models/DKTForgetting.py) | 0.7462 |   6.2s    |     √      |                |
| [KTM](https://github.com/THUwangcy/HawkesKT/blob/main/src/models/KTM.py) | 0.7535 |   49.8s   |     √      |                |
| [AKT-R](https://github.com/THUwangcy/HawkesKT/blob/main/src/models/AKT.py) | 0.7555 |   13.8s   |     √      |                |
| [HawkesKT](https://github.com/THUwangcy/HawkesKT/blob/main/src/models/HawkesKT.py) | 0.7676 |   3.2s    |     √      |       √        |

Current running commands are listed in [run.sh](https://github.com/THUwangcy/HawkesKT/blob/main/src/run.sh).  We adopt 5-fold cross validation and report the average score (see [run_exp.py](https://github.com/THUwangcy/HawkesKT/blob/main/src/utils/run_exp.py)). All experiments are conducted with a single GTX-1080Ti GPU.



## Contact

Chenyang Wang (THUwangcy@gmail.com)