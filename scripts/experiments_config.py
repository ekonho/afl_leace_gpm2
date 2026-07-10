"""
================================ 实验配置文件 ================================

使用方法:
    1. 在 EXPERIMENTS 列表中定义你的实验
    2. 每个实验是一个 dict，key 是 main.py 的命令行参数名（去掉 -- 前缀）
    3. 不写的参数使用 main.py 的默认值
    4. 运行:  python scripts/run_experiments.py
    5. 也可以指定只跑某些实验:  python scripts/run_experiments.py --exp_ids 0 2 3

实验日志会单独保存到 logs/<exp_name>/ 下，互不干扰。
如果某个实验中断或报错，不影响其他实验的日志。

命名规则（沿用 main.py 原始逻辑）:
    {dataset}_{model}_{partition}_a{alpha}_{timestamp}
=============================================================================
"""


# ===================== 在这里定义你的实验 =====================

EXPERIMENTS = [

    # ---- 实验 0 ----
    {
        "params": {

            "global_rounds": 100,

        },
    },

    # ---- 实验 1 ----
    {
        "params": {

            "global_rounds": 30,
            "drift_var_th":0.95,
            "memory_var_th":0.95,
            "local_lr":0.005
        },
    },

    # ---- 实验 2 ----
    {
        "params": {

            "global_rounds": 50,
            "drift_var_th":0.95,
            "memory_var_th":0.95,
            "local_lr":0.005

        },
    },

    # ---- 实验 3 ----
    {
        "params": {
            "global_rounds": 100,
            "drift_var_th": 0.95,
            "memory_var_th": 0.95,
            "local_lr": 0.005
        },
    },
]
