# dynamic_intermodal

一种考虑多源不确定性的多式联运方案动态优化方法的基本实现：联合场景 → 观测驱动更新 → 风险传播 → 节点—时段—订单冲突与协同范围 → 权限约束下的 Gurobi 场景 MILP → 复核与主体确认反馈，在滚动时域中闭环运行。

设计、模型公式、与专利要点及课题 4 的对应关系、实验结果见 [docs/design.md](docs/design.md)。

## 环境

Python ≥ 3.10，需要 numpy、pandas、openpyxl、scipy、gurobipy（需要 Gurobi 许可证）和 pytest。

```bash
pip install -e ".[dev]"
```

不安装时，可以在项目根目录用 `PYTHONPATH=.` 运行。

## 运行

```bash
python3 -m pytest -q
```

```bash
python3 scripts/run_demo.py --method full --seed 13
```

```bash
python3 scripts/run_experiments.py --seeds 48 --workers 8
```

`run_demo.py` 打印逐时点日志，并把辅助决策报告写到 `results/demo/`（JSON + Markdown）。`run_experiments.py` 在相同隐藏真值上对比全部方法（`--methods`、`--requests`、`--correlation-scale`、`--yard-scale` 可调），结果写入 CSV 并打印汇总。

## 结构

```
dynamic_intermodal/
  config.py              全部参数（数据、成本、网络/堆场、不确定性、风险、主体权限、优化）
  data/                  Excel 数据加载（复制自 intermodal_refactor）+ 订单扩充
  network.py             服务网络：班期/卡车服务、时间可行服务链枚举
  plan.py                方案（订单 → 预约的服务链）
  uncertainty/           要点 1：公共因子 Gaussian-copula 联合模型、场景生成与缩减
  update/                要点 2：观测（含删失）生成、场景固定/修复/重加权/重生成
  risk/                  要点 3：事件驱动风险传播、风险指标（P(延误)、CVaR …）
  conflict/              要点 4：节点—时段—订单冲突记录、协同调整范围
  coordination/          要点 5：主体与权限、候选方案、Gurobi 场景 MILP
  feedback/              要点 6：联合复核与样本外验证、主体确认模拟
  simulation/            执行状态同步、滚动时域主控、方法预设（基线/消融）
  reporting.py           辅助决策输出：方案生成、运行监测、异常预警、调整建议
scripts/                 run_demo.py, run_experiments.py
tests/                   单元测试与端到端测试
data_input/2trans_instances/   算例数据
```
