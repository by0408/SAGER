# SAGER2：仅主实验代码和数据

独立的SAGER主实验项目：六模型配置、探针训练、GPU编辑、解码与毒性/PPL_g评分。
主实验使用全部1,199条Challenge prompt，每条最多生成10个token。

按用户要求，LLaMA-7B完整对齐通过后已清理本项目内全部六模型训练好的MLP探针副本，
以及本次测试的原始输出、诊断和编译缓存。原SAGER项目未删除或修改。
简要核验报告保留在项目之外；本项目不附带生成结果或预训练探针。

## 目录

- `src/sager2/`：主实验、数据准备、训练与GPU编辑代码。
- `configs/`：六模型主配置，全部采用Top100、每20轮刷新。
- `data/`：真实训练/验证/测试JSONL及1,199条Challenge prompt。
- `docs/`：方法、数据来源、指标定义与复现说明。
- `tests/`：导入、候选选择、数据与小模型解码检查。

没有加入消融、测速、人工/AI评阅、案例筛选、通用能力实验、语言模型大权重或hidden缓存。

## 使用

```bash
cp local_paths.example.json local_paths.json
# 填写已有语言模型和Detoxify权重的位置
python run.py doctor --config configs/llama7b.json --data-only
CUDA_VISIBLE_DEVICES=0 python run.py extract --config configs/llama7b.json \
  --hidden-dir cache/retrain_llama7b
CUDA_VISIBLE_DEVICES=0 python run.py train --config configs/llama7b.json \
  --hidden-dir cache/retrain_llama7b --probe-dir outputs/retrained_llama7b_probes
python run.py doctor --config configs/llama7b.json --probe-dir outputs/retrained_llama7b_probes
python run.py evaluate --config configs/llama7b.json --probe-dir outputs/retrained_llama7b_probes --gpus 0 3
```

训练产物保存在新目录，再用`--probe-dir`显式指定。未训练探针时，`--data-only`仅检查
数据与配置，正式评测会明确拒绝启动。重新训练不保证权重逐字节复现原探针。
以下命令使用当前环境的`python`。示例配置中的模型路径均相对于项目根目录。
`local_paths.json`由使用者按本机情况生成，不随项目发布或提交Git。

`data/`仅保留`labeled.jsonl`和`challenge_prompts.jsonl`。
13B、30B的专用训练数据已按要求移除，配置中的`training_data`留空。
使用这些模型时，给doctor、extract、train、evaluate显式增加
`--training-data data/external/original.jsonl`；程序仍核对原数据SHA和训练条数。

每卡加载完整FP16模型，batch=1，基础/编辑配对固定同一物理GPU；多卡是分片，
不是独立重复。启动后后台进程和日志留在服务器，关闭本机或SSH不影响已经启动的任务。

```bash
python run.py status --run-dir outputs/llama7b
python run.py stop --run-dir outputs/llama7b
```

## 主设置与核验范围

L15到末层、原良性TRAIN均值μ、FP64探针/搜索、最多500个旋转槽、每轮±5°、
800不同坐标限制、Top100每20轮刷新、输入末tokenID13跳过编辑、达到μ不提前停止。
原生BOS、无chat模板、greedy、KV cache、EOS停止、最多10新token。

四个7B模型使用随项目提供的48,500条训练文本。13B原训练集为43,053条，30B为10,000条，
这两份数据不再随项目提供。30B统一refresh20后的完整生成尚未复跑。
本次只对LLaMA-7B完整1,199条做了新旧代码核验；没有重新运行其余五模型。
完整旋转坐标/角度轨迹未记录，不能声称每轮选点轨迹都相同。

## GitHub整理

依赖、配置、训练代码及真实数据均保留；训练权重、缓存、运行输出、机器专用路径和ZIP
已加入Git忽略规则。公开许可证由作者确定，本次未发布到GitHub。
第三方文本仍遵循原来源条款，见`NOTICE.md`和`docs/DATASETS.md`。
