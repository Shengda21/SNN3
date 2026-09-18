# 实验协议 v1：训练与执行双对照

日期：2026-09-08。所有新增数值均为设计参数，不是实验结果。基于已有先导结果制定，不能称整个项目为事先预注册。

## 1. 固定实现与数据角色

### 模型

- 语言主实验：本地 vendor/SmoothSpike 的12层BERT，沿用 common.make_model('folded') 的融合模型、同一官方权重来源、原神经元与reset逻辑；T∈{2,4}。
- “full”沿用现有完整适配配置，其 bert.H1 在 common.py 中冻结。须记录可训练参数名及数量，不能写成所有参数均更新。
- 视觉执行复现：SpikingResformer / CIFAR100，原 screen_identity_p0.0/best.pt，seed12450、30 epoch、原T4；仅切换T2/T4，保留权重和buffer，不新增视觉训练。
- FP32，TF32关闭，固定线程4。视觉沿用 cudnn benchmark=False、deterministic=True。记录训练和推理后端、库版本和任何必要兼容改动。
- 不用重新随机初始化模型代替缺失checkpoint；不得用新下载的浮动版本不加说明地替代历史权重。

### 数据角色与新发现

|角色|定义|允许用途|
|---|---|---|
|训练|Wikitext-2 raw，原tokenizer；每62个正文token加CLS/SEP组成64-token块；train共37156块|训练采样|
|train留出|seed7600排列train的前128块，固定mask seed7601；其余37028块用于训练|全部lr选择与曲线监控，明确它已在前期使用|
|旧确认|validation按seed6101排列后的[3424:3616)，192块，mask seed7602|仅历史复现、实现调试和探索对照|
|233块补充|同排列[3616:3849)，固定新mask seed8701|模型列表锁定后的补充确认；不称完全未见数据|
|主确认候选|同一Wikitext-2 raw版本的官方test split，完整切块，固定mask seed8702|E0核验全项目未用于选择后，E5一次性评估；本轮未下载、读取标签或运行评分|

掩码沿用 mask_batch：正文位置15%选择，80% MASK、10%随机词、10%不变，排除CLS/SEP。每个确认split一次性生成并保存固定输入和标签；所有模型共享，不能按batch重新采样。若切块数或tokenizer结果与旧环境不一致，先查版本，不静默改索引。

**233块的历史暴露核验。** 本轮只读取已有JSON索引和源码。来源为 environment/data.json，results/{mechanism/data,mechanism/replication_data,confidence/protocol,time_budget/protocol,local_warmup/protocol,context_length/protocol,compensated_polar/protocol,trained_state/data}.json。排除train_holdout_indices后合并validation评分索引，得到3616个唯一块，全集3849块的补集为233块。

context_length_probe.create 的 contiguous 路径从 (index+1)×62 开始取 length−64 个token，length最大512；映射到原始块为(index+1)…(index+8)，末块部分重合，越界按原代码循环。历史context_length的tune和confirm共288个目标，与该上下文集合重合的保留块为104个；该上下文配置在历史代码和结果文件中均有记录。另检查前2040-token前缀候选覆盖，7个候选交集已包含在上述104个中。是否存在其他暴露仍需E0全项目审计。此发现不等于训练污染，不抹去旧质量结果，但不足以把233块称为完整独立确认集。

E0核验官方test的使用历史（含同工作区其他方向），并检查其与训练/选择材料的重叠；预训练来源是否用过该语料另列为未知，不把“本项目未使用”扩成“模型从未见过”。如果test已有选择使用，暂停E5独立确认分支，先在看模型结果前指定新的隔离数据协议；E2/E3/E4可继续。不得从233块中按有利结果挑子集。

## 2. E0：资产与云端准备

输入与恢复路径见 CLOUD_HANDOFF。输出 environment.json、asset_inventory.json、data_roles.json 和源码改动记录，保存到新 results/paper_experiments_20260908/，旧结果目录只读。

必须先补齐的入口能力：

1. ROOT与视觉项目路径可配置，不依赖 /home/upone；保持模型语义。
2. 模型构建与最终确认数据解耦。原 make_adapted→data_for_state 会载入旧确认；新训练入口只允许train/tune，最终确认接口独立。
3. 训练支持 max_steps、lr、seed、T、output_dir、resume；每256步保存原子checkpoint，含model、optimizer、step、累计训练秒数、数据generator及Python/NumPy/CPU/CUDA随机数状态。
4. 保存配置后逐任务执行；不沿用旧queue的wait-pid、旧complete.json跳过规则及自动research_target_round10报告改写。
5. 评估支持不整除batch的尾批；233不能被batch4整除，图评估须另建batch1尾图或明确尾批使用eager；禁止用广播复制最后一条后重复计数。
6. 质量输出保留block_id、mask计数、loss总和、正确数和预测；计时输出保留每次调用记录。
7. 正式实现接收本目录任务配置。本轮交付的是协议与任务清单，新的可运行入口尚待服务器环境确定后实现，不能把本文件当成已验证启动器。

## 3. E1：只检查改变的关键路径

只在train留出和旧192块上进行。

- 各T加载一个旧512步checkpoint，在新环境复核旧预测及块损失，保存差异。跨GPU/库版本不预设逐位相等；有变化则区分数值迁移与权重/实现错误。跨GPU旧预测不等不自动否定新环境内的eager/graph比较。
- 每个架构batch1、T2/T4进行一次A/B/A动态输入与状态reset检查，比较同checkpoint、同batch、同精度的全部logits。后续E4各shape自身正确性检查是图捕获条件，不是额外模型实验。
- 新训练保存/恢复路径只做一次短检查：同seed连续两步，对比一步保存恢复后第二步的权重、optimizer和采样一致性。该输出不得计入正式训练曲线或证据。
- 从该检查和首条正式任务的计时估算新服务器资源需求。出现可定位的实现问题修复一次后复查受影响路径，不重复整套预实验。

输出保持的严格主条件为新环境内 torch.equal(eager_logits, graph_logits)。若失败，保存最大绝对差、预测变化、CE差并排查reset/图输入；仍失败则该单元不属于“保持输出的执行干预”，不能通过放宽容差混入主比较。

## 4. E2：对称学习率选择

|项目|固定值|
|---|---|
|T|2、4|
|lr候选|6e-6、2e-5、6e-5、2e-4|
|筛选seed|8611、8612|
|每条长度|2048更新，从同一融合初始化重新开始|
|优化器|AdamW，weight_decay=0，clip_grad_norm=1|
|有效batch|8，microbatch2 × 累积4|
|训练样本及mask|沿用原有有放回采样与动态mask；同seed的T共享采样流|
|学习率调度|固定lr，无warmup、无scheduler，与旧固定lr轨迹口径一致|
|tune评估点|512、1024、2048|
|lr选择规则|各T分别按两个seed在2048步的token加权CE均值最小选择；精确并列取较小lr|

共16条轨迹、32768更新。训练前列出完整任务矩阵；不因早期质量差提前淘汰某个lr。不查看官方test或233块分数。数值发散按失败候选保留，不能悄悄补跑更多lr；若某T全部候选失效，先修实现或在确认前声明协议修订。

记录每个T的搜索GPU时间、更新数、样本数。相同候选与更新预算不是相同GPU小时；论文应说明主设计控制的是训练机会/数据暴露。

## 5. E3：正式训练与两种预算比较

正式seed为8621、8622、8623。各T使用E2选中的单一lr，在同一融合初始化上从零适配4096更新。不要把旧不同lr的512步权重接上新lr并称同一轨迹。

共6条轨迹、24576更新。保留每256步checkpoint，tune在512/1024/2048/4096评估。任何评估保存并恢复随机数状态，不能改变训练采样轨迹。累计训练时间包含取样、传输、前后向及更新和必要同步；不包含tune评估、checkpoint写盘和进程初始化，这些额外成本分别记录。

**同更新比较：** 4096步为新实验主终点；512/1024/2048为预算曲线。4096并不代表充分收敛，结论限于本lr集合和长度。即使tune早期更好，也不事后将主终点换成最佳checkpoint；曲线可解释过拟合。

**等训练时间比较：** 复用上述六条轨迹，定义 h 为六条完整轨迹累计训练秒数的最小值，预定时间阈值为 h/4、h/2、h。对每条轨迹选最后一个不超过阈值的256步checkpoint，不插值模型质量。记录实际秒数、更新数与距阈值差。若距阈值超过10%，该点只作离散预算观察，不称严格等时间结果；不因此增加新训练。时间阈值由运行时间决定，不由质量挑选。该分析沿用按相同更新选出的lr，不能声称已完成等GPU小时超参搜索。

执行次序按seed配对轮换：8621 T2→T4，8622 T4→T2，8623 T2→T4；同GPU无其他训练或计时任务争用。记录中断和续训，不混合不同GPU测得的训练时间。

## 6. E4：执行方式、batch和读出控制

### E4a：历史模型新GPU复现

- SmoothSpike冻结原始模型与SpikingResformer原30 epoch模型；各T2/T4、batch1/4/16/64。
- 每架构三个独立Python进程会话：8801、8802、8803。每会话重新加载模型与捕获graph。
- 三方式：eager、graph、graph_with_copy。后者是GPU常驻源输入到静态图输入的D2D复制，**不是CPU→GPU传输或端到端服务延迟**。
- 每单元3轮交替顺序×10次调用/方式；按会话轮换初始方式与T顺序。共2×3×2×4×3×3×10=4320次记录，不含预热和正确性检查。
- 输入用原train留出；视觉用原训练索引的确定性评估变换。正确性可复用旧192块/1024图，不动最终确认数据。
- 每shape检查A/B/A与reset。统一推理模式；记录wall_ms、cuda_ms、显存、预热+捕获秒数、GPU/驱动/软件环境。
- OOM或不支持图捕获时保留缺失单元及原因。不能只为T4降低batch或精度；在两T均可运行的共同网格比较。
- 若新机器与RTX3090不同型号，提供新的硬件配置证据；若软件也变化，结论称跨硬件/软件环境复测，不能把差异全归因于GPU。

### E4b：同一正式模型的直接时延

对E3六份4096步checkpoint执行与E4a相同三方式，batch1/4/16/64，每checkpoint独立进程；三轮×10次。共6×4×3×3×10=2160次。每个点绑定checkpoint_id，不将三个训练seed冒充三个同权重新会话。

旧512步六权重若可恢复，新GPU上补测batch1/4用于连接历史质量结果：1080次。该项缺失只影响旧模型的新GPU图，不以冻结模型时延替代。

### E4c：必要输出位置控制

SmoothSpike原冻结模型T2/T4，batch1/4/16/64，full_head和masked_head × eager/graph，沿用旧mask位置数量与输入，一形状一图。共960次调用。同route的eager/graph必须逐位等价；full与masked之间矩阵shape变化，报告logit最大误差、masked预测变化及CE差，不预设FP32逐位一致。

旧实验最大差约4.6e-5且预测无变化；新环境仍需检查。沿用旧工程门槛max_abs_logit<1e-3且预测变化为0后才将其称为任务预测保持控制；这不等于形式等价证明。旧192块质量足够，本控制不消耗新确认集。正式4096步模型的masked_head扩展不在必做范围。

### 时延统计与成本解释

每会话/单元以30次调用的中位数为汇总；报告三个会话的独立比值，不把90次调用当90个独立实验。逐调用p10/p90可说明波动，不作生产服务p99宣称。batch吞吐=B/(wall_ms/1000)，明确单位为序列/秒或图像/秒。

主eager/graph均不含输入复制，复制敏感性另以同样D2D复制的eager_with_copy和graph_with_copy比较。为避免额外完整网格，eager_with_copy仅在两架构batch1/4、T2/T4的三个会话测三轮×10次，共720次，记为E4d；实际嵌入E4a的同一进程会话与交替计时轮次，不能事后换会话后仍称严格配对。GPU源输入、attention mask及必要位置索引的更新口径必须匹配。

预热+捕获成本单列。若单调用节省d>0，报告保守摊销调用数 ceil((预热+捕获秒数)×1000/d)；若d≤0，记无摊销收益。当前不是请求排队、tokenization、网络及CPU预处理的端到端服务测量，不报告能耗。

## 7. E5：确认集解封与统计

E2/E3完成后先保存 confirm_manifest.json：固定数据版本、mask、完整checkpoint清单、主次比较、统计代码版本及预算规则；E4性能结果不得改变质量模型选择。

一次性批处理同时评估：
1. 冻结T2/T4；
2. 可恢复的旧六份512步模型（历史假设确认）；
3. 新六条轨迹的512/1024/2048/4096端点；
4. E3预定三个时间阈值对应的checkpoint，去重后列入清单。

所有以上模型对主确认集和233块补充集分别评分，禁止合并它们扩大“独立测试”样本量。233块的主用途是延续旧计划，保留其上下文暴露说明。完整官方test块数、mask token数到E0时才确定，当前为TBD。

主新比较是4096步的Δmatch和Δunmatched；旧512步同类比较单独列为历史假设复核。其他预算与等时间为次分析。每seed报告CE、accuracy及配对差；CE为sum(loss_sum)/sum(mask_count)，不能平均每块CE忽略mask数量。

统计：
- 主表显示三训练seed均值±样本SD，另列每seed配对差。
- 数据不确定性：10000次配对块bootstrap，固定seed8901，同一次重采样对所有模型使用同一块索引；每次先按token总数求每seed CE，再平均seed。其95%区间明确条件于已训练seed。
- 连续文本块可能相关。保留原始顺序/文档映射，补充按原顺序8块分组的cluster bootstrap敏感性（不重复加入token）；若可恢复文档边界则优先按文档聚类。233块按原始块编号分组，不按随机排列当作连续语篇。
- 跨训练随机性：列出三配对seed差的均值、SD及t区间（df=2），说明仅三seed精度有限。不把块bootstrap当作训练总体置信区间。
- 判据不以“显著”代替效果大小。均值符号决定观察方向，三seed同号且条件块区间不跨0记为稳定方向证据；任一不满足均如实报告为不稳定/不确定，不追加seed直到通过。
- 不做多终点中择优的显著性声明；如后续需要正式多重假设检验，先修订并固定方案，不能解封后临时选检验。
- 报告实际幅度。计划中不设置“低于阈值就不展示”的筛除规则。

## 8. E6：结果结构与论文论断

|产物|必要行/轴|字段/解释|
|---|---|---|
|quality.csv|split × checkpoint × T × seed ×预算|CE、accuracy%、mask数、实际更新和训练秒数|
|quality_differences.csv|matched/unmatched ×预算 × split|各seed差、均值、SD、条件块区间、聚类敏感性|
|latency.csv|GPU/软件 ×架构 ×checkpoint ×batch ×engine ×session ×round ×repeat|wall_ms、cuda_ms、复制口径、显存与输出检查|
|budget_curves|横轴分别为更新数、训练秒数|train留出曲线与最终确认曲线分开，不混用|
|batch_boundary|batch→时延/比值|r_cross、r_same、各会话范围，保留无反转单元|
|quality_latency|x=该模型实际时延，y=CE或accuracy|相同GPU、batch、split、引擎分面；按全部seed展示|
|claim_ledger.md|RQ1—RQ5|支持/不支持/不确定、原始路径、可写范围|

质量—延迟前沿先报告非支配点，不事后按最有利阈值挑部署赢家。若需要质量容忍阈值，提供完整敏感性图，说明这是后续决策分析。

论文进入写作条件：主比较全部已报告或明确缺失；关键数字可回溯；确认数据身份准确；同引擎与batch边界齐全。假设反向也能进入写作，但题目、摘要和贡献必须按证据改写。尚无第二架构训练对照时，语言训练结论与两架构执行结论必须分开陈述。
