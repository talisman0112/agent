======================================================================
  FinSight RAGAS 端到端评估报告 v2
======================================================================

评测集: golden_set.yml (37 题，8 个类别，清理虚构 demo 后)
新题: q31-q40 基于宁德时代 2026Q1 真实财报 + 补充金融术语
管道: RAGSummarize (local), rerank_enabled=True
运行时间: 894s (~15 min)

—— 总体指标 ——
  answer_relevancy      mean=0.7109  std=0.3679  n=37
  context_recall        mean=0.4054  std=0.4402  n=37
  faithfulness          mean=0.7147  std=0.2783  n=37

—— 按类别汇总 ——
  金融术语         ( 9 题): faithfulness=0.647  answer_relevancy=0.630  context_recall=0.519
  估值方法         ( 5 题): faithfulness=0.871  answer_relevancy=0.876  context_recall=0.500
  财务报表         ( 4 题): faithfulness=0.610  answer_relevancy=0.862  context_recall=0.625
  新能源汽车        ( 5 题): faithfulness=0.680  answer_relevancy=0.732  context_recall=0.400
  半导体          ( 4 题): faithfulness=0.757  answer_relevancy=0.937  context_recall=0.500
  AI 算力        ( 5 题): faithfulness=0.783  answer_relevancy=0.808  context_recall=0.267
  宁德时代（新增）     ( 5 题): faithfulness=0.696  answer_relevancy=0.272  context_recall=0.000

—— 各题详情 ——
  q01  f=0.750 r=0.821 c=0.000 ctx=5
  q02  f=0.714 r=0.963 c=1.000 ctx=3
  q03  f=1.000 r=0.932 c=1.000 ctx=4
  q04  f=0.909 r=0.988 c=0.667 ctx=3
  q05  f=0.778 r=0.987 c=1.000 ctx=4
  q06  f=0.833 r=0.983 c=1.000 ctx=3
  q07  f=0.393 r=0.000 c=0.000 ctx=3
  q08  f=0.955 r=0.996 c=1.000 ctx=4
  q09  f=1.000 r=0.967 c=1.000 ctx=4
  q10  f=0.750 r=0.994 c=0.000 ctx=2
  q11  f=0.900 r=0.834 c=0.000 ctx=1
  q12  f=0.389 r=0.772 c=1.000 ctx=2
  q13  f=0.617 r=0.928 c=0.000 ctx=2
  q14  f=1.000 r=0.999 c=1.000 ctx=3
  q15  f=0.435 r=0.748 c=0.500 ctx=2
  q16  f=0.824 r=0.920 c=0.000 ctx=2
  q17  f=0.917 r=0.832 c=0.500 ctx=2
  q18  f=0.407 r=0.914 c=1.000 ctx=2
  q19  f=1.000 r=0.992 c=0.500 ctx=5
  q20  f=0.944 r=0.985 c=0.500 ctx=1
  q21  f=1.000 r=0.880 c=1.000 ctx=3
  q22  f=0.250 r=0.922 c=0.000 ctx=3
  q23  f=0.833 r=0.958 c=0.500 ctx=2
  q24  f=1.000 r=0.975 c=1.000 ctx=3
  q25  f=0.833 r=0.768 c=0.333 ctx=2
  q26  f=0.870 r=0.820 c=0.000 ctx=2
  q27  f=0.444 r=0.974 c=0.000 ctx=3
  q31  f=0.857 r=0.689 c=0.000 ctx=5
  q32  f=0.833 r=0.673 c=0.000 ctx=5
  q33  f=0.900 r=0.000 c=0.000 ctx=1
  q34  f=0.000 r=0.000 c=0.000 ctx=2
  q35  f=0.889 r=0.000 c=0.000 ctx=1
  q36  f=0.200 r=0.000 c=0.000 ctx=3
  q37  f=0.250 r=0.000 c=0.000 ctx=3
  q38  f=0.750 r=0.586 c=0.500 ctx=4
  q39  f=0.769 r=0.501 c=0.000 ctx=3
  q40  f=0.250 r=0.000 c=0.000 ctx=3

—— v1 vs v2 对比 ——
  v1: 30 题 (含虚构 demo)  faithfulness=0.767  answer_relevancy=0.841  context_recall=0.517
  v2: 37 题 (删虚构+增宁德时代) faithfulness=0.715  answer_relevancy=0.711  context_recall=0.405

  v2 分数下降主要原因:
  1. 新增 company_filings(宁德时代) 题要求精确数字检索，Rerank 过滤后仅保留 1-2 个 chunk
  2. 新增 q36/q37 的参考答案包含 KB 未覆盖的通识知识，拉低了 context_recall
  3. q33/q35 只检索到 1-2 个 chunk，导致 answer_relevancy 为 0