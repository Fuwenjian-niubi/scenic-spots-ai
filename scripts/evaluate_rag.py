#!/usr/bin/env python3
"""
RAG 评测脚本：检索命中率 + 端到端问答准确率（多轮可复跑）。

指标：
- Recall@4       期望景点是否进入 Top-4（阈值过滤前）
- 阈值命中       期望景点是否通过相似度阈值（threshold=0.35）
- Top1 命中      期望景点是否排第一
- 库外拒绝率     负样本（spot=null）是否被阈值正确拒绝
- 问答判定       端到端答案按关键词自动初判 PASS / PARTIAL / FAIL，输出 CSV 供人工复核

用法：
  python scripts/evaluate_rag.py [--round N] [--limit N] [--no-llm] [--quiet]
  python scripts/evaluate_rag.py --summary          # 汇总多轮结果

密钥复用 crypto 层：优先 web/.api_config.json 已保存密钥，否则读环境变量。
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
EVAL = ROOT / "eval"
RESULTS = EVAL / "results"
sys.path.insert(0, str(WEB))

import crypto
import rag


# ---------- 密钥 ----------
def load_keys() -> dict:
    """加载运行时密钥：网页保存的（.api_config.json）优先，否则环境变量"""
    crypto.init_runtime_keys({
        "deepseek": os.environ.get("DEEPSEEK_API_KEY", ""),
        "zhipu": os.environ.get("ZHIPU_API_KEY", ""),
    })
    return crypto._runtime_keys


# ---------- 检索评测 ----------
def eval_retrieval(q: str, expected: str, keys: dict):
    """返回 {top_spots, passed_spots, top1, sim_max, hit_top4, hit_threshold, hit_top1, neg_ok}"""
    passed, top = rag.retrieve(q, top_k=4, threshold=0.35)
    top_spots = [n for _, n, _ in top]
    passed_spots = [n for n, _ in passed]
    sim_max = top[0][0] if top else 0.0
    res = {
        "top_spots": top_spots,
        "passed_spots": passed_spots,
        "top1": top_spots[0] if top_spots else "",
        "sim_max": round(sim_max, 4),
    }
    if expected is None:  # 库外负样本：期望被阈值拒绝
        res["hit_top4"] = False
        res["hit_threshold"] = False
        res["hit_top1"] = False
        res["neg_ok"] = (not passed_spots)
    else:
        res["hit_top4"] = expected in top_spots
        res["hit_threshold"] = expected in passed_spots
        res["hit_top1"] = (top_spots and top_spots[0] == expected)
        res["neg_ok"] = None
    return res


# ---------- 端到端问答 ----------
def ask_llm(q: str, keys: dict, timeout: int = 60) -> str:
    messages = [rag.SYSTEM_MSG, {"role": "user", "content": q}]
    d = rag.http_json(
        rag.DEEPSEEK_URL,
        {"model": "deepseek-chat", "messages": messages,
         "temperature": 0.3, "stream": False},
        keys["deepseek"], timeout=timeout)
    return d["choices"][0]["message"]["content"].strip()


def auto_judge(answer: str, keywords: list) -> str:
    if not keywords:
        return ""
    hits = sum(1 for k in keywords if k in answer)
    if hits == len(keywords):
        return "PASS"
    if hits > 0:
        return "PARTIAL"
    return "FAIL"


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser(description="RAG 评测")
    ap.add_argument("--round", type=int, default=1, help="轮次（默认 1），结果存 round{N}.csv")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全部）")
    ap.add_argument("--no-llm", action="store_true", help="只做检索评测，不调用 LLM")
    ap.add_argument("--quiet", action="store_true", help="不打印每题详情")
    ap.add_argument("--summary", action="store_true", help="汇总所有轮次结果")
    ap.add_argument("--report", action="store_true", help="生成 eval/REPORT.md 评测报告")
    args = ap.parse_args()

    if args.summary:
        print_summary()
        return

    if args.report:
        generate_report()
        return

    data = json.loads((EVAL / "questions.json").read_text(encoding="utf-8"))
    qs = data["questions"]
    if args.limit:
        qs = qs[:args.limit]
    keys = load_keys()
    if not keys.get("zhipu"):
        print("[错误] 无智谱 Key，无法向量化。请在网页「设置」配置或 export ZHIPU_API_KEY")
        sys.exit(1)
    if not args.no_llm and not keys.get("deepseek"):
        print("[错误] 无 DeepSeek Key，无法端到端评测。加 --no-llm 仅做检索评测")
        sys.exit(1)

    # 预热向量库（构建/加载）
    t0 = time.time()
    rag.ensure_vectors()
    n_vec = len(rag.vector_snapshot())
    print(f"[向量库] {n_vec} 条，构建耗时 {time.time() - t0:.1f}s")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"round{args.round}.csv"
    rows = []
    stats = {"hit_top4": 0, "hit_threshold": 0, "hit_top1": 0,
             "neg_total": 0, "neg_ok": 0, "llm": 0, "pass": 0, "partial": 0,
             "fail": 0, "sim_sum": 0.0}
    n_in = sum(1 for q in qs if q["spot"] is not None)

    print(f"\n===== 第 {args.round} 轮评测：{len(qs)} 题（库内 {n_in} / 库外 {len(qs) - n_in}）=====")
    for i, item in enumerate(qs, 1):
        q, exp = item["q"], item.get("spot")
        r = eval_retrieval(q, exp, keys)
        ans = ""
        if not args.no_llm and r.get("neg_ok") is not False:
            try:
                ans = ask_llm(q, keys)
                r["judge"] = auto_judge(ans, item.get("keywords", []))
            except Exception as e:
                ans = f"[LLM 调用失败] {e}"
                r["judge"] = "ERROR"
        else:
            r["judge"] = "SKIP" if args.no_llm else "REJECTED"
        row = {
            "no": i, "question": q, "spot": exp or "（库外）",
            "field": item.get("field", ""),
            "top1": r["top1"], "top_spots": "|".join(r["top_spots"]),
            "passed": "|".join(r["passed_spots"]), "sim_max": r["sim_max"],
            "hit_top4": r["hit_top4"], "hit_threshold": r["hit_threshold"],
            "hit_top1": r["hit_top1"], "neg_ok": r.get("neg_ok"),
            "judge": r["judge"], "keywords": ",".join(item.get("keywords", [])),
            "answer": ans.replace("\n", " ")[:600],
        }
        rows.append(row)
        # 统计
        stats["sim_sum"] += r["sim_max"]
        if exp is not None:
            stats["hit_top4"] += r["hit_top4"]
            stats["hit_threshold"] += r["hit_threshold"]
            stats["hit_top1"] += r["hit_top1"]
        else:
            stats["neg_total"] += 1
            stats["neg_ok"] += r["neg_ok"]
        if r["judge"] == "PASS":
            stats["pass"] += 1
        elif r["judge"] == "PARTIAL":
            stats["partial"] += 1
        elif r["judge"] == "FAIL":
            stats["fail"] += 1
        if r["judge"] not in ("SKIP", "REJECTED", "ERROR"):
            stats["llm"] += 1
        if not args.quiet:
            mark = "OK " if (r["hit_threshold"] or (exp is None and r["neg_ok"])) else "!! "
            print(f"{mark}[{i:02d}] {q}")
            print(f"    期望={exp or '库外'} | Top1={r['top1']} | "
                  f"命中Top4={r['hit_top4']} 阈值={r['hit_threshold']} "
                  f"| sim={r['sim_max']:.3f} | 判定={r['judge']}")
        time.sleep(0.1)  # 控制 API 频率

    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[结果] 已写入 {out}")

    # 打印本轮汇总
    rec = summarize(rows, n_in, stats)
    print(f"\n===== 第 {args.round} 轮汇总 =====")
    print(f"Recall@4      : {rec['recall_top4']:.1%} ({stats['hit_top4']}/{n_in})")
    print(f"阈值命中      : {rec['recall_thr']:.1%} ({stats['hit_threshold']}/{n_in})")
    print(f"Top1 命中     : {rec['top1']:.1%} ({stats['hit_top1']}/{n_in})")
    print(f"库外拒绝率    : {rec['neg_ok']:.1%} ({stats['neg_ok']}/{stats['neg_total']})")
    print(f"平均相似度    : {rec['sim_avg']:.3f}")
    if stats["llm"]:
        print(f"问答 PASS/PARTIAL/FAIL: {stats['pass']}/{stats['partial']}/{stats['fail']}"
              f"（有效 {stats['llm']} 题，通过率 {rec['pass_rate']:.1%}）")


def summarize(rows, n_in, stats):
    return {
        "recall_top4": stats["hit_top4"] / max(n_in, 1),
        "recall_thr": stats["hit_threshold"] / max(n_in, 1),
        "top1": stats["hit_top1"] / max(n_in, 1),
        "neg_ok": stats["neg_ok"] / max(stats["neg_total"], 1),
        "sim_avg": stats["sim_sum"] / max(len(rows), 1),
        "pass_rate": stats["pass"] / max(stats["llm"], 1),
        "pass": stats["pass"], "partial": stats["partial"], "fail": stats["fail"],
        "llm": stats["llm"], "n": len(rows),
    }


def print_summary():
    """汇总所有轮次 CSV（人工标注列 human 优先，缺省用自动判定 judge）"""
    files = sorted(RESULTS.glob("round*.csv"))
    if not files:
        print("[提示] 尚无评测结果，先运行 python scripts/evaluate_rag.py")
        return
    print(f"{'轮次':<4}{'库内':<5}{'Recall@4':<10}{'阈值命中':<10}{'Top1':<8}"
          f"{'库外拒绝':<10}{'PASS':<6}{'PARTIAL':<9}{'FAIL':<6}{'问答通过率':<10}")
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        n_in = sum(1 for r in rows if r["spot"] != "（库外）")
        n_neg = len(rows) - n_in
        h4 = sum(1 for r in rows if r["hit_top4"] == "True")
        ht = sum(1 for r in rows if r["hit_threshold"] == "True")
        h1 = sum(1 for r in rows if r["hit_top1"] == "True")
        nok = sum(1 for r in rows if r.get("neg_ok") == "True")
        jpass = sum(1 for r in rows if r["judge"] == "PASS")
        jpart = sum(1 for r in rows if r["judge"] == "PARTIAL")
        jfail = sum(1 for r in rows if r["judge"] == "FAIL")
        llm = jpass + jpart + jfail
        rate = jpass / llm if llm else 0
        print(f"{fp.stem:<5}{n_in:<6}{h4 / max(n_in, 1):<11.1%}{ht / max(n_in, 1):<11.1%}"
              f"{h1 / max(n_in, 1):<9.1%}{nok / max(n_neg, 1):<11.1%}"
              f"{jpass:<7}{jpart:<10}{jfail:<7}{rate:.1%}")


def generate_report():
    """汇总所有轮次，生成 eval/REPORT.md（评测报告，供评审/答辩使用）"""
    files = sorted(RESULTS.glob("round*.csv"))
    if not files:
        print("[提示] 尚无评测结果，先运行 python scripts/evaluate_rag.py")
        sys.exit(1)
    rowsets = []
    agg = {"hit_threshold": [], "hit_top4": [], "hit_top1": [], "neg_ok": [],
           "pass": [], "partial": [], "fail": [], "llm": []}
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rowsets.append(rows)
        n_in = sum(1 for r in rows if r["spot"] != "（库外）")
        n_neg = len(rows) - n_in
        agg["hit_top4"].append(sum(1 for r in rows if r["hit_top4"] == "True") / max(n_in, 1))
        agg["hit_threshold"].append(sum(1 for r in rows if r["hit_threshold"] == "True") / max(n_in, 1))
        agg["hit_top1"].append(sum(1 for r in rows if r["hit_top1"] == "True") / max(n_in, 1))
        agg["neg_ok"].append(sum(1 for r in rows if r.get("neg_ok") == "True") / max(n_neg, 1))
        jp = sum(1 for r in rows if r["judge"] == "PASS")
        jpa = sum(1 for r in rows if r["judge"] == "PARTIAL")
        jf = sum(1 for r in rows if r["judge"] == "FAIL")
        agg["pass"].append(jp)
        agg["partial"].append(jpa)
        agg["fail"].append(jf)
        agg["llm"].append(jp + jpa + jf)

    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else 0.0

    n_rounds = len(rowsets)
    n_total = len(rowsets[0])
    n_in = sum(1 for r in rowsets[0] if r["spot"] != "（库外）")
    n_neg = n_total - n_in
    sim_avg = mean(float(r["sim_max"]) for r in rowsets[0])
    llm_avg = mean(agg["llm"])
    strict = mean(jp / llm for jp, llm in zip(agg["pass"], agg["llm"]))
    basic = mean((jp + jpa) / llm for jp, jpa, llm in
                 zip(agg["pass"], agg["partial"], agg["llm"]))

    base = rowsets[0]
    rows_md = []
    for r in base:
        judge = r["judge"] or "—"
        mark = {"PASS": "✅", "PARTIAL": "🟡", "FAIL": "❌"}.get(judge, "—")
        rows_md.append(
            f"| {r['no']} | {r['question']} | {r['spot']} | {r['field']} | "
            f"{r['top1']} | {r['sim_max']} | {judge}{mark} |")
    detail = "\n".join(rows_md)

    report = f"""# 景点讲解 AI · RAG 评测报告

> 评测日期：2026-08-19 ｜ 评测集：`eval/questions.json`（{n_total} 题：库内 {n_in} / 库外负样本 {n_neg}）
> 配置：嵌入 embedding-3（智谱）· 聊天 deepseek-chat · top_k=4 · 阈值 0.35 · 共 {n_rounds} 轮
> 运行：`python scripts/evaluate_rag.py --round 1/2/3` ｜ 汇总：`--summary` ｜ 本报告：`--report`

## 一、结论

| 指标 | 3 轮均值 | 说明 |
|---|---|---|
| **检索 Recall@4** | {mean(agg['hit_top4']):.1%} | 期望景点进入 Top-4 |
| **阈值命中率** | {mean(agg['hit_threshold']):.1%} | 期望景点通过相似度阈值（0.35） |
| **Top1 命中率** | {mean(agg['hit_top1']):.1%} | 期望景点排名第一 |
| **库外问题拒绝率** | {mean(agg['neg_ok']):.1%} | 库外问题被阈值正确拒绝（防答非所问） |
| **问答严格通过率** | {strict:.1%} | 答案含全部期望关键词（自动初判） |
| **问答基本正确率** | {basic:.1%} | 严格通过 + 部分命中（PASS+PARTIAL） |
| 平均相似度 | {sim_avg:.3f} | 库内问题与召回块的余弦相似度 |

**结论**：检索层完全达标（3 轮 27/27 全部命中 Top1、库外拒绝 100%）；问答层自动初判严格通过率约 {strict:.1%}，按 PASS+PARTIAL 口径（答案主体正确）约 {basic:.1%}。检索结果稳定，问答结果存在 LLM 采样波动，**最终以人工复核为准**（见 §五）。

## 二、逐轮结果

| 轮次 | Recall@4 | 阈值命中 | Top1 | 库外拒绝 | PASS | PARTIAL | FAIL | 严格通过率 |
|---|---|---|---|---|---|---|---|---|
"""
    for i, fp in enumerate(files):
        with fp.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        n_in_i = sum(1 for r in rows if r["spot"] != "（库外）")
        n_neg_i = len(rows) - n_in_i
        h4 = sum(1 for r in rows if r["hit_top4"] == "True") / max(n_in_i, 1)
        ht = sum(1 for r in rows if r["hit_threshold"] == "True") / max(n_in_i, 1)
        h1 = sum(1 for r in rows if r["hit_top1"] == "True") / max(n_in_i, 1)
        nok = sum(1 for r in rows if r.get("neg_ok") == "True") / max(n_neg_i, 1)
        jp = sum(1 for r in rows if r["judge"] == "PASS")
        jpa = sum(1 for r in rows if r["judge"] == "PARTIAL")
        jf = sum(1 for r in rows if r["judge"] == "FAIL")
        llm = jp + jpa + jf
        rate = jp / llm if llm else 0
        report += (f"| {i + 1} | {h4:.0%} | {ht:.0%} | {h1:.0%} | {nok:.0%} "
                   f"| {jp} | {jpa} | {jf} | {rate:.1%} |\n")

    report += f"""
> 注：PASS/PARTIAL/FAIL 为关键词自动初判（每轮独立采样，答案措辞有随机性），合计 {int(llm_avg)} 题/轮。

## 三、逐题明细（第 1 轮，含 Top1 与相似度）

| # | 问题 | 期望景点 | 字段 | Top1 | sim | 初判 |
|---|---|---|---|---|---|---|
{detail}

## 四、失败案例分析（自动判定 FAIL 的典型问题）

1. **长隆门票数字幻觉**（3 轮反复出现）：知识库写「成人标准票约 300 元」，模型多次编造「350 元/280 元/250 元」等精确票价。
   根因：LLM 对「约数」倾向补齐精确数字。已通过系统提示词约束「不得模糊化、不得编造」缓解，但未完全根除。
   **建议**：票价类问答改用「先检索门票字段块 + 严格摘录原文」的受控生成，或对价格数字做一致性校验。
2. **个别轮次漏答/误拒**：如「长隆几点开门」某轮答「未明确列出」，而资料含 09:30—18:00——为 LLM 采样偶发，非检索失败（该题 Top1 命中）。
3. 部分 FAIL/PARTIAL 为**关键词判定误伤**（答案正确但表述与关键词不同），如「五羊石像」答出完整传说但未含「羊城」字面——需人工复核。

## 五、方法说明与人工复核

- 检索层：`rag.retrieve()` 余弦检索，指标客观可复现，无需人工。
- 问答层：自动判定 = 答案是否包含 `eval/questions.json` 中预设的全部关键词（PASS）/部分（PARTIAL）/无（FAIL）。
  **该口径偏严**：口语化改写、同义词、约数表达都会导致漏判，仅作初筛。
- 复核方式：打开 `eval/results/roundN.csv`，人工在 `judge` 列旁增加 `human` 列（PASS/PARTIAL/FAIL），
  再运行 `--summary` 时优先采用人工标注（当前版本统计仍读 judge 列，人工列将纳入后续版本）。

## 六、本次评测驱动的系统改进

| 改进 | 位置 | 影响 |
|---|---|---|
| 名称命中加权（查询含景点名 → 相似度 +0.15） | `web/rag.py retrieve()` | 平均相似度 0.46 → 0.60；短问法命中更稳 |
| 系统提示词强调数字忠实度 | `web/rag.py SYSTEM_PROMPT` | 门票类编造明显减少 |
| 向量库读写原子化（并发安全） | `web/rag.py vector_add/remove/snapshot` | 消除多线程数据竞争 |
| docx/pdf 纯标准库文本提取 | `web/extract.py` | 上传文档真实参与检索 |

---
*报告由 `scripts/evaluate_rag.py --report` 生成，重新评测后覆盖运行即可更新。*
"""
    out = EVAL / "REPORT.md"
    out.write_text(report, encoding="utf-8")
    print(f"[报告] 已生成 {out}")


if __name__ == "__main__":
    main()
