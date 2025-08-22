import json
from collections import Counter

gold_counter = Counter()
pred_counter = Counter()
tp_counter = Counter()

with open('../output/falcon_diversity.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())

        gold_types = [e['type'] for e in data.get('gold_entities', [])]
        pred_types = [e['type'] for e in data.get('pred_entities', [])]

        gold_counter.update(gold_types)
        pred_counter.update(pred_types)

        # 类型级别 TP: 预测中出现在真实中的
        matched = Counter(gold_types) & Counter(pred_types)
        tp_counter.update(matched)

# 所有出现过的类型
all_labels = sorted(set(gold_counter) | set(pred_counter))

print("Label\tPrec\tRec\tF1\tTP\tFP\tFN")
for label in all_labels:
    tp = tp_counter[label]
    pred = pred_counter[label]
    gold = gold_counter[label]

    prec = tp / pred if pred > 0 else 0.0
    rec = tp / gold if gold > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print(f"{label}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t{tp}\t{pred - tp}\t{gold - tp}")

# Micro平均
total_tp = sum(tp_counter.values())
total_pred = sum(pred_counter.values())
total_gold = sum(gold_counter.values())

micro_prec = total_tp / total_pred if total_pred > 0 else 0.0
micro_rec = total_tp / total_gold if total_gold > 0 else 0.0
micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

print("\nMicro Precision: {:.4f}".format(micro_prec))
print("Micro Recall:    {:.4f}".format(micro_rec))
print("Micro F1 Score:  {:.4f}".format(micro_f1))