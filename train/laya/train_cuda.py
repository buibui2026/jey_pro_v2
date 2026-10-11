"""T5.4 云端训练脚本：RTX 4090 / 24GB / CUDA 12.4.1 —— 单卡全参微调 300 条。

官方 train_ddp.py 改造（T5.2 试跑门已过）：
- DDP → 单卡；autocast fp16 + GradScaler 启用（4090 支持）
- max_len 1024 → 3200（M4 配比 v1.3 真实上沿）；head_max_len 256
- 数据：laya_trainset.jsonl 300 条 → encode_record（含 calib 切片隔离）
- 训练核心一行不动：REINFORCE＋GRPO×4＋高斯探索＋软CE＋LBFGS 温度＋梯度裁剪＋余弦退火
- 4090 24GB vs T4×2 16GB×2：batch 可放大（MICRO_BATCH=12，官方 8×2GPU），余量更足

运行：
  python3 laya_ft/train_cuda.py --dry-run        # 10 条干跑验证
  python3 laya_ft/train_cuda.py --epochs 4       # 全量训练
产物：laya_ft/output/（model.safetensors + rl_agent_config.json + encoder/ + tokenizer/ + train_log.json）
"""
import argparse, json, os, random, sys, time
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))
from rl_common import build_model, proper_reward, QTYPES, encode_record, collate_items
TEMP_MIN, TEMP_MAX = 0.1, 10.0

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, 'model', 'multilingual')
INS = {'ctx_relevance': '检索到的法条对回答该问题有多大帮助？', 'ctx_sufficiency': '仅凭检索到的法条，是否足以回答该问题？',
       'groundedness': '答案中的论断被检索法条支持的程度？', 'citation_correct': '答案引用的法条号是否正确？',
       'answers_question': '答案是否正面回答了所问的问题？', 'completeness': '答案的完整程度？',
       'answer_correctness': '答案与参考答案的实质一致程度？', 'overall': '综合检索与答案，这次问答的整体质量？'}

def row_to_rec(r):
    qs = []
    for qid, lab in r['labels'].items():
        t = {'ctx_relevance': 'choice', 'groundedness': 'choice', 'citation_correct': 'choice',
             'answer_correctness': 'choice', 'ctx_sufficiency': 'noul', 'answers_question': 'noul',
             'completeness': 'score', 'overall': 'score'}[qid]
        pr = lab.get('probs', {})
        if t == 'choice':
            crit = {k: None for k in pr}; soft = [pr[k] for k in pr]
        elif t == 'score':
            crit = list(pr); soft = [pr[k] for k in crit]
        else:
            crit = None; soft = [pr.get('false', 0.0), pr.get('true', 0.0)]
        qs.append({'t': t, 'ins': INS[qid], 'crit': crit, 'soft': soft})
    return {'state': r['state'], 'qs': qs, 'src': f"m4_k{r['k']}"}

def fit_one_temp(sel):
    if len(sel) < 10: return 1.0
    Z = torch.stack([z for z, _ in sel]); T = torch.stack([t for _, t in sel])
    log_t = torch.zeros(1, requires_grad=True)
    lo = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    def closure():
        lo.zero_grad(); l = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean(); l.backward(); return l
    lo.step(closure)
    return float(torch.clamp(log_t.exp(), TEMP_MIN, TEMP_MAX).item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--micro-batch', type=int, default=12)
    args = ap.parse_args()

    device = torch.device('cuda')
    torch.cuda.set_device(0)
    print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB', flush=True)

    cfg = json.load(open(os.path.join(D, 'rl_agent_config.json')))
    cfg.update(max_len=3200, head_max_len=256, gradient_checkpointing=True)
    tok = AutoTokenizer.from_pretrained(os.path.join(D, 'tokenizer'))
    model = build_model(cfg, encoder_dir=os.path.join(D, 'encoder'))
    model.load_state_dict(load_file(os.path.join(D, 'model.safetensors')), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.head_checkpointing = True
    model.to(device); model.train()

    # 数据
    rows = [json.loads(l) for l in open(os.path.join(BASE, 'data', 'laya_trainset.jsonl'))]
    if args.dry_run:
        rows = rows[:10]; args.epochs = 1; print('=== DRY RUN（10 条×1 epoch）===', flush=True)
    items_all = []
    for rec in map(row_to_rec, rows):
        items_all += encode_record(rec, tok, cfg, random.Random(42), train=True)
    # calib 切片隔离（官方逻辑：1/10，固定种子，calib 标记优先）
    order = list(range(len(items_all)))
    random.Random(20260922).shuffle(order)
    n_calib = max(10, len(items_all) // 10)
    calib_idx = set(order[:n_calib])
    calib_items = [items_all[i] for i in sorted(calib_idx)]
    train_items = [items_all[i] for i in sorted(order[n_calib:])]
    print(f'{len(rows)} 条 → {len(items_all)} 序列 | 训练 {len(train_items)} | calib {len(calib_items)}', flush=True)

    # 超参（官方值，仅 batch 放大）
    EPOCHS = args.epochs
    MICRO_BATCH = args.micro_batch
    GRAD_ACCUM = 2
    GROUP_SIZE = 4
    LR_ENC, LR_HEAD = 2.5e-5, 1.0e-4
    SIGMA_START, SIGMA_END = 0.4, 0.1
    enc_p = [p for n, p in model.named_parameters() if 'encoder.' in n]
    head_p = [p for n, p in model.named_parameters() if 'encoder.' not in n]
    opt = torch.optim.AdamW([{'params': enc_p, 'lr': LR_ENC}, {'params': head_p, 'lr': LR_HEAD}], weight_decay=0.01)
    total_updates = max(1, len(train_items) // (MICRO_BATCH * GRAD_ACCUM)) * EPOCHS
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    t0 = time.time(); loss_log = []; rew_log = []
    for epoch in range(EPOCHS):
        random.Random(42 + epoch).shuffle(train_items)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * (epoch / max(1, EPOCHS - 1))
        opt.zero_grad(set_to_none=True)
        for bi in range(0, len(train_items), MICRO_BATCH):
            chunk = train_items[bi:bi + MICRO_BATCH]
            if not chunk: continue
            b = collate_items([chunk], tok.pad_token_id)
            with torch.autocast('cuda', dtype=torch.float16):
                logits, act = model(b['input_ids'].to(device), b['attention_mask'].to(device),
                                    b['marker_pos'].to(device), b['marker_mask'].to(device), b['qtype'].to(device))
            logits = logits.float()
            mask = b['marker_mask'].to(device); k = mask.sum(-1, keepdim=True).float()
            target = b['target'].to(device)
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), b['qtype'].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True); adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()
            scaler.scale(loss).backward()
            if (bi // MICRO_BATCH + 1) % GRAD_ACCUM == 0 or bi + MICRO_BATCH >= len(train_items):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
            loss_log.append(round(loss.item() * GRAD_ACCUM, 4)); rew_log.append(round(r.mean().item(), 3))
            if len(loss_log) % 20 == 0:
                print(f'  ep{epoch+1} step{len(loss_log)} loss={loss.item()*GRAD_ACCUM:.4f} rew={r.mean().item():.3f} '
                      f'lr={sched.get_last_lr()[0]:.2e} vram={torch.cuda.memory_allocated()/1e9:.1f}G ({time.time()-t0:.0f}s)', flush=True)
        ep_avg = sum(loss_log[-(len(train_items)//MICRO_BATCH):]) / max(1, len(train_items)//MICRO_BATCH)
        print(f'=== Epoch {epoch+1}/{EPOCHS} avg={ep_avg:.4f} ({time.time()-t0:.0f}s) ===', flush=True)

    # 温度拟合（calib 切片，LBFGS 逐题型）
    print('Fitting temperatures on calib slice...', flush=True)
    del opt, scaler, sched; torch.cuda.empty_cache(); model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(calib_items), 16):
            cb = collate_items([calib_items[i:i+16]], tok.pad_token_id)
            with torch.autocast('cuda', dtype=torch.float16):
                lg, _ = model(cb['input_ids'].to(device), cb['attention_mask'].to(device),
                              cb['marker_pos'].to(device), cb['marker_mask'].to(device), cb['qtype'].to(device))
            for rr in range(lg.size(0)):
                kk = int(cb['marker_mask'][rr].sum())
                preds.append((int(cb['qtype'][rr]), lg[rr, :kk].float().cpu(), cb['target'][rr, :kk]))
    temps = []
    for qt in range(3):
        sel = [(z, t) for q, z, t in preds if q == qt]
        temps.append(round(fit_one_temp(sel), 3) if sel else 1.0)
    print(f'Temperatures (choice, score, noul): {temps}', flush=True)

    # 保存
    out = os.path.join(BASE, 'output'); os.makedirs(out, exist_ok=True)
    sd = {kk: v.half().contiguous().cpu() for kk, v in model.state_dict().items()}
    save_file(sd, os.path.join(out, 'model.safetensors'))
    model.encoder.config.save_pretrained(os.path.join(out, 'encoder'))
    tok.save_pretrained(os.path.join(out, 'tokenizer'))
    cfg.update(fine_tuned=True, model_name='laya-legal-rag-judge', temperature=temps)
    cfg.pop('temperature_by_options', None)
    json.dump(cfg, open(os.path.join(out, 'rl_agent_config.json'), 'w'), indent=2)
    json.dump({'loss_curve': loss_log, 'reward_curve': rew_log, 'temps': temps,
               'train_items': len(train_items), 'calib_items': len(calib_items),
               'epochs': EPOCHS, 'total_secs': round(time.time() - t0)},
              open(os.path.join(out, 'train_log.json'), 'w'), indent=1)
    print(f'✅ 训练完成 → {out}/', flush=True)

if __name__ == '__main__':
    main()
