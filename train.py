from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import math
import os
import random
import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm
from models.generator import Generator
from utils import helper
from utils.data_utils import prepare_dataset, MultiWozDataset
from utils.constant import track_slots, ansvocab, slot_map, n_slot, TURN_SPLIT, TEST_TURN_SPLIT
from utils.data_utils import make_slot_meta, domain2id, OP_SET, make_turn_label, postprocessing
from evaluation import op_evaluation, joint_evaluation
# from transformers.configuration_albert import AlbertConfig
from transformers import AlbertConfig
# from transformers.tokenization_albert import AlbertTokenizer
from transformers import AlbertTokenizer
from transformers.optimization import AdamW
# from pytorch_transformers import AdamW, WarmupLinearSchedule
from transformers import get_linear_schedule_with_warmup
from utils.logger import get_logger

import sys
import csv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

logger_trainInfo = get_logger("train_logger","./saved_models/logger.log")
logger_trainInfo.info("")

csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

'''compute joint operation scores based on logits of two stages 
'''
def compute_jointscore(start_scores, end_scores, gen_scores, pred_ops, ans_vocab, slot_mask):
    seq_lens = start_scores.shape[-1]
    joint_score = start_scores.unsqueeze(-2) + end_scores.unsqueeze(-1)
    triu_mask = np.triu(np.ones((joint_score.size(-1), joint_score.size(-1))))
    triu_mask[0, 1:] = 0
    triu_mask = (torch.Tensor(triu_mask) ==  0).bool()
    joint_score = joint_score.masked_fill(triu_mask.unsqueeze(0).unsqueeze(0).cuda(),-1e9).masked_fill(
        slot_mask.unsqueeze(1).unsqueeze(-2)  == 0, -1e9)
    joint_score = F.softmax(joint_score.view(joint_score.size(0), joint_score.size(1), -1),
                            dim = -1).view(joint_score.size(0), joint_score.size(1), seq_lens, -1)

    score_diff = (joint_score[:, :, 0, 0] - joint_score[:, :, 1:, 1:].max(dim = -1)[0].max(dim = -1)[
        0])
    score_noans = pred_ops[:, :, -1] - pred_ops[:, :, 0]
    slot_ans_count = (ans_vocab.sum(-1) != 0).sum(dim=-1)-2
    ans_idx = torch.where(slot_ans_count < 0, torch.zeros_like(slot_ans_count), slot_ans_count)
    neg_ans_mask = torch.cat((torch.linspace(0, ans_vocab.size(0) - 1,
                                             ans_vocab.size(0)).unsqueeze(0).long(),
                              ans_idx.unsqueeze(0)),
                             dim = 0)
    neg_ans_mask = torch.sparse_coo_tensor(neg_ans_mask, torch.ones(ans_vocab.size(0)),
                                           (ans_vocab.size(0),
                                            ans_vocab.size(1))).to_dense().cuda()
    score_neg = gen_scores.masked_fill(neg_ans_mask.unsqueeze(0) == 0, -1e9).max(dim=-1)[0]
    score_has = gen_scores.masked_fill(neg_ans_mask.unsqueeze(0) == 1, -1e9).max(dim=-1)[0]
    cate_score_diff = score_neg - score_has
    score_diffs = score_diff.view(-1).cpu().detach().numpy().tolist()
    cate_score_diffs = cate_score_diff.view(-1).cpu().detach().numpy().tolist()
    score_noanses = score_noans.view(-1).cpu().detach().numpy().tolist()
    return score_diffs, cate_score_diffs, score_noanses


def saveOperationLogits(model, device, dataset, save_path, turn):
    score_ext_map = {}
    model.eval()
    for batch in tqdm(dataset, desc = "Evaluating"):
        batch = [b.to(device) if not isinstance(b, int) and not isinstance(b, list) else b for b in batch]
        input_ids, input_mask, slot_mask, segment_ids, state_position_ids, op_ids, pred_ops, domain_ids, gen_ids, start_position, end_position, max_value, max_update, slot_ans_ids, start_idx, end_idx, sid = batch
        batch_size = input_ids.shape[0]
        seq_lens = input_ids.shape[1]
        start_logits, end_logits, has_ans, gen_scores, _, _, _ = model(input_ids = input_ids,
                                                                       token_type_ids = segment_ids,
                                                                       state_positions = state_position_ids,
                                                                       attention_mask = input_mask,
                                                                       slot_mask = slot_mask,
                                                                       max_value = max_value,
                                                                       op_ids = op_ids,
                                                                       max_update = max_update)

        score_ext = has_ans.cpu().detach().numpy().tolist()
        for i, sd in enumerate(score_ext):
            score_ext_map[sid[i]] = sd
    with open(os.path.join(save_path, "cls_score_test_turn{}.json".format(turn)), "w") as writer:
        writer.write(json.dumps(score_ext_map, indent = 4) + "\n")

def compute_span_loss(gen_ids, input_ids, fuse_score, start_scores, end_scores, generate_turn, sample_mask, generate_mask):
    loss = 0
    for i in range(start_scores.shape[0]):
        if i >= 2:
            select_mask = [generate_turn[i][t] in fuse_score[i].argsort(dim=-1, descending=True)[:, :2][t] for t in range(30)]
            select_mask = torch.from_numpy(np.array(select_mask)).cuda()
            batch_mask = sample_mask[i] & ((generate_mask[i] == 0) | select_mask)
        else:
            batch_mask = sample_mask[i]
        start_idx = [-1 for i in range(n_slot)]
        end_idx = [-1 for i in range(n_slot)]
        for ti in range(n_slot):
            if not batch_mask[ti]:
                continue
            value = gen_ids[i][ti][0]
            value = value[:-1] if isinstance(value, list) else [value]
            batch_input = input_ids[i][ti].cpu().detach().numpy().tolist()
            for text_idx in range(len(input_ids[i][ti]) - len(value)):
                if batch_input[text_idx: text_idx + len(value)] == value:
                    start_idx[ti] = text_idx
                    end_idx[ti] = text_idx + len(value) - 1
                    break
        start_idx = torch.from_numpy(np.array(start_idx)).cuda()
        end_idx = torch.from_numpy(np.array(end_idx)).cuda()
        loss += masked_cross_entropy_for_value(start_scores[i].contiguous(),
                                                                start_idx.contiguous(),
                                                                sample_mask = batch_mask,
                                                                pad_idx = -1
                                                                )
        batch_loss = masked_cross_entropy_for_value(end_scores[i].contiguous(),
                                                end_idx.contiguous(),
                                                sample_mask=batch_mask,
                                                pad_idx=-1
                                                )
        loss += batch_loss
    loss /= start_scores.shape[0]
    return loss

def masked_cross_entropy_for_value(logits, target, sample_mask = None, slot_mask = None, pad_idx = -1):
    mask = logits.eq(0)
    pad_mask = target.ne(pad_idx)
    target = target.masked_fill(target < 0, 0)
    sample_mask = pad_mask & sample_mask if sample_mask is not None else pad_mask
    sample_mask = slot_mask & sample_mask if slot_mask is not None else sample_mask
    target = target.masked_fill(sample_mask ==  0, 0)
    logits = logits.masked_fill(mask, 1)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim = 1, index = target_flat)
    losses = losses_flat.view(*target.size())
    # if mask is not None:
    sample_num = sample_mask.sum().float()
    losses = losses * sample_mask.float()
    loss = (losses.sum() / sample_num) if sample_num != 0 else losses.sum()
    return loss

# [SLOT], [NULL], [EOS]
def addSpecialTokens(tokenizer, specialtokens):
    special_key = "additional_special_tokens"
    tokenizer.add_special_tokens({special_key: specialtokens})

def fixontology(ontology, turn, tokenizer):
    ans_vocab = []
    esm_ans_vocab = []
    esm_ans = ansvocab
    slot_mm = np.zeros((len(slot_map), len(esm_ans)))
    max_anses_length = 0
    max_anses = 0
    for i, k in enumerate(ontology.keys()):
        if k in track_slots:
            s = ontology[k]
            s['name'] = k
            if not s['type']:
                s['db'] = []
            slot_mm[i][slot_map[s['name']]] = 1
            ans_vocab.append(s)
    for si in esm_ans:
        slot_anses = []
        for ans in si:
            enc_ans=tokenizer.encode(ans)
            max_anses_length = max(max_anses_length, len(ans))
            slot_anses.append(enc_ans)
        max_anses = max(max_anses, len(slot_anses))
        esm_ans_vocab.append(slot_anses)
    for s in esm_ans_vocab:
        for ans in s:
            gap = max_anses_length - len(ans)
            ans += [0] * gap
        gap = max_anses - len(s)
        s += [[0] * max_anses_length] * gap
    esm_ans_vocab = np.array(esm_ans_vocab)
    ans_vocab_tensor = torch.from_numpy(esm_ans_vocab)
    slot_mm = torch.from_numpy(slot_mm).float()
    return ans_vocab, slot_mm, ans_vocab_tensor

def mask_ans_vocab(ontology, slot_meta, tokenizer):
    ans_vocab = []
    max_anses = 0
    max_anses_length = 0
    change_k = []
    cate_mask = []
    for k in ontology:
        if (' range' in k['name']) or (' at' in k['name']) or (' by' in k['name']):
            change_k.append(k)
    for key in change_k:
        new_k = key['name'].replace(' ', '')
        key['name'] = new_k
    for s in ontology:
        cate_mask.append(s['type'])
        v_list = s['db']
        slot_anses = []
        for v in v_list:
            ans = tokenizer.encode(v)
            max_anses_length = max(max_anses_length, len(ans))
            slot_anses.append(ans)
        max_anses = max(max_anses, len(slot_anses))
        ans_vocab.append(slot_anses)
    for s in ans_vocab:
        for ans in s:
            gap = max_anses_length - len(ans)
            ans += [0] * gap
        gap = max_anses - len(s)
        s += [[0] * max_anses_length] * gap
    ans_vocab = np.array(ans_vocab)
    ans_vocab_tensor = torch.from_numpy(ans_vocab)
    return ans_vocab_tensor, ans_vocab, cate_mask

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--model_type", default = 'albert', type = str,
                        help = "Model type selected in the list: ")
    parser.add_argument("--model_name_or_path", default = 'pretrained_models/albert_large/', type = str,
                        help = "Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--output_dir", default = "saved_models/", type = str,
                        help = "The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--config_name", default = "", type = str,
                        help = "Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default = "", type = str,
                        help = "Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default = "", type = str,
                        help = "Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_train", default = True, action = 'store_true',
                        help = "Whether to run training.")
    parser.add_argument("--evaluate_during_training", action = 'store_true',
                        help = "Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action = 'store_true',
                        help = "Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default = 32, type = int,
                        help = "Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default = 32, type = int,
                        help = "Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type = int, default = 1,
                        help = "Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default = 3e-5, type = float,
                        help = "The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default = 0.1, type = float,
                        help = "Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default = 1e-8, type = float,
                        help = "Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=5.0, type=float,
                        help = "Max gradient norm.")
    parser.add_argument("--num_train_epochs", default = 3.0, type = float,
                        help = "Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default = -1, type = int,
                        help = "If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default = 0, type = int,
                        help = "Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type = int, default = 50,
                        help = "Log every X updates steps.")
    parser.add_argument('--save_steps', type = int, default = 50,
                        help = "Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action = 'store_true',
                        help = "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action = 'store_true',
                        help = "Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', default = True, action = 'store_true',
                        help = "Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action = 'store_true',
                        help = "Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type = int, default = 42,
                        help = "random seed for initialization")

    parser.add_argument('--fp16', action = 'store_true',
                        help = "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type = str, default = 'O1',
                        help = "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--local_rank", type = int, default = -1,
                        help = "For distributed training: local_rank")  # DST params
    parser.add_argument("--data_root", default = 'data/mwz2.2/', type = str)
    parser.add_argument("--train_data", default = 'test_dials.json', type = str)
    parser.add_argument("--dev_data", default = 'test_dials.json', type = str)
    parser.add_argument("--test_data", default = 'test_dials.json', type = str)
    parser.add_argument("--ontology_data", default = 'schema.json', type = str)
    parser.add_argument("--vocab_path", default = 'assets/vocab.txt', type = str)
    parser.add_argument("--save_dir", default = 'saved_models', type = str)
    parser.add_argument("--load_model", default = False, action = 'store_true')
    parser.add_argument("--load_ckpt_epoch", default='checkpoint_epoch_9.bin', type=str)
    parser.add_argument("--load_test_op_data_path", default='cls_score_test_state_update_predictor_output.json', type=str)
    parser.add_argument("--random_seed", default = 42, type = int)
    parser.add_argument("--num_workers", default = 4, type = int)
    parser.add_argument("--batch_size", default = 1, type = int)
    parser.add_argument("--enc_warmup", default = 0.01, type = float)
    parser.add_argument("--dec_warmup", default = 0.01, type = float)
    parser.add_argument("--enc_lr", default = 5e-6, type = float)
    parser.add_argument("--base_lr", default = 1e-4, type = float)
    parser.add_argument("--n_epochs", default = 10, type = int)
    parser.add_argument("--eval_epoch", default = 1, type = int)
    parser.add_argument("--eval_step", default=5, type=int)
    parser.add_argument("--turn", default = 2, type = int)
    parser.add_argument("--op_code", default = "2", type = str)
    parser.add_argument("--slot_token", default = "[SLOT]", type = str)
    parser.add_argument("--dropout", default = 0.0, type = float)
    parser.add_argument("--hidden_dropout_prob", default = 0.0, type = float)
    parser.add_argument("--attention_probs_dropout_prob", default = 0.1, type = float)
    parser.add_argument("--decoder_teacher_forcing", default = 0.5, type = float)
    parser.add_argument("--word_dropout", default = 0.1, type = float)
    parser.add_argument("--not_shuffle_state", default = True, action = 'store_true')

    parser.add_argument("--n_history", default = 1, type = int)
    parser.add_argument("--max_seq_length", default = 256, type = int)
    parser.add_argument("--sketch_weight", default = 0.55, type = float)
    parser.add_argument("--answer_weight", default = 0.6, type = float)
    parser.add_argument("--generation_weight", default = 0.2, type = float)
    parser.add_argument("--extraction_weight", default = 0.1, type = float)
    parser.add_argument("--msg", default = None, type = str)
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend = 'nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)
    def worker_init_fn(worker_id):
        np.random.seed(args.random_seed + worker_id)

    # Prepare GLUE task
    n_gpu = 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    turn = args.turn

    ontology = json.load(open(args.data_root + args.ontology_data))

    _, slot_meta = make_slot_meta(ontology)
    with torch.cuda.device(0):
        op2id = OP_SET[args.op_code]

        rng = random.Random(args.random_seed)
        print(op2id)
        logger_trainInfo.info(op2id)
        tokenizer = AlbertTokenizer.from_pretrained(args.model_name_or_path + "spiece.model")
        addSpecialTokens(tokenizer, ['[SLOT]', '[NULL]', '[EOS]', '[dontcare]', '[negans]', '[noans]', '[TURN]'])
        turn_id = tokenizer.encode('[TURN]')[1]
        args.vocab_size = len(tokenizer)
        ontology, slot_mm, esm_ans_vocab = fixontology(slot_meta, turn, tokenizer)
        ans_vocab, ans_vocab_nd, cate_mask = mask_ans_vocab(ontology, slot_meta, tokenizer)

        if turn == 2:
            train_op_data_path = None
            test_op_data_path = args.data_root + "cls_score_test_state_update_predictor_output.json"
            isfilter = True

        model = Generator(args, len(op2id), len(domain2id), op2id['update'], esm_ans_vocab, slot_mm, turn=turn, turn_id=turn_id)
        train_data_raw, _, _ = prepare_dataset(data_path = args.data_root + args.train_data,
                                               tokenizer = tokenizer,
                                               slot_meta = slot_meta,
                                               n_history = args.n_history,
                                               max_seq_length = args.max_seq_length,
                                               op_code = args.op_code,
                                               slot_ans = ontology,
                                               turn = turn,
                                               op_data_path = None,
                                               isfilter = isfilter,
                                               if_train=True
                                               )
        train_data = MultiWozDataset(train_data_raw,
                                     tokenizer,
                                     slot_meta,
                                     args.max_seq_length,
                                     ontology,
                                     args.word_dropout,
                                     turn = turn)
        print("# train examples %d" % len(train_data_raw))
        logger_trainInfo.info("# train examples %d" % len(train_data_raw))

        dev_data_raw, idmap, _ = prepare_dataset(data_path = args.data_root + args.dev_data,
                                                 tokenizer = tokenizer,
                                                 slot_meta = slot_meta,
                                                 n_history = args.n_history,
                                                 max_seq_length = args.max_seq_length,
                                                 op_code = args.op_code,
                                                 turn = turn,
                                                 slot_ans = ontology,
                                                 op_data_path = test_op_data_path,
                                                 isfilter = False,
                                                 if_train=False)
        dev_data = MultiWozDataset(dev_data_raw,
                                   tokenizer,
                                   slot_meta,
                                   args.max_seq_length,
                                   ontology,
                                   word_dropout = 0,
                                   turn = turn)
        print("# dev examples %d" % len(dev_data_raw))
        logger_trainInfo.info("# dev examples %d" % len(dev_data_raw))

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data,
                                      sampler = train_sampler,
                                      batch_size = args.batch_size,
                                      collate_fn = train_data.collate_fn,
                                      num_workers = args.num_workers,
                                      worker_init_fn = worker_init_fn)

        dev_sampler = RandomSampler(dev_data)
        dev_dataloader = DataLoader(dev_data,
                                    sampler=dev_sampler,
                                    batch_size=args.batch_size,
                                    collate_fn=dev_data.collate_fn,
                                    num_workers=args.num_workers,
                                    worker_init_fn=worker_init_fn)

        if args.load_model:
            # load best
            # checkpoint = torch.load(os.path.join(args.save_dir, 'model_best_turn{}.bin'.format(turn)))
            # model.load_state_dict(checkpoint['model'])
            # load inter
            checkpoint = torch.load(os.path.join(args.save_dir, args.load_ckpt_epoch))
            model.load_state_dict(checkpoint)

        model.to(args.device)
        logger.info("Training/evaluation parameters %s", args)
        if args.do_train:
            num_train_steps = int(len(train_dataloader) / args.batch_size * args.n_epochs)
            bert_params_ids = list(map(id, model.albert.parameters()))
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            enc_param_optimizer = list(model.named_parameters())

            enc_optimizer_grouped_parameters = [
                {'params': [p for n, p in enc_param_optimizer if
                            (id(p) in bert_params_ids and not any(nd in n for nd in no_decay))], 'weight_decay': 0.01,
                 'lr': args.enc_lr},
                {'params': [p for n, p in enc_param_optimizer if
                            id(p) in bert_params_ids and any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': args.enc_lr},
                {'params': [p for n, p in enc_param_optimizer if
                            id(p) not in bert_params_ids and not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
                 'lr': args.base_lr},
                {'params': [p for n, p in enc_param_optimizer if
                            id(p) not in bert_params_ids and any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': args.base_lr}]

            enc_optimizer = AdamW(enc_optimizer_grouped_parameters, lr = args.base_lr)

            enc_scheduler = get_linear_schedule_with_warmup(enc_optimizer, num_warmup_steps=int(num_train_steps * args.enc_warmup), num_training_steps=num_train_steps) # 线性warmup
            
            best_score = {'epoch': 0, 'overall_jga': 0, 'cate_jga': 0, 'noncate_jga': 0}
            file_logger = helper.FileLogger(args.save_dir + '/log.txt',
                                            header="# epoch\tstep\ttrain_loss\tbest_jointacc\tbest_catejointacc\tbest_noncatejointacc\tnow_jointacc\tnow_catejointacc\tnow_noncatejointacc")
            model.train()
            loss = 0
            sketchy_weight, answer_weight, generation_weight, extraction_weight = args.sketch_weight, args.answer_weight, args.generation_weight, args.extraction_weight
            verify_weight = 1 - sketchy_weight
            for epoch in range(args.n_epochs):
                batch_loss = []
                for step, batch in enumerate(train_dataloader):
                    batch = [b.to(device) if not isinstance(b, int) and not isinstance(b, list) else b for b in batch]
                    input_ids, input_mask, slot_mask, segment_ids, state_position_ids, op_ids, pred_ops, domain_ids, gen_ids, start_position, end_position, max_value, max_update, slot_ans_ids, start_idx, end_idx, position_ids, sample_mm, generate_turn, generate_mask,ref_slot,gold_ans_label,sid, update_current_mm, slot_all_connect, update_mm, slot_domain_connect = batch
                    if input_ids.numel() == 0 or sample_mm.numel() == 0:
                        continue

                    assert len(input_ids) <= TURN_SPLIT  # train在之前已经切分了，这里不应该有任何false的情况
                    sample_mask = (pred_ops.argmax(dim=-1) == 0) if turn == 1 else (op_ids == 0)
                    start_logits, end_logits, gen_scores, _, _, _, fuse_score, input_ids,  = model(input_ids = input_ids,
                                                                                    token_type_ids = segment_ids,
                                                                                    state_positions = state_position_ids,
                                                                                    attention_mask = input_mask,
                                                                                    slot_mask = slot_mask,
                                                                                    first_edge_mask = update_current_mm,
                                                                                    second_edge_mask = slot_all_connect,
                                                                                    third_edge_mask = update_mm,
                                                                                    fourth_edge_mask = slot_domain_connect,
                                                                                    max_value = max_value,
                                                                                    op_ids = op_ids,
                                                                                    max_update = max_update,
                                                                                    position_ids = position_ids,
                                                                                    sample_mm = sample_mm)


                    if turn == 2:
                        loss_selector = masked_cross_entropy_for_value(fuse_score.contiguous(),
                                                                generate_turn.contiguous(),
                                                                sample_mask = sample_mask & (generate_mask==1)
                                                                )
                        loss_classifier= masked_cross_entropy_for_value(gen_scores.contiguous(),
                                                                slot_ans_ids.contiguous(),
                                                                sample_mask = sample_mask
                                                                )
                        loss_extractor = compute_span_loss(gen_ids, input_ids, fuse_score,start_logits, end_logits, generate_turn, sample_mask, generate_mask)
                        loss = loss_classifier + loss_extractor
                        loss = math.log(input_ids.shape[0], 8) * loss

                        loss.backward()
                        for name, par in model.named_parameters():
                            if par.requires_grad and par.grad is not None:
                                if torch.sum(torch.isnan(par.grad)) != 0:
                                    model.zero_grad()

                    batch_loss.append(loss.item())
                    torch.cuda.empty_cache()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    enc_optimizer.step()
                    enc_scheduler.step()
                    model.zero_grad()
                    if step % 100 == 0:
                        print("[%d/%d] [%d/%d] mean_loss : %.3f" \
                              % (epoch + 1, args.n_epochs, step,
                                 len(train_dataloader), np.mean(batch_loss),), end='\t')
                        logger_trainInfo.debug("[%d/%d] [%d/%d] mean_loss : %.3f" \
                              % (epoch + 1, args.n_epochs, step,
                                 len(train_dataloader), np.mean(batch_loss),))
                        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        logger_trainInfo.debug(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

                        file_logger.log(
                            "Epoch {}\t Step {}\t loss {:.6f}\t time {}".format(epoch, step, loss, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

                        logger_trainInfo.info("Epoch {}\t Step {}\t loss {:.6f}\t time {}".format(epoch, step, loss, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                        batch_loss = []
                if True: 
                    joint_acc, catejoint_acc, noncatejoint_acc = evaluate(dev_dataloader, model, device, ans_vocab_nd, cate_mask, turn=2, tokenizer=tokenizer, ontology=ontology)
                    if joint_acc > best_score['overall_jga']:  
                        best_score['epoch'] = epoch
                        best_score['overall_jga'] =joint_acc
                        best_score['cate_jga'] = catejoint_acc
                        best_score['noncate_jga'] = noncatejoint_acc
                        saved_name = 'model_best_turn' + str(turn) + '.bin'
                        save_path = os.path.join(args.save_dir, saved_name)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        params = {
                            'model': model_to_save.state_dict(),
                            'optimizer': enc_optimizer.state_dict(),
                            'scheduler': enc_scheduler.state_dict(),
                            'args': args
                        }
                        torch.save(params, save_path)

                    file_logger.log("{}\t{}\t{:.6f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(epoch, step, loss, best_score['overall_jga'], best_score['cate_jga'], best_score['noncate_jga'], joint_acc, catejoint_acc, noncatejoint_acc))
                    logger_trainInfo.warning("{}\t{}\t{:.6f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(epoch, step, loss, best_score['overall_jga'], best_score['cate_jga'], best_score['noncate_jga'], joint_acc, catejoint_acc, noncatejoint_acc))
                    print("Current: Epoch_{}\tstep_{}\tloss_{:.6f}\tJointAcc: {:.4f}\tCategorical-JointAcc: {:.4f}\tnon-Categorical-JointAcc: {:.4f}".format(epoch, step, loss, joint_acc, catejoint_acc, noncatejoint_acc))
                    logger_trainInfo.warning("Current: Epoch_{}\tstep_{}\tloss_{:.6f}\tJointAcc: {:.4f}\tCategorical-JointAcc: {:.4f}\tnon-Categorical-JointAcc: {:.4f}".format(epoch, step, loss, joint_acc, catejoint_acc, noncatejoint_acc))
                    print("Best: Epoch_{}\tJointAcc: {:.4f}\tCategorical-JointAcc: {:.4f}\tnon-Categorical-JointAcc: {:.4f}".format(best_score['epoch'], best_score['overall_jga'], best_score['cate_jga'], best_score['noncate_jga']))
                    logger_trainInfo.warning("Best: Epoch_{}\tJointAcc: {:.4f}\tCategorical-JointAcc: {:.4f}\tnon-Categorical-JointAcc: {:.4f}".format(best_score['epoch'], best_score['overall_jga'], best_score['cate_jga'], best_score['noncate_jga']))
                    del loss
                
                # 每个epoch保存模型
                model_to_save = model.module if hasattr(model, 'module') else model
                save_path = os.path.join(args.save_dir, 'checkpoint_epoch_' + str(epoch) + '.bin')
                torch.save(model_to_save.state_dict(), save_path)

        else:
            joint_acc, catejoint_acc, noncatejoint_acc = evaluate(dev_dataloader, model, device, ans_vocab_nd, cate_mask, turn=2, tokenizer=tokenizer, ontology=ontology)
            print("Test Result:\tJointAcc: {:.4f}\tCategorical-JointAcc: {:.4f}\tnon-Categorical-JointAcc: {:.4f}".format(joint_acc, catejoint_acc, noncatejoint_acc))
            logger_trainInfo.warning("Test Result:\tJointAcc: {:.4f}\tCategorical-JointAcc: {:.4f}\tnon-Categorical-JointAcc: {:.4f}".format(joint_acc, catejoint_acc, noncatejoint_acc))


def evaluate(dev_dataloader, model, device, ans_vocab_nd, cate_mask,turn =2, tokenizer = None, ontology = None):
    model.eval()
    start_predictions = []
    start_ids = []
    end_predictions = []
    end_ids = []
    has_ans_predictions = []
    has_ans_labels = []
    gen_predictions = []
    gen_labels = []
    score_diffs = []
    cate_score_diffs = []
    score_noanses = []
    all_input_ids = []
    sample_ids = []
    ref_scores = []
    fuse_scores = []
    gen_correct = 0
    gen_guess = 0
    catecorrect = 0.0
    noncatecorrect = 0.0
    cate_slot_correct = 0.0
    nocate_slot_correct = 0.0
    domain_joint = {"hotel": 0, "train": 0, "attraction": 0, "taxi": 0, "restaurant": 0}
    joint_correct = 0
    update_guess = 0
    update_correct = 0
    update_gold = 0
    samples = 0
    gen_id_labels = []
    gold_ans_labels = []

    for step, batch in enumerate(tqdm(dev_dataloader)):
        batch = [b.to(device) if not isinstance(b, int) and not isinstance(b, list) else b for b in
                 batch]
        input_ids, input_mask, slot_mask, segment_ids, state_position_ids, op_ids, pred_ops, domain_ids, gen_ids, start_position, end_position, max_value, max_update, slot_ans_ids, start_idx, end_idx, position_ids, sample_mm, generate_turn, generate_mask,ref_slot,gold_ans_label,sid, update_current_mm, slot_all_connect, update_mm, slot_domain_connect = batch
        if input_ids.numel() == 0 or sample_mm.numel() == 0:
            continue

        if turn == 2:
            has_ans_predictions += pred_ops.argmax(dim=-1).cpu().detach().numpy().tolist()
            start_ids += start_idx.cpu().detach().numpy().tolist()
            end_ids += end_idx.cpu().detach().numpy().tolist()
            has_ans_labels += op_ids.cpu().detach().numpy().tolist()
            gen_labels += slot_ans_ids.cpu().detach().numpy().tolist()
            gen_id_labels += gen_ids
            gold_ans_labels += gold_ans_label
            sample_ids += sid

            # assert len(input_ids) <= TEST_TURN_SPLIT  # test的之前没切分，在这里应该有False的情况出现
            # 测试样本在这里切分
            if len(input_ids) <= TEST_TURN_SPLIT:
                start_logits, end_logits, gen_scores, _, _, _, fuse_score, input_ids, = model(input_ids=input_ids,
                                                                                              token_type_ids=segment_ids,
                                                                                              state_positions=state_position_ids,
                                                                                              attention_mask=input_mask,
                                                                                              slot_mask=slot_mask,
                                                                                              first_edge_mask=update_current_mm,
                                                                                              second_edge_mask=slot_all_connect,
                                                                                              third_edge_mask=update_mm,
                                                                                              fourth_edge_mask=slot_domain_connect,
                                                                                              max_value=max_value,
                                                                                              op_ids=op_ids,
                                                                                              max_update=max_update,
                                                                                              position_ids=position_ids,
                                                                                              sample_mm=sample_mm)
                start_predictions += start_logits.argmax(dim=-1).cpu().detach().numpy().tolist()
                end_predictions += end_logits.argmax(dim=-1).cpu().detach().numpy().tolist()
                gen_predictions += gen_scores.argmax(dim=-1).cpu().detach().numpy().tolist()
                fuse_scores += fuse_score.argmax(dim=-1).cpu().detach().numpy().tolist()
                all_input_ids += input_ids[:, :, 1:].cpu().detach().numpy().tolist()

                del start_logits, end_logits, gen_scores, _, fuse_score, input_ids,
                torch.cuda.empty_cache()

            else:
                tmp_input_ids = [input_ids[:TEST_TURN_SPLIT, :], input_ids[TEST_TURN_SPLIT:, :]]
                tmp_segment_ids = [segment_ids[:TEST_TURN_SPLIT, :], segment_ids[TEST_TURN_SPLIT:, :]]
                tmp_state_position_ids = [state_position_ids[:TEST_TURN_SPLIT, :], state_position_ids[TEST_TURN_SPLIT:, :]]
                tmp_input_mask = [input_mask[:TEST_TURN_SPLIT, :], input_mask[TEST_TURN_SPLIT:, :]]
                tmp_slot_mask = [slot_mask[:TEST_TURN_SPLIT, :], slot_mask[TEST_TURN_SPLIT:, :]]
                tmp_update_current_mm = [update_current_mm[:TEST_TURN_SPLIT, :, :TEST_TURN_SPLIT], update_current_mm[TEST_TURN_SPLIT:, :, TEST_TURN_SPLIT:]]
                tmp_slot_all_connect = [slot_all_connect[:TEST_TURN_SPLIT, :, :], slot_all_connect[TEST_TURN_SPLIT:, :, :]]
                tmp_update_mm = [update_mm[:TEST_TURN_SPLIT, :, :TEST_TURN_SPLIT], update_mm[TEST_TURN_SPLIT:, :, TEST_TURN_SPLIT:]]
                tmp_slot_domain_connect = [slot_domain_connect[:TEST_TURN_SPLIT, :, :], slot_domain_connect[TEST_TURN_SPLIT:, :, :]]
                tmp_max_value = [max_value, max_value]
                tmp_op_ids = [op_ids[:TEST_TURN_SPLIT, :], op_ids[TEST_TURN_SPLIT:, :]]
                tmp_max_update = [max_update, max_update]
                tmp_position_ids = [position_ids[:TEST_TURN_SPLIT, :], position_ids[TEST_TURN_SPLIT:, :]]
                tmp_sample_mm = [sample_mm[:TEST_TURN_SPLIT, :, :], sample_mm[TEST_TURN_SPLIT:, :, :]]

                for cnt in range(2):
                    start_logits, end_logits, gen_scores, _, _, _, fuse_score, input_ids, = model(
                        input_ids=tmp_input_ids[cnt],
                        token_type_ids=tmp_segment_ids[cnt],
                        state_positions=tmp_state_position_ids[cnt],
                        attention_mask=tmp_input_mask[cnt],
                        slot_mask=tmp_slot_mask[cnt],
                        first_edge_mask=tmp_update_current_mm[cnt],
                        second_edge_mask=tmp_slot_all_connect[cnt],
                        third_edge_mask=tmp_update_mm[cnt],
                        fourth_edge_mask=tmp_slot_domain_connect[cnt],
                        max_value=tmp_max_value[cnt],
                        op_ids=tmp_op_ids[cnt],
                        max_update=tmp_max_update[cnt],
                        position_ids=tmp_position_ids[cnt],
                        sample_mm=tmp_sample_mm[cnt])
                    start_predictions += start_logits.argmax(dim=-1).cpu().detach().numpy().tolist()
                    end_predictions += end_logits.argmax(dim=-1).cpu().detach().numpy().tolist()
                    gen_predictions += gen_scores.argmax(dim=-1).cpu().detach().numpy().tolist()
                    fuse_scores += fuse_score.argmax(dim=-1).cpu().detach().numpy().tolist()
                    all_input_ids += input_ids[:, :, 1:].cpu().detach().numpy().tolist()

                    del start_logits, end_logits, gen_scores, _, fuse_score, input_ids,
                    torch.cuda.empty_cache()

        if (step + 1) % 50 ==0:
            if turn == 2:
                gen_acc, op_acc, opguess, opgold, opcorrect, gen_correct, gen_guess, cate_slot_correct_b, nocate_slot_correct_b, catecorrect_b, noncatecorrect_b, domain_correct_b, joint_cor, sample_l = joint_evaluation(start_predictions,
                                                                                                  end_predictions,
                                                                                                  gen_predictions,
                                                                                                  has_ans_predictions,
                                                                                                  start_ids, end_ids,
                                                                                                  gen_labels, gen_id_labels,
                                                                                                  has_ans_labels,
                                                                                                  all_input_ids,
                                                                                                  ans_vocab_nd, ref_scores,
                                                                                                  fuse_scores,
                                                                                                  gold_ans_labels,
                                                                                                  tokenizer = tokenizer,
                                                                                                  score_diffs=None,
                                                                                                  cate_score_diffs=None,
                                                                                                  score_noanses=None,
                                                                                                  sketchy_weight=None,
                                                                                                  verify_weight=None,
                                                                                                  sid=sample_ids,
                                                                                                  catemask=cate_mask,
                                                                                                ontology = ontology)
                gen_correct += gen_correct
                gen_guess += gen_guess
                joint_correct += joint_cor
                samples += sample_l
                update_gold += opgold
                update_guess += opguess
                update_correct += opcorrect
                catecorrect += catecorrect_b
                noncatecorrect += noncatecorrect_b
                cate_slot_correct += cate_slot_correct_b
                nocate_slot_correct += nocate_slot_correct_b
                for key in domain_joint.keys():
                    domain_joint[key] += domain_correct_b[key]

            start_predictions = []
            start_ids = []
            end_predictions = []
            end_ids = []
            has_ans_predictions = []
            has_ans_labels = []
            gen_predictions = []
            gen_labels = []
            score_diffs = []
            cate_score_diffs = []
            score_noanses = []
            all_input_ids = []
            sample_ids = []
            ref_scores = []
            fuse_scores = []
            gen_id_labels = []
            gold_ans_labels = []
    # print(gen_correct / gen_guess)
    # print(update_correct/update_gold)
    # print(update_correct/update_guess)
    # print(cate_slot_correct/(samples * n_slot))
    # print(nocate_slot_correct/(samples * n_slot))
    # print(catecorrect/samples)
    # print(noncatecorrect/samples)
    # for key in domain_joint.keys():
    #     print("{}, {}".format(key, domain_joint[key]/samples))
    # print(joint_correct/samples)
    return joint_correct/samples, catecorrect/samples, noncatecorrect/samples


if __name__ ==  "__main__":
    main()
