from utils.constant import n_slot
import torch
import difflib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def op_evaluation_sketchy(op_prediction, op_ids):
    op_guess = 0.0
    op_correct = 0.0
    op_update_guess = 0.0
    op_update_correct = 0.0
    op_update_gold = 0.0
    for i, op_pred in enumerate(op_prediction):
        op_guess += 1
        if op_pred == op_ids[i]:
            op_correct += 1
            if op_ids[i] == 0:
                op_update_correct += 1
        if op_ids[i] == 0:
            op_update_gold += 1
        if op_prediction[i] == 0:
            op_update_guess += 1

    op_acc = op_correct / op_guess if op_guess != 0 else 0
    op_prec = op_update_correct / op_update_guess if op_update_guess != 0 else 0
    op_recall = op_update_correct / op_update_gold if op_update_gold != 0 else 0
    op_F1 = 2 * (op_prec * op_recall) / (op_prec + op_recall) if op_prec + op_recall != 0 else 0
    return op_acc, op_prec, op_recall, op_F1


def op_evaluation(start_prediction, end_prediction, gen_prediction, op_prediction, start_idx, end_idx, slot_ans_idx,
                  gen_ids, op_ids, input_ids, ans_vocab, ref_score, fuse_score, gold_ans_labels, score_diffs=None,
                  cate_score_diffs=None, score_noanses=None, sketchy_weight=0.55, verify_weight=0.45, sid=None,
                  catemask=None):
    gen_guess = 0.0
    gen_correct = 0.0
    op_guess = 0.0
    op_correct = 0.0
    op_update_guess = 0.0
    op_update_correct = 0.0
    op_update_gold = 0.0
    ans_pad_size = ans_vocab.shape[-1]
    ans_vocab = ans_vocab.tolist()
    isverify = (score_diffs is not None)
    score_reveryfy = {}
    sample_op = []
    count = 0
    for i, op_pred in enumerate(op_prediction):
        sample_id = i // n_slot
        slot_id = i % n_slot
        op_guess += 1
        extract_ans = [2] + input_ids[sample_id][slot_id][start_prediction[i] - 1:end_prediction[i]] + [3]
        extract_ans += [0] * (ans_pad_size - len(extract_ans))
        if isverify:
            isvalid = (start_prediction[i] == 0 and end_prediction[i] == 0) or (extract_ans in ans_vocab[slot_id])
            if isvalid:
                score_logit = score_diffs[i] * verify_weight + score_noanses[i] * sketchy_weight
                op_pred = (score_logit > 0) * 1
            else:
                score_logit = cate_score_diffs[i] * verify_weight + score_noanses[i] * sketchy_weight
                op_pred = (score_logit > 0) * 1

            if slot_id == 29:
                score_reveryfy[sid[sample_id]] = sample_op
                sample_op = []
        else:
            isvalid = (extract_ans in ans_vocab[slot_id])
        sample_op += [[1 - op_pred, op_pred]]
        if op_ids[i] == 0:
            op_update_gold += 1
        if op_pred == 0:
            op_update_guess += 1
        if op_pred == op_ids[i]:
            op_correct += 1
            if op_ids[i] == 0:
                op_update_correct += 1
        if op_ids[i] == 0:
            gen_guess += 1
            if catemask[slot_id]:
                if isvalid:
                    gen_correct += 1 * (((input_ids[sample_id][slot_id][start_prediction[i] - 1:end_prediction[i]] + [
                        30002]) in gen_ids[sample_id][slot_id]) or (input_ids[sample_id][slot_id][
                                                                    start_prediction[i] - 1:end_prediction[i]] ==
                                                                    gen_ids[sample_id][slot_id]))
                else:
                    gen_correct += 1 * (gen_prediction[i] == slot_ans_idx[i])
            else:
                gen_correct += 1 * ((input_ids[sample_id][slot_id][start_prediction[i] - 1:end_prediction[i]] + [
                    30002] in gen_ids[sample_id][slot_id]) or (input_ids[sample_id][slot_id][
                                                               start_prediction[i] - 1:end_prediction[i]] ==
                                                               gen_ids[sample_id][slot_id]))
        if slot_id == 29:
            score_reveryfy[sid[sample_id]] = sample_op
            sample_op = []

    gen_acc = gen_correct / gen_guess if gen_guess != 0 else 0
    op_acc = op_correct / op_guess if op_guess != 0 else 0
    op_prec = op_update_correct / op_update_guess if op_update_guess != 0 else 0
    op_recall = op_update_correct / op_update_gold if op_update_gold != 0 else 0
    op_F1 = 2 * (op_prec * op_recall) / (op_prec + op_recall) if op_prec + op_recall != 0 else 0
    print(op_update_correct)
    print(op_update_gold)
    print(op_update_guess)
    print(gen_correct)
    print(gen_guess)
    print("Update score: operation precision: %.3f, operation_recall : %.3f,operation F1:%.3f" % (
        op_prec, op_recall, op_F1))
    return gen_acc, op_acc, op_prec, op_recall, op_F1, gen_correct, gen_guess


def joint_evaluation(start_prediction, end_prediction, gen_prediction, op_prediction, start_idx, end_idx, slot_ans_idx,
                     gen_ids, op_ids, input_ids, ans_vocab, ref_score, fuse_score, gold_ans_labels, tokenizer=None,
                     score_diffs=None, cate_score_diffs=None,
                     score_noanses=None, sketchy_weight=0.55, verify_weight=0.45, sid=None, catemask=None,
                     ontology=None):
    ans_pad_size = ans_vocab.shape[-1]
    ans_vocab = ans_vocab.tolist()
    isverify = (score_diffs is not None)
    score_reveryfy = {}
    sample_op = []
    count = 0
    gen_guess = 0.0
    gen_correct = 0.0
    op_guess = 0.0
    op_correct = 0.0
    op_update_guess = 0.0
    op_update_correct = 0.0
    op_update_gold = 0.0
    joint_correct = 0.0
    slot_correct = 0.0
    cateslot = 0.0
    nocateslot = 0.0
    catecorrect = 0.0
    noncatecorrect = 0.0
    cate_slot_correct = 0.0
    nocate_slot_correct = 0.0
    domain_joint = {"hotel": 0, "train": 0, "attraction": 0, "taxi": 0, "restaurant": 0}
    error = 0
    for i, op_pred_turn in enumerate(op_prediction):
        op_guess += 1
        if int(sid[i].split('_')[-1]) == 0:
            last_state = [[] for k in op_pred_turn]
        current_state = [[] for k in op_pred_turn]
        gold_state = gold_ans_labels[i]
        iscate_correct = 1
        isnoncate_correct = 1
        domain_correct = {"hotel": 1, "train": 1, "attraction": 1, "taxi": 1, "restaurant": 1}
        for si, op_pred in enumerate(op_pred_turn):
            extract_ans = [2] + input_ids[i][si][start_prediction[i][si] - 1:end_prediction[i][si]] + [3]
            extract_ans += [0] * (ans_pad_size - len(extract_ans))

            if isverify:
                isvalid = (start_prediction[i][si] == 0 and end_prediction[i][si] == 0) or (
                            extract_ans in ans_vocab[si])
                if isvalid:
                    score_logit = score_diffs[i] * verify_weight + score_noanses[i] * sketchy_weight
                    op_pred = (score_logit > 0) * 1
                else:
                    score_logit = cate_score_diffs[i] * verify_weight + score_noanses[i] * sketchy_weight
                    op_pred = (score_logit > 0) * 1

                if si == 29:
                    score_reveryfy[sid[i]] = sample_op
                    sample_op = []
            else:
                isvalid = (extract_ans in ans_vocab[si])
            sample_op += [[1 - op_pred, op_pred]]
            if op_ids[i][si] == 0:
                op_update_gold += 1
            if op_pred == 0:
                op_update_guess += 1
            if op_pred == op_ids[i][si]:
                op_correct += 1
                if op_pred == 0:
                    op_update_correct += 1
            else:
                error += 1

            if op_pred == 0:
                batch_correct = 0
                gen_guess += 1
                if catemask[si]:
                    if isvalid:

                        batch_correct += 1 * (((input_ids[i][si][start_prediction[i][si] - 1:end_prediction[i][si]] + [
                            30002]) in gen_ids[i][si]) or (input_ids[i][si][
                                                           start_prediction[i][si] - 1:end_prediction[i][si]] ==
                                                           gen_ids[i][si]))
                        current_state[si] = input_ids[i][si][
                                            start_prediction[i][si] - 1:end_prediction[i][si]] + [30002]
                        if ans_vocab[si].index(extract_ans) == slot_ans_idx[i][si]:
                            current_state[si] = gold_state[si]
                    else:
                        batch_correct += 1 * (gen_prediction[i][si] == slot_ans_idx[i][si])
                        current_state[si] = list(
                            filter(lambda x: x not in [0, 2, 3], ans_vocab[si][gen_prediction[i][si]])) + [30002]
                        if gen_prediction[i][si] == slot_ans_idx[i][si]:
                            current_state[si] = gold_state[si]
                else:
                    batch_correct += 1 * ((input_ids[i][si][start_prediction[i][si] - 1:end_prediction[i][si]] + [
                        30002] in gen_ids[i][si]) or (input_ids[i][si][
                                                      start_prediction[i][si] - 1:end_prediction[i][si]] ==
                                                      gen_ids[i][si]))
                    current_state[si] = input_ids[i][si][
                                        start_prediction[i][si] - 1:end_prediction[i][si]] + [30002]
                gen_correct += batch_correct
            else:
                current_state[si] = last_state[si]

        correct_mask = []
        for slot_id in range(len(current_state)):
            if current_state[slot_id] == 3 or (30004 in current_state[slot_id]):
                current_state[slot_id] = []
            if current_state[slot_id] == gold_state[slot_id]:
                correct_mask.append(1)
            else:
                if current_state[slot_id] != [] and gold_state[slot_id] != []:
                    sim = match(current_state[slot_id], gold_state[slot_id], tokenizer)
                    if sim > 0.9:
                        current_state[slot_id] = gold_state[slot_id]
                        correct_mask.append(1)
                    else:
                        correct_mask.append(0)
                else:
                    correct_mask.append(0)
            if correct_mask[-1] == 0:
                if catemask[slot_id]:
                    iscate_correct = 0
                else:
                    isnoncate_correct = 0
                name = ontology[slot_id]['name'].split("-")[0]
                domain_joint[name] = 0
            else:
                if catemask[slot_id]:
                    cate_slot_correct += 1
                else:
                    nocate_slot_correct += 1

        correct = 1 if sum(correct_mask) == len(current_state) else 0
        joint_correct += correct
        catecorrect += iscate_correct
        noncatecorrect += isnoncate_correct
        for k in domain_joint.keys():
            domain_correct[k] += domain_correct[k]
        last_state = current_state

    gen_acc = gen_correct / gen_guess if gen_guess != 0 else 0
    op_acc = op_correct / op_guess if op_guess != 0 else 0
    op_prec = op_update_correct / op_update_guess if op_update_guess != 0 else 0
    op_recall = op_update_correct / op_update_gold if op_update_gold != 0 else 0
    op_F1 = 2 * (op_prec * op_recall) / (op_prec + op_recall) if op_prec + op_recall != 0 else 0
    # print(op_update_correct)
    # print(op_update_gold)
    # print(op_update_guess)
    # print(gen_correct)
    # print(gen_guess)
    # print(catecorrect)
    # print(noncatecorrect)
    # print(domain_correct)
    # print(joint_correct)
    # print(len(op_prediction))
    # print("Update score: operation precision: %.3f, operation_recall : %.3f,operation F1:%.3f" % (
    #     op_prec, op_recall, op_F1))
    return gen_acc, op_acc, op_update_guess, op_update_gold, op_update_correct, gen_correct, gen_guess, cate_slot_correct, nocate_slot_correct, catecorrect, noncatecorrect, domain_correct, joint_correct, len(
        op_prediction)


def match(a, b, tokenizer):
    a = "".join(tokenizer.convert_ids_to_tokens(a))
    b = "".join(tokenizer.convert_ids_to_tokens(b))
    similarity = difflib.SequenceMatcher(None, a, b).quick_ratio()
    return similarity


if __name__ == "__main__":
    pass
