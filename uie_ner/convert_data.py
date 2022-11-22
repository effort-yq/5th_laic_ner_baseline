# -*- coding: utf-8 -*-

# author:Administrator
# contact: test@test.com
# datetime:2022/11/21 16:32
# software: PyCharm

"""
文件说明：

"""

import os
import time
import argparse
import json
import random
from decimal import Decimal
import numpy as np
import yaml
import random
from tqdm import tqdm
import math
import torch

entity_dict = {'11339': '被告人交通工具', '11340': '被告人交通工具情况及行驶情况', '11341': '被告人违规情况', '11342': '行为地点',
               '11343': '搭载人姓名', '11344': '其他事件参与人', '11345': '参与人交通工具', '11346': '参与人交通工具情况及行驶情况',
               '11347': '参与人违规情况', '11348': '被告人责任认定', '11349': '参与人责任认定', '11350': '被告人行为总结'}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def generate_cls_example(text, labels, prompt_prefix, options):
    random.shuffle(options)
    cls_options = ",".join(options)
    prompt = prompt_prefix + "[" + cls_options + "]"

    result_list = []
    example = {"content": text, "result_list": result_list, "prompt": prompt}
    for label in labels:
        start = prompt.rfind(label[0]) - len(prompt) - 1
        end = start + len(label)
        result = {"text": label, "start": start, "end": end}
        example["result_list"].append(result)
    return example


def add_negative_example(examples, texts, prompts, label_set, negative_ratio):
    negative_examples = []
    positive_examples = []
    with tqdm(total=len(prompts)) as pbar:
        for i, prompt in enumerate(prompts):
            negative_sample = []
            redundants_list = list(set(label_set) ^ set(prompt))
            redundants_list.sort()

            num_positive = len(examples[i])
            if num_positive != 0:
                actual_ratio = math.ceil(len(redundants_list) / num_positive)
            else:
                # Set num_positive to 1 for text without positive example
                num_positive, actual_ratio = 1, 0

            if actual_ratio <= negative_ratio or negative_ratio == -1:
                idxs = [k for k in range(len(redundants_list))]
            else:
                idxs = random.sample(range(0, len(redundants_list)),
                                     negative_ratio * num_positive)

            for idx in idxs:  # 构建负样本
                negative_result = {
                    "content": texts[i],
                    "result_list": [],
                    "prompt": redundants_list[idx]  # 选择样本中不存在的实体label
                }
                negative_examples.append(negative_result)
            positive_examples.extend(examples[i])
            pbar.update(1)
    return positive_examples, negative_examples


def add_full_negative_example(examples, texts, relation_prompts, predicate_set,
                              subject_goldens):
    with tqdm(total=len(relation_prompts)) as pbar:
        for i, relation_prompt in enumerate(relation_prompts):
            negative_sample = []
            for subject in subject_goldens[i]:
                for predicate in predicate_set:
                    # The relation prompt is constructed as follows:
                    # subject + "的" + predicate
                    prompt = subject + "的" + predicate
                    if prompt not in relation_prompt:
                        negative_result = {
                            "content": texts[i],
                            "result_list": [],
                            "prompt": prompt
                        }
                        negative_sample.append(negative_result)
            examples[i].extend(negative_sample)
            pbar.update(1)
    return examples


def construct_relation_prompt_set(entity_name_set, predicate_set):
    relation_prompt_set = set()
    for entity_name in entity_name_set:
        for predicate in predicate_set:
            # The relation prompt is constructed as follows:
            # subject + "的" + predicate
            relation_prompt = entity_name + "的" + predicate
            relation_prompt_set.add(relation_prompt)
    return sorted(list(relation_prompt_set))


def convert_ext_examples(raw_examples,
                         negative_ratio,
                         prompt_prefix="情感倾向",
                         options=["正向", "负向"],
                         separator="##",
                         is_train=True):
    """
    Convert labeled data export from doccano for extraction and aspect-level classification task.
    """

    def _sep_cls_label(label, separator):
        label_list = label.split(separator)
        if len(label_list) == 1:
            return label_list[0], None
        return label_list[0], label_list[1:]

    def _concat_examples(positive_examples, negative_examples, negative_ratio):
        examples = []
        if math.ceil(len(negative_examples) /
                     len(positive_examples)) <= negative_ratio:
            examples = positive_examples + negative_examples
        else:
            # Random sampling the negative examples to ensure overall negative ratio unchanged.
            idxs = random.sample(range(0, len(negative_examples)),
                                 negative_ratio * len(positive_examples))
            negative_examples_sampled = []
            for idx in idxs:
                negative_examples_sampled.append(negative_examples[idx])
            examples = positive_examples + negative_examples_sampled
        return examples

    texts = []
    entity_examples = []
    relation_examples = []
    entity_cls_examples = []
    entity_prompts = []
    relation_prompts = []
    entity_label_set = []  # 实体类型集合
    entity_name_set = []
    predicate_set = []
    subject_goldens = []

    with tqdm(total=len(raw_examples)) as pbar:
        for line in raw_examples:
            items = json.loads(line)
            entity_id = 0

            text, relations, entities = items["context"], [], items["entities"]
            texts.append(text)

            entity_example = []
            entity_prompt = []
            entity_example_map = {}
            entity_map = {}  # id to entity name
            for entity in entities:
                span = entity['span']
                if len(span) == 1:
                    start = int(span[0].split(';')[0])
                    end = int(span[0].split(';')[1])
                    entity_name = text[start:end]
                    entity_map[entity["label"]] = {
                        "name": entity_name,
                        "start": start,
                        "end": end
                    }
                    entity_label, entity_cls_label = _sep_cls_label(
                        entity_dict[entity["label"]], separator)

                    # Define the prompt prefix for entity-level classification
                    entity_cls_prompt_prefix = prompt_prefix + "的" + entity_name
                    if entity_cls_label is not None:
                        entity_cls_example = generate_cls_example(
                            text, entity_cls_label, entity_cls_prompt_prefix,
                            options)

                        entity_cls_examples.append(entity_cls_example)

                    result = {
                        "text": entity_name,
                        "start": start,
                        "end": end
                    }
                    if entity_label not in entity_example_map.keys():
                        entity_example_map[entity_label] = {
                            "content": text,
                            "result_list": [result],
                            "prompt": entity_label
                        }
                    else:
                        entity_example_map[entity_label]["result_list"].append(
                            result)

                    if entity_label not in entity_label_set:
                        entity_label_set.append(entity_label)
                    if entity_name not in entity_name_set:
                        entity_name_set.append(entity_name)
                    entity_prompt.append(entity_label)

                else:
                    for s in span:
                        start = int(s.split(';')[0])
                        end = int(s.split(';')[1])
                        entity_name = text[start:end]
                        entity_map[entity["label"]] = {
                            "name": entity_name,
                            "start": start,
                            "end": end
                        }

                    entity_label, entity_cls_label = _sep_cls_label(
                        entity_dict[entity["label"]], separator)

                    # Define the prompt prefix for entity-level classification
                    entity_cls_prompt_prefix = prompt_prefix + "的" + entity_name
                    if entity_cls_label is not None:
                        entity_cls_example = generate_cls_example(
                            text, entity_cls_label, entity_cls_prompt_prefix,
                            options)

                        entity_cls_examples.append(entity_cls_example)

                    result = {
                        "text": entity_name,
                        "start": start,
                        "end": end
                    }
                    if entity_label not in entity_example_map.keys():
                        entity_example_map[entity_label] = {
                            "content": text,
                            "result_list": [result],
                            "prompt": entity_label
                        }
                    else:
                        entity_example_map[entity_label]["result_list"].append(
                            result)

                    if entity_label not in entity_label_set:
                        entity_label_set.append(entity_label)
                    if entity_name not in entity_name_set:
                        entity_name_set.append(entity_name)
                    entity_prompt.append(entity_label)

            for v in entity_example_map.values():
                entity_example.append(v)

            entity_examples.append(entity_example)
            entity_prompts.append(entity_prompt)

            subject_golden = []  # Golden entity inputs
            relation_example = []
            relation_prompt = []
            relation_example_map = {}
            # for relation in relations:
            #     predicate = relation["type"]
            #     subject_id = relation["from_id"]
            #     object_id = relation["to_id"]
            #     # The relation prompt is constructed as follows:
            #     # subject + "的" + predicate
            #     prompt = entity_map[subject_id]["name"] + "的" + predicate
            #     if entity_map[subject_id]["name"] not in subject_golden:
            #         subject_golden.append(entity_map[subject_id]["name"])
            #     result = {
            #         "text": entity_map[object_id]["name"],
            #         "start": entity_map[object_id]["start"],
            #         "end": entity_map[object_id]["end"]
            #     }
            #     if prompt not in relation_example_map.keys():
            #         relation_example_map[prompt] = {
            #             "content": text,
            #             "result_list": [result],
            #             "prompt": prompt
            #         }
            #     else:
            #         relation_example_map[prompt]["result_list"].append(result)
            #
            #     if predicate not in predicate_set:
            #         predicate_set.append(predicate)
            #     relation_prompt.append(prompt)

            # for v in relation_example_map.values():
            #     relation_example.append(v)

            relation_examples.append(relation_example)
            relation_prompts.append(relation_prompt)
            subject_goldens.append(subject_golden)
            pbar.update(1)

    positive_examples, negative_examples = add_negative_example(
        entity_examples, texts, entity_prompts, entity_label_set,
        negative_ratio)
    if len(positive_examples) == 0:
        all_entity_examples = []
    elif is_train:
        all_entity_examples = _concat_examples(positive_examples,
                                               negative_examples,
                                               negative_ratio)
    else:
        all_entity_examples = positive_examples + negative_examples

    all_relation_examples = []
    if len(predicate_set) != 0:
        if is_train:
            relation_prompt_set = construct_relation_prompt_set(
                entity_name_set, predicate_set)
            positive_examples, negative_examples = add_negative_example(
                relation_examples, texts, relation_prompts, relation_prompt_set,
                negative_ratio)
            all_relation_examples = _concat_examples(positive_examples,
                                                     negative_examples,
                                                     negative_ratio)
        else:
            relation_examples = add_full_negative_example(
                relation_examples, texts, relation_prompts, predicate_set,
                subject_goldens)
            all_relation_examples = [
                r for relation_example in relation_examples
                for r in relation_example
            ]
    return all_entity_examples, all_relation_examples, entity_cls_examples


def do_convert():
    set_seed(args.seed)

    tic_time = time.time()
    if not os.path.exists(args.doccano_file):
        raise ValueError("Please input the correct path of doccano file.")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if len(args.splits) != 0 and len(args.splits) != 3:
        raise ValueError("Only []/ len(splits)==3 accepted for splits.")

    def _check_sum(splits):
        return Decimal(str(splits[0])) + Decimal(str(splits[1])) + Decimal(
            str(splits[2])) == Decimal("1")

    if len(args.splits) == 3 and not _check_sum(args.splits):
        raise ValueError(
            "Please set correct splits, sum of elements in splits should be equal to 1."
        )

    with open(args.doccano_file, "r", encoding="utf-8") as f:
        raw_examples = f.readlines()
    with open(r'E:\ai studio\第五届法研杯\选手数据集\train_second.json', 'r', encoding='utf-8') as f:
        raw_examples += f.readlines()

    def _create_ext_examples(examples,
                             negative_ratio,
                             prompt_prefix="情感倾向",
                             options=["正向", "负向"],
                             separator="##",
                             shuffle=False,
                             is_train=True):
        entities, relations, aspects = convert_ext_examples(
            examples, negative_ratio, prompt_prefix, options, separator,
            is_train)
        examples = entities + relations + aspects
        if shuffle:
            indexes = np.random.permutation(len(examples))
            examples = [examples[i] for i in indexes]
        return examples

    def _save_examples(save_dir, file_name, examples):
        count = 0
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, "a", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1

    if len(args.splits) == 0:
        if args.task_type == "ext":
            examples = _create_ext_examples(raw_examples, args.negative_ratio,
                                            args.prompt_prefix, args.options,
                                            args.separator, args.is_shuffle)
        else:
            examples = _create_ext_examples(raw_examples, args.negative_ratio,
                                            args.prompt_prefix, args.options,
                                            args.separator, args.is_shuffle)
        _save_examples(args.save_dir, "train.txt", examples)
    else:
        if args.is_shuffle:
            indexes = np.random.permutation(len(raw_examples))
            raw_examples = [raw_examples[i] for i in indexes]

        i1, i2, _ = args.splits
        p1 = int(len(raw_examples) * i1)
        p2 = int(len(raw_examples) * (i1 + i2))

        train_examples = _create_ext_examples(raw_examples[:p1],
                                              args.negative_ratio,
                                              args.prompt_prefix,
                                              args.options, args.separator,
                                              args.is_shuffle)
        dev_examples = _create_ext_examples(raw_examples[p1:p2],
                                            -1,
                                            args.prompt_prefix,
                                            args.options,
                                            args.separator,
                                            is_train=False)
        test_examples = _create_ext_examples(raw_examples[p2:],
                                             -1,
                                             args.prompt_prefix,
                                             args.options,
                                             args.separator,
                                             is_train=False)

        _save_examples(args.save_dir, "train_data.json", train_examples)
        _save_examples(args.save_dir, "dev_data.json", dev_examples)
        _save_examples(args.save_dir, "test_data.json", test_examples)


if __name__ == "__main__":
    # # yapf: disable
    # 事件抽取
    parser = argparse.ArgumentParser()
    parser.add_argument("--doccano_file", default=r"E:\ai studio\第五届法研杯\选手数据集\train.json", type=str,
                        help="The doccano file exported from doccano platform.")
    parser.add_argument("--save_dir", default="./", type=str, help="The path of data that you wanna save.")
    parser.add_argument("--negative_ratio", default=5, type=int,
                        help="Used only for the extraction task, the ratio of positive and negative samples, number of negtive samples = negative_ratio * number of positive samples")
    parser.add_argument("--splits", default=[0.8, 0.2, 0], type=float, nargs="*",
                        help="The ratio of samples in datasets. [0.6, 0.2, 0.2] means 60% samples used for training, 20% for evaluation and 20% for test.")
    parser.add_argument("--task_type", choices=['ext', 'cls'], default="ext", type=str,
                        help="Select task type, ext for the extraction task and cls for the classification task, defaults to ext.")
    parser.add_argument("--is_shuffle", default=True, type=bool,
                        help="Whether to shuffle the labeled dataset, defaults to True.")
    parser.add_argument("--seed", type=int, default=2022, help="Random seed for initialization")
    parser.add_argument("--separator", type=str, default='##',
                        help="Used only for entity/aspect-level classification task, separator for entity label and classification label")
    parser.add_argument("--options", default=["超速", "正常"], type=str, nargs="+",
                        help="Used only for the classification task, the options for classification")
    parser.add_argument("--prompt_prefix", default="危险驾驶罪", type=str,
                        help="Used only for the classification task, the prompt prefix for classification")
    args = parser.parse_args()
    do_convert()