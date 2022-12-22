import os
import sys
import csv
import json
import logging
import collections
import torch
from torch.utils.data import TensorDataset

from e2eAIOK.DeNas.module.nlp.tokenization import BertTokenizer, whitespace_tokenize
from e2eAIOK.common.trainer.data.data_builder_nlp import DataBuilderNLP, DataProcessor, InputExample
import e2eAIOK.common.trainer.utils.extend_distributed as ext_dist

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

csv.field_size_limit(sys.maxsize)

class DataBuilderSQuAD(DataBuilderNLP):
    def __init__(self, cfg):
        super().__init__(cfg)

    def prepare_dataset(self):
        dataset_train, train_examples, train_dataset, labels = build_dataset(is_train=True, args=self.cfg)
        dataset_val, val_examples, val_dataset, val_features, tokenizer = build_dataset(is_train=False, args=self.cfg)
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        return (train_examples, val_examples, val_dataset, val_features, tokenizer)

    def get_dataloader(self):
        """
            create training/evaluation dataloader
        """
        other_data = self.prepare_dataset()

        if ext_dist.my_size > 1:
            num_tasks = ext_dist.dist.get_world_size()
            global_rank = ext_dist.dist.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                self.dataset_train, num_replicas=num_tasks, rank=global_rank
            )
            
            #sampler_val = torch.utils.data.DistributedSampler(
            #    dataset_val, num_replicas=num_tasks, rank=global_rank)
            sampler_val = None
        else:
            sampler_val = torch.utils.data.SequentialSampler(self.dataset_val)
            sampler_train = torch.utils.data.RandomSampler(self.dataset_train)
        
        shuffle = True
        if sampler_train is not None:
            shuffle = False

        dataloader_train = torch.utils.data.DataLoader(
            self.dataset_train, 
            sampler=sampler_train,
            batch_size=self.cfg.train_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=shuffle,
            drop_last=True,
        )

        dataloader_val = torch.utils.data.DataLoader(
            self.dataset_val, 
            batch_size=self.cfg.eval_batch_size,
            sampler=sampler_val, 
            num_workers=self.cfg.num_workers,
            shuffle=False,
            drop_last=False
        )
        
        return dataloader_train, dataloader_val, other_data


class Squad1Processor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SquadExample(object):
    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class QAInputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def qa_convert_examples_to_features(examples, tokenizer, max_seq_length,
                                    doc_stride, max_query_length,
                                    is_training):
    """Loads a data file into a list of `InputBatch`s."""
    question_part_length_list = []
    passage_part_length_list = []

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)
            question_part_length_list.append(len(tokens))                                    

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            passage_part_length_list.append(len(tokens))                                    

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0

            features.append(
                QAInputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible))
            unique_id += 1

    return features


def read_squad_examples(input_file, is_training, version_2_with_negative):
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:

                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        if 'is_impossible' not in qa:
                            qa['is_impossible'] = True
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer")
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    logger.info('load {} examples!'.format(len(examples)))
    return input_data, examples


def build_dataset(is_train, args):
    processors = {
        "squad1": Squad1Processor
    }

    output_modes = {
        "squad1": "qa_classification"
    }

    model_dir = args.model_dir
    if args.data_set == "SQuADv1.1":
        task_name = "squad1"
        data_dir = args.data_dir
        
        train_file = os.path.join(data_dir, 'train-v1.1.json')
        predict_file = os.path.join(data_dir, 'dev-v1.1.json')

    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=args.do_lower_case)

    def get_data(examples, label_list, max_seq_length, tokenizer, output_mode,
                   is_dev=False, is_training=False):
        
        features = qa_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=is_training)

        all_input_ids = torch.unsqueeze(torch.tensor([f.input_ids for f in features], dtype=torch.long), -1)
        all_input_mask = torch.unsqueeze(torch.tensor([f.input_mask for f in features], dtype=torch.long), -1)
        all_segment_ids = torch.unsqueeze(torch.tensor([f.segment_ids for f in features], dtype=torch.long), -1)
        all_inputs = torch.concat((all_input_ids, all_input_mask, all_segment_ids), -1)
        if is_training:
            all_start_positions = torch.unsqueeze(torch.tensor([f.start_position for f in features], dtype=torch.long), -1)
            all_end_positions = torch.unsqueeze(torch.tensor([f.end_position for f in features], dtype=torch.long), -1)
            all_positions = torch.concat((all_start_positions, all_end_positions), -1)
            data = TensorDataset(all_inputs,
                                 all_positions)
        else:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            data = TensorDataset(all_inputs, all_example_index)

        #if not is_dev:
        #    sampler = RandomSampler(data)
        #else:
        #    sampler = SequentialSampler(data)
        #return features, DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)
        return features, data

    if is_train:
        train_dataset, train_examples = read_squad_examples(
            input_file=train_file, is_training=is_train,
            version_2_with_negative=args.version_2_with_negative)

        labels, train_data = get_data(train_examples, label_list, args.max_seq_length,
                                        tokenizer, output_mode, is_training=is_train)
        return train_data, train_examples, train_dataset, labels 
    else:   
        eval_dataset, eval_examples = read_squad_examples(
            input_file=predict_file, is_training=False,
            version_2_with_negative=args.version_2_with_negative)

        eval_features, eval_data = get_data(eval_examples, label_list, args.max_seq_length,
                                        tokenizer, output_mode, is_dev=True, is_training=is_train)
        return eval_data, eval_examples, eval_dataset, eval_features, tokenizer