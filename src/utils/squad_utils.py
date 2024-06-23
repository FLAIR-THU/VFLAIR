import logging
import json
import collections
import math
import string
import re

# from gpt2sqa.tokenization import ( BasicTokenizer)


logger = logging.getLogger(__name__)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _get_best_indexes(logits, n_best_size=20):
    """Get the n-best logits from a list."""
    # logits tensor 512
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    # list 512
    best_indexes = []

    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])

    return best_indexes


def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(a_gold == a_pred)


def compute_f1(a_gold, a_pred):
    gold_toks = a_gold
    pred_toks = a_pred

    set_common = set(a_gold) & set(a_pred)
    list_common = list(set_common)
    num_same = len(list_common)  # sum(common.values())

    if len(a_gold) == 0 or len(a_pred) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(a_gold == a_pred)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(a_pred)
    recall = 1.0 * num_same / len(a_gold)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# def compute_exact(a_gold, a_pred):
#     return int(normalize_answer(a_gold) == normalize_answer(a_pred))

# def compute_f1(a_gold, a_pred):
#     gold_toks = get_tokens(a_gold)
#     pred_toks = get_tokens(a_pred)
#     common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
#     num_same = sum(common.values())
#     if len(gold_toks) == 0 or len(pred_toks) == 0:
#         # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
#         return int(gold_toks == pred_toks)
#     if num_same == 0:
#         return 0
#     precision = 1.0 * num_same / len(pred_toks)
#     recall = 1.0 * num_same / len(gold_toks)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return f1

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

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

    # def __str__(self):
    #     return self.__repr__()

    # def __repr__(self):
    #     s = ""
    #     s += "qas_id: %s" % (self.qas_id)
    #     s += ", question_text: %s" % (
    #         self.question_text)
    #     s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    #     if self.start_position:
    #         s += ", start_position: %d" % (self.start_position)
    #     if self.end_position:
    #         s += ", end_position: %d" % (self.end_position)
    #     if self.is_impossible:
    #         s += ", is_impossible: %r" % (self.is_impossible)
    #     return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens, orig_answer_text,
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
        self.orig_answer_text = orig_answer_text
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def get_dict(self):
        return {
            "unique_id": self.unique_id,
            "example_index": self.example_index,
            "doc_span_index": self.doc_span_index,
            "tokens": self.tokens,
            "orig_answer_text": self.orig_answer_text,
            "token_to_orig_map": self.token_to_orig_map,
            "token_is_max_context": self.token_is_max_context,
            "input_ids": self.input_ids,
            "input_mask": self.input_mask,
            "segment_ids": self.segment_ids,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "is_impossible": self.is_impossible
        }


def standard_read_squad_examples(input_file, is_training, version_2_with_negative=False):
    """Read a SQuAD json file into a list of SquadExample."""
    # with tf.io.gfile.Open(input_file, "r") as reader:
    with open(input_file, "r") as reader:
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
                        end_position = char_to_word_offset[answer_offset + answer_length -
                                                           1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(
                            doc_tokens[start_position:(end_position + 1)])
                        # print('actual_text:',actual_text)

                        # cleaned_answer_text = " ".join(
                        #     tokenization.whitespace_tokenize(orig_answer_text))
                        # if actual_text.find(cleaned_answer_text) == -1:
                        #     tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                        #                         actual_text, cleaned_answer_text)
                        #     continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""
                else:
                    orig_answer_text = []
                    start_position = []
                    end_position = []
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]

                    if not is_impossible:
                        for answer in qa["answers"]:
                            orig_answer_text.append(answer["text"])
                            # print('orig_answer_text:',orig_answer_text)
                            answer_offset = answer["answer_start"]
                            answer_length = len(orig_answer_text[-1])
                            start_position.append(char_to_word_offset[answer_offset])
                            end_position.append(char_to_word_offset[answer_offset + answer_length - 1])

                            # Only add answers where the text can be exactly recovered from the
                            # document. If this CAN'T happen it's likely due to weird Unicode
                            # stuff so we will just skip the example.
                            #
                            # Note that this means for training mode, every example is NOT
                            # guaranteed to be preserved.

                            actual_text = " ".join(doc_tokens[start_position[-1]:(end_position[-1] + 1)])
                            # print('actual_text:',actual_text)

                            # cleaned_answer_text = " ".join(
                            #     tokenization.whitespace_tokenize(orig_answer_text))
                            # if actual_text.find(cleaned_answer_text) == -1:
                            #     tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                            #                         actual_text, cleaned_answer_text)
                            #     continue
                        else:
                            start_position.append(-1)
                            end_position.append(-1)
                            orig_answer_text.append("")

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)

                # print(examples[0].doc_tokens)
                # print(examples[0].question_text)
                # print(examples[0].orig_answer_text)
                # print(examples[0].start_position, train_examples[0].end_position )

    return examples


def read_squad_examples(dst, is_training, version_2_with_negative=True):
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    paragraph = list(dst)
    for qa in paragraph:  # each sample
        paragraph_text = qa['context']
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

        qas_id = qa['id']

        question_text = qa['question']
        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False
        # is_impossible = qa['is_impossible']

        # if (len(qa['answers']['text']) != 1) and (not is_impossible):  # 有多个答案的情况
        #     raise ValueError(
        #         "For training, each question should have exactly 1 answer."
        #     )

        if not is_impossible:
            answer = qa['answers']
            # print('answer:',len(answer['text']),answer)
            orig_answer_text = answer['text'][0]
            # print('orig_answer_text:',orig_answer_text)
            answer_offset = answer['answer_start'][0]
            # print('answer_offset:',answer_offset)
            answer_length = len(orig_answer_text)
            # print('answer_length:',answer_length)
            start_position = char_to_word_offset[answer_offset]
            end_position = char_to_word_offset[answer_offset + answer_length - 1]
            # print('start_position:',start_position,' end_position:',end_position)
            actual_text = "".join(doc_tokens[start_position:(end_position + 1)])
            # print('actual_text:',actual_text)

            # cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
            # if actual_text.find(cleaned_answer_text) == -1:
            #     print("Could not find answer: '%s' vs. '%s'",
            #                      actual_text, cleaned_answer_text)
            #     continue

        else:
            start_position = -1
            end_position = -1
            orig_answer_text = ""

        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,  #
            doc_tokens=doc_tokens,  #
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            is_impossible=is_impossible,
        )
        examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    #  output_fn):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    features = []

    for (example_index, example) in enumerate(examples):
        # print(' === convert_examples_to_features === ')
        # print('example.question_text:',example.question_text)
        # print('example.orig_answer_text:',example.orig_answer_text) #["ans","ans","ans","ans"]
        # print('example.start_position:',example.start_position) #[28,28,28,28]
        # print('example.doc_tokens:',type(example.doc_tokens))
        query_tokens = tokenizer.tokenize(example.question_text)
        # tokenizer(example.question_text,padding='max_length', # Pad to max_length
        #                             truncation='longest_first',  # Truncate to max_length
        #                             max_length=int(max_query_length),padding_side="left", return_tensors="pt")   
        max_query_length = int(max_query_length)
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

        # if is_training and example.is_impossible:
        #     tok_start_position = -1
        #     tok_end_position = -1
        # if is_training and not example.is_impossible:
        #     tok_start_position = orig_to_tok_index[example.start_position]

        #     if example.end_position < len(example.doc_tokens) - 1:
        #         tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        #     else:
        #         tok_end_position = len(all_doc_tokens) - 1

        #     (tok_start_position, tok_end_position) = _improve_answer_span(
        #         all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
        #         example.orig_answer_text)
        if is_training:
            if example.is_impossible:
                tok_start_position = -1
                tok_end_position = -1
            else:
                tok_start_position = orig_to_tok_index[example.start_position]

                if example.end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1

                (tok_start_position, tok_end_position) = _improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                    example.orig_answer_text)
        else:  # multiple answers
            tok_start_position_list = []
            tok_end_position_list = []

            if example.is_impossible:
                tok_start_position = -1
                tok_end_position = -1
            else:
                for _id in range(len(example.start_position)):
                    tok_start_position = orig_to_tok_index[example.start_position[_id]]

                    if example.end_position[_id] < len(example.doc_tokens) - 1:
                        tok_end_position = orig_to_tok_index[example.end_position[_id] + 1] - 1
                    else:
                        tok_end_position = len(all_doc_tokens) - 1

                    (tok_start_position, tok_end_position) = _improve_answer_span(
                        all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                        example.orig_answer_text[_id])

                    tok_start_position_list.append(tok_start_position)
                    tok_end_position_list.append(tok_end_position)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0

        # print('all_doc_tokens:',len(all_doc_tokens))
        # print('max_tokens_for_doc:',max_tokens_for_doc)
        # print('doc_stride:',doc_stride)
        # print('-'*25)
        while start_offset < len(all_doc_tokens):
            # print('start_offset:',start_offset)
            length = len(all_doc_tokens) - start_offset
            # print('length:',length)

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

            # print('doc_span.length:',type(doc_span.length),doc_span.length)
            for i in range(int(doc_span.length)):
                split_token_index = int(doc_span.start + i)
                # print('split_token_index:',split_token_index,type(split_token_index))
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

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

            # if is_training and not example.is_impossible:
            #     # For training, if our document chunk does not contain an annotation
            #     # we throw it out, since there is nothing to predict.
            #     doc_start = doc_span.start
            #     doc_end = doc_span.start + doc_span.length - 1
            #     out_of_span = False
            #     if not (tok_start_position >= doc_start and
            #             tok_end_position <= doc_end):
            #       out_of_span = True
            #     if out_of_span:
            #     start_position = 0
            #     end_position = 0
            #     else:
            #     doc_offset = len(query_tokens) + 2
            #     start_position = tok_start_position - doc_start + doc_offset
            #     end_position = tok_end_position - doc_start + doc_offset
            # if is_training and example.is_impossible:
            #     start_position = 0
            #     end_position = 0

            if is_training:
                if not example.is_impossible:
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
                else:
                    start_position = 0
                    end_position = 0

            else:
                start_position = []
                end_position = []

                for tok_start_position, tok_end_position in zip(tok_start_position_list, tok_end_position_list):
                    if not example.is_impossible:
                        # For training, if our document chunk does not contain an annotation
                        # we throw it out, since there is nothing to predict.
                        doc_start = doc_span.start
                        doc_end = doc_span.start + doc_span.length - 1
                        out_of_span = False
                        if not (tok_start_position >= doc_start and
                                tok_end_position <= doc_end):
                            out_of_span = True
                        if out_of_span:
                            _start_position = 0
                            _end_position = 0
                        else:
                            doc_offset = len(query_tokens) + 2
                            _start_position = tok_start_position - doc_start + doc_offset
                            _end_position = tok_end_position - doc_start + doc_offset
                    else:
                        _start_position = 0
                        _end_position = 0
                    start_position.append(_start_position)
                    end_position.append(_end_position)

            feature = InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                orig_answer_text=example.orig_answer_text,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_position,
                end_position=end_position,
                is_impossible=example.is_impossible)
            # Run callback
            # output_fn(feature)
            unique_id += 1
            features.append(feature.get_dict())

            # print(len(features[0]['tokens']), features[0]['tokens'] )
            # print(len(features[0]['input_ids']), features[0]['input_ids'] )
            # print( features[0]['start_position'], features[0]['end_position'])
            # print( features[0]['tokens'][start_position: end_position+1])
            # print( features[0]['orig_answer_text'])
            # print( features[0]['doc_span_index'])
            # assert 1>2

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
        if text_span == tok_answer_text:
            return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
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
