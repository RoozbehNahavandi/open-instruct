from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import PreTrainedModel
import torch
from typing import List, Dict, Tuple, Any
from abc import abstractmethod
import numpy as np
# from datasets import load_metric
import evaluate
from datasets import load_dataset
import random
from gem_metrics.msttr import MSTTR
from gem_metrics.ngrams import NGramStats
from gem_metrics.texts import Predictions
from tqdm import tqdm
import copy
import rouge
import pdb
from abc import abstractclassmethod
from dataclasses import dataclass
from datasets.arrow_dataset import Dataset



@dataclass(init=True)
class Sample:
    id: str
    prompt_or_input_text: str
    references: List[str]
    meta_data: Dict[str, Any] = None



class TextGenPool:
    def __init__(self, samples: List[Sample]):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, ix: int) -> Sample:
        if ix >= len(self):
            raise StopIteration
        sample = self._samples[ix]
        return sample, 1.0

    def sample(self) -> Sample:
        random_sample = random.choice(self._samples)
        return random_sample

    @abstractclassmethod
    def prepare(cls, **args) -> 'TextGenPool':
        """
        A factory method to instantiate data pool
        """
        raise NotImplementedError

    def split(self, split_ratios: List[float]) -> List['TextGenPool']:
        start_ix = 0
        pools = []
        for ratio in split_ratios:
            count = int(len(self) * ratio)
            end_ix = start_ix + count
            pools.append(type(self)(self._samples[start_ix: end_ix]))
            start_ix = end_ix
        return pools



class CommonGen(TextGenPool):
    @classmethod
    def prepare(cls, split: str,
                concept_separator_token: str = " ",
                concept_end_token=" ",
                prefix: str = "summarize: ") -> 'TextGenPool':
        ds = load_dataset("gem", "common_gen")
        samples = []
        split_id = CommonGen.gen_split_name(split)
        for ix, item in enumerate(ds[split_id]):
            concepts = concept_separator_token.join(item["concepts"])
            concepts = prefix + concepts
            concepts += concept_end_token
            if item["target"] == "":
                # just to avoid breaking of metric computation
                item["target"] = "empty reference"
            targets = [item["target"]]
            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=concepts,
                            references=targets,
                            meta_data={
                                "concepts": item["concepts"]
                            }
                            )
            samples.append(sample)
        pool_instance = cls(samples)
        return pool_instance

    @staticmethod
    def gen_split_name(split: str):
        if split == "train":
            split_name = "train"
        elif split == "val":
            split_name = "validation"
        elif split == "test":
            split_name = "test"
        else:
            raise NotImplementedError
        return split_name




class DailyDialog(TextGenPool):
    EOU_TOKEN = "<EOU>"
    @classmethod
    def prepare(cls, split: str, context_size: int):
        split = CommonGen.gen_split_name(split)
        dataset = load_dataset("daily_dialog", split=split, keep_in_memory=True)
        samples = []
        utterance_id = 0
        for item in dataset:
            contexts = []
            for utterance, emotion, intent in zip(item["dialog"],
                                                  item["emotion"],
                                                  item["act"]):
                if len(contexts) >= context_size:
                    context = DailyDialog.EOU_TOKEN.join(contexts[-context_size:]) 
                    context += " " + DailyDialog.EOU_TOKEN
                    target = utterance + DailyDialog.EOU_TOKEN
                    sample = Sample(id=utterance_id, 
                                    prompt_or_input_text=context, 
                                    references=[target],
                                    meta_data={
                                        "emotion": [emotion],
                                        "intent": [intent]
                                    })
                    samples.append(sample)
                contexts.append(utterance)
                utterance_id += 1

        dp_instance = cls(samples)
        return dp_instance


class BaseMetric:
    @abstractmethod
    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        """
        Returns a dict where key is the metric name and value is again a dict consisting of tuple of individual scores (if any) and corpus level score

        eg. {
            metric_name: (individual_scores, corpus_level_score)
            "metric_1": ([0.5, 0.5, 0.8], 0.1)
        }

        """
        raise NotImplementedError


class LearnedRewardMetric(BaseMetric):
    def __init__(
        self,
        model_name: str,
        label_ix: int,
        batch_size: int,
        include_prompt_for_eval: bool = True,
    ) -> None:
        super().__init__()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._tokenizer.truncation_side = "left"
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self._device
        )
        self._label_ix = label_ix
        self._batch_size = batch_size
        self._include_prompt_for_eval = include_prompt_for_eval

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Dict[str, float]:
        all_scores = []
        current_ix = 0
        n_texts = len(generated_texts)
        while current_ix < n_texts:
            batch_gen_texts = generated_texts[
                current_ix : current_ix + self._batch_size
            ]
            batch_prompt_texts = prompt_texts[
                current_ix : current_ix + self._batch_size
            ]

            if self._include_prompt_for_eval:
                batch_gen_texts = [
                    (prompt + gen)
                    for gen, prompt in zip(batch_gen_texts, batch_prompt_texts)
                ]
            encoded = self._tokenizer(
                batch_gen_texts, return_tensors="pt", truncation=True, padding=True
            )
            with torch.no_grad():
                outputs = self._model(
                    input_ids=encoded.input_ids.to(self._device),
                    attention_mask=encoded.attention_mask.to(self._device),
                )
                scores = torch.softmax(outputs.logits, dim=1)
                scores = scores[:, self._label_ix].tolist()
                all_scores.extend(scores)
            current_ix += self._batch_size

        metric_dict = {
            "semantic/learned_automodel_metric": (all_scores, np.mean(all_scores))
        }
        return metric_dict


class MeteorMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = evaluate.load("meteor")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[str],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):

        score = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )["meteor"]
        # print(score, 'this is meteor score', generated_texts, reference_texts)

        metric_dict = {"lexical/meteor": (None, score)}
        return metric_dict
    

class CRLHFEvaluationMetric(BaseMetric):
    def __init__(self, **args) -> None:
        super().__init__()
        # lexical metrics
        self._bleu_metric = BLEUMetric()
        self._sacrebleu_metric = SacreBLEUMetric(**args)
        self._rouge_metric = RougeMetric()
        # diversity metrics
        self._diversity_metrics = DiversityMetrics()
        self._diversity_metric_names = [
            "diversity_metrics/max_pred_length-nopunct",
            "diversity_metrics/vocab_size-3-nopunct",
            "diversity_metrics/unique-3"]


    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        # print(f'prompt_texts: {prompt_texts}')
        # print(f'generated_texts: {generated_texts}')
        # print(f'reference_texts: {reference_texts}')
        _MIN_BLEU, _MAX_BLEU = 0.0, 0.004676984628247904
        _MIN_SACREBLEU, _MAX_SACREBLEU = 5.330883519077344e-06, 0.11063353821330388
        _MIN_ROUGE2, _MAX_ROUGE2 = 0.0, 0.014568765083436451

        _MIN_MAXPREDLENGTH, _MAX_MAXPREDLENGTH = 9, 183
        _MIN_VOCABSIZE, _MAX_VOCABSIZE = 41, 32_227
        _MIN_UNIQUE, _MAX_UNIQUE = 314, 36_880
        reference_texts = [[ref] for ref in reference_texts]

        args = [prompt_texts, generated_texts, reference_texts,]
        kwargs = {"meta_infos": meta_infos, "model": model, "split_name": split_name,}
        bleu_score = self._bleu_metric.compute(*args, **kwargs)["lexical/bleu"][1]
        sacrebleu_score = self._sacrebleu_metric.compute(*args, **kwargs)["lexical/sacrebleu"][1]
        rouge_score = self._rouge_metric.compute(*args, **kwargs)["lexical/rouge_rouge2"][1]
        diversity_scores = self._diversity_metrics.compute(*args, **kwargs)
        diversity_scores = {
            k: v[1] for k, v in diversity_scores.items() if k in self._diversity_metric_names}
        maxpredlength_score = diversity_scores["diversity_metrics/max_pred_length-nopunct"]
        vocabsize_score = diversity_scores["diversity_metrics/vocab_size-3-nopunct"]
        unique_score = diversity_scores["diversity_metrics/unique-3"]
        
        try:
            # normalize
            bleu_score = (bleu_score - _MIN_BLEU) / (_MAX_BLEU - _MIN_BLEU)
            sacrebleu_score = (sacrebleu_score - _MIN_SACREBLEU) / (_MAX_SACREBLEU - _MIN_SACREBLEU)
            rouge_score = (rouge_score - _MIN_ROUGE2) / (_MAX_ROUGE2 - _MIN_ROUGE2)
            maxpredlength_score = (maxpredlength_score - _MIN_MAXPREDLENGTH) / (_MAX_MAXPREDLENGTH - _MIN_MAXPREDLENGTH)
            vocabsize_score = (vocabsize_score - _MIN_VOCABSIZE) / (_MAX_VOCABSIZE - _MIN_VOCABSIZE)
            unique_score = (unique_score - _MIN_UNIQUE) / (_MAX_UNIQUE - _MIN_UNIQUE)
            # average lexical scores
            lexical_score = (bleu_score + sacrebleu_score + rouge_score) / 3
            # average diversity scores
            diversity_score = (maxpredlength_score + vocabsize_score + unique_score) / 3
            # average lexical and diversity together
            crlhfeval_score = (lexical_score + diversity_score) / 2

            print(f'bleu_score: {bleu_score}, sacrebleu_score: {sacrebleu_score}, rouge_score: {rouge_score}')
            print(f'maxpredlength_score: {maxpredlength_score}, vocabsize_score: {vocabsize_score}, unique_score: {unique_score}')
            print(f'crlhfeval_score: {crlhfeval_score}')
            # return
            return {"lexical/CRLHFEval_Score": (None, crlhfeval_score)}
        except:
            return {"lexical/CRLHFEval_Score": (None, "n/a")}
        




class RougeMetric(BaseMetric):
    def __init__(self, use_single_ref: bool = True) -> None:
        super().__init__()
        self._metric = evaluate.load("rouge")
        self._use_single_ref = use_single_ref

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        if self._use_single_ref:
            # TBD: this is required for CNN/DM dataset, without this we get low scores
            # TBD: needs investigation
            ref_texts = [ref[0] for ref in reference_texts]
        else:
            ref_texts = reference_texts

        metric_results = self._metric.compute(
            predictions=generated_texts, references=ref_texts, use_stemmer=True
        )
        # print('we are in rougemetric, ', metric_results)
        # print(generated_texts, ref_texts)
        score_keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        metric_dict = {}
        for rouge_type in score_keys:
            rouge_score = metric_results[rouge_type]
            metric_dict[f"lexical/rouge_{rouge_type}"] = (None, rouge_score)
        return metric_dict



class BLEUMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = evaluate.load("bleu")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        tokenized_predictions = []
        tokenized_reference_texts = []
        for prediction, refs in zip(generated_texts, reference_texts):
            tokenized_prediction = prediction.split()
            tokenized_refs = [ref.split() for ref in refs]
            tokenized_predictions.append(tokenized_prediction)
            tokenized_reference_texts.append(tokenized_refs)

        try:
            # print(generated_texts, reference_texts)
            metric_results = self._metric.compute(
                predictions=generated_texts, references=reference_texts
            )
            # print(metric_results, 'metric results')
            bleu_score = metric_results["bleu"]
            metric_dict = {"lexical/bleu": (None, bleu_score)}
            return metric_dict
        except Exception as e:
            return {"lexical/bleu": (None, "n/a")}


class BLEURTMetric(BaseMetric):
    def __init__(self, config_name: str = None) -> None:
        super().__init__()
        self._metric = evaluate.load("bleurt", config_name=config_name)

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )
        corpus_score = np.mean(metric_results["scores"])
        metric_dict = {"semantic/bleurt": (metric_results["scores"], corpus_score)}
        return metric_dict


def get_generated_and_predictions(
    prompt_texts: List[str],
    generated_texts: List[str],
    reference_texts: List[List[str]],
    split_name: str,
):
    split_name = "" if split_name is None else split_name
    preds = {}
    refs = {}
    for ix, (prompt_text, gen_text, ref_text) in enumerate(
        zip(prompt_texts, generated_texts, reference_texts)
    ):
        preds[split_name + prompt_text] = [gen_text]
        refs[split_name + prompt_text] = ref_text
    return preds, refs


def get_individual_scores(
    prompt_texts: List[str], split_name: str, scores_dict: Dict[str, float]
):
    split_name = "" if split_name is None else split_name
    scores = []
    for prompt_text in prompt_texts:
        scores.append(scores_dict.get(split_name + prompt_text, "n/a"))
    return scores


class DiversityMetrics(BaseMetric):
    def __init__(self, window_size: int = 100) -> None:
        self._msttr_metric = MSTTR(window_size=window_size)
        self._n_gram_metric = NGramStats()

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        predictions = Predictions(data={"filename": "", "values": generated_texts})
        # print(f'predictions: {predictions}')
        # print(f'type of generated_texts: {type(generated_texts)}, {generated_texts}')
        diversity_metrics = {}
        msttr_metrics = self._msttr_metric.compute(None, predictions)
        n_gram_metrics = self._n_gram_metric.compute(None, predictions)

        for key, value in msttr_metrics.items():
            diversity_metrics[f"diversity_metrics/{key}"] = (None, value)
        for key, value in n_gram_metrics.items():
            diversity_metrics[f"diversity_metrics/{key}"] = (None, value)

        return diversity_metrics



class Perplexity(BaseMetric):
    def __init__(
        self,
        stride: int,
        tokenizer_id: str,
        model_type: str = "causal",
        use_text_from_meta_data: bool = False,
    ) -> None:
        super().__init__()
        self._tokenizer_id = tokenizer_id
        self._model_type = model_type
        self._stride = stride
        self._use_text_from_meta_data = use_text_from_meta_data

    def get_device(self, model: PreTrainedModel):
        try:
            return model.transformer.first_device
        except:
            return model.device

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        if split_name == "train":
            return {}

        if self._model_type != "causal":
            raise NotImplementedError

        # we compute perplexity on reference texts
        if self._use_text_from_meta_data:
            reference_texts = [info["reference"] for info in meta_infos]
        else:
            reference_texts = [ref for refs in reference_texts for ref in refs]
        tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_id)
        encodings = tokenizer("\n\n".join(reference_texts), return_tensors="pt")

        device = self.get_device(model)

        nlls = []
        max_length = model.config.n_positions
        for i in tqdm(range(0, encodings.input_ids.size(1), self._stride)):
            begin_loc = max(i + self._stride - max_length, 0)
            end_loc = min(i + self._stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop

            # run on last device
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

        return {
            "fluency_metrics/perplexity": (
                None,
                torch.exp(torch.stack(nlls).sum() / end_loc).item(),
            )
        }



class RougeLMax(BaseMetric):
    def __init__(self, **args) -> None:
        super().__init__()
        self._metric = rouge.Rouge(metrics=["rouge-l"], **args)

    def _rouge_max_over_ground_truths(self, prediction, ground_truths):
        """
        Computes max of Rouge-L (https://github.com/allenai/unifiedqa/blob/bad6ef339db6286f0d8bd0661a2daeeb0f800f59/evaluation/evaluate_narrativeqa.py#L25)
        """
        # load stemmer
        self._metric.load_stemmer(self._metric.ensure_compatibility)

        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = self._metric.get_scores(prediction, [ground_truth])
            scores_for_ground_truths.append(score)
        max_score = copy.deepcopy(score)
        max_score = max([score["rouge-l"]["f"] for score in scores_for_ground_truths])
        return max_score

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        all_scores = []
        for gen_text, ref_texts in zip(generated_texts, reference_texts):
            rouge_max_score = self._rouge_max_over_ground_truths(gen_text, ref_texts)
            all_scores.append(rouge_max_score)

        metric_dict = {"lexical/rouge_l_max": (all_scores, np.mean(all_scores))}
        return metric_dict


class SacreBLEUMetric(BaseMetric):
    def __init__(self, **args) -> None:
        super().__init__()
        self._args = args
        self._metric = evaluate.load("sacrebleu")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts, **self._args
        )
        bleu_score = metric_results["score"] / 100
        metric_dict = {"lexical/sacrebleu": (None, bleu_score)}
        return metric_dict


class TERMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = evaluate.load("ter")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )
        score = metric_results["score"] / 100
        metric_dict = {"lexical/ter": (None, score)}
        return metric_dict


class chrFmetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._metric = evaluate.load("chrf")

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:

        metric_results = self._metric.compute(
            predictions=generated_texts, references=reference_texts
        )
        score = metric_results["score"] / 100
        metric_dict = {"lexical/chrf": (None, score)}
        return metric_dict


class IntentAccuracyDailyDialog(BaseMetric):
    def __init__(self) -> None:
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(
            "rajkumarrrk/roberta-daily-dialog-intent-classifier"
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            "rajkumarrrk/roberta-daily-dialog-intent-classifier"
        )
        # self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # self._device = f"cuda:{torch.cuda.device_count() - 1}"
        if torch.cuda.is_available():
            self._device = f"cuda:{torch.cuda.device_count() - 1}"
        else:
            self._device = "cpu"
        self._model = self._model.to(self._device)


    def get_model(self):
        return self._model
    
    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        def get_input_for_classifier(prompt, generated_text):
            history = prompt.split(DailyDialog.EOU_TOKEN)
            history = [utt for utt in history if utt != ""]
            last_utterance = history[-1]
            input_text = last_utterance + generated_text
            return input_text


        input_texts = [
            get_input_for_classifier(prompt, gen)
            for prompt, gen in zip(prompt_texts, generated_texts)
        ]

        # extract target intents
        target_intents = [info["intent"][0] - 1 for info in meta_infos]

        # tokenize
        encoded = self._tokenizer(
            input_texts, return_tensors="pt", truncation=True, padding=True
        )

        with torch.no_grad():
            outputs = self._model(
                input_ids=encoded.input_ids.to(self._device),
                attention_mask=encoded.attention_mask.to(self._device),
            )
            pred_labels = torch.argmax(outputs.logits, dim=1).tolist()

        matching_scores = (np.array(pred_labels) == np.array(target_intents)).astype(
            np.int32
        )
        intent_accuracy = np.mean(matching_scores)

        metric_dict = {"intent/accuracy": (matching_scores.tolist(), intent_accuracy)}
        return metric_dict


def get_dataset(datapool, label="intent"):
    # get the training data in text, label format
    texts = []
    labels = []
    for sample, _ in datapool:

        history = sample.prompt_or_input_text.split(DailyDialog.EOU_TOKEN)
        history = [utt for utt in history if utt != ""]
        last_utterance = history[-1]

        # just consider the utterance
        input_text = last_utterance + DailyDialog.EOU_TOKEN + sample.references[0]
        
        texts.append(input_text)
        labels.append(sample.meta_data[label][0]-1)
    
    print(np.unique(labels, return_counts=True))

    dataset = Dataset.from_dict(
            {
                "text": texts,
                "labels": labels
            },
            split="train"
        )
    return dataset

if __name__ == "__main__":
    prompt_texts = [""]
    gen_texts = ["Hello there general kenobi", "foo bar foobar", 'hi', 'hi2', 'hi3', 'hi4']
    reference_texts = ["Hello there general kenobi", "foo bar foobar", 'hi', 'hi22', 'hi33', 'hi44']
    # dd_instance = DailyDialog.prepare('train', 1)
    # ds_train = get_dataset(dd_instance, 'intent')

    # print(f'ds_train: {ds_train["text"][:10]}')
    # metric = MeteorMetric()
    # print("MeteorMetric:", metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = RougeMetric()
    # print("RougeMetric:", metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = SacreBLEUMetric(tokenize="intl")
    # print("SacreBLEUMetric:", metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = BLEUMetric()
    # print("BLEUMetric:", metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = DiversityMetrics()
    # print("DiversityMetrics:", metric.compute(prompt_texts, gen_texts, reference_texts))

    metric = CRLHFEvaluationMetric()
    print("metric:", metric.compute(prompt_texts, gen_texts, reference_texts))

    # document = """Jeff joined Microsoft in 1992 to lead corporate developer evangelism for Windows NT. He then served as a Group Program manager in Microsoft’s Internet Business Unit. In 1998, he led the creation of SharePoint Portal Server, which became one of Microsoft’s fastest-growing businesses, exceeding $2 billion in revenues. Jeff next served as Corporate Vice President for Program Management across Office 365 Services and Servers, which is the foundation of Microsoft’s enterprise cloud leadership. He then led Corporate Strategy supporting Satya Nadella and Amy Hood on Microsoft’s mobile-first/cloud-first transformation and acquisitions. Prior to joining Microsoft, Jeff was vice president for software development for an investment firm in New York. He leads Office shared experiences and core applications, as well as OneDrive and SharePoint consumer and business services in Office 365. Jeff holds a Master of Business Administration degree from Harvard Business School and a Bachelor of Science degree in information systems and finance from New York University."""
    # summary = "Jeff joined Microsoft in 1992 to lead the company's corporate evangelism. He then served as a Group Manager in Microsoft's Internet Business Unit. In 1998, Jeff led Sharepoint Portal Server, which became the company's fastest-growing business, surpassing $3 million in revenue. Jeff next leads corporate strategy for SharePoint and Servers which is the basis of Microsoft's cloud-first strategy. He leads corporate strategy for Satya Nadella and Amy Hood on Microsoft's mobile-first."

    # metric = SummaCZSMetric(granularity="sentence",
    #                         use_ent=True,
    #                         use_con=False)
    # print(metric.compute([document], [summary], []))

    # metric = SummaCConvMetric(granularity="sentence")
    # print(metric.compute([document], [summary], []))

    # prompt_texts = ["1", "2"]
    # gen_texts = [
    #     "The dog is the boy's cat.",
    #     "A boy is picking apples from trees and put them into bags.",
    # ]
    # reference_texts = [
    #     ["The dog is the boy's cat.", "The dog eats the cat of the boy."],
    #     ["A boy is picking apples from trees."],
    # ]
    # metric = CIDERMetric()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

    # metric = SpiceMetric()
    # print(metric.compute(prompt_texts, gen_texts, reference_texts))

