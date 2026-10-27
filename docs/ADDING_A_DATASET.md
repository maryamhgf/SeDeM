# Adding a dataset

SeDeM reads datasets through small transform functions registered in `sedem/data.py`. Two are needed.

## 1. Stage 1 transform (raw text)

Maps a batch of HuggingFace rows to `{"text": [...]}`; each text is one long context that will be segmented and compressed.

```python
def mydata_pretrain_transform(examples):
    return {"text": [" ".join(p) for p in examples["paragraphs"]]}
```

## 2. Stage 2 / evaluation transform (QA triples)

Maps rows to `input` (the context), `prompt` (the question) and `answer`. For evaluation also return `all_answers`
(list of acceptable strings) and, if the dataset has evidence annotations, `evidence` (list of evidence strings) so that
`label` selection and the selector's training labels work.

```python
def mydata_qa_transform(examples):
    return {
        "input":  [" ".join(p) for p in examples["paragraphs"]],
        "prompt": examples["question"],
        "answer": examples["answer"],
    }

def mydata_qa_eval_transform(examples):
    out = mydata_qa_transform(examples)
    out["all_answers"] = [[a] for a in examples["answer"]]
    out["evidence"] = examples.get("supporting_text", [[] for _ in examples["answer"]])
    return out
```

Register them:

```python
PRETRAIN_TRANSFORMS["mydata"] = mydata_pretrain_transform
QA_TRANSFORMS["mydata"] = mydata_qa_transform
QA_EVAL_TRANSFORMS["mydata"] = mydata_qa_eval_transform
```

and in your config set `DATASET_NAME=<hf id or local path>`, `STAGE1_TRANSFORM=mydata`, `STAGE2_TRANSFORM=mydata`,
`EVAL_HARNESS=qa`, `EVAL_TRANSFORM=mydata`. Local JSONL files (`{"input": ..., "prompt": ..., "answer": ...}` per line)
can be used for Stage 2 directly with `STAGE2_DATA_PATH` / `STAGE2_EVAL_DATA_PATH`, no transform needed.

## Selector training labels

`train_selector.py` needs, per question, the set of context blocks that contain the evidence. For paragraph-structured
datasets this is `sedem/evidence_labels_hotpot.py` (supporting-fact titles -> paragraph blocks) and
`sedem/blocks_{hotpot,twowiki,musique}.py` (one block per paragraph); for long documents it is
`sedem/evidence_labels_qasper.py` (evidence spans -> the segments that overlap them). To add a dataset to the selector
trainer, add a branch in `train_selector.py` (`--dataset`) that builds `(blocks, num_blocks, positive_mask)` per example
from your transform's `evidence` field, following the `qasper` branch for long documents or the `hotpot` branch for
paragraph lists.

Without evidence labels you can still run SeDeM selector-disabled (`EVAL_SELECTION_MODE=off`: every block is
decompressed).
