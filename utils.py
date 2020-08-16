import torch

def get_mnli_map_fn(lang_id, max_length, tokenizer):
    def convert_to_features(examples):
        premises = []
        for ex in examples["premise"]:
            ex = f"{lang_id} {ex} </s> </s>"
            premises.append(ex)
        
        hyps = examples["hypothesis"]
        
        enc = tokenizer.prepare_seq2seq_batch(
            src_texts=[(premise, hyp) for premise, hyp in zip(premises, hyps)],
            src_lang=lang_id,
            max_length=192,
            padding="max_length",
            truncation=True,
            return_tensors=None
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "lables": examples["label"]
        }
  
    return convert_to_features