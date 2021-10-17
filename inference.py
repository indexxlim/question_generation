
from fastapi import APIRouter
from loguru import logger
import transformers
import torch
import random
from pydantic import BaseModel

from typing import Optional, List, Dict

from configloader import server_config




class responseJSON(BaseModel):
    candidates: List[str]

        
class requestJSON(BaseModel):
    context: str
    answer: str
    answer_p: Optional[int]



router = APIRouter()

device=server_config.misc.device
model_class = getattr(transformers, server_config.misc.model_class)
model = model_class.from_pretrained(server_config.misc.model)
model.to(device)

# Define Tokenizer
tokenizer_class = getattr(transformers, server_config.misc.tokenizer_class)
tokenizer = tokenizer_class.from_pretrained(server_config.misc.tokenizer)
source_span_len = 1000

def inference(model, tokenizer, input_data, configs):
    context     = input_data['context']
    answer      = input_data['answer']
    tokenizer

    if 'answer_p' in input_data:
        start_position = input_data['answer_p']
    else:
        context.index(answer)

    end_position = start_position + len(answer)
    if len(context) > source_span_len:   #check source_span_len
        before_start = int(max(start_position-source_span_len/2,0))
        before_answer = context[before_start:start_position]
        after_answer = context[end_position:end_position+int(source_span_len/2)]
        
        if len(before_answer) < int(source_span_len/2):
            after_end = end_position + int(source_span_len/2 - len(before_answer))
            after_answer = context[end_position:after_end]
        elif len(after_answer) < int(source_span_len/2):
            before_start = before_start - int(source_span_len/2 - len(after_answer))
            before_start = int(max(before_start,0))
            before_answer = context[before_start:start_position]
        
        source = before_answer + '<extra_id_0>' + context[start_position:end_position] + '<extra_id_0>' + after_answer
    else:
        source = context[:start_position] + '<extra_id_0>' + context[start_position:end_position] + '<extra_id_0>' + context[end_position:]

    source_batch = tokenizer.batch_encode_plus(source,
                                    padding='max_length',
                                    max_length=tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors='pt')

    source_ids, source_mask= (
            source_batch.input_ids.to(device),
            source_batch.attention_mask.to(device))
            
    repetition_penalty = 2.5
    length_penalty=1.0
    no_repeat_ngram_size=3

    pred_ids = model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            max_length=200, 
            num_beams=3,
            early_stopping=True
        )

    decoded_preds = [tokenizer.decode(c, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False) 
                    for c in pred_ids]


@router.post("/generate", response_model=responseJSON)
def generate(input_data: requestJSON):
    try:
        results = inference(
            model,
            tokenizer,
            dict(input_data),
            server_config
        )
        return results
    except Exception as exc:  #add type of Exception 
        raise(exc)
    