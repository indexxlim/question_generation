'''
    Generation Question by MRC dataset
'''
import time
import tqdm
import torch
from transformers import set_seed, AdamW
from apex.parallel import DistributedDataParallel as DDP
from torch.nn.parallel.data_parallel import DataParallel
import os


SEED = 42
set_seed(SEED)


def serialize_args(args):
    def is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except:
            return False

def single_epoch_train(model, optimizer, train_loader, pad_id, args):
    model.train()
    logger = args.logger
    loader = tqdm.tqdm(train_loader)
    device = args.device
    print(args.distributed)


    for idx, batch in enumerate(loader):
        if args.distributed:
            source_ids, source_mask, target_ids, target_mask = (
                batch['source_ids'].cuda(),
                batch['source_mask'].cuda(),
                batch['target_ids'].cuda(),
                batch['target_mask'].cuda()
            )        
    
        else:
            source_ids, source_mask, target_ids, target_mask = (
                batch['source_ids'].to(device),
                batch['source_mask'].to(device),
                batch['target_ids'].to(device),
                batch['target_mask'].to(device)
            )

        target_ids[target_ids==pad_id] = -100

        outputs = model(
            input_ids = source_ids,
            attention_mask = source_mask,
            labels=target_ids,
            decoder_attention_mask=target_mask
        )

        loss = outputs[0]
        loss = loss.mean()
        loader.set_description(f"Batch Loss: {loss.item():.3f}")
        loader.refresh()
        try:
            import wandb
            wandb.log({'loss': loss.item()})
        except:
            pass
        
        

        loss = loss / args.gradient_accumulation_steps
        loss.backward()

        if idx % args.gradient_accumulation_steps:
            optimizer.step()
            optimizer.zero_grad()
            
        if not idx % args.log_every:
            logger.info(f"Loss: {loss.item()}")
            
@torch.no_grad()          
def single_epoch_validate(model, tokenizer, valid_loader, pad_id, args):
    '''
        Validating or Testing for a single epoch.
    '''
    model.eval()
    logger = args.logger
    loader = tqdm.tqdm(valid_loader)
    device = args.device
    
    output = []
    losses = []
    for idx, batch in enumerate(loader):
        source_ids, source_mask, target_ids, target_mask, kind_batch = (
            batch['source_ids'].to(device),
            batch['source_mask'].to(device),
            batch['target_ids'].to(device),
            batch['target_mask'].to(device),
            batch['kinds']
        )
        
        repetition_penalty = 2.5
        length_penalty=1.0
        no_repeat_ngram_size=3
        #with torch.cuda.amp.autocast():
        pred_ids = model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            max_length=200, 
            num_beams=3,
            repetition_penalty=repetition_penalty, 
            length_penalty=length_penalty, 
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
            # top_k=50,
            # top_p=1.0,
            # do_sample=False,
            # temperature=1.0,
            # num_return_sequences=10,
            # length_penalty=2,
            # min_length=3,
            # decoder_start_token_id=model.config.eos_token_id,
        )
        
        decoded_inputs = [tokenizer.decode(c, 
                          skip_special_tokens=False, 
                          clean_up_tokenization_spaces=False) 
                          for c in source_ids]
        
        decoded_preds = [tokenizer.decode(c, 
                          skip_special_tokens=False, 
                          clean_up_tokenization_spaces=False) 
                          for c in pred_ids]
        
        decoded_labels = [tokenizer.decode(c, 
                          skip_special_tokens=False, 
                          clean_up_tokenization_spaces=False) 
                          for c in target_ids]

        for inputs, preds, labels, kinds in zip(decoded_inputs, decoded_preds, decoded_labels, kind_batch):
            O = { 'inputs': inputs, 'preds': preds, 'labels': labels, 'kinds': kinds}
            outputs.appends(o)

        target_ids[target_ids==pad_id] = -100

        model_ouptuts = model(
            input_ids = source_ids,
            attention_mask = source_mask,
            labels=target_ids,
            decoder_attention_mask=target_mask
        )
        loss = model_outputs[0]

        losses.append(loss.item())

        loader.set_description(f"Batch Loss: { loss.item(): .3f}")
        lodaer.refresh()

        try:
            import wandb
            wandb.log({'val_loss': loss.item()})
        except:
            pass

        if not idx % args.log_every:
            logger.info(f"Loss: {loss.item()}")

    avg_loss = sum(losses) / len(losses)

    metrics = compute_metric(tokenizer, outputs)
    metrics['val_avg_loss'] = avg_loss

    return outputs, metrics



def train(model, optimizer, tokenizer, train_loader, test_loader, args):
    logger = args.logger


    # if args.distributed:
    #     # FOR DISTRIBUTED:  Set the device according to local_rank.
    #     torch.cuda.set_device(args.local_rank)

    #     # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
    #     # environment variables, and requires that you use init_method=`env://`.
    
    #     model.cuda()
    #     model = DDP(model, delay_allreduce=True)
    # else:
    #     model = DataParallelModel(model)

    for epoch in range(args.epochs):

        start_time = time.time()
        logger.info(f"Epoch {epoch+1} Globally {args.checkpoint_count}")

        #Training
        logger.info(f"Begin Training ...")
        single_epoch_train(model, optimizer, train_loader, tokenizer.pad_token_id, args)        
        mins = round((time.time() - start_time) / 60 ,2)
        logger.info(f"Training Fisished!")
        logger.info(f"Time taken for training epoch {epoch+1} globally {args.checkpoint_count}: {mins} min(s)")
        
        #Validating
        #ogger.info(f"Begin Validation ...")
        #outputs, metrics = single_epoch_validate(model, tokenizer, valid_loader, tokenizer.pad_token_id, args)

        #log_metric(metrics, args.loader, prefix='valid')

        #outputs, metrics = sin

        #log_metric(metrics ,args)
        model.save_pretrained(f"{args.checkpoint}/saved_checkpoint_{args.checkpoint_count}")
        logger.info(f"Checkpoint saved at {args.checkpoint}/saved_checkpoint_{args.checkpoint_count}")

        args.checkpoint_count += 1
