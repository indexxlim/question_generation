'''
    REFERENCE: BART Data loader
'''

from torch.utils.data import Dataset, DataLoader
import json

# class QGDataModule(pl.LightningDataModule):
#    def __init__(self, train_datapath, test_datapath, QGBatchGenerator, batch_size: int = 4):
#      super().__init__()
#      self.train_datapath = train_datapath
#      self.test_datapath = test_datapath
#      self.batch_size = batch_size
#      self.QGBatchGenerator = QGBatchGenerator
#    def setup(self):
#      self.train_dataset = QGDataset(self.train_datapath)
#      self.test_dataset = QGDataset(self.test_datapath)
#    def train_dataloader(self):
#      return DataLoader(
#          self.train_dataset,
#          batch_size=self.batch_size,
#          shuffle=True,
#          collate_fn=self.QGBatchGenerator,
#          num_workers=4
#          )
#    def val_dataloader(self):
#      return DataLoader(
#          self.test_dataset,
#          batch_size=self.batch_size,
#          collate_fn=self.QGBatchGenerator,
#          num_workers=4
#          )
#    def test_dataloader(self):
#      return DataLoader(
#          self.test_dataset,
#          batch_size=1,
#          collate_fn=self.QGBatchGenerator,
#          num_workers=4
#          )


class QGDataset(Dataset):
    '''
        To read Korean MRC Corpus
        /home/choyh/0.RPAi/3.MRC/0.data/MRC_Text_train_data_300K_2021-08-09.json
    '''
    def __init__(self,data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)["data"]
        

    def __getitem__(self, index):
        return self.data[index]
        
    def __len__(self):
        return len(self.data)
    

class QGBatchGenerator:
    '''
        collate function
    '''
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer        


    def __call__(self, batch):
        source = []
        target = []
        for i in range(len(batch)):
            source.append(f"answer: {batch[i]['a']}  context: {batch[i]['c']}")
        
        target = [item['q'] for item in batch]
        

        source_batch = self.tokenizer.batch_encode_plus(source,
                                                        padding='max_length',
                                                        max_length=self.tokenizer.model_max_length,
                                                        truncation=True,
                                                        return_tensors='pt')

        target_batch = self.tokenizer.batch_encode_plus(target,
                                                        padding='max_length',
                                                        max_length=self.tokenizer.model_max_length,
                                                        truncation=True,
                                                        return_tensors='pt')

        return {'source_ids': source_batch.input_ids, # tensor of shape (BS, MAX_SRC_LEN_IN_BATCH)
                'source_mask': source_batch.attention_mask, # tensor of shape (BS, MAX_SRC_LEN_IN_BATCH)
                'target_ids': target_batch.input_ids, # tensor of shape (BS, MAX_TGT_LEN_IN_BATCH)
                'target_mask': target_batch.attention_mask, # tensor of shape (BS, MAX_TGT_LEN_IN_BATCH
               } 

def get_dataloader(dataset, batch_generator, batch_size=4, shuffle=True):
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             collate_fn=batch_generator,
                             num_workers=4)
    return data_loader

                        

if __name__ =="__main__":
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    model_name = 'KETI-AIR/ke-t5-base'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    dataset = QGDataset("/home/choyh/0.RPAi/3.MRC/0.data/MRC_Text_train_data_300K_2021-08-09.json")
    datagenerator = QGBatchGenerator(tokenizer)
    data_loader = get_dataloader(dataset,datagenerator)
