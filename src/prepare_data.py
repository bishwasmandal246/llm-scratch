import torch
from torch.utils.data import Dataset, DataLoader

class TinyStoriesDataset(Dataset):
    def __init__(self, tokenized_text_file, context_length, stride):
        self.input_ids = []
        self.target_ids = []

        with open(tokenized_text_file, 'r') as file:
            token_ids = [int(token_id) for token_id in file.read().split()]
        # token_ids = token_ids[:1000000] # for debugging
        
        for i in range(0, len(token_ids) - context_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i : i+context_length]))
            self.target_ids.append(torch.tensor(token_ids[i+1: i+context_length+1]))
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]



def create_dataloader(tokenized_text_file_train, 
                      tokenized_text_file_validation,
                      batch_size=4, 
                      context_length=256, 
                      stride=128, 
                      shuffle=True, 
                      drop_last=True, 
                      num_workers=0):

    train_dataset = TinyStoriesDataset(tokenized_text_file=tokenized_text_file_train, 
                                       context_length=context_length, 
                                       stride=stride)
    
    validation_dataset = TinyStoriesDataset(tokenized_text_file=tokenized_text_file_validation, 
                                            context_length=context_length, 
                                            stride=stride)

    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=shuffle, 
                                  drop_last=drop_last, 
                                  num_workers=num_workers)
    
    validation_dataloader = DataLoader(dataset=validation_dataset, 
                                       batch_size=batch_size, 
                                       shuffle=False, 
                                       drop_last=drop_last, 
                                       num_workers=num_workers)
    
    return train_dataloader, validation_dataloader

