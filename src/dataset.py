import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

class IMDBDataset(Dataset):
    """
    A PyTorch Dataset class for loading and tokenizing the IMDB dataset using the Hugging Face Transformers library.

    Attributes:
        dataset (datasets.Dataset): The loaded IMDB dataset.
        reviews (List[str]): List of review texts.
        classes (List[int]): Corresponding sentiment labels (0 = negative, 1 = positive).
        tokenizer (AutoTokenizer): Pre-trained BERT tokenizer for text processing.
    """
    
    def __init__(self, split: str = "train"):
        """
        Initializes the IMDBDataset by loading the dataset and setting up the tokenizer.

        Args:
            split (str): The dataset split to load ("train" or "test"). Default is "train".
        """
        super(IMDBDataset, self).__init__()
        self.dataset = load_dataset("imdb", split=split)
        self.reviews = self.dataset['text']
        self.classes = self.dataset['label']
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Load tokenizer once

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves and tokenizes a single review from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'review' (torch.Tensor): Tokenized review input IDs.
                - 'attention_mask' (torch.Tensor): Attention mask for padding.
                - 'label' (int): The sentiment label (0 for negative, 1 for positive).
        """
        tokens = self.tokenizer(
            self.reviews[idx], 
            padding="max_length", 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )

        return {
            'review': tokens["input_ids"].squeeze(0),  # Remove batch dim (1,512) → (512)
            'attention_mask': tokens["attention_mask"].squeeze(0),
            'label': self.classes[idx]
        }
