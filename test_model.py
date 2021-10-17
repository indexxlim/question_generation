"""
	simple Test Code
"""

import datatime
import os
import logging

from configloader import train_config
import transformers
from dataloader import QGDataset, QGBatchGenerator, get_dataloader
from train import train


