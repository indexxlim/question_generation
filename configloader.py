import yaml
import json

from easydict import EasyDict

TRAIN_CONFIG_FILE = "./configurations/train.yml"
SERVER_CONFIG_FILE = "./configurations/server.yml"


train_config = EasyDict(yaml.load(open(TRAIN_CONFIG_FILE).read(), Loader=yaml.Loader))
server_config = EasyDict(yaml.load(open(SERVER_CONFIG_FILE).read(), Loader=yaml.Loader))