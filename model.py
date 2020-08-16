from transformers import BartForSequenceClassification
from transformers import MBartConfig

class MBartForSequenceClassification(BartForSequenceClassification):
    config_class = MBartConfig
