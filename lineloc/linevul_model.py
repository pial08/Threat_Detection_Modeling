import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification

import torchvision

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)
        

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaLocalizationHead(nn.Module):
    """Head for function-level localization tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)
        

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        # Adding Range
        x = torch.sigmoid(x) * 512
        return x
      

class Model(RobertaForSequenceClassification):   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__(config=config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.localizer = RobertaLocalizationHead(config)
        self.args = args
    
        
    def forward(self, input_embed=None, labels=None, lines=None, output_attentions=False, input_ids=None):
        #print("You are here ...")
        if output_attentions:
            
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            logits = self.classifier(last_hidden_state)
            localizerLogits = self.localizer(last_hidden_state)


            prob = torch.softmax(logits, dim=-1)

            if labels is not None:
                loss_cel = CrossEntropyLoss()
                loss_mse = nn.MSELoss()
                #print("Loss shapes ... ", loss_cel.shape, loss.mse.shape)
                loss = loss_cel(logits, labels) + loss_mse(logits, labels)
                return loss, prob, attentions
            else:
                return prob, attentions
        else:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)[0]
            logits = self.classifier(outputs)

            localizerLogits = self.localizer(outputs)
            #print("Localization Logits ...", localizerLogits, lines)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_cel = CrossEntropyLoss()
                loss_mse = nn.MSELoss()
                #print("Loss shapes ... ", logits.shape, localizerLogits.shape, lines.shape)
                #print("TYPE ",type(loss_cel(logits, labels)), type(loss_mse(localizerLogits, lines)))
                #loss = loss_cel(logits, labels) + loss_mse(localizerLogits, lines.float()) * 3.0
                loss = loss_mse(localizerLogits, lines.float())

                #loss_2 = torchvision.ops.sigmoid_focal_loss(localizerLogits, lines.float(), alpha=-1, gamma=5.0, reduction="mean")
                #loss = loss_1 + loss_2
                #print("Calculating Focal loss ...")
                return loss, prob, localizerLogits
            else:
                return prob