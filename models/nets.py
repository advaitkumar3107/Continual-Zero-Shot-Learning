import os
from old_16_frames import VideoDataset
from torch.utils.data import DataLoader
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from torchvision.models import resnet152, resnet50

##############################
#         Encoder
##############################


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        resnet = resnet152(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.final = nn.Sequential(
            nn.Linear(resnet.fc.in_features, latent_dim), nn.BatchNorm1d(latent_dim, momentum=0.01)
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.final(x)

##############################
#           LSTM
##############################


class LSTM(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, bidirectional):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        x, self.hidden_state = self.lstm(x, self.hidden_state)
        return x


##############################
#      Attention Module
##############################


class Attention(nn.Module):
    def __init__(self, latent_dim, hidden_dim, attention_dim):
        super(Attention, self).__init__()
        self.latent_attention = nn.Linear(latent_dim, attention_dim)
        self.hidden_attention = nn.Linear(hidden_dim, attention_dim)
        self.joint_attention = nn.Linear(attention_dim, 1)

    def forward(self, latent_repr, hidden_repr):
        if hidden_repr is None:
            hidden_repr = [
                Variable(
                    torch.zeros(latent_repr.size(0), 1, self.hidden_attention.in_features), requires_grad=False
                ).float()
            ]
        h_t = hidden_repr[0]
        latent_att = self.latent_attention(latent_att)
        hidden_att = self.hidden_attention(h_t)
        joint_att = self.joint_attention(F.relu(latent_att + hidden_att)).squeeze(-1)
        attention_w = F.softmax(joint_att, dim=-1)
        return attention_w


##############################
#         ConvLSTM
##############################
class ConvLSTM(nn.Module):
    def __init__(
        self, latent_dim=512, lstm_layers=1, hidden_dim=1024, bidirectional=True, attention=True
    ):
        super(ConvLSTM, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.lstm = LSTM(latent_dim, lstm_layers, hidden_dim, bidirectional)
        self.process_layers = nn.Sequential(
            nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            # nn.Softmax(dim=-1),
        )

        self.attention = attention
        self.attention_layer = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, 1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        self.lstm.reset_hidden_state()
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.encoder(x)
        x = x.view(batch_size, seq_length, -1)
        x = self.lstm(x)
        if self.attention:
            attention_w = F.softmax(self.attention_layer(x).squeeze(-1), dim=-1)
            x = torch.sum(attention_w.unsqueeze(-1) * x, dim=1)
        else:
            x = x[:, -1]
        return self.process_layers(x)

##############################
#  Generator and Discriminator
##############################

class Generator(nn.Module):
    def __init__(self,semantic_dim,noise_dim):
        super(Generator, self).__init__()

        # self.label_emb = nn.Embedding(final_total_class, final_total_class)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            return layers

        self.model = nn.Sequential(
            *block(semantic_dim + noise_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 8192),
            nn.Tanh()
        )

    def forward(self, semantic, noise):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((semantic, noise), -1)
        feature = self.model(gen_input)
        means, stds = feature.mean(dim = 1, keepdim = True), feature.std(dim = 1, keepdim = True)
        feature = (feature - means)/stds
        return feature

class Modified_Generator(nn.Module):
    def __init__(self,semantic_dim,noise_dim,latent_dim = 2048):
        super(Modified_Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            return layers

        self.model = nn.Sequential(
            *block(semantic_dim + noise_dim, 128, normalize=False),
            *block(128, 128),
            *block(128, 256),
            *block(256, 256),
            *block(256, 512),
            *block(512, 512),
            *block(512, 1024),
            nn.Linear(1024, 8192),
            nn.Tanh()
        )
         
    def forward(self, semantic, noise):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((semantic, noise), -1)
        feature = self.model(gen_input)
        means, stds = feature.mean(dim = 1, keepdim = True), feature.std(dim = 1, keepdim = True)
        feature = (feature - means)/stds
        return feature


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        # self.label_embedding = nn.Embedding(final_total_class, final_total_class)

        self.model = nn.Sequential(
            # nn.Linear(final_total_class + 8192, 512),
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # Concatenate label embedding and image to produce input
        d_in = img.view(img.size(0), -1)
        # d_in = img.view(img.size(0), -1)
        validity = self.model(d_in)
        return validity

##############################
#  Final classifier
##############################
class Classifier(nn.Module):
    def __init__(self, num_classes, bias = True):
        super(Classifier, self).__init__()
        self.extractor = nn.Sequential(nn.Linear(8192, 1024),
            nn.ReLU(), 
            nn.BatchNorm1d(1024, momentum=0.01), nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512, momentum = 0.01))
     
        self.classifier_out = nn.Linear(512, num_classes, bias = bias) 

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.extractor(x)
        x = self.classifier_out(x)
        return (x)

class Modified_Classifier(nn.Module):
    def __init__(self, num_classes, bias = True):
        super(Modified_Classifier, self).__init__()
        self.extractor = nn.Sequential(nn.Linear(8192, 1024),
            nn.ReLU(), 
            nn.BatchNorm1d(1024, momentum=0.01), nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512, momentum = 0.01))
     
        self.classifier_out = nn.Linear(512, num_classes, bias = bias) 

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.extractor(x)
        return (x)

################################################################
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    train_dataloader = DataLoader(VideoDataset(dataset='ucf101', split='train',clip_len=16), batch_size=20, shuffle=True, num_workers=4)
    model = ConvLSTM(
        num_classes=101,
        latent_dim=512,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
    )
    model.to('cuda')
    classifier_model = Classifier(10)
    classifier_model.to('cuda')

    for inputs, labels in (train_dataloader):
        inputs = inputs.permute(0,2,1,3,4)
        image_sequences = Variable(inputs.to("cuda"), requires_grad=True)        
        print("image_sequences.shape:",image_sequences.shape)
        out = model(image_sequences)
        print("out.shape:",out.shape)
        cls_out = classifier_model(out)
        print("cls_out.shape:",cls_out.shape)
        
        pdb.set_trace()
   