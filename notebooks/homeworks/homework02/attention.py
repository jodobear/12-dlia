import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import torch.nn.functional as F

import random
import math
import time

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = hid_dim
        self.dec_hid_dim = hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim
        )

        self.rnn = nn.GRU(
            imput_size=emb_dim,
            hidden_size=hid_dim,
            bidirectional=True
        )

        self.fc = nn.Linear(
            in_features=hid_dim * 2,
            out_features=hid_dim
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src):

        #src shape = [src sent len, batch size]
        # Compute an embedding from the src data and apply dropout to it
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        #embedded shape = [src sent len, batch size, emb dim]

        outputs, hidden = self.rnn(embedded)
        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        #outputs are always from the top hidden layer

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hid_dim, attn_dim):
        super().__init__()

        self.enc_hid_dim = hid_dim
        self.dec_hid_dim = hid_dim
        self.attn_in = (hid_dim * 2) + hid_dim

        self.attn = nn.Linear(
            in_features=self.attn_in,
            out_features=attn_dim
        )

    def forward(self, decoder_hidden, encoder_outputs) :

        src_len = encoder_outputs.shape[0]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)))

        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = hid_dim
        self.dec_hid_dim = hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(
            num_embeddings=output_dim,  # input_dim == output_dim
            embedding_dim=emb_dim
        )

        self.rnn = nn.GRU(
            imput_size=(hid_dim * 2) + emb_dim,
            hidden_size=hid_dim
        )

        self.out = nn.Linear(
            in_features=self.attention.attn_in + emb_dim,
            out_features=output_dim
        )

        self.dropout = nn.Dropout(dropout)


    def _weighted_encoder_rep(self, decoder_hidden, encoder_outputs):

        a = self.attention(decoder_hidden, encoder_outputs).unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs).permute(1, 0, 2)

        return weighted_encoder_rep


    def forward(self, input, decoder_hidden, encoder_outputs):

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                            encoder_outputs)

        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)

        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(torch.cat((output, weighted_encoder_rep, embedded), dim=1))

        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5) :

        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        # Again, now batch is the first dimention instead of zero
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        output = trg[0,:]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs
