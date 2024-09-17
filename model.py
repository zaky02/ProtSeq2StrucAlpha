import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Architecture following: Attention is all you need (original transformers paper)

We will implement transformer encoder-decoder architecture with a multi 
head-attention within the encoder and the decoder and a cross attention between 
them.

The encoder will receive the input embedding together with the positional
encoding in order to maintain position information and will perform a
forward pass to the self-attention layer in order to learn dependencies 
between the sequence elements.

The decoder will receive the output which will also go through an embedding
and positional transformation. For the decoder there will be two multi 
head-attention layers, one for the decoder only and one to maintain relevant
information from the encoder to the decoder, this way we don't lose context.

The output layer will be a softmax layer in order to transform the predicted
output as a probability which will serve to calculate the loss and hence to
carry out the backpropagation of the model.
'''


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        
        super(PositionalEncoding, self).__init__()
        
        self.encoding = torch.zeros(max_len, d_model)
        
        # Positional encoding tensor from 0 to the max length of the sequence in 1D
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # It captures different dimensions and different patters accordingly
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        # Applies sin function to all even indices
        self.encoding[:, 0::2] = torch.sin(pos * div_term)
        # Applies cos function to all odd indices
        self.encoding[:, 1::2] = torch.cos(pos * div_term)
        
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        x = x + self.encoding[:, :x.size(1), :].to(x.device)
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, d_model=512, nhead=8,
                 num_layers=6, dim_feedforward=2048, dropout=0.1):
        
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead,
                                                   dim_feedforward, dropout)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, encoder_input, encoder_mask=None, encoder_key_padding_mask=None):
        # Embedding and Positional Encoding
        encoder_emb = self.embedding(encoder_input) * torch.sqrt(torch.tensor(self.embedding.embedding_dim, dtype=torch.float32)).to(encoder_input.device)
        encoder_emb = self.pos_encoder(encoder_emb)

        # Encoder forward pass
        memory = self.encoder(encoder_emb.transpose(0, 1),
                              src_mask=encoder_mask,
                              src_key_padding_mask=encoder_key_padding_mask)
        return memory


class Decoder(nn.Module):
    def __init__(self, output_dim, d_model=512, nhead=8,
                 num_layers=6, dim_feedforward=2048, dropout=0.1):
        
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(output_dim, d_model)
        self.pos_decoder = PositionalEncoding(d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model,
                                                   nhead,
                                                   dim_feedforward,
                                                   dropout)

        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers)
        self.fc_out = nn.Linear(d_model,
                                output_dim)
    
    def forward(self, decoder_input, memory, decoder_mask=None, memory_mask=None,
                decoder_key_padding_mask=None, memory_key_padding_mask=None):
        
        # Embedding and Positional Encoding
        decoder_emb = self.embedding(decoder_input) * torch.sqrt(torch.tensor(self.embedding.embedding_dim, dtype=torch.float32)).to(decoder_input.device)
        decoder_emb = self.pos_decoder(decoder_emb)

        # Decoder forward pass
        output = self.decoder(decoder_emb.transpose(0, 1),
                              memory, decoder_mask=decoder_mask,
                              tgt_mask=memory_mask,
                              tgt_key_padding_mask=decoder_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        
        # Linear output
        output = self.fc_out(output.transpose(0, 1))
        return output


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        
        super(TransformerModel, self).__init__()

        self.encoder = Encoder(input_dim, d_model, nhead,
                               num_encoder_layers, dim_feedforward, dropout)

        self.decoder = Decoder(output_dim, d_model, nhead,
                               num_decoder_layers, dim_feedforward, dropout)
    
    def forward(self, encoder_input, decoder_input,
                encoder_mask=None, decoder_mask=None,
                memory_mask=None, encoder_key_padding_mask=None,
                decoder_key_padding_mask=None):
        
        # Encoder pass
        memory = self.encoder(encoder_input, encoder_mask=encoder_mask,
                              encoder_key_padding_mask=encoder_key_padding_mask)
        
        # Decoder pass
        output = self.decoder(decoder, memory, decoder_mask=decoder_mask,
                              memory_mask=memory_mask,
                              decoder_key_padding_mask=decoder_key_padding_mask)
        return output

