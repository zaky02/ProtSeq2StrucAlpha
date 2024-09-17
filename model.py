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
    def __init__(self, dim_model, max_len):
        
        super(PositionalEncoding, self).__init__()
       
        # max_len + 2 to account for aa_max_len + cls + eos
        self.pe = torch.zeros(max_len + 2, dim_model)
        
        # Positional encoding tensor from 0 to the max length of the sequence in 1D
        pos = torch.arange(0, max_len + 2, dtype=torch.float).unsqueeze(1)
        # It captures different dimensions and different patters accordingly
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim_model))
        
        # Applies sin function to all even indices
        self.pe[:, 0::2] = torch.sin(pos * div_term)
        # Applies cos function to all odd indices
        self.pe[:, 1::2] = torch.cos(pos * div_term)
        
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, max_len, dim_model,
                 num_heads, num_layers, ff_hidden_layer, dropout):
        
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(input_dim, dim_model)
        self.pos_encoder = PositionalEncoding(dim_model, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(dim_model, num_heads,
                                                   dim_feedforward=ff_hidden_layer,
                                                   dropout=dropout, batch_first=False)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, encoder_input,
                encoder_mask=None,
                encoder_key_padding_mask=None):
        # Embedding () and Positional Encoding
        encoder_emb = self.embedding(encoder_input) * torch.sqrt(torch.tensor(self.embedding.embedding_dim, dtype=torch.float32)).to(encoder_input.device)
        encoder_emb = self.pos_encoder(encoder_emb)

        # Encoder forward pass
        memory = self.encoder(encoder_emb,
                              mask=encoder_mask,
                              src_key_padding_mask=encoder_key_padding_mask)
        return memory


class Decoder(nn.Module):
    def __init__(self, output_dim, max_len, dim_model, num_heads,
                 num_layers, ff_hidden_layer, dropout):
        
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(output_dim, dim_model)
        self.pos_decoder = PositionalEncoding(dim_model, max_len)

        decoder_layer = nn.TransformerDecoderLayer(dim_model,
                                                   num_heads,
                                                   dim_feedforward=ff_hidden_layer,
                                                   dropout=dropout, batch_first=False)

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(dim_model, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    
    def forward(self, decoder_input, memory, decoder_mask=None, memory_mask=None,
                decoder_key_padding_mask=None, memory_key_padding_mask=None):
        
        # Ensure tensor shapes are correct
        #print(f"decoder_input shape: {decoder_input.shape}")
        #print(f"memory shape: {memory.shape}")
        #print(f"decoder_mask shape: {decoder_mask.shape if decoder_mask is not None else None}")
        #print(f"memory_mask shape: {memory_mask.shape if memory_mask is not None else None}")
        #print(f"decoder_key_padding_mask shape: {decoder_key_padding_mask.shape if decoder_key_padding_mask is not None else None}")
        #print(f"memory_key_padding_mask shape: {memory_key_padding_mask.shape if memory_key_padding_mask is not None else None}")
        
        # Embedding and Positional Encoding
        decoder_emb = self.embedding(decoder_input) * torch.sqrt(torch.tensor(self.embedding.embedding_dim, dtype=torch.float32)).to(decoder_input.device)
        decoder_emb = self.pos_decoder(decoder_emb)

        # Decoder forward pass
        output = self.decoder(decoder_emb,
                              memory,
                              tgt_mask=decoder_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=decoder_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        
        # Linear output
        output = self.fc_out(output)
        output = self.softmax(output)
        return output


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, max_len, dim_model, num_heads,
                 num_layers, ff_hidden_layers, dropout, verbose=False):
        
        super(TransformerModel, self).__init__()

        self.encoder = Encoder(input_dim, max_len, dim_model, num_heads,
                               num_layers, ff_hidden_layer, dropout)

        self.decoder = Decoder(output_dim, max_len, dim_model, num_heads,
                               num_layers, ff_hidden_layer, dropout)
    
    def forward(self, encoder_input, decoder_input,
                encoder_mask=None, decoder_mask=None,
                memory_mask=None, encoder_key_padding_mask=None,
                decoder_key_padding_mask=None):
        
        # Encoder pass
        memory = self.encoder(encoder_input,
                              encoder_mask=encoder_mask,
                              encoder_key_padding_mask=encoder_key_padding_mask)
        
        # Decoder pass
        output = self.decoder(decoder_input,
                              memory,
                              decoder_mask=decoder_mask,
                              memory_mask=memory_mask,
                              decoder_key_padding_mask=decoder_key_padding_mask,
                              memory_key_padding_mask=encoder_key_padding_mask)
        return output

