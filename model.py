import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from data_module import sort_batch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import Encoder, EncoderTransformer, make_src_mask, XMU_MultiHeadAttention, LayerNorm, XMU_masking_bias, GRUAtt
import math

class Model_new(nn.Module):

	def __init__(self, params):
		super().__init__()

		self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)

		self.encoder = Encoder(in_channels=params.embedding_dim,
								out_channels=params.embedding_dim,
								kernel_size=3,
								padding="same",
								n_layers=3)
		# self.encoder2 = EncoderConv(in_channels=params.embedding_dim,
		# 						out_channels=params.embedding_dim,
		# 						kernel_size=20,
		# 						padding="same",
		# 						n_layers=1)
		# self.encoder = EncoderTransformer(d_model=params.embedding_dim,
		# 						n_head=4,
		# 						ffn_hidden=512,
		# 						drop_prob=0.1,
		# 						n_layers=6,
		# 						)

		self.biGRU = nn.LSTM(params.embedding_dim, params.hidden_size1, 2, bidirectional=True, batch_first=True)
		self.GRU = nn.LSTM(params.hidden_size1*2, params.hidden_size2, 1, batch_first=True)

		# self.attention1 = XMU_MultiHeadAttention(params.hidden_size2, 8, 0.2)
		# self.layer_norm1 = LayerNorm(params.hidden_size2)
		# self.dropout = nn.Dropout(0.2)
		# self.GRUAtt = GRUAtt(params.hidden_size2, n_layers=3)

		# self.case_fc = nn.Sequential(
		# 					nn.Dropout(params.dropout),
		# 					nn.Linear(params.hidden_size2*2, 1536),
		# 					nn.Dropout(params.dropout),
		# 					nn.ReLU(),
		# 					)
		# self.punct_fc = nn.Sequential(
		# 					nn.Dropout(params.dropout),
		# 					nn.Linear(params.hidden_size2*2, 1536),
		# 					nn.Dropout(params.dropout),
		# 					nn.ReLU(),
		# 					)
		# self.decoder_case = nn.Linear(1536, params.out_size_case)
		# self.decoder_punct = nn.Linear(1536, params.out_size_punct)



		############ Linear Attention
		# self.attention_case = nn.Linear(params.hidden_size2, params.hidden_size2)
		# self.attention_punct = nn.Linear(params.hidden_size2, params.hidden_size2)


		########### ScaledDotProduct Attention
		# self.w_q_case = nn.Linear(params.hidden_size2, params.hidden_size2)
		# self.w_k_case = nn.Linear(params.hidden_size2, params.hidden_size2)
		# self.w_v_case = nn.Linear(params.hidden_size2, params.hidden_size2)
		# self.w_q_punct = nn.Linear(params.hidden_size2, params.hidden_size2)
		# self.w_k_punct = nn.Linear(params.hidden_size2, params.hidden_size2)
		# self.w_v_punct = nn.Linear(params.hidden_size2, params.hidden_size2)
		# self.softmax = nn.Softmax(dim=-1)

		# self.grn_case = nn.Sequential(
		# 					nn.Linear(params.hidden_size2*2, 1536),
		# 					nn.GELU(),
		# 					nn.Linear(1536, params.hidden_size2*2),
		# 					nn.Dropout(params.dropout),
		# 				)
		# self.layer_norm_case = nn.LayerNorm(params.hidden_size2*2)
		# self.gate_case = nn.Linear(params.hidden_size2*2, params.hidden_size2*2)

		# self.grn_punct = nn.Sequential(
		# 					nn.Linear(params.hidden_size2*2, 1536),
		# 					nn.GELU(),
		# 					nn.Linear(1536, params.hidden_size2*2),
		# 					nn.Dropout(params.dropout),
		# 				)
		# self.layer_norm_punct = nn.LayerNorm(params.hidden_size2*2)
		# self.gate_punct = nn.Linear(params.hidden_size2*2, params.hidden_size2*2)

		# self.decoder_case = nn.Linear(params.hidden_size2, params.out_size_case)
		# self.decoder_punct = nn.Linear(params.hidden_size2, params.out_size_punct)



		self.decoder_case = nn.Linear(params.hidden_size2*2, params.out_size_case)
		self.decoder_punct = nn.Linear(params.hidden_size2*2, params.out_size_punct)
		self.dropout1 = nn.Dropout(params.dropout)
		self.dropout2 = nn.Dropout(params.dropout)

		self.config = params

	def forward(
		self, 
		input_token_ids,
		valid_ids=None,
		label_lens=None,
		label_masks=None,
		labels=None,
	):
		src_mask = make_src_mask(input_token_ids)

		embedding_out = self.embedding(input_token_ids) # [batch_size, max_seq_length, embedding_dim]

		embedding_out = self.encoder(embedding_out)
		# embedding_out = self.encoder(embedding_out, src_mask)

		batch_size, max_seq_length, embedding_dim = embedding_out.shape
		# Placeholder for the output with the same shape as `embedding_out`
		valid_output = torch.zeros_like(embedding_out)

		# Create a mask for valid positions
		valid_mask = valid_ids.to(torch.bool)

		# Flatten the mask and the embedding output to map valid positions
		flat_valid_mask = valid_mask.view(-1)
		flat_embedding_out = embedding_out.view(-1, embedding_dim)

		# Filter out the valid embeddings
		valid_embeddings = flat_embedding_out[flat_valid_mask]

		# We need a cumulative sum of the valid_ids to determine the correct indices in valid_output
		cumulative_valid_counts = valid_ids.cumsum(dim=1) - 1

		# Flatten cumulative_valid_counts to use it for indexing
		flat_cumulative_valid_counts = cumulative_valid_counts.view(-1)

		# Use the cumulative sum as indices to place valid_embeddings in the valid_output
		# We also need a range for each example in the batch
		batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, max_seq_length).reshape(-1)
		batch_indices = batch_indices.to(flat_valid_mask.device)
		batch_indices = batch_indices[flat_valid_mask]  # Select only indices for valid embeddings

		# Now we place the valid embeddings into the correct positions in valid_output
		valid_output[batch_indices, flat_cumulative_valid_counts[flat_valid_mask]] = valid_embeddings

		label_lens, indx = label_lens.sort(dim=0, descending=True)
		valid_output = valid_output[indx]

		# valid_output = valid_output.permute(1, 0, 2)

		embedding_out = pack_padded_sequence(valid_output, lengths=label_lens.cpu(), batch_first=True) # unpad
		biGRU_out, _ = self.biGRU(embedding_out) # [batch_size, max_seq_length, 2*hidden_size1]
		biGRU_out, label_lens = pad_packed_sequence(biGRU_out, batch_first=True) # pad sequence to max length in batch
		GRU_out, _ = self.GRU(biGRU_out) # [batch_size, max_label_lens_in_this_batch, hidden_size2]

		# GRU_out = GRU_out.permute(1, 0, 2)

		############# GRU + attention
		# x1 = self.dropout(GRU_out)
		# y1 = self.attention1(x1)
		# y1 = self.dropout(y1)
		# y1 = self.layer_norm1(x1 + y1)
		# GRU_out = self.GRUAtt(y1)


		# for case prediction, concat with previous token
		# Pad the tensor at the beginning of the T dimension to duplicate the first vector
		padded_tensor_case = torch.nn.functional.pad(GRU_out, (0, 0, 1, 0), mode='replicate')
		# Concatenate the original tensor with the padded tensor, which includes the duplicated first vector
		concat_adjacent_case = torch.cat((padded_tensor_case[:, :-1, :], GRU_out), dim=-1)

		# for punctuation prediction, concat with next token
		# Pad the tensor at the end of the T dimension to duplicate the last vector
		padded_tensor_punct = torch.nn.functional.pad(GRU_out, (0, 0, 0, 1), mode='replicate')
		# Concatenate the original tensor with the padded tensor, which includes the duplicated last vector
		concat_adjacent_punct = torch.cat((GRU_out, padded_tensor_punct[:, 1:, :]), dim=-1)


		# ################# concat with previous and next token
		# prev_elements = torch.cat((GRU_out[:, :1, :], GRU_out[:, :-1, :]), dim=1)
		# next_elements = torch.cat((GRU_out[:, 1:, :], GRU_out[:, -1:, :]), dim=1)
		# concat_previous_next = torch.cat((prev_elements, GRU_out, next_elements), dim=-1)


		#################### Linear Attention
		# att_case = torch.tanh(self.attention_case(GRU_out)) # (N, T, C)
		# attention_case = torch.matmul(att_case, att_case.transpose(-2, -1)) # (N, T, T)
		# concat_adjacent_case = torch.bmm(attention_case, GRU_out) # (N, T, C)

		# att_punct = torch.tanh(self.attention_punct(GRU_out))
		# attention_punct = torch.matmul(att_punct, att_punct.transpose(-2, -1))
		# concat_adjacent_punct = torch.bmm(attention_punct, GRU_out)


		################### ScaledDotProduct Attention
		## use attention will predict "how are you" as "How? are? you?", seems not suitable for this model
		# N, T, C = GRU_out.size()
		# q_case, k_case, v_case = self.w_q_case(GRU_out), self.w_k_case(GRU_out), self.w_v_case(GRU_out)
		# q_punct, k_punct, v_punct = self.w_q_punct(GRU_out), self.w_k_punct(GRU_out), self.w_v_punct(GRU_out)

		# k_t_case = k_case.transpose(1, 2)
		# score_case = (q_case @ k_t_case) / math.sqrt(C)
		# score_case = self.softmax(score_case)
		# concat_adjacent_case = score_case @ v_case

		# k_t_punct = k_punct.transpose(1, 2)
		# score_punct = (q_punct @ k_t_punct) / math.sqrt(C)
		# score_punct = self.softmax(score_punct)
		# concat_adjacent_punct = score_punct @ v_punct


		# hidden_case = self.grn_case(concat_adjacent_case)
		# gate_case = torch.sigmoid(self.gate_case(concat_adjacent_case))
		# concat_adjacent_case = self.layer_norm_case((1-gate_case)*concat_adjacent_case + gate_case*hidden_case)

		# hidden_punct = self.grn_punct(concat_adjacent_punct)
		# gate_punct = torch.sigmoid(self.gate_punct(concat_adjacent_punct))
		# concat_adjacent_punct = self.layer_norm_punct((1-gate_punct)*concat_adjacent_punct + gate_punct*hidden_punct)


		case_logits = self.decoder_case(self.dropout1(concat_adjacent_case)) # [batch_size, max_label_lens_in_this_batch, 6]
		punct_logits = self.decoder_punct(self.dropout2(concat_adjacent_punct))
		# case_logits = self.decoder_case(self.dropout1(concat_previous_next)) # [batch_size, max_label_lens_in_this_batch, 6]
		# punct_logits = self.decoder_punct(self.dropout2(concat_previous_next))

		if labels is not None:
			labels = labels[indx]
			label_masks = label_masks[indx]

			loss_fn_case = nn.CrossEntropyLoss()
			loss_fn_punct = nn.CrossEntropyLoss()
			# label_masks shape: [batch_size, max_seq_length]
			label_masks = label_masks[:, :case_logits.shape[1]]
			active_ones = label_masks.reshape(-1) == 1

			labels = labels[:, :, :case_logits.shape[1]] # labels shape:[batch_size, 2, max_seq_len]
			active_case_labels = labels[:, 0, :].reshape(-1)[active_ones] 
			active_punct_labels = labels[:, 1, :].reshape(-1)[active_ones]

			active_case_logits = case_logits.view(-1, self.config.out_size_case)[active_ones]
			active_punct_logits = punct_logits.view(-1, self.config.out_size_punct)[active_ones]

			case_loss = loss_fn_case(active_case_logits, active_case_labels)
			punct_loss = loss_fn_punct(active_punct_logits, active_punct_labels)

			return case_loss, punct_loss
		else:
			valid_ids = valid_ids[indx]
			# move all ones to the left
			valid_ids_sorted, _ = valid_ids.sort(dim=1, descending=True)
			valid_ids_sorted_sliced = valid_ids_sorted[:, :case_logits.shape[1]]

			non_zero_mask = valid_ids_sorted_sliced != 0

			# exclude the first token, i.e. <bos>
			cumulative_non_zeros = valid_ids_sorted_sliced.cumsum(dim=1)
			exclude_first = cumulative_non_zeros != 1

			# exclude the last token, i.e. <eos>
			cumulative_non_zeros_flip = valid_ids_sorted_sliced.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
			exclude_last = cumulative_non_zeros_flip != 1

			final_mask = non_zero_mask & exclude_first & exclude_last

			active_case_logits = case_logits[final_mask] # (T', out_size_case)
			active_punct_logits = punct_logits[final_mask] # (T', out_size_punct)
			
			# torch.argmax should be handled in C++ code, it does not work if exported in ONNX
			# case_pred = torch.argmax(active_case_logits, dim=1)
			# punct_pred = torch.argmax(active_punct_logits, dim=1)	

			return active_case_logits, active_punct_logits, final_mask


