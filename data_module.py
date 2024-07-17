import argparse
import os
import sentencepiece
import torch
from torch.utils.data import (Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np

class InputFeatures(object):

	def __init__(self, token_ids = None, label_ids = None, valid_ids = None, token_masks = None, label_masks = None, label_len = None):
		self.token_ids = token_ids
		self.label_ids = label_ids
		self.valid_ids = valid_ids
		self.token_masks = token_masks
		self.label_masks = label_masks
		self.label_len = label_len

class TextDataset(Dataset):

	def __init__(self, text_path, label_path):
		self.text_path = text_path
		self.label_path = label_path
		self.text_lines = []
		self.label_lines = [[], []]

	def __len__(self):
		return len(self.features)

	def __getitem__(self, idx):
		if isinstance(idx, slice):
			print(f"idx:{idx}, idx.start:{idx.start}, idx.stop:{idx.stop}, idx.step:{idx.step}")
			print(self.features[idx])
			for i in range(*idx.indices(len(self))):
				print(i)
			return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]
		else:
			token_ids = self.features[idx].token_ids
			label_ids = self.features[idx].label_ids
			valid_ids = self.features[idx].valid_ids
			label_len = self.features[idx].label_len
			label_masks = self.features[idx].label_masks
			# print(f"getitem, token_ids:{token_ids}")
			# print(f"getitem {idx}")
			return np.array(token_ids), np.array(label_ids), np.array(valid_ids), label_len, np.array(label_masks)

	def readLines(self):
		with open(self.text_path, "r") as hand:
			for line in hand:
				self.text_lines.append(line)

		print("Reading examples...")
		with open(self.label_path, "r") as hand:
			lines = hand.readlines()
			for i, line in enumerate(tqdm(lines)):

				# covert string line "1 2 3 4 5" to int list [1, 2, 3, 4, 5]
				numbers = line.split()
				label_list = [int(n) for n in numbers]

				if i % 2 == 0:
					self.label_lines[0].append(label_list) # case labels
				else:
					self.label_lines[1].append(label_list) # punctuation labels

		assert(len(self.text_lines) == len(self.label_lines[0]) and len(self.text_lines) == len(self.label_lines[1]))

	def convert_examples_to_features(self, max_seq_length, tokenizer):

		self.readLines()

		# all examples:
		self.features = []
		self.max_seq_length = max_seq_length

		# one example:
		tokens = []
		labels = [[], []]
		valid = []
		token_masks = []
		label_masks = []
		label_len = 0

		print("Converting examples to features...")
		for il, line in enumerate(tqdm(self.text_lines)):
			words = line.split()
			for iw, word in enumerate(words):
				word_tokens = tokenizer.encode(word, out_type=int)

				if len(tokens) + len(word_tokens) > max_seq_length:
					token_masks = [1] * len(tokens)
					label_masks = [1] * len(labels[0])
					label_len = len(labels[0])

					while len(tokens) < max_seq_length:
						tokens.append(0)
						token_masks.append(0)
						valid.append(0)
					while len(labels[0]) < max_seq_length:
						labels[0].append(0)
						labels[1].append(0)
						label_masks.append(0)
					assert len(tokens) == max_seq_length
					assert len(token_masks) == max_seq_length
					assert len(valid) == max_seq_length
					assert len(labels[0]) == max_seq_length
					assert len(labels[1]) == max_seq_length
					assert len(label_masks) == max_seq_length

					self.features.append( InputFeatures(token_ids = tokens,
												   label_ids = labels,
												   valid_ids = valid,
												   token_masks = token_masks,
												   label_masks = label_masks,
												   label_len = label_len) )
					tokens = []
					labels = [[], []]
					valid = []
					token_masks = []
					label_masks = []
					label_len = 0

				tokens.extend(word_tokens)
				for m in range(len(word_tokens)):
					if m == 0:
						labels[0].append(self.label_lines[0][il][iw])
						labels[1].append(self.label_lines[1][il][iw])
						valid.append(1)
					else:
						valid.append(0)

	def getTokensNum(self, words, tokenizer):
		num = 0
		for word in words:
			word_tokens = tokenizer.encode(word, out_type=int)
			num += len(word_tokens)
		return num

	def convert_examples_to_features_bos_eos(self, max_seq_length, tokenizer):

		self.readLines()

		# all examples:
		self.features = []
		self.max_seq_length = max_seq_length

		# one example:
		tokens = [tokenizer.piece_to_id('<s>')]
		labels = [[0], [0]]
		valid = [1]
		token_masks = []
		label_masks = []
		label_len = 0

		self.last_part_sentence = InputFeatures()

		print(f"Converting examples to features... bos_id:{tokenizer.piece_to_id('<s>')}, eos_id:{tokenizer.piece_to_id('</s>')}")
		for il, line in enumerate(tqdm(self.text_lines)):
			words = line.split()

			tokens_num = self.getTokensNum(words, tokenizer)
			if tokens_num < max_seq_length - 10:
				for iw, word in enumerate(words):
					word_tokens = tokenizer.encode(word, out_type=int)

					if len(tokens) + len(word_tokens) > max_seq_length - 1:

						tokens.append(tokenizer.piece_to_id('</s>'))
						labels[0].append(0)
						labels[1].append(0)
						valid.append(1)

						token_masks = [1] * len(tokens)
						label_masks = [1] * len(labels[0])
						label_len = len(labels[0])

						while len(tokens) < max_seq_length:
							tokens.append(0)
							token_masks.append(0)
							valid.append(0)
						while len(labels[0]) < max_seq_length:
							labels[0].append(0)
							labels[1].append(0)
							label_masks.append(0)
						assert len(tokens) == max_seq_length
						assert len(token_masks) == max_seq_length
						assert len(valid) == max_seq_length
						assert len(labels[0]) == max_seq_length
						assert len(labels[1]) == max_seq_length
						assert len(label_masks) == max_seq_length

						self.features.append( InputFeatures(token_ids = tokens,
													   label_ids = labels,
													   valid_ids = valid,
													   token_masks = token_masks,
													   label_masks = label_masks,
													   label_len = label_len) )

						tokens = [tokenizer.piece_to_id('<s>')]
						labels = [[0], [0]]
						valid = [1]
						token_masks = []
						label_masks = []
						label_len = 0

						### iw = 0 means this word is the start of a new sentence, no need to insert last part sentence
						if iw > 0:
							tokens.extend(self.last_part_sentence.tokens)
							labels[0].extend(self.last_part_sentence.labels[0])
							labels[1].extend(self.last_part_sentence.labels[1])
							valid.extend(self.last_part_sentence.valid)
					
					if iw == 0:
						self.last_part_sentence.tokens = []
						self.last_part_sentence.labels = [[], []]
						self.last_part_sentence.valid = []

					tokens.extend(word_tokens)
					self.last_part_sentence.tokens.extend(word_tokens)
					for m in range(len(word_tokens)):
						if m == 0:
							labels[0].append(self.label_lines[0][il][iw])
							labels[1].append(self.label_lines[1][il][iw])
							valid.append(1)

							self.last_part_sentence.labels[0].append(self.label_lines[0][il][iw])
							self.last_part_sentence.labels[1].append(self.label_lines[1][il][iw])
							self.last_part_sentence.valid.append(1)
						else:
							valid.append(0)

							self.last_part_sentence.valid.append(0)
			else:
				print(f"tokens num:[{tokens_num}] ----> {line}")

	def save_features(self, filename):
		with open(filename, "w") as fp:
			for f in tqdm(self.features):
				for i in range(self.max_seq_length):
					fp.write(str(f.token_ids[i]) + " ")
				fp.write("\n")
				for i in range(self.max_seq_length):
					fp.write(str(f.label_ids[0][i]) + " ")
				fp.write("\n")
				for i in range(self.max_seq_length):
					fp.write(str(f.label_ids[1][i]) + " ")
				fp.write("\n")
				for i in range(self.max_seq_length):
					fp.write(str(f.valid_ids[i]) + " ")
				fp.write("\n")
				for i in range(self.max_seq_length):
					fp.write(str(f.token_masks[i]) + " ")
				fp.write("\n")
				for i in range(self.max_seq_length):
					fp.write(str(f.label_masks[i]) + " ")
				fp.write("\n")
				fp.write(str(f.label_len))
				fp.write("\n")

	def load_features(self, filename, max_seq_length):
		self.features = []
		with open(filename, "r") as fp:
			lines = fp.readlines()
			indx = 0
			tokens = []
			labels = [[], []]
			valid = []
			token_masks = []
			label_masks = []
			label_len = 0
			for i, line in enumerate(tqdm(lines)):
				numbers = line.split()
				n_list = [int(n) for n in numbers]	
				if indx == 0:
					tokens = n_list
				elif indx == 1:
					labels[0] = n_list
				elif indx == 2:
					labels[1] = n_list
				elif indx == 3:
					valid = n_list
				elif indx == 4:
					token_masks = n_list
				elif indx == 5:
					label_masks = n_list
				elif indx == 6:
					assert len(n_list) == 1
					label_len = n_list[0]

				# print(f"len(tokens):{len(tokens)}, len(token_masks):{len(token_masks)}")
				indx += 1
				if indx == 7:
					assert len(tokens) == max_seq_length
					assert len(token_masks) == max_seq_length
					assert len(valid) == max_seq_length
					assert len(labels[0]) == max_seq_length
					assert len(labels[1]) == max_seq_length
					assert len(label_masks) == max_seq_length
					assert (label_len > 0 & label_len <= 200)

					self.features.append( InputFeatures(token_ids = tokens,
												   	label_ids = labels,
												   	valid_ids = valid,
												   	token_masks = token_masks,
												   	label_masks = label_masks,
												   	label_len = label_len) )
					indx = 0

					tokens = []
					labels = [[], []]
					valid = []
					token_masks = []
					label_masks = []
					label_len = 0	

class DataModule(object):

	def __init__(self, args:argparse.Namespace, sp:sentencepiece):
		self.args = args
		self.sp = sp

		self.data_dir = self.args.data_dir
		train_text = f"{self.data_dir}/train_text.txt"
		train_label = f"{self.data_dir}/train_label.txt"
		valid_text = f"{self.data_dir}/valid_text.txt"
		valid_label = f"{self.data_dir}/valid_label.txt"
		self.test_text = f"{self.data_dir}/0_IWSLT2011_asr_test_text.txt"
		test_label = f"{self.data_dir}/0_IWSLT2011_asr_test_label.txt"

		# print("Reading train examples...")
		self.train_dataset = TextDataset(train_text, train_label)
		# print("Reading valid examples...")
		self.valid_dataset = TextDataset(valid_text, valid_label)
		### test set
		self.test_dataset = TextDataset(self.test_text, test_label)

		self.train_features_file = f"{self.data_dir}/train_features.txt"
		self.valid_features_file = f"{self.data_dir}/valid_features.txt"
		self.test_features_file = f"{self.data_dir}/test_features.txt"

	def train_dataloader(self) -> DataLoader:
	# 	print("Converting train examples to features...")
	# 	train_features = self.train_processor.convert_examples_to_features(self.args.max_seq_length, self.sp)
	# 	token_ids = [f.token_ids for f in train_features]
	# 	token_ids = torch.tensor(token_ids, dtype=torch.long)
	# 	label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.long)
	# 	valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
	# 	train_data = TensorDataset(token_ids, label_ids, valid_ids)

		if not os.path.isfile(self.train_features_file):
			print("Extracting train features:")
			self.train_dataset.convert_examples_to_features_bos_eos(self.args.max_seq_length, self.sp)

			print("First time to extract features, save features to local file for next time quick load...")
			self.train_dataset.save_features(self.train_features_file)
		else:
			print("Train feature file already exists, loading...")
			self.train_dataset.load_features(self.train_features_file, self.args.max_seq_length)
		# print(f"print first 2 example in train dataset:\n{self.train_dataset[:2]}")

		if self.args.world_size > 1:
			train_sampler = DistributedSampler(self.train_dataset)
			# shuffle = False
		else:
			train_sampler = RandomSampler(self.train_dataset)
			# shuffle = True
		train_dataloader = DataLoader(
							dataset=self.train_dataset, 
							sampler=train_sampler, 
							batch_size=self.args.batch_size, 
							# shuffle=shuffle,
						)

		return train_dataloader

	def valid_dataloader(self) -> DataLoader:
		# print(f"valid features len:{len(valid_features)}")
		# token_ids = [f.token_ids for f in valid_features]
		# print(f"token_ids len:{len(token_ids)}")
		# for i, t in enumerate(token_ids):
		# 	print(f"{i}: {len(t)}")
		# token_ids = torch.tensor(token_ids, dtype=torch.long)
		# # for f in valid_features:
		# # 	print(f"{f.label_ids.shape}")
		# label_ids = torch.tensor([f.label_ids for f in valid_features], dtype=torch.long)
		# valid_ids = torch.tensor([f.valid_ids for f in valid_features], dtype=torch.long)
		# valid_data = TensorDataset(token_ids, label_ids, valid_ids)
		if not os.path.isfile(self.valid_features_file):
			print("Extracting valid features:")
			self.valid_dataset.convert_examples_to_features_bos_eos(self.args.max_seq_length, self.sp)

			print("First time to extract features, save features to local file for next time quick load...")
			self.valid_dataset.save_features(self.valid_features_file)
		else:
			print("Valid feature file already exists, loading...")
			self.valid_dataset.load_features(self.valid_features_file, self.args.max_seq_length)
		# print(f"print first 2 example in valid dataset:\n{self.valid_dataset[:2]}")

		if self.args.world_size > 1:
			valid_sampler = DistributedSampler(self.valid_dataset)
		else:
			valid_sampler = RandomSampler(self.valid_dataset)
		valid_dataloader = DataLoader(
								dataset=self.valid_dataset, 
								sampler=valid_sampler, 
								batch_size=self.args.batch_size
							)

		return valid_dataloader

	def test_dataloader(self) -> DataLoader:
		if not os.path.isfile(self.test_features_file):
			print("Extracting test features:")
			self.test_dataset.convert_examples_to_features_bos_eos(self.args.max_seq_length, self.sp)

			print("First time to extract features, save features to local file for next time quick load...")
			self.test_dataset.save_features(self.test_features_file)
		else:
			print("Test feature file already exists, loading...")
			self.test_dataset.load_features(self.test_features_file, self.args.max_seq_length)
		# print(f"print first 2 example in test dataset:\n{self.test_dataset[:2]}")

		if self.args.world_size > 1:
			test_sampler = DistributedSampler(self.test_dataset)
		else:
			test_sampler = RandomSampler(self.test_dataset)
		test_dataloader = DataLoader(
								dataset=self.test_dataset, 
								sampler=test_sampler, 
								batch_size=self.args.batch_size
							)

		return test_dataloader, self.test_text

# def sort_batch(token_ids, label_ids, valid_ids, label_lens, label_masks):
# 	print(f"before, label_lens:{label_lens}")
# 	label_lens, indx = label_lens.sort(dim=0, descending=True)
# 	print(f"after, label_lens:{label_lens}")
# 	print(f"indx:{indx}")
# 	# print(f"token_ids:{token_ids}")
# 	# indx = indx.to(dtype=torch.long, device="cuda:0")
# 	# indx = [i.tolist() for i in indx]
# 	print(f"token_ids len:{len(token_ids)}")
# 	print(token_ids)
# 	print(f"before, token_ids[0] len:{len(token_ids[0])}, token_ids[1] len:{len(token_ids[1])}")
# 	# token_ids = token_ids[indx]
# 	token_ids = sorted(token_ids, key=lambda x: x.size()[0], reverse=True)
# 	print(f"after, token_ids[0] len:{len(token_ids[0])}, token_ids[1] len:{len(token_ids[1])}")
# 	label_ids = label_ids[indx]
# 	valid_ids = valid_ids[indx]
# 	label_masks = label_masks[indx]
# 	# transpose (batch_size, seq_length, _) to (seq_length, batch_size, _) ?
# 	return token_ids, label_ids, valid_ids, label_lens, label_masks

def sort_batch(label_lens, valid_output, labels, label_masks, valid_ids = None):
	# print(f"before, label_lens:{label_lens}")
	label_lens, indx = label_lens.sort(dim=0, descending=True)

	valid_output = valid_output[indx]
	if labels is not None:
		labels = labels[indx]
	label_masks = label_masks[indx]
	if valid_ids is not None:
		valid_ids = valid_ids[indx]
		return label_lens, valid_output, labels, label_masks, valid_ids
	else:
		return label_lens, valid_output, labels, label_masks