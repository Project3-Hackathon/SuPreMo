#!/usr/bin/env python
# coding: utf-8

'''
Functions that accompany SuPreMo get_scores for scoring variants for nearby gene expression 
'''


# # # # # # # # # # # # # # # # # # 
# # # # # Import packages # # # # #
import os
import sys
import torch
from torch import nn
import numpy as np
import math
import pandas as pd
import h5py
from pathlib import Path
import json
import tabix
import xgboost as xgb
from six.moves import reduce


# # # # # # # # # # # # # # # # # # 
# # # # # # Load model # # # # # #

# This file path and model path
repo_path = Path(__file__).parents[1]

#beluga_model_file  = f'{repo_path}/ExPecto_model/resources/deepsea.beluga.pth'
#tss_tabix_file_hg19 = f'{repo_path}/ExPecto_model/resources/geneanno.pc.sorted.bed.gz'
#tss_tabix_file_hg38 = f'{repo_path}/ExPecto_model/resources/geneanno.hg38.sorted.bed.gz'
params_file = f'{repo_path}/ExPecto_model/params.json'
#models = []

########################################
# class for chromatin profile prediction
########################################

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class Beluga(nn.Module):
    def __init__(self):
        super(Beluga, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4,320,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(320,320,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(320,480,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(480,480,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(480,640,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(640,640,(1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0),-1)),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(67840,2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2003,2002)),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def load_model(model_path = f'{repo_path}/ExPecto_model/resources/deepsea.beluga.pth', is_cuda = False, verbose_level = 0):
    if os.path.exists(model_path):
        model = Beluga()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        if is_cuda:
            model = model.cuda()
        if verbose_level > 0:
            print(f"> ExPecto -> Model: {model_path} loaded.")
    else:
        print("Model file: %s not found" % (model_path))
        sys.exit(1)
    return model


# --------------------------------------
# Main function to get Expecto scores
# ---------------------------------------

class GeneTSS(object): 
    def __init__(self, chrom, start, end, strand, gene_id):
        self.chrom = chrom
        self.start = start
        self.end = end
        self.strand = strand
        self.gene_id = gene_id
        self.dist = 0
        self.score = None
    def __repr__(self):
        return "GeneTSS(chrom=%s, start=%d, end=%d, strand=%s, gene_id=%s, dist=%d)" % (self.chrom, self.start, self.end, self.strand, self.gene_id, self.dist)
    def cal_dist(self, query_pos):
        self.dist = self.start - query_pos


class ExPectoPrediction(object):
    def __init__(self, chr, pos, ref, alt, svtype, svlen, sequence, shift, revcomp, var_index, inputsize = 2000):
        # verbose levels
        self.verbose_level = 0
        self.chr = chr
        self.pos = pos
        self.ref = ref
        self.alt = alt
        self.svtype = svtype
        self.variant_type_detail = None
        self.svlen = svlen
        self.sequence = sequence
        self.shift = shift
        self.revcomp = revcomp
        self.var_index = var_index
        self.inputsize = inputsize
        # variables below are for sequence encoding 
        self.ref_seq = None
        self.alt_seq = None
        self.ref_encode = None
        self.alt_encode = None
        # variables below are for the nearby gene search 
        self.nearby_genes = []
        self.BND_type = None
        self.var_relative_pos = None
        # variables below are for chromatin effect prediction
        self.chromatin_effect_ref = None
        self.chromatin_effect_alt = None
        self.chromatin_effect_diff = None
        self.chromatin_effect_hdf5_file = None
        # variables below are for gene expression prediction
    def report_variant_score(self):
        # build a row a dataframe
        variant_score = []
        if pd.isna(self.svtype):
            variant_type = 'non-sv'
        else:
            variant_type = self.svtype
        for gene in self.nearby_genes:
            for cell_id in gene.score.keys():
                variant_score.append([self.var_index, variant_type, gene.gene_id, self.shift, cell_id, gene.score[cell_id], gene.dist, gene.strand])
        # convert to pandas dataframe
        variant_score = pd.DataFrame(variant_score, columns = ['var_index', 'var_type', 'gene_id', 'shift', 'cell_id', 'gene_score', 'distance', 'strand'])
        return variant_score
    def adjust_sequence_length(self):
        ref_seq = self.sequence[0]
        alt_seq = self.sequence[1]
        self.var_relative_pos = self.sequence[2]
        assert len(ref_seq) == len(alt_seq)
        # check if the sequence is long enough
        seq_length = len(ref_seq)
        if seq_length < self.inputsize:
            print("> ExPecto -> Sequence Encoding: Warning: Sequence length is too short, skip it!", sys.stderr)
            return None
        elif seq_length == self.inputsize: # do nothing
            if self.verbose_level > 0:
                print("> ExPecto -> Sequence Encoding: Sequence length is 2000bp, no need to center the sequence.")
            pass
        else: # get the centered 2000bp sequence
            if self.verbose_level > 0:
                print("> ExPecto -> Sequence Encoding: Sequence length is longer than 2000bp, centering the sequence.")
            ref_seq = ref_seq[seq_length//2 - self.inputsize//2: seq_length//2 + self.inputsize//2]
            alt_seq = alt_seq[seq_length//2 - self.inputsize//2: seq_length//2 + self.inputsize//2]
        #
        self.ref_seq = ref_seq
        self.alt_seq = alt_seq
    def encode_seq(self):
        if self.verbose_level > 0:
            print("> ExPecto -> Sequence Encoding: Encoding the sequence.")
        self.ref_encode = encodeSeqs([self.ref_seq], inputsize=self.inputsize).astype(np.float32)
        self.alt_encode = encodeSeqs([self.alt_seq], inputsize=self.inputsize).astype(np.float32)
        if self.verbose_level > 0:
            print("> ExPecto -> Sequence Encoding: Sequence encoding completed.")
    def predict_chromatin_diff(self, beluga_model, is_cuda = False):
        if self.verbose_level > 0:
            print("> ExPecto -> Chromatin Effect Prediction: Predicting the chromatin effect.")
        # reference sequence prediction
        ref_preds = []
        input = torch.from_numpy(self.ref_encode).unsqueeze(2)
        if is_cuda:
            input = input.cuda()
        ref_preds.append(beluga_model.forward(input).cpu().detach().numpy().copy())
        ref_preds = np.vstack(ref_preds)
        # alternative sequence prediction
        alt_preds = []
        input = torch.from_numpy(self.alt_encode).unsqueeze(2)
        if is_cuda:
            input = input.cuda()
        alt_preds.append(beluga_model.forward(input).cpu().detach().numpy().copy())
        alt_preds = np.vstack(alt_preds)
        #
        self.chromatin_effect_ref = ref_preds
        self.chromatin_effect_alt = alt_preds
        effects_temp = alt_preds - ref_preds 
        self.chromatin_effect_diff = effects_temp.mean(axis = 0).reshape(1, -1)
    def save_chromatin_effect(self, hdf5_file):
        if self.verbose_level > 0:
            print("> ExPecto -> Chromatin Effect Prediction: Saving the chromatin effect to %s" % (hdf5_file))
        f = h5py.File(hdf5_file, 'w')
        f.create_dataset('pred', data = self.chromatin_effect_diff)
        f.create_dataset('ref', data = self.chromatin_effect_ref)
        f.create_dataset('alt', data = self.chromatin_effect_alt)
        f.close()
        self.chromatin_effect_hdf5_file = hdf5_file
    def get_nearby_genes(self, tss_tabix_file, distance_threshold = 20000):
        if self.verbose_level > 0:
            print("> ExPecto -> Nearby Gene Search: Searching for nearby genes.")
        # get gene tss tabix
        tb = tabix.open(tss_tabix_file)
        # in different cases search nearby genes
        if pd.isna(self.svtype): # SNP, MNP, or small INDEL
            query_chrom = self.chr
            query_start = self.pos - distance_threshold
            #
            if '<' not in self.alt:
                query_end = self.pos + len(self.ref) + distance_threshold
            else:
                query_end = self.pos + abs(self.svlen) + distance_threshold
            self.nearby_genes = self.search_gene(tb, query_chrom, query_start, query_end, self.pos)
        else:
            if self.svtype == 'DEL':
                query_chrom = self.chr
                query_start = self.pos - distance_threshold
                query_end = self.pos + abs(self.svlen) + distance_threshold
                self.nearby_genes = self.search_gene(tb, query_chrom, query_start, query_end, self.pos)
            elif self.svtype == 'INS' or self.svtype == 'DUP':
                query_chrom = self.chr
                query_start = self.pos - distance_threshold
                query_end = self.pos + distance_threshold
                self.nearby_genes = self.search_gene(tb, query_chrom, query_start, query_end, self.pos)
            elif self.svtype == 'INV':
                query_chrom = self.chr
                query_start = self.pos - distance_threshold
                query_end = self.pos + abs(self.svlen) + distance_threshold
                self.nearby_genes = self.search_gene(tb, query_chrom, query_start, query_end, self.pos)
            elif self.svtype == 'BND':
                first = self.alt[0]
                last = self.alt[-1]
                another_chr, another_pos = parse_BND_alt_to_pos(self.alt, verbose_label = self.verbose_level)
                if first in ['A', 'C', 'G', 'T', 'a', 'c', 'g', 't', 'N', 'n'] and last == '[':
                    self.BND_type = 'A_left->B_right' 
                elif first == '[' and last in ['A', 'C', 'G', 'T', 'a', 'c', 'g', 't', 'N', 'n']:
                    self.BND_type = 'A_rc_right->B_right'
                elif first in ['A', 'C', 'G', 'T', 'a', 'c', 'g', 't', 'N', 'n'] and last == ']':
                    self.BND_type = 'A_left->B_rc_right'
                elif first == ']' and last in ['A', 'C', 'G', 'T', 'a', 'c', 'g', 't', 'N', 'n']:
                    self.BND_type = 'B_left->A_right'
                else: 
                    print("> ExPecto -> Nearby Gene Search: Error: BND type is not correct.", file = sys.stderr)
                    exit(1)
                # 
                if self.BND_type == 'A_left->B_right':
                    query_chrom_A = self.chr
                    query_start_A = self.pos - distance_threshold
                    query_end_A = self.pos
                    query_chrom_B = another_chr
                    query_start_B = another_pos 
                    query_end_B = another_pos + distance_threshold
                    self.nearby_genes = self.search_gene(tb, query_chrom_A, query_start_A, query_end_A, self.pos) + self.search_gene(tb, query_chrom_B, query_start_B, query_end_B, another_pos)
                elif self.BND_type == 'A_rc_right->B_right':
                    query_chrom_A = self.chr
                    query_start_A = self.pos 
                    query_end_A = self.pos + distance_threshold
                    query_chrom_B = another_chr
                    query_start_B = another_pos
                    query_end_B = another_pos + distance_threshold
                    self.nearby_genes = self.search_gene(tb, query_chrom_A, query_start_A, query_end_A, self.pos) + self.search_gene(tb, query_chrom_B, query_start_B, query_end_B, another_pos)
                elif self.BND_type == 'A_left->B_rc_right':
                    query_chrom_A = self.chr
                    query_start_A = self.pos - distance_threshold
                    query_end_A = self.pos
                    query_chrom_B = another_chr
                    query_start_B = another_pos - distance_threshold
                    query_end_B = another_pos
                    self.nearby_genes = self.search_gene(tb, query_chrom_A, query_start_A, query_end_A, self.pos) + self.search_gene(tb, query_chrom_B, query_start_B, query_end_B, another_pos)
                elif self.BND_type == 'B_left->A_right':
                    query_chrom_A = another_chr
                    query_start_A = another_pos - distance_threshold
                    query_end_A = another_pos
                    query_chrom_B = self.chr
                    query_start_B = self.pos
                    query_end_B = self.pos + distance_threshold
                    self.nearby_genes = self.search_gene(tb, query_chrom_A, query_start_A, query_end_A, another_pos) + self.search_gene(tb, query_chrom_B, query_start_B, query_end_B, self.pos)
                else:
                    pass
            else:
                if self.verbose_level > 0:
                    print("> ExPecto -> Nearby Gene Search: Error: SV type (%s) is not known" % (self.svtype), file = sys.stderr)
    def search_gene(self, tb, query_chrom, query_start, query_end, query_pos):
        result = []
        query_start = max(0, int(query_start))
        query_end = int(query_end)
        for record in tb.query(query_chrom, query_start, query_end):
            chrom = record[0]
            start = int(record[1])
            end = int(record[2])
            strand = record[3]
            gene_id = record[4]
            gene_tss = GeneTSS(chrom, start, end, strand, gene_id)
            gene_tss.cal_dist(query_pos)
            result.append(gene_tss)
        return result
    def compute_effects(self, ExPecto_params, models):
        if self.verbose_level > 0:
            print("> ExPecto -> Computing ExPecto scores.")
        for gene in self.nearby_genes:
            scores = self.compute_effect_by_gene(np.array([gene.dist]), np.array([gene.strand]), self.chromatin_effect_diff, models, self.shift, ExPecto_params['old_format'], ExPecto_params['n_features'])
            gene_score = {}
            for cell_id,cell_score in zip(ExPecto_params['model_name'], scores):
                gene_score[cell_id] = cell_score
            gene.score = gene_score
    def compute_effect_by_gene(self, snpdists, snpstrands, snpeffects, models, shift_dist, old_format, n_features):
        n_variant = 1
        if n_variant != len(snpdists):
            print("> ExPecto -> Error: The number of variants is not consistent with the number of distances.", sys.stderr)
            exit(1)
        snpdists = snpdists * ((snpstrands == '+') * 2 - 1)
        Xreducedall_diffs = [np.vstack([
        np.exp(-0.01 * np.floor(np.abs((snpdists + dist * ((snpstrands == '+') * 2 - 1)
               ) / 200.0))) * ((snpdists + dist * ((snpstrands == '+') * 2 - 1)) <= 0),
        np.exp(-0.02 * np.floor(np.abs((snpdists + dist * ((snpstrands == '+') * 2 - 1)
               ) / 200.0))) * ((snpdists + dist * ((snpstrands == '+') * 2 - 1)) <= 0),
        np.exp(-0.05 * np.floor(np.abs((snpdists + dist * ((snpstrands == '+') * 2 - 1)
               ) / 200.0))) * ((snpdists + dist * ((snpstrands == '+') * 2 - 1)) <= 0),
        np.exp(-0.1 * np.floor(np.abs((snpdists + dist * ((snpstrands == '+') * 2 - 1)
               ) / 200.0))) * ((snpdists + dist * ((snpstrands == '+') * 2 - 1)) <= 0),
        np.exp(-0.2 * np.floor(np.abs((snpdists + dist * ((snpstrands == '+') * 2 - 1)
               ) / 200.0))) * ((snpdists + dist * ((snpstrands == '+') * 2 - 1)) <= 0),
        np.exp(-0.01 * np.floor(np.abs((snpdists + dist * ((snpstrands == '+') * 2 - 1)
               ) / 200.0))) * ((snpdists + dist * ((snpstrands == '+') * 2 - 1)) >= 0),
        np.exp(-0.02 * np.floor(np.abs((snpdists + dist * ((snpstrands == '+') * 2 - 1)
               ) / 200.0))) * ((snpdists + dist * ((snpstrands == '+') * 2 - 1)) >= 0),
        np.exp(-0.05 * np.floor(np.abs((snpdists + dist * ((snpstrands == '+') * 2 - 1)
               ) / 200.0))) * ((snpdists + dist * ((snpstrands == '+') * 2 - 1)) >= 0),
        np.exp(-0.1 * np.floor(np.abs((snpdists + dist * ((snpstrands == '+') * 2 - 1)
               ) / 200.0))) * ((snpdists + dist * ((snpstrands == '+') * 2 - 1)) >= 0),
        np.exp(-0.2 * np.floor(np.abs((snpdists + dist * ((snpstrands == '+') * 2 - 1)
               ) / 200.0))) * ((snpdists + dist * ((snpstrands == '+') * 2 - 1)) >= 0)
         ]).T for dist in [shift_dist] ]
        # compute gene expression change with models
        diff = reduce(lambda x, y: x + y, [np.tile(np.asarray(snpeffects[:, :]), 10)
                                 * np.repeat(Xreducedall_diffs[j][:, :], n_features, axis=1) for j in range(len(Xreducedall_diffs))])
        if old_format:
            # backward compatibility
            diff = np.concatenate([np.zeros((diff.shape[0], 10, 1)), diff.reshape(
                (-1, 10, 2002))], axis=2).reshape((-1, 20030))
        dtest_ref = xgb.DMatrix(diff * 0)
        dtest_alt = xgb.DMatrix(diff)

        effect = np.zeros(len(models))
        for j in range(len(models)):
            effect[j] = models[j].predict(dtest_alt) - models[j].predict(dtest_ref)
        return effect

def parse_BND_alt_to_pos(alt, verbose_level = 0):   
    # A]chr6:73541678]
    # ]chr5:45700000]T
    if '[' in alt:
        chrom,pos = alt.split('[')[1].split(':')
    elif ']' in alt:
        chrom,pos = alt.split(']')[1].split(':')
    else:
        if verbose_level > 0:
            print("> ExPecto -> Nearby Gene Search: Error: BND alt format is not correct.", sys.stderr)
            return None, None
    return chrom, int(pos)


def encodeSeqs(seqs, inputsize=2000):
    """Convert sequences to 0-1 encoding and truncate to the input size.
    The output concatenates the forward and reverse complement sequence
    encodings.

    Args:
        seqs: list of sequences (e.g. produced by fetchSeqs)
        inputsize: the number of basepairs to encode in the output

    Returns:
        numpy array of dimension: (2 x number of sequence) x 4 x inputsize

    2 x number of sequence because of the concatenation of forward and reverse
    complement sequences.
    """
    seq_encoding = np.zeros((len(seqs), 4, inputsize), np.bool_)

    mydict = {'A': np.asarray([1, 0, 0, 0]), 'G': np.asarray([0, 1, 0, 0]),
            'C': np.asarray([0, 0, 1, 0]), 'T': np.asarray([0, 0, 0, 1]),
            'N': np.asarray([0, 0, 0, 0]), 'H': np.asarray([0, 0, 0, 0]),
            'a': np.asarray([1, 0, 0, 0]), 'g': np.asarray([0, 1, 0, 0]),
            'c': np.asarray([0, 0, 1, 0]), 't': np.asarray([0, 0, 0, 1]),
            'n': np.asarray([0, 0, 0, 0]), '-': np.asarray([0, 0, 0, 0])}

    n = 0
    for line in seqs:
        cline = line[int(math.floor(((len(line) - inputsize) / 2.0))):int(math.floor(len(line) - (len(line) - inputsize) / 2.0))]
        for i, c in enumerate(cline):
            seq_encoding[n, :, i] = mydict[c]
        n = n + 1

    # get the complementary sequences
    dataflip = seq_encoding[:, ::-1, ::-1]
    seq_encoding = np.concatenate([seq_encoding, dataflip], axis=0)
    return seq_encoding


def get_effects(chr, pos, ref, alt, svtype, svlen, sequence, shift, revcomp, var_index, is_cuda = False, inputsize = 2000, beluga_model = None, beluga_model_file = None, beluga_checkpoint_prefix = None, ExPecto_params = None):
    ########################################
    # construct ExPecto object
    ########################################
    verbose_level = ExPecto_params['verbose_level']
    if verbose_level > 0:
        print(f'> ExPecto -> Getting ExPecto scores for {var_index} ({shift} revcomp:{revcomp})')
    expecto_task = ExPectoPrediction(chr, pos, ref, alt, svtype, svlen, sequence, shift, revcomp, var_index, inputsize)
    expecto_task.verbose_level = verbose_level


    ########################################
    # check beluga model
    ########################################
    # load beluga model if not given
    if beluga_model is None:
        if verbose_level > 0:
            print("> Expecto -> Loading the model.")
            beluga_model = load_model(model_path = ExPecto_params['beluga_model_file'], is_cuda = is_cuda)

    ########################################
    # check prediction models
    ########################################
    models = []
    if ExPecto_params['model_name'] == 'all':
        use_all_models = True
    else:
        use_all_models = False
    with open(ExPecto_params['model_list_file']) as fin:
        N = 0
        for line in fin:
            if line[0] == '#':
                continue
            row = line.strip().split('\t')
            if N == 0: # skip header
                N += 1
                continue
            model_path, model_name = row[0], row[1]
            model_file = os.path.join(ExPecto_params['model_dir'], model_path)
            # check if model file exists
            if not os.path.isfile(model_file):
                print("> ExPecto -> Error: Model file %s does not exist." % (model_file), file = sys.stderr)
            # load model
            if not use_all_models:
                if model_name in ExPecto_params['model_name']:
                    bst = xgb.Booster({'nthread': ExPecto_params['threads']})
                    bst.load_model(model_file.strip())
                    models.append(bst)
            else:
                bst = xgb.Booster({'nthread': ExPecto_params['threads']})
                bst.load_model(model_file.strip())
                models.append(bst)
        # backward compatibility with earlier model format
        if len(models) > 0:
            if len(models[0].get_dump()[0].split('\n')) == 20034:
                ExPecto_params['old_format'] = True
            else:
                ExPecto_params['old_format'] = False


    ########################################
    # Skip reverse complement sequence as ExPecto will automatically use reverse complement sequence
    ########################################
    if revcomp:
        print("> ExPecto -> Preparation: Reverse complement sequence skipped because ExPecto model will automatically use reverse complement sequence.")
        return None


    ########################################
    # check sequence length
    ########################################
    expecto_task.adjust_sequence_length()
    # encode sequence
    expecto_task.encode_seq() 


    ########################################
    # get all genes within 40kb of the variant
    ########################################
    expecto_task.get_nearby_genes(tss_tabix_file = ExPecto_params["gene_tss_file"], distance_threshold = 40000)
    if len(expecto_task.nearby_genes) == 0:
        print("> ExPecto -> Nearby Gene Search: No nearby genes found, skip variant: %s" % (expecto_task.var_index))
        return None


    ########################################
    # predict chromatin effect
    ########################################
    expecto_task.predict_chromatin_diff(beluga_model, is_cuda)
    # save chromatin effect prediction into HDF5 file
    #expecto_task.save_chromatin_effect(beluga_checkpoint_prefix + str(var_index) + ".shift_" + str(shift) + ".diff.h5")


    ########################################
    # calculate the ExPecto scores
    ########################################
    expecto_task.compute_effects(ExPecto_params, models)


    # --------------------------------------
    # return epecto task: variant score
    # --------------------------------------
    if verbose_level > 0:
        print("> ExPecto -> ExPecto scores calculation done")
    # variant score variant_index, gene_id, gene_score, distance, strand 
    return expecto_task.report_variant_score()


def load_expecto_params(params_file):
    models = []
    # load json file
    with open(params_file) as f:
        expecto_params = json.load(f)
    # load model list
    if expecto_params['model_name'] == 'all':
        use_all_models = True
    else:
        use_all_models = False
    with open(expecto_params['model_list_file']) as fin:
        N = 0
        for line in fin:
            if line[0] == '#':
                continue
            row = line.strip().split('\t')
            if N == 0: # skip header
                N += 1
                continue
            model_path, model_name = row[0], row[1]
            model_file = os.path.join(expecto_params['model_dir'], model_path)
            # check if model file exists
            if not os.path.isfile(model_file):
                print("> ExPecto -> Error: Model file %s does not exist." % (model_file), file = sys.stderr)
            # load model
            if not use_all_models:
                if model_name in expecto_params['model_name']:
                    bst = xgb.Booster({'nthread': expecto_params['threads']})
                    bst.load_model(model_file.strip())
                    models.append(bst)
            else:
                bst = xgb.Booster({'nthread': expecto_params['threads']})
                bst.load_model(model_file.strip())
                models.append(bst)
        # backward compatibility with earlier model format
        if len(models) > 0:
            if len(models[0].get_dump()[0].split('\n')) == 20034:
                expecto_params['old_format'] = True
            else:
                expecto_params['old_format'] = False
    #print(models)
    return expecto_params


    # load model list
    # 