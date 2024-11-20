
import os

from transformers import T5EncoderModel, T5Tokenizer
import re
import numpy as np
import pandas as pd
import torch
import  torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import esm
import glob


def read_data_file_trip(queryfile):
    results=[]
    f = open(queryfile) #'customer_test/query.txt'
    data = f.readlines()
    f.close()
    length=len(data)
    for index in range(length):
        results.append(data[index].strip())
    return results

def coll_paddding(batch_traindata):
    batch_traindata.sort(key=lambda data: len(data[0]), reverse=True)
    feature_plms = []
    train_y = []
    task_ids=[]


    for data in batch_traindata:
        feature_plms.append(data[0])

        train_y.append(data[1])
        task_ids.append(data[2])
    data_length = [len(data) for data in feature_plms]

    feature_plms = torch.nn.utils.rnn.pad_sequence(feature_plms, batch_first=True, padding_value=0)
    train_y = torch.nn.utils.rnn.pad_sequence(train_y, batch_first=True, padding_value=0)
    task_ids = torch.nn.utils.rnn.pad_sequence(task_ids, batch_first=True, padding_value=0)
    return feature_plms,train_y,task_ids,torch.tensor(data_length)

class BioinformaticsDataset(Dataset):
    # X: list of filename
    def __init__(self, X):
        self.X = X
    def __getitem__(self, index):
        filename = self.X[index]
        #esm_embedding1280 prot_embedding  esm_embedding2560 msa_embedding
        df0 = pd.read_csv('customer_test/' + filename + '_T5.data', header=None)
        prot0 = df0.values.astype(float).tolist()
        prot0 = torch.tensor(prot0)

        df1 = pd.read_csv('customer_test/' + filename + '_ESM2.data', header=None)
        prot1 = df1.values.astype(float).tolist()
        prot1 = torch.tensor(prot1)

        #combine two plms
        prot=torch.cat((prot0,prot1),dim=1)#prot0#
        return prot


    def __len__(self):
        return len(self.X)

class AttentionModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AttentionModel, self).__init__()
        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
        self._norm_fact = 1 / torch.sqrt(torch.tensor(out_dim))

    def forward(self, plms1, seqlengths):
        Q = self.q(plms1)
        K = self.k(plms1)
        V = self.v(plms1)
        atten=self.masked_softmax((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact,seqlengths)
        output = torch.bmm(atten, V)
        return output + V
    def create_src_lengths_mask(self, batch_size: int, src_lengths):
        max_src_len = int(src_lengths.max())
        src_indices = torch.arange(0, max_src_len).unsqueeze(0).type_as(src_lengths)
        src_indices = src_indices.expand(batch_size, max_src_len)
        src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_src_len)
        # returns [batch_size, max_seq_len]
        return (src_indices < src_lengths).int().detach()

    def masked_softmax(self, scores, src_lengths, src_length_masking=False):
        # scores [batchsize,L*L]
        if src_length_masking:
            bsz, src_len, max_src_len = scores.size()
            # compute masks
            src_mask = self.create_src_lengths_mask(bsz, src_lengths)
            src_mask = torch.unsqueeze(src_mask, 2)
            # Fill pad positions with -inf
            scores = scores.permute(0, 2, 1)
            scores = scores.masked_fill(src_mask == 0, -np.inf)
            scores = scores.permute(0, 2, 1)
        return F.softmax(scores.float(), dim=-1)


class Task_shared(nn.Module):
    def __init__(self,inputdim):
        super(Task_shared,self).__init__()
        self.inputdim=inputdim


        self.ms1cnn1=nn.Conv1d(self.inputdim,512,1,padding='same')
        self.ms1cnn2=nn.Conv1d(512,256,1,padding='same')
        self.ms1cnn3=nn.Conv1d(256,128,1,padding='same')


        self.ms2cnn1 = nn.Conv1d(self.inputdim, 512, 3, padding='same')
        self.ms2cnn2 = nn.Conv1d(512, 256, 3, padding='same')
        self.ms2cnn3 = nn.Conv1d(256, 128, 3, padding='same')


        self.ms3cnn1 = nn.Conv1d(self.inputdim, 512, 5, padding='same')
        self.ms3cnn2 = nn.Conv1d(512, 256, 5, padding='same')
        self.ms3cnn3 = nn.Conv1d(256, 128, 5, padding='same')

        self.relu=nn.ReLU(True)

        self.AttentionModel1 = AttentionModel(512, 128)
        self.AttentionModel2 = AttentionModel(256, 128)
        self.AttentionModel3 = AttentionModel(128, 128)



    def forward(self,prot_input,seqlengths):

        prot_input_share = prot_input.permute(0, 2, 1)

        m1=self.relu(self.ms1cnn1(prot_input_share))
        m2 = self.relu(self.ms2cnn1(prot_input_share))
        m3 = self.relu(self.ms3cnn1(prot_input_share))

        att=m1+m2+m3
        att=att.permute(0,2,1)
        s1=self.AttentionModel1(att, seqlengths)

        m1 = self.relu(self.ms1cnn2(m1))
        m2 = self.relu(self.ms2cnn2(m2))
        m3 = self.relu(self.ms3cnn2(m3))

        att = m1 + m2 + m3
        att = att.permute(0, 2, 1)
        s2 = self.AttentionModel2(att, seqlengths)

        m1 = self.relu(self.ms1cnn3(m1))
        m2 = self.relu(self.ms2cnn3(m2))
        m3 = self.relu(self.ms3cnn3(m3))

        att = m1 + m2 + m3
        att = att.permute(0, 2, 1)
        s3 = self.AttentionModel3(att, seqlengths)

        mscnn=m1+m2+m3
        mscnn=mscnn.permute(0,2,1)
        s=s1+s2+s3

        return mscnn+s


class MTLModule(nn.Module):
    def __init__(self,inputdim,istrain,tasklen):
        super(MTLModule,self).__init__()
        self.istrain=istrain
        self.inputdim = inputdim
        self.tasklen=tasklen
        self.ShardEncoder=Task_shared(self.inputdim)

        self.tasks_fcs = nn.ModuleList()

        for i in range(self.tasklen):

            self.tasks_fcs.append(nn.Sequential(nn.Linear(128, 512),
                      nn.Dropout(0.5),
                      nn.Linear(512, 64),
                      nn.Dropout(0.5),
                      nn.Linear(64, 2)))


    def forward(self,prot_input,datalengths):
        sharedembedding=self.ShardEncoder(prot_input,datalengths)

        task_outs=[]
        for i in range(self.tasklen):
            task_embeddingi=self.tasks_fcs[i](sharedembedding)
            task_outs.append(task_embeddingi)
        return task_outs


def test(queryseq,taks_length, modelstoreapl):
    model = MTLModule(1024+1280,False,taks_length)
    model = model.to(device)
    model.load_state_dict(torch.load(modelstoreapl))
    model.eval()
    tmresult = {}

    test_set = BioinformaticsDataset([queryseq])
    test_load = DataLoader(dataset=test_set, batch_size=1,
                           num_workers=16, pin_memory=True, persistent_workers=True)

    print("===Test with ",modelstoreapl," ====================")

    with torch.no_grad():
        for prot_xs in  test_load:
            task_preds=[]
            task_outs = model(prot_xs.to(device),torch.tensor([1,1]).to(device))
            for i in range(taks_length):
                task_pred = task_outs[i]
                task_pred = task_pred.to('cpu')
                task_pred=task_pred[0]
                task_pred = torch.nn.functional.softmax(task_pred, dim=1)
                task_preds.append(np.array(task_pred[:,1]))
            return task_preds



def generatePLMs(seq,seqtmpname):
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    model = model.to(device)
    model = model.eval()

    newseq = ' '.join(s for s in seq)
    newseq = re.sub(r"[UZOB]", "X", newseq)
    # print('newseq length',len(newseq))
    ids = tokenizer.batch_encode_plus([newseq], add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)

    embedding = embedding.last_hidden_state.cpu().numpy()


    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][:seq_len - 1]
        # features.append(seq_emd)
        with open(os.path.join('customer_test', seqtmpname+'_T5' + '.data'), 'w') as f:
            np.savetxt(f, seq_emd, delimiter=',', fmt='%s')
    print('embedding generated using ProtT5 cpmpleted')
    torch.cuda.empty_cache()
    model=None
    torch.cuda.empty_cache()

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    model = model.to(device)

    data = [(seqtmpname, seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    # Extract per-residue representations (on CPU)
    with torch.no_grad():

        batch_tokens = batch_tokens.to(device)
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        results = results['representations'][33].to(device="cpu")
    token_representations = results

    for token_representation, tokens_len, batch_label in zip(token_representations, batch_lens, batch_labels):
        with open(os.path.join('customer_test', seqtmpname+'_ESM2' + '.data'), 'w') as f:
            np.savetxt(f, token_representation[1: tokens_len - 1], delimiter=',', fmt='%s')
    print('embedding generated using ESM2 cpmpleted')
    print("Feature Embedding completed!")


if __name__ == "__main__":

    cuda = torch.cuda.is_available()
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")

    #you can remove models in  tasks_pre_models for specific nucleotides binding residues prediction
    tasks_pre_models={'task_ADP_ATP_AMP':3,'task_GDP':4,'task_CDP':2,'task_GMP':2,'task_IMP':9,
                     'task_TTP':8,'task_UMP':4,'task_UTP':6,'task_CMP':3,'task_CTP':4,'task_TMP':3,
                     'task_GTP':3, 'task_UDP':1}
    queryseqfile='customer_test/query.txt'
    filecontents = read_data_file_trip(queryseqfile)
    protein_sequence=filecontents[0]
    tmp_name='TMP_2024'
    # plms generation
    generatePLMs(protein_sequence,tmp_name)
    seqs=[s for s in protein_sequence]
    pdata={}
    print('run prediction')
    pdata['sequence']=seqs
    for modelfold,fv in tasks_pre_models.items():
        if modelfold == 'task_ADP_ATP_AMP':
            pdata['ADP']=[]
            pdata['ATP']=[]
            pdata['AMP']=[]
        else:
            pdata[modelfold.replace('task_', '')]=[]
        for storeapl in glob.iglob('pre_model/'+modelfold+'/*.pkl'):
            #load each pre_trained model
            tmresults = test(tmp_name,fv, storeapl)
            if modelfold=='task_ADP_ATP_AMP':
                pdata['ADP'].append(tmresults[0])
                pdata['ATP'].append(tmresults[1])
                pdata['AMP'].append(tmresults[2])
            else:
                pdata[modelfold.replace('task_','')].append(tmresults[0])
    for pk in pdata:
        if pk=='sequence':
            continue
        pdata[pk]=np.mean(pdata[pk], axis=0)

    df = pd.DataFrame(pdata)  # create DataFrame and save to xlsx
    df.to_excel('customer_test/result.xlsx', index=False)

