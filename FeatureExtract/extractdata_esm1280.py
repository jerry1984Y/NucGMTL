import torch
import esm
import numpy as np
import os


def read_data_file_trip(filename):
    f = open(filename)
    data = f.readlines()
    f.close()

    results=[]
    block=len(data)//2
    for index in range(block):
        item1=data[index*2+0].split()
        name =item1[0].strip()
        seq=item1[1].strip()
        #item2 = data[index * 2 + 1].split()
        item = []
        item.append(name)
        item.append(seq)
        results.append(item)
    return results
def extratdata(file,destfolder):
    student_tuples=read_data_file_trip(file)
    for name,seq in student_tuples:
        # if os.path.exists(os.path.join(destfolder, name + '.data')):
        #     print(name,' existed')
        #     continue
        # with open(os.path.join(destfolder,name+ '.label'), 'w') as f:
        #     f.write(','.join(l for l in label))
        print(name)
        data = [(name, seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            if torch.cuda.is_available():
                batch_tokens = batch_tokens.to(device="cuda:0")
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            if torch.cuda.is_available():
                results = results['representations'][33].to(device="cpu")
            else:
                results = results['representations'][33]

        token_representations = results

        for token_representation, tokens_len, batch_label in zip(token_representations, batch_lens, batch_labels):
            with open(os.path.join(destfolder, batch_label + '.data'), 'w') as f:
                np.savetxt(f, token_representation[1: tokens_len - 1], delimiter=',', fmt='%s')

if __name__ == "__main__":
    # Load ESM-2 model
    print('----loading esm-2 model-------')
    #esm2_t36_3B_UR50D esm2_t33_650M_UR50D
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    if torch.cuda.is_available():
        #model = model.cuda()
        model = model.to('cuda:0')
    print('----prepare dataset------')

    trainfiles1 = ['../DataSet/Train/ATP30_Train.txt',
                   '../DataSet/Train/ADP30_Train.txt',
                   '../DataSet/Train/AMP30_Train.txt',
                   '../DataSet/Train/GDP30_Train.txt',
                   '../DataSet/Train/GTP30_Train.txt',
                   '../DataSet/Train/TMP30_Train.txt',
                   '../DataSet/Train/CTP30_Train.txt',
                   '../DataSet/Train/CMP30_Train.txt',
                   '../DataSet/Train/UTP30_Train.txt',
                   '../DataSet/Train/UMP30_Train.txt',
                   '../DataSet/Train/UDP30_Train.txt',
                   '../DataSet/Train/IMP30_Train.txt',
                   '../DataSet/Train/GMP30_Train.txt',
                   '../DataSet/Train/CDP30_Train.txt',
                   '../DataSet/Train/TTP30_Train.txt',
                   ]
    testfiles1 = ['../DataSet/Test/ATP30_Test.txt',
                  '../DataSet/Test/ADP30_Test.txt',
                  '../DataSet/Test/AMP30_Test.txt',
                  '../DataSet/Test/GDP30_Test.txt',
                  '../DataSet/Test/GTP30_Test.txt',
                  '../DataSet/Test/TMP30_Test.txt',
                  '../DataSet/Test/CTP30_Test.txt',
                  '../DataSet/Test/CMP30_Test.txt',
                  '../DataSet/Test/UTP30_Test.txt',
                  '../DataSet/Test/UMP30_Test.txt',
                  '../DataSet/Test/UDP30_Test.txt',
                  '../DataSet/Test/IMP30_Test.txt',
                  '../DataSet/Test/GMP30_Test.txt',
                  '../DataSet/Test/CDP30_Test.txt',
                  '../DataSet/Test/TTP30_Test.txt'
                  ]
    for item in trainfiles1:
        print(item)
        extratdata(item, '../embedding/esm_embedding1280/')

    for item in testfiles1:
        print(item)
        extratdata(item, '../embedding/esm_embedding1280/')

    print('----finish-------')



