import pandas as pd

def gendictinfo():
    df0 = pd.read_excel('A2Relation.xlsx', sheet_name='Sheet1')
    # print(df0)
    dict = {}
    columns = df0.columns.values
    columnlen = len(columns)
    rows = len(df0.index.values)
    # print(rows)
    for index in range(rows):
        data = df0.loc[index].values
        # print(data)
        dict[data[0]] = {}
        for columnindex in range(1, columnlen):
            dict[data[0]][columns[columnindex]] = data[columnindex]
            columnindex += 1
    # print(dict)
    return dict

def runtask_aux_with_weight2():
    dict=gendictinfo()
    keys=dict.keys()
    taskgroups= {}
    #taskTrainRate={'ADP':510,'AMP':296,'ATP':448,'GDP':151,'GTP':116,'TMP':26,'CTP':53,'CMP':29,'UTP':45,'UMP':12,'UDP':100,'TTP':32,'IMP':22,'GMP':21,'CDP':31}
    taskTrainRate = {'ADP': 1131+68974, 'AMP': 2505+100429, 'ATP': 4688+183273, 'GDP': 11523+56859, 'GTP': 1204+50405, 'TMP': 224+7652, 'CTP': 484+20888, 'CMP': 315+8273,
                     'UTP': 444+19712, 'UMP': 103+2398, 'UDP': 974+37356, 'TTP': 347+14685, 'IMP': 209+6826, 'GMP': 178+7060, 'CDP': 311+10922}
    map=0
    for key in keys:
        taskgroups[key]=[]
        #AUPRC in descending order decend
        d_ordered=sorted(dict[key].items(),key=lambda x:x[1],reverse=True)  #[('ss',3),('ee',2),('ff',1)]
        #print(d_ordered)

        lastMaxAUPRC=0
        totalcross=0
        for okey,ovalue in d_ordered:
            #do not compared with itself
            if okey==key:
                continue
            #remove useless task
            if ovalue<=0.01:
                continue
            if len(taskgroups[key])==0:
                taskgroups[key].append(okey)
                lastMaxAUPRC=ovalue
            else:
                #newnodeavg=ovalue/2
                crosseffect=totalcross
                totalrecords=0
                for itkey in taskgroups[key]:
                    totalrecords+=taskTrainRate[itkey]

                    # if p1<=0.001 and p1>=-0.001:
                    #     p1=0
                    # if p2<=0.001 and p2>=-0.001:
                    #     p2=0
                    #crosseffect += (p1 * taskTrainRate[okey] / (taskTrainRate[okey] +taskTrainRate[itkey]) + p2 * taskTrainRate[itkey] /
                                    #(taskTrainRate[okey]+taskTrainRate[itkey])) / 4
                    transfer_itkey=1 if dict[itkey][key]>0 else -1
                    transfer_okey = 1 if dict[okey][key] > 0 else -1

                    p1 = dict[okey][itkey]*transfer_okey
                    p2 = dict[itkey][okey]*transfer_itkey

                    #p1 = dict[itkey][okey] * transfer_okey
                    #p2 = dict[okey][itkey] * transfer_itkey
                    crosseffect+=(p1*taskTrainRate[okey]/taskTrainRate[itkey]+p2*taskTrainRate[itkey]/taskTrainRate[okey])/1
                tmpAUPRC=0
                totalrecords+=taskTrainRate[okey]
                singleAUPRC=0
                for itkey in taskgroups[key]:
                    singleAUPRC+=taskTrainRate[itkey]/totalrecords*dict[key][itkey]
                singleAUPRC +=taskTrainRate[okey]/totalrecords*dict[key][okey]

                tmpAUPRC =  singleAUPRC + crosseffect
                if tmpAUPRC>lastMaxAUPRC:
                    taskgroups[key].append(okey)
                    totalcross+=crosseffect
                    lastMaxAUPRC=tmpAUPRC
                #calc average
                #avg=sum(taskgroups[key],d_ordered[okey])/2
                #calc each effect
        map+=lastMaxAUPRC
    print(map)
    print(taskgroups)


if __name__ == "__main__":
    runtask_aux_with_weight2()
