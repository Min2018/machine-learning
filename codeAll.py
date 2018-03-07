import pandas as pd
import numpy as np
#from anaCommonValues import *
import datetime
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import xlrd
import xlutils.copy
import datetime
import time

DATA_PATH="F:/DataAna/cf_all/data/"
RESULT_PATH="F:/DataAna/cf_all/data/result/"

def getAget():
    agedata = pd.read_csv(DATA_PATH+'cf_age_all.csv',encoding='gbk')
    return agedata
def getDegree():
    degreedata = pd.read_csv(DATA_PATH+'cf_final_all.csv',encoding='gbk')
    degreedata = degreedata[['ecif_id','educationDegree']]
    return degreedata
def load_age():
    with open(DATA_PATH+"cf_age_all.csv","r") as f:
        header=f.readline().strip().split(",")
        recs=f.readlines()
    recs=map(lambda a:a.strip().split(","),recs)
    eff=[]
    for rec in recs:
        try:
            rec=map(lambda a:int(a),rec)
            eff.append(rec)
        except:
            eff.append([int(rec[0]),None])
    age_pd=pd.DataFrame(eff,columns=header)
    #age_pd.set_index('ecif_id',inplace=True)
    return age_pd
def intx(x):
    try:
        return float(x)
    except:
        return ""
def load_yl():
    min_list=set(['S0121','S0124','S0127','S0130','S0297','S0298','S0299','S0300','S0301','S0302'])
    max_list=set(['S0098','S0101','S0109','S0111','S0112','S0113','S0115','S0118','S0003','S0017','S0080','S0089','S0095','S0133','S0174','S0175','S0274','S0277','S0279','S0280','S0281','S0282','S0283','S0284','S0292','S0294','S0295','S0296','S0316','S0319','S0321','S0322','S0323','S0478','S0479'])
    sum_list=set(['S0006','S0007','S0010','S0011','S0012','S0015','S0026','S0029','S0056','S0059','S0062','S0065','S0067','S0068','S0069','S0070','S0071','S0072','S0073','S0074','S0075','S0076','S0077','S0078','S0180','S0181','S0182','S0184','S0187','S0190','S0193','S0196','S0199','S0202','S0205','S0208','S0211','S0217','S0220','S0223','S0226','S0229','S0232','S0235','S0238','S0241','S0244','S0247','S0250','S0252','S0253','S0254','S0256','S0258','S0259','S0260','S0262','S0265','S0268','S0271','S0303','S0304','S0305','S0307','S0310','S0324','S0325','S0326','S0328','S0354','S0355','S0356','S0358','S0360','S0361','S0362','S0363','S0364','S0365','S0372','S0373','S0374','S0375','S0376','S0377','S0378','S0379','S0380','S0381','S0382','S0383','S0391','S0394','S0397','S0400','S0409','S0412','S0415','S0418','S0426','S0427','S0428','S0430','S0433','S0436','S0438','S0439','S0442','S0451','S0454'])
    merge_list=set(['S0002','S0013','S0014','S0020','S0023','S0083','S0086','S0140','S0143','S0146','S0149','S0151','S0152','S0153','S0155','S0158','S0166','S0167','S0169','S0172','S0176','S0177','S0385','S0388','S0462','S0463','S0464','S0465','S0466','S0467','S0468','S0469','S0482','S0483','S0501'])
    with open(DATA_PATH+"cf_yl_all.csv","r") as f:
        header=f.readline().strip().split(",")
        recs=f.readlines()
    i=0
    eff_dict={}
    header_dict={}
    header=header[4:]
    for idx,h in enumerate(header):
        header_dict[h]=idx
    for rec in recs:
        rec=rec.strip().split(",")
        ecif_id=rec[2]
        query_status=rec[-1]
        if query_status == "" or int(float(query_status)) != 0:
            continue
        eff=rec[4:]
        if len(eff) != 197:
            print "len wrong in %s"%ecif_id
        if not eff_dict.has_key(ecif_id):
            eff_dict[ecif_id]=[]
        eff_dict[ecif_id].append(eff)
    output_dict={}
    for ecif_id,recs in eff_dict.iteritems():
        output_dict[ecif_id]={}
        s0001_min=datetime.datetime.strptime("20200101","%Y%m%d")
        s0001_max=datetime.datetime.strptime("19000101","%Y%m%d")
        card_nums=len(recs)
        S0045_51=[{},{},{},{},{}]
        S0045_51_name=['S0045','S0046','S0047','S0048','S0051']
        for rec in recs:
            s0001=rec[0]
            try:
                s0001 = datetime.datetime.strptime(str(int(float(s0001))),"%Y%m%d")
            except:
                s0001=""
            if s0001 != "":
                if s0001 > s0001_max:
                    s0001_max=s0001
                if s0001 < s0001_min:
                    s0001_min=s0001
            S0045,S0046,S0047,S0048,S0051=rec[16:21]
            for idx,s in enumerate([S0045,S0046,S0047,S0048,S0051]):
                if s != "":
                    s=s.split(";")
                    for r in s:
                        r=r.split("_")
                        mon=r[0]
                        cash=r[1]
                        try:
                            cash=float(cash)
                        except:
                            continue
                        if not S0045_51[idx].has_key(mon):
                            S0045_51[idx][mon]=0.0
                        if idx ==4:
                            S0045_51[idx][mon]=max(S0045_51[idx][mon],cash)
                        else:
                            S0045_51[idx][mon]+=cash
        for idx,s in enumerate(S0045_51):
            if len(s)==0:
                continue
            h=S0045_51_name[idx]
            h_merge=sorted(s.iteritems(),key=lambda a:a[0],reverse=True)
            h_merge=";".join(map(lambda a:a[0]+"_"+str(a[1]),h_merge))
            h_v=s.values()
            h_max=max(h_v)
            h_min=min(h_v)
            h_sum=sum(h_v)
            h_avg=h_sum/len(h_v)
            output_dict[ecif_id][h+"_merge"]=h_merge
            output_dict[ecif_id][h+"_max"]=h_max
            output_dict[ecif_id][h+"_min"]=h_min
            output_dict[ecif_id][h+"_sum"]=h_sum
            output_dict[ecif_id][h+"_avg"]=h_avg
        output_dict[ecif_id]["card_nums"]=card_nums
        output_dict[ecif_id]["S0001_min"]=str(s0001_min.date())
        output_dict[ecif_id]["S0001_max"]=str(s0001_max.date())
        for idx,h in enumerate(header):
            if h in min_list:
                var_name=h+"_min"
                ana_list=map(lambda a:intx(a[idx]),recs)
                ana_list=filter(lambda a: a != "",ana_list)
                if len(ana_list)>0:
                    var_value=min(ana_list)
                    output_dict[ecif_id][var_name]=var_value
            if h in max_list:
                var_name=h+"_max"
                ana_list=map(lambda a:intx(a[idx]),recs)
                ana_list=filter(lambda a: a != "",ana_list)
                if len(ana_list)>0:
                    var_value=max(ana_list)
                    output_dict[ecif_id][var_name]=var_value
            if h in sum_list:
                var_name=h+"_sum"
                ana_list=map(lambda a:intx(a[idx]),recs)
                ana_list=filter(lambda a: a != "",ana_list)
                if len(ana_list)>0:
                    var_value=sum(ana_list)
                    output_dict[ecif_id][var_name]=var_value
            if h in merge_list:
                var_name=h+"_merge"
                ana_list=map(lambda a:a[idx].decode("gbk").encode("utf-8"),recs)
                var_value="|".join(ana_list)
                output_dict[ecif_id][var_name]=var_value
    frame_or={}
    for ecif_id,v_dict in output_dict.iteritems():
        for var_name,var_value in v_dict.iteritems():
            if not frame_or.has_key(var_name):
                frame_or[var_name]={}
            frame_or[var_name][ecif_id]=var_value

    yl_pd=pd.DataFrame(frame_or)

    yl_pd["S0462_renminbi"]=yl_pd.S0462_merge.apply(lambda a:a.count("人民币卡"))
    yl_pd["S0462_feirenminbi"]=yl_pd.S0462_merge.apply(lambda a:a.count("多币种卡")+a.count("人民币境外卡"))
    yl_pd["S0464_jieji"]=yl_pd.S0464_merge.apply(lambda a:a.count("借记卡"))
    yl_pd["S0464_feijieji"]=yl_pd.S0464_merge.apply(lambda a:a.count("贷记卡")+a.count("准贷记卡"))

    yl_pd["S0467_A"]=yl_pd.S0467_merge.apply(lambda a:a.count("A"))
    yl_pd["S0467_baijin"]=yl_pd.S0467_merge.apply(lambda a:a.count("白金卡"))
    yl_pd["S0467_jin"]=yl_pd.S0467_merge.apply(lambda a:a.count("金卡"))
    yl_pd["S0467_zuanshi"]=yl_pd.S0467_merge.apply(lambda a:a.count("钻石卡"))

    yl_pd["S0468_baijin"]=yl_pd.S0468_merge.apply(lambda a:a.count("白金"))
    yl_pd["S0468_jin"]=yl_pd.S0468_merge.apply(lambda a:a.count("金"))
    yl_pd["S0468_zuanshi"]=yl_pd.S0468_merge.apply(lambda a:a.count("钻石"))

    yl_pd["ecif_id"]=yl_pd.index

    return yl_pd
def load_wifi():
    wifi_pd=pd.read_csv(DATA_PATH+"wifi_match_new.csv",encoding="gbk")
    wifi_columns=list(wifi_pd.columns)
    wifi_columns=map(lambda a:"wifi_"+a.encode("utf_8"),wifi_columns)
    wifi_columns[0]="ecif_id"
    wifi_pd.columns=wifi_columns
    return wifi_pd
def getsample():
    rawData = pd.read_csv(DATA_PATH+'sample.csv',encoding='gbk',engine = 'python')
    return rawData
def gethobby():
    rawData = pd.read_csv(DATA_PATH+'customer_hobby.csv',encoding='gbk',engine = 'python')
    return rawData
def getactivity():
    rawData = pd.read_csv(DATA_PATH+'activity.csv')
    return rawData

def load_app():
    app_pd=pd.read_csv(DATA_PATH+"app_list.csv")
    columns=list(app_pd.columns)
    columns[0]='ecif_id'
    app_pd['app_inst_time']=app_pd.app_inst_time.apply(lambda a:datetime.datetime.strptime(a,"%Y/%m/%d %H:%M"))
    app_pd.columns=columns
    app_cat=pd.read_csv(DATA_PATH+"app_cat.csv")
    columns=list(app_cat.columns)
    columns[0]='app_name'
    app_cat.columns=columns
    print "total has ecif_id %s"%len(app_pd.ecif_id.unique())
    data=app_pd.merge(app_cat,on='app_name')
    for cat in data.cat2.unique():
        d=data[data.cat2==cat]
        ecif_ids=len(d.ecif_id.unique())
        print "%s|%s"%(cat,ecif_ids)
    data_output=data.groupby(['ecif_id','cat2']).size()
    d_l=list(data_output.index)
    d_ecif=map(lambda a:a[0],d_l)
    d_cat=map(lambda a:a[1],d_l)
    data_output=pd.DataFrame({'app_nums':data_output.values,'ecif_id':d_ecif,'cat':d_cat})
    last_inst=data.groupby(['ecif_id','cat2'])['app_inst_time'].max()
    d_l=list(last_inst.index)
    d_ecif=map(lambda a:a[0],d_l)
    d_cat=map(lambda a:a[1],d_l)
    last_inst=pd.DataFrame({'last_inst':last_inst.values,'ecif_id':d_ecif,'cat':d_cat})
    data_output=data_output.merge(last_inst,on=['ecif_id','cat'])
    return data_output
def load_xianxia(wifi_pd):
    #data_output=pd.DataFrame(columns=['ecif_id','cat','nums'])
    car=wifi_pd.ix[pd.notnull(wifi_pd.wifi_4S),['ecif_id','wifi_4S']]
    car.columns=['ecif_id','wifi_nums']
    car['cat']='汽车游艇'
    data_output=car
    fit=wifi_pd.ix[pd.notnull(wifi_pd['wifi_运动健身']),['ecif_id','wifi_运动健身']]
    fit.columns=['ecif_id','wifi_nums']
    fit['cat']='运动综合'
    data_output=data_output.append(fit)
    fit=wifi_pd.ix[pd.notnull(wifi_pd['wifi_海外留学']),['ecif_id','wifi_海外留学']]
    fit.columns=['ecif_id','wifi_nums']
    fit['cat']='教育'
    data_output=data_output.append(fit)
    fit=wifi_pd.ix[pd.notnull(wifi_pd['wifi_美容美体']),['ecif_id','wifi_美容美体']]
    fit.columns=['ecif_id','wifi_nums']
    fit['cat']='美容SPA'
    data_output=data_output.append(fit)
    fit=wifi_pd.ix[pd.notnull(wifi_pd['wifi_五星级酒店']),['ecif_id','wifi_五星级酒店']]
    fit.columns=['ecif_id','wifi_nums']
    fit['cat']='五星级酒店'
    data_output=data_output.append(fit)
    fit=wifi_pd.ix[pd.notnull(wifi_pd['wifi_儿童英语']),['ecif_id','wifi_儿童英语']]
    fit.columns=['ecif_id','wifi_nums']
    fit['cat']='教育'
    data_output=data_output.append(fit)
    fit=wifi_pd.ix[pd.notnull(wifi_pd['wifi_早教中心']),['ecif_id','wifi_早教中心']]
    fit.columns=['ecif_id','wifi_nums']
    fit['cat']='智力拓展'
    data_output=data_output.append(fit)

    data_output.wifi_nums=1
    data_output=data_output.drop_duplicates()
    return data_output
def load_yyl(sample,yl_pd):
    data=sample.merge(yl_pd,on="ecif_id",how="inner")
    temp=data.ix[data['S0415_sum']>0,['ecif_id','S0415_sum','S0418_sum']]
    temp.columns=['ecif_id','amount','counts']
    temp['cat']='珠宝首饰'
    data_output=temp
    temp=data.ix[data['S0247_sum']>0,['ecif_id','S0247_sum','S0250_sum']]
    temp.columns=['ecif_id','amount','counts']
    temp['cat']='汽车游艇'
    data_output=data_output.append(temp)
    temp=data.ix[data['S0199_sum']>0,['ecif_id','S0199_sum','S0202_sum']]
    temp.columns=['ecif_id','amount','counts']
    temp['cat']='医疗体检'
    data_output=data_output.append(temp)
    temp=data.ix[data['S0241_sum']>0,['ecif_id','S0241_sum','S0244_sum']]
    temp.columns=['ecif_id','amount','counts']
    temp['cat']='运动综合'
    data_output=data_output.append(temp)
    temp=data.ix[data['S0259_sum']>0,['ecif_id','S0259_sum','S0262_sum']]
    temp.columns=['ecif_id','amount','counts']
    temp['cat']='音乐'
    data_output=data_output.append(temp)
    temp=data.ix[data['S0235_sum']>0,['ecif_id','S0235_sum','S0238_sum']]
    temp.columns=['ecif_id','amount','counts']
    temp['cat']='美容SPA'
    data_output=data_output.append(temp)
    temp=data.ix[data['S0193_sum']>0,['ecif_id','S0193_sum','S0196_sum']]
    temp.columns=['ecif_id','amount','counts']
    temp['cat']='旅游综合'
    data_output=data_output.append(temp)
    temp=data.ix[data['S0187_sum']>0,['ecif_id','S0187_sum','S0190_sum']]
    temp.columns=['ecif_id','amount','counts']
    temp['cat']='美食发现'
    data_output=data_output.append(temp)
    temp=data.ix[data['S0451_sum']>0,['ecif_id','S0451_sum','S0454_sum']]
    temp.columns=['ecif_id','amount','counts']
    temp['cat']='保险'
    data_output=data_output.append(temp)
    return data_output
def process_app(seg_app_t):
    sel=(seg_app_t.cat=='医疗体检')&(seg_app_t.app_nums <2)
    sel=(sel==False)
    seg_app_t=seg_app_t[sel]
    sel=(seg_app_t.cat=='影视剧集')&(seg_app_t.app_nums <3)
    sel=(sel==False)
    seg_app_t=seg_app_t[sel]
    sel=(seg_app_t.cat=='旅游综合')&(seg_app_t.app_nums <3)
    sel=(sel==False)
    seg_app_t=seg_app_t[sel]
    sel=(seg_app_t.cat=='美食发现')&(seg_app_t.app_nums <2)
    sel=(sel==False)
    seg_app_t=seg_app_t[sel]
    sel=(seg_app_t.cat=='阅读')&(seg_app_t.app_nums <3)
    sel=(sel==False)
    seg_app_t=seg_app_t[sel]
    sel=(seg_app_t.cat=='运动综合')&(seg_app_t.app_nums <2)
    sel=(sel==False)
    seg_app_t=seg_app_t[sel]
    sel=(seg_app_t.cat=='音乐')&(seg_app_t.app_nums <3)
    sel=(sel==False)
    seg_app_t=seg_app_t[sel]
    seg_app_t['cat']=seg_app_t.cat.apply(lambda a:'旅游综合' if a == '户外探险' else a,)
    seg_app_t=seg_app_t[['app_nums','cat','ecif_id']]
    seg_app_t.app_nums=1
    seg_app_t=seg_app_t.drop_duplicates()
    return seg_app_t
def process_yl(seg_yl_t):
    sel=(seg_yl_t.cat=='珠宝首饰')&(seg_yl_t.amount <50000)
    sel=(sel==False)
    seg_yl_t=seg_yl_t[sel]
    sel=(seg_yl_t.cat=='汽车游艇')&(seg_yl_t.amount <400000)
    sel=(sel==False)
    seg_yl_t=seg_yl_t[sel]
    sel=(seg_yl_t.cat=='医疗体检')&(seg_yl_t.counts <3)
    sel=(sel==False)
    seg_yl_t=seg_yl_t[sel]
    sel=(seg_yl_t.cat=='运动综合')&(seg_yl_t.amount <7000)
    sel=(sel==False)
    seg_yl_t=seg_yl_t[sel]
    sel=(seg_yl_t.cat=='音乐')
    sel=(sel==False)
    seg_yl_t=seg_yl_t[sel]
    sel=(seg_yl_t.cat=='旅游综合')&(seg_yl_t.counts <2)
    sel=(sel==False)
    seg_yl_t=seg_yl_t[sel]
    sel=(seg_yl_t.cat=='美食发现')
    sel=(sel==False)
    seg_yl_t=seg_yl_t[sel]
    sel=(seg_yl_t.cat=='保险')&(seg_yl_t.amount/seg_yl_t.counts <3000)
    sel=(sel==False)
    seg_yl_t=seg_yl_t[sel]
    return seg_yl_t[['ecif_id','amount','cat']]
def load_hobby(hobbydata):
    hobbylist = []
    hobbydata1 = hobbydata.dropna(subset = ['t_customer_hobby_desc'])
    hobbydata1 = hobbydata1.reset_index(drop=True)
    for i in range(len(hobbydata1)):
        ecif_id =  int(hobbydata1.loc[i,['ecif_id']].values[0])
        t_customer_hobby_desc =   hobbydata1.loc[i,['t_customer_hobby_desc']]
        s = str(t_customer_hobby_desc.values[0]).split('|')
        for j in s:
            act_hob = j
            data = (ecif_id, j)
            hobbylist.append(data)
    hobby = pd.DataFrame(hobbylist, columns=['ecif_id', 'act_hob'])
    hobbyCat = pd.read_csv(DATA_PATH+'act_hob_cat.csv')
    hobbyCat.columns = ['act_hob','cat']
    hobby = hobby.merge(hobbyCat,how = 'left',on = 'act_hob')
    hobby.columns = ['ecif_id','hobby','cat']
    hobby = hobby.dropna(subset = ['cat'])
    return hobby
def load_activity(activity):
    activity1 = activity[['ecif_id','category']].dropna(subset = ['category'])
    activity1 = activity1.reset_index(drop=True)
    activity1.columns = ['ecif_id', 'act_hob']
    actCat = pd.read_csv(DATA_PATH+'act_hob_cat.csv')
    actCat.columns = ['act_hob','cat']
    activity1 = activity1.merge(actCat,how = 'left',on = 'act_hob')
    activity1.columns = ['ecif_id','hobby','cat']
    activity1 = activity1.dropna(subset = ['cat'])
    return activity1

#取源数据
age = getAget()
degree = getDegree()
sample = getsample()
yl_pd = load_yl()
wifi_pd=load_wifi()
hobbydata = gethobby()
activitydata = getactivity()
yl_pd['ecif_id']=yl_pd.ecif_id.astype(np.int64)

seg_app_t=load_app()
seg_app=process_app(seg_app_t)
seg_wifi=load_xianxia(wifi_pd)
seg_yl_t=load_yyl(sample,yl_pd)
seg_yl=process_yl(seg_yl_t)
seg_hobby = load_hobby(hobbydata)
seg_activity = load_activity(activity =activitydata)
seg_app.columns=['nums','cat','ecif_id']
seg_wifi.columns=['ecif_id','nums','cat']
seg_yl.columns=['ecif_id','nums','cat']


#数据汇总得到sampleAll
def mergeData(sample,age,degree,seg_app,seg_wifi,seg_yl,seg_hobby,seg_activity):
    sample1 = pd.merge(sample, age[['ecif_id', 'age']].dropna().drop_duplicates(), how='left', on='ecif_id')
    sample1 = pd.merge(sample1, degree[['ecif_id', 'educationDegree']].dropna().drop_duplicates(), how='left',on='ecif_id')
    ecifidList = []
    ageList = []
    sample1 = sample1.reset_index(drop=False)
    for i in range(len(sample1) ):
        ecif_id = sample1.ix[i,'ecif_id']
        if  sample1.ix[i,'age_y'] <= 40 :
            age_range = '-40'
        elif sample1.ix[i,'age_y'] <= 50 and  sample1.ix[i,'age_y']>40   :
            age_range = '40-50'
        elif sample1.ix[i,'age_y'] <= 60 and  sample1.ix[i,'age_y']>50   :
            age_range = '50-60'
        elif sample1.ix[i,'age_y']>60 :
            age_range = '60-'
        else:
            age_range = 'Unknown'
        ecifidList.append(ecif_id)
        ageList.append(age_range)
    ageRange =  pd.DataFrame({'ecif_id':ecifidList, 'age_range':ageList})
    sample2 = pd.merge(sample1,ageRange,how = 'left' ,on = 'ecif_id')
    sample2 = sample2.drop(['age_x','age_y'],1)
    ecifidList = []
    EOPList = []
    sample2 = sample2.reset_index(drop=False)
    for i in range(len(sample2)):
        ecif_id = sample2.ix[i, 'ecif_id']
        if sample2.ix[i, 'EOP'] < 6000000:
            eop_range = '<600万'
        elif sample2.ix[i, 'EOP'] <= 10000000 and sample2.ix[i, 'EOP'] >= 6000000:
            eop_range = '600-1000万'
        elif sample2.ix[i, 'EOP'] > 1000000:
            eop_range = '大于1000万'
        else:
            eop_range = 'Unknown'
        ecifidList.append(ecif_id)
        EOPList.append(eop_range)
    eop_range =  pd.DataFrame({'ecif_id': ecifidList, 'eop_range': EOPList})
    sample2 = pd.merge(sample2, eop_range, how='left', on='ecif_id')
    ecifidList = []
    degreeList = []
    for i in range(len(sample2)):
        ecif_id = sample2.ix[i, 'ecif_id']
        if sample2.ix[i, 'educationDegree'] == '专科' or sample2.ix[i, 'educationDegree'] == '专科(高职)':
            degree = '专科'
        elif sample2.ix[i, 'educationDegree'] == '硕士研究生' or sample2.ix[i, 'educationDegree'] == '博士研究生':
            degree = '硕博'
        elif sample2.ix[i, 'educationDegree'] == '本科':
            degree = '本科'
        else:
            degree = 'Unknown'
        ecifidList.append(ecif_id)
        degreeList.append(degree)
    degree = pd.DataFrame({'ecif_id': ecifidList, 'degree': degreeList})
    sample2 = pd.merge(sample2, degree, how='left', on='ecif_id')

    seg_app1 = pd.DataFrame(seg_app[['ecif_id','cat']].append(seg_wifi[['ecif_id','cat']]).drop_duplicates(),columns =['ecif_id','cat'])
    seg_yl1 = pd.DataFrame(seg_yl[['ecif_id','cat']].drop_duplicates(),columns =['ecif_id','cat'])
    seg_yl1['ecif_id'] = seg_yl1['ecif_id'].astype('int64')
    seg_hobby1 =  pd.DataFrame(seg_hobby[['ecif_id','cat']].drop_duplicates(),columns =['ecif_id','cat'])
    seg_activity1 =  pd.DataFrame(seg_activity[['ecif_id','cat']].drop_duplicates(),columns =['ecif_id','cat'])
    seg_cat = pd.read_csv(DATA_PATH + 'seg_cat.csv')
    sampleApp = pd.merge(sample2, seg_app1, how='inner', on='ecif_id')
    sampleApp = pd.merge(sampleApp, seg_cat, how='left', left_on='cat', right_on='cat2')
    sampleApp['sources'] = '发现者'
    sampleHob = pd.merge(sample2, seg_hobby1, how='inner', on='ecif_id')
    sampleHob = pd.merge(sampleHob, seg_cat, how='left', left_on='cat', right_on='cat2')
    sampleHob['sources'] = '内部'
    sampleActi = pd.merge(sample2, seg_activity1, how='inner', on='ecif_id')
    sampleActi = pd.merge(sampleActi, seg_cat, how='left', left_on='cat', right_on='cat2')
    sampleActi['sources'] = '内部'
    sampleYl1 = pd.merge(sample2, seg_yl1, how='inner', on='ecif_id')
    sampleYl1 = pd.merge(sampleYl1, seg_cat, how='left', left_on='cat', right_on='cat2')
    sampleYl2 = sampleYl1[(sampleYl1.cat == '美容SPA') & (sampleYl1.sex == '男')]
    sampleYl2 = sampleYl2.drop(['cat', 'cat1', 'cat2'], 1)
    sampleYl2['cat'] = '保健养生'
    sampleYl2['cat1'] = '养生/美容'
    sampleYl2['cat2'] = '保健养生'
    sampleYl = pd.concat([sampleYl1[(sampleYl1.cat != '美容SPA') | (sampleYl1.sex != '男')], sampleYl2])
    sampleYl['sources'] = '银联'
    product = pd.read_csv(DATA_PATH + 'product.csv')
    product.columns = ['ecif_id', 'product']
    # product.columns = ['ecif_id', 'cat']
    product['ecif_id'] = product.ecif_id.astype(np.int64)
    frames = [sampleApp, sampleYl, sampleHob, sampleActi]
    sampleAllTemp = pd.concat(frames)
    # sampleAllTemp = sampleAllTemp.drop(['cat', 'cat1', 'cat2'], 1)
    # sampleProduct = pd.merge(sampleAllTemp.drop_duplicates(), product, how='inner', on='ecif_id')
    # sampleProduct = pd.merge(sampleProduct, seg_cat, how='left', left_on='cat', right_on='cat2')
    # sampleProduct['sources'] = '产品'
    # frames = [sampleApp, sampleYl, sampleHob, sampleActi,sampleProduct]
    # sampleAll = pd.concat(frames)
    sampleAll = pd.merge(sampleAllTemp,product,how = 'left',on = 'ecif_id')
    sampleAll = sampleAll.fillna('Unknown')
    return sampleAll
sampleAll = mergeData(sample,age,degree,seg_app,seg_wifi,seg_yl,seg_hobby,seg_activity)

sampleAll.ecif_id.nunique()
sampleAll.cat1.unique()
sampleAll[['ecif_id','cat']]

sampleAll.columns

CAT1_PATH = "F:/DataAna/cf_all/result/CROSS1/"
CAT2_PATH ="F:/DataAna/cf_all/result/CROSS2/"
#大类交叉分析
def getCrossCat1(sampleAll,colList,sourceList,CAT1_PATH,dataFlag):
    if dataFlag == '':
        sampleAll1 = sampleAll
    else:
        sampleAll1 = sampleAll[sampleAll.flag == dataFlag ]
    for col in colList:
        collist = []
        collistdata = sampleAll1[[col]].dropna().drop_duplicates()
        collistdata = collistdata[collistdata[col] != 'Unknown']
        collistdata = collistdata.reset_index(drop = True)
        for i in range(len(collistdata)):
            addlist = collistdata.loc[i, [col]].values[0]
            collist.append(addlist)
        collist.sort(reverse = True)
        collist.insert(0,col )
        collist.append('source')
        dataSubAll = pd.DataFrame(columns = ['cat1','source','allCnt'])
        dataFinal = pd.DataFrame(columns = collist)

        for source in sourceList:
            if source == '':
                data = sampleAll1[sampleAll1[col] != 'Unknown']
                datasource = 'ALL'
                dataSubt = data.groupby(['cat1']).ecif_id.nunique()
                dataSubt = dataSubt.reset_index(drop=False)
                dataSubt['source'] = datasource
                dataSubt.columns = ['cat1','allCnt','source']
                sustcCntAll = data.ecif_id.nunique()
                list = [('ALL', datasource, sustcCntAll)]
                dataSourceAll = pd.DataFrame(list, columns=['cat1', 'source', 'allCnt'])
            else:
                data = sampleAll1[(sampleAll1.sources == source)& (sampleAll1[col] != 'Unknown')].dropna(subset = [col])
                datasource = source
                dataSubt = data.groupby(['cat1', 'sources']).ecif_id.nunique()
                dataSubt = dataSubt.reset_index(drop=False)
                dataSubt.columns = ['cat1', 'source', 'allCnt']
                sustcCntAll = data.ecif_id.nunique()
                list = [('ALL', datasource, sustcCntAll)]
                dataSourceAll = pd.DataFrame(list, columns=['cat1', 'source', 'allCnt'])
            concatData = [dataSubAll,dataSubt,dataSourceAll]
            dataSubAll = pd.concat(concatData)
            dataSubAll = dataSubAll.drop_duplicates()

            sourceSub = data.groupby(col).ecif_id.nunique()
            sourceSub = sourceSub.reset_index(drop = False)
            sourceSub.columns = [col,'ALL']
            sourceSub = sourceSub.set_index(col)
            sourceSub = sourceSub.T
            sourceSub = sourceSub.reset_index(drop = False)
            sourceSub['source'] = datasource
            sourceSub.rename(columns={'index': 'cat1'}, inplace=True)
            dataFinal = pd.concat([dataFinal, sourceSub])

            dfR = data[[col]].drop_duplicates()
            for cat1 in data[data.cat1 != 'Unknown'].cat1.unique():
                datasub = data[data.cat1 ==cat1] .groupby([col]).ecif_id.nunique()
                datasub = datasub.reset_index(drop = False)
                datasub.columns = [col,cat1]
                dfR = pd.merge(dfR, datasub, how='left', on=col)
            dfR = dfR.set_index(col)
            dfF = dfR.T.fillna(0)
            dfF = dfF.reset_index(drop = False)
            dfF['source'] = datasource
            dfF.rename(columns={'index': 'cat1'}, inplace=True)
            dataFinal =  pd.concat([dataFinal,dfF])
        dataFinal.rename(columns = {'index':'cat1'},inplace =  True)
        dataFinal = pd.merge(dataFinal,dataSubAll,how = 'left' ,on = ['cat1','source'])
        colnames = []
        newColNmes = []
        for colname in  data[col].unique():
            print colname
            newColNme = str(colname )+'占比'
            dataFinal[newColNme] = dataFinal[colname]/dataFinal['allCnt']
            colnames.append(colname)
            newColNmes.append(newColNme)
        colnames.insert(0, col)
        colnames.insert(0, 'source')
        colnames.append('cat1')
        colnames.append('allCnt')
        colnames = colnames + newColNmes
        dataFinal = dataFinal.fillna(0)
        dataFinal = dataFinal[colnames]
        runDate = time.strftime('%Y%m%d', time.localtime(time.time()))
        dataFinal.to_csv(CAT1_PATH + 'cat1_%s_sub_%s_%s.csv'.decode('utf-8') % (col,dataFlag, runDate), encoding='gbk')

#小类交叉分析
def getCrossCat2(sampleAll,colList,sourceList,CAT2_PATH,dataFlag):
    if dataFlag == '':
        sampleAll1 = sampleAll
    else:
        sampleAll1 = sampleAll[sampleAll.flag == dataFlag ]
    for col in colList:
        collist = []
        collistdata = sampleAll1[[col]].dropna().drop_duplicates()
        collistdata = collistdata[collistdata[col] != 'Unknown']
        collistdata = collistdata.reset_index(drop = True)
        for i in range(len(collistdata)):
            addlist = collistdata.loc[i, [col]].values[0]
            collist.append(addlist)
        collist.sort(reverse = True)
        collist.insert(0,col )
        collist.append('source')
        dataSubAll = pd.DataFrame(columns = ['cat2','source','allCnt'])
        dataFinal = pd.DataFrame(columns = collist)

        for source in sourceList:
            if source == '':
                data = sampleAll1[sampleAll1[col] != 'Unknown']
                datasource = 'ALL'
                dataSubt = data.groupby(['cat2']).ecif_id.nunique()
                dataSubt = dataSubt.reset_index(drop=False)
                dataSubt['source'] = datasource
                dataSubt.columns = ['cat2','allCnt','source']
                sustcCntAll = data.ecif_id.nunique()
                list = [('ALL', datasource, sustcCntAll)]
                dataSourceAll = pd.DataFrame(list, columns=['cat2', 'source', 'allCnt'])
            else:
                data = sampleAll1[(sampleAll1.sources == source)& (sampleAll1[col] != 'Unknown')].dropna(subset = [col])
                datasource = source
                dataSubt = data.groupby(['cat2', 'sources']).ecif_id.nunique()
                dataSubt = dataSubt.reset_index(drop=False)
                dataSubt.columns = ['cat2', 'source', 'allCnt']
                sustcCntAll = data.ecif_id.nunique()
                list = [('ALL', datasource, sustcCntAll)]
                dataSourceAll = pd.DataFrame(list, columns=['cat2', 'source', 'allCnt'])
            concatData = [dataSubAll,dataSubt,dataSourceAll]
            dataSubAll = pd.concat(concatData)
            dataSubAll = dataSubAll.drop_duplicates()

            sourceSub = data.groupby(col).ecif_id.nunique()
            sourceSub = sourceSub.reset_index(drop = False)
            sourceSub.columns = [col,'ALL']
            sourceSub = sourceSub.set_index(col)
            sourceSub = sourceSub.T
            sourceSub = sourceSub.reset_index(drop = False)
            sourceSub['source'] = datasource
            sourceSub.rename(columns={'index': 'cat2'}, inplace=True)
            dataFinal = pd.concat([dataFinal, sourceSub])

            dfR = data[[col]].drop_duplicates()
            for cat2 in data[data.cat2 != 'Unknown'].cat2.unique():
                datasub = data[data.cat2 ==cat2] .groupby([col]).ecif_id.nunique()
                datasub = datasub.reset_index(drop = False)
                datasub.columns = [col,cat2]
                dfR = pd.merge(dfR, datasub, how='left', on=col)
            dfR = dfR.set_index(col)
            dfF = dfR.T.fillna(0)
            dfF = dfF.reset_index(drop = False)
            dfF['source'] = datasource
            dfF.rename(columns={'index': 'cat2'}, inplace=True)
            dataFinal =  pd.concat([dataFinal,dfF])
        dataFinal.rename(columns = {'index':'cat2'},inplace =  True)
        dataFinal = pd.merge(dataFinal,dataSubAll,how = 'left' ,on = ['cat2','source'])
        colnames = []
        newColNmes = []
        for colname in  data[col].unique():
            print colname
            newColNme = str(colname )+'占比'
            dataFinal[newColNme] = dataFinal[colname]/dataFinal['allCnt']
            colnames.append(colname)
            newColNmes.append(newColNme)
        colnames.insert(0, col)
        colnames.insert(0, 'source')
        colnames.append('cat2')
        colnames.append('allCnt')
        colnames = colnames + newColNmes
        dataFinal = dataFinal.fillna(0)
        dataFinal = dataFinal[colnames]
        runDate = time.strftime('%Y%m%d', time.localtime(time.time()))
        dataFinal.to_csv(CAT2_PATH + 'cat2_%s_sub_%s_%s.csv'.decode('utf-8') % (col,dataFlag, runDate), encoding='gbk')

dataFinal1 = getCrossCat1(sampleAll = sampleAll,colList = ('age_range', 'sex', 'degree', 'flag','eop_range'),sourceList = ('','发现者', '银联', '内部','产品'),CAT1_PATH = CAT1_PATH,dataFlag = '')
dataFinal2 = getCrossCat2(sampleAll = sampleAll,colList = ('age_range', 'sex', 'degree', 'flag','eop_range'),sourceList = ('','发现者', '银联', '内部','产品'),CAT2_PATH = CAT2_PATH,dataFlag = '')
dataFinal1 = getCrossCat1(sampleAll = sampleAll,colList = ('product',),sourceList = ('','发现者', '银联', '内部'),CAT1_PATH = CAT1_PATH,dataFlag = '')
dataFinal2 = getCrossCat2(sampleAll = sampleAll,colList = ('product',),sourceList = ('','发现者', '银联', '内部'),CAT2_PATH = CAT2_PATH,dataFlag = '')

#大类小类人数统计
sampleAll.groupby(sampleAll['cat1']).ecif_id.nunique()
sampleAll.groupby(sampleAll['product']).ecif_id.nunique()
sampleAll[sampleAll.cat1!= 'Unknown'].ecif_id.nunique()
sampleAll.groupby(sampleAll['cat2']).ecif_id.nunique()
sampleAll.groupby(sampleAll['product']).ecif_id.nunique()
sampleAll[sampleAll.cat2!= 'Unknown'].ecif_id.nunique()

# test = sampleAll.groupby(['sources']).cat2.unique()
# test = test.reset_index(drop = False)
# test.to_csv('test.csv',encoding = 'gbk')

SamSubPATH="F:/DataAna/cf_all/result/sample/"
#兴趣偏好统计
def catSub(sampleAll):
    cat1Sub = pd.DataFrame(columns=['source', 'cat1', u'人数'])
    cat2Sub = pd.DataFrame(columns=['source', 'cat1','cat2', u'人数'])
    for i in  ('','发现者','内部','银联'):
        if i == '':
            source = 'ALL'
            data = sampleAll[sampleAll.cat1 != 'Unknown']
        else:
            source = i
            data = sampleAll[(sampleAll.cat1 != 'Unknown')&(sampleAll.sources == i)]
        cat1Sub1 = data.groupby(['cat1']).ecif_id.nunique()
        cat1Sub1list = []
        for i in range(len(cat1Sub1)):
            cat1 = cat1Sub1.index[i]
            cnt = cat1Sub1.values[i]
            data1 = (source,cat1,cnt)
            cat1Sub1list.append(data1)
        cat1Subt = pd.DataFrame(cat1Sub1list,columns = ['source','cat1',u'人数'])
        cat2Sub1 = data.groupby(['cat1','cat2']).ecif_id.nunique()
        cat2Sub1list = []
        for i in range(len(cat2Sub1)):
            cat1 = cat2Sub1.index[i][0]
            cat2 = cat2Sub1.index[i][1]
            cnt = cat2Sub1.values[i]
            data2 = (source, cat1,cat2, cnt)
            cat2Sub1list.append(data2)
        cat2Subt = pd.DataFrame(cat2Sub1list, columns=['source', 'cat1','cat2', u'人数'])
        cat1Sub = pd.concat([cat1Sub,cat1Subt])
        cat2Sub = pd.concat([cat2Sub,cat2Subt])
    cat1Sub.to_csv(SamSubPATH+'cat1Sub.csv',encoding = 'gbk')
    cat2Sub.to_csv(SamSubPATH+'cat2Sub.csv', encoding='gbk')
catSub(sampleAll = sampleSeg)
# 人口属性统计
def sampleDataSub(sample2,sampleAll):
    for i in  ('age_range', 'sex', 'educationDegree', 'flag','eop_range'):
        dataf = sample2.groupby([i]).ecif_id.nunique()
        dataf.to_csv(SamSubPATH+'all_%s_sub.csv'%i,encoding = 'gbk')
    for i in  ('age_range', 'sex', 'educationDegree', 'flag','eop_range'):
        dataf = sampleAll.groupby([i]).ecif_id.nunique()
        dataf.to_csv(SamSubPATH+'sample_%s_sub.csv'%i,encoding = 'gbk')
sampleDataSub(sample2,sampleAll = sampleSeg)

# 数据覆盖统计
def dataCover():
    ecifidList = []
    is_fxz_list = []
    is_yl_list = []
    is_nb_list = []
    sampleAll = sampleAll.reset_index(drop = True)
    for i in range(len(sampleAll)):
        ecif_id = sampleAll.ix[i, 'ecif_id']
        if sampleAll.ix[i, 'sources'] == '发现者':
            is_fxz = 1
        else :
            is_fxz = 0
        if sampleAll.ix[i, 'sources'] == '银联':
            is_yl = 1
        else :
            is_yl = 0
        if sampleAll.ix[i, 'sources'] == '内部':
            is_nb = 1
        else :
            is_nb = 0
        ecifidList.append(ecif_id)
        is_fxz_list.append(is_fxz)
        is_yl_list.append(is_yl)
        is_nb_list.append(is_nb)

    samplef = sampleAll[['ecif_id']].drop_duplicates()
    fxzs = sampleAll[sampleAll['is_fxz'] == 1][['ecif_id','is_fxz']].drop_duplicates()
    yls = sampleAll[sampleAll['is_yl'] == 1][['ecif_id','is_yl']].drop_duplicates()
    nbs = sampleAll[sampleAll['is_nb'] == 1][['ecif_id','is_nb']].drop_duplicates()
    for i in (fxzs,yls,nbs):
        samplef = pd.merge(samplef,i,how = 'left',on = 'ecif_id')
    samplef = samplef.fillna(0)
    samplef.groupby(['is_fxz','is_yl','is_nb']).ecif_id.nunique()



# 小类在大类中占比
catPerSubPATH="F:/DataAna/cf_all/result/catPerSub/"
colList = ('age_range', 'sex', 'degree', 'flag','eop_range')
def catPerSub(sampleAll,colList):
    catsub  = sampleAll[sampleAll.cat1 != 'Unknown'].groupby(['cat1','cat2']).ecif_id.nunique().reset_index(drop = False)
    cat1sub  = sampleAll.groupby(['cat1']).ecif_id.nunique().reset_index(drop = False)
    catsub = pd.merge(catsub,cat1sub,how = 'left',on = 'cat1')
    allCnt = sampleAll[sampleAll.cat1 != 'Unknown'].ecif_id.nunique()
    catsub['allCnt'] = allCnt
    catsub['catPer'] = catsub['ecif_id_x'] /catsub['ecif_id_y']
    catsub['allPer'] = catsub['ecif_id_x'] /catsub['allCnt']
    catsub.columns = ['cat1','cat2',u'人数',u'大类人数',u'总人数',u'大类中占比',u'总量中占比']
    catsub.to_csv(catPerSubPATH+'catsub.csv',encoding = 'gbk')
    for col in colList:
        catsub = sampleAll[(sampleAll[col] !='Unknown')&(sampleAll['cat1'] !='Unknown')].groupby(['cat1', 'cat2',col]).ecif_id.nunique().reset_index(drop=False)
        catsub1 = sampleAll[sampleAll[col] !='Unknown'].groupby(['cat1', col]).ecif_id.nunique().reset_index(drop=False)
        catsub2 = sampleAll[sampleAll[col] !='Unknown'].groupby(['cat1']).ecif_id.nunique().reset_index(drop=False)
        catsub = pd.merge(catsub,catsub1,how = 'left',on = ['cat1',col])
        catsub = pd.merge(catsub,catsub2,how = 'left',on = ['cat1'])
        catsub['allCnt'] = allCnt

        catsub['cat1PoPer'] = catsub['ecif_id_x'] / catsub['ecif_id_y']
        catsub['cat1Per'] = catsub['ecif_id_x'] / catsub['ecif_id']
        catsub['allPer'] = catsub['ecif_id_x'] / allCnt
        catsub.columns = ['cat1','cat2',col,u'人数',u'大类&%s人数'%col,u'大类人数',u'总人数',u'大类&%s中占比'%col,u'大类中占比',u'总量中占比']
        catsub.to_csv(catPerSubPATH+'catsub_%s.csv'%col,encoding = 'gbk')
catPerSub(sampleAll,colList)

#朱澄后续需求
def zhucheng():
    for i in  ('','发现者','内部','银联'):
        allCnt = sampleAll[sampleAll.cat1 != 'Unknown'].ecif_id.nunique()
        if i == '':
            source = 'ALL'
            data = sampleAll[sampleAll.cat1 != 'Unknown']
            cat1Sub1 = data.groupby(['cat1']).ecif_id.nunique()
            cat1Sub1list = []
            for i in range(len(cat1Sub1)):
                cat1 = cat1Sub1.index[i]
                cnt = cat1Sub1.values[i]
                data1 = ( cat1, cnt)
                cat1Sub1list.append(data1)
            cat1Subt = pd.DataFrame(cat1Sub1list, columns=[ 'cat1', u'cat1人数'])
            cat2Sub1 = data.groupby(['cat1', 'cat2']).ecif_id.nunique()
            cat2Sub1list = []
            for i in range(len(cat2Sub1)):
                cat1 = cat2Sub1.index[i][0]
                cat2 = cat2Sub1.index[i][1]
                cnt = cat2Sub1.values[i]
                data2 = ( cat1, cat2, cnt)
                cat2Sub1list.append(data2)
            cat2Subt = pd.DataFrame(cat2Sub1list, columns=['cat1', 'cat2', '%scat2人数' % source])
            cat2Sub = pd.merge(cat2Subt,cat1Subt,how = 'left',on = 'cat1')
            cat2Sub['%scat2Per'%source] = cat2Sub[ '%scat2人数'%source] / allCnt
        else:
            source = i
            data = sampleAll[(sampleAll.cat1 != 'Unknown')&(sampleAll.sources == i)]
            cat2Sub1 = data.groupby(['cat1','cat2']).ecif_id.nunique()
            cat2Sub1list = []
            for i in range(len(cat2Sub1)):
                cat1 = cat2Sub1.index[i][0]
                cat2 = cat2Sub1.index[i][1]
                cnt = cat2Sub1.values[i]
                data2 = ( cat1,cat2, cnt)
                cat2Sub1list.append(data2)
            cat2Subt = pd.DataFrame(cat2Sub1list, columns=[ 'cat1','cat2', '%scat2人数'%source])
            cat2Sub = pd.merge(cat2Sub, cat2Subt, how='left', on=['cat1','cat2'])
            cat2Sub['%scat2Per'%source] = cat2Sub[ '%scat2人数'%source] / allCnt
    cat2Sub.fillna(0)
    cat2Sub = cat2Sub.sort_values(by =u'cat1人数',ascending  = False)
    cat2Sub.set_index('cat2')
    cat2Sub.to_csv(catPerSubPATH+'ZHUCHENG.CSV',encoding = 'gbk')


def haoshijie():
    samplet = sample[['ecif_id','age','sex','EOP','flag']]
    samplet = samplet.dropna()
    samplet.ecif_id.nunique()

    sample1 = sampleAll[sampleAll.sources == '内部'][['ecif_id','sources']].drop_duplicates()

    samplef =    pd.merge(samplet[['ecif_id','flag']].drop_duplicates(),sample1[['ecif_id','sources']].drop_duplicates(),how = 'left',on = 'ecif_id')
    samplef[samplef.sources != '内部'].groupby(samplef.flag).ecif_id.nunique()






