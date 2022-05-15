import pandas as pd
import pickle
from collections import Counter

train = pd.read_csv('/home/johnso/PycharmProjects/News_recommendation/CTR_prediction/avazu_CTR/train.csv',chunksize=20000)
test = pd.read_csv('/home/johnso/PycharmProjects/News_recommendation/CTR_prediction/avazu_CTR/test.csv',chunksize=20000)

C14 = {}
C17 = {}
C19 = {}
C21 = {}
site_id = {}
site_domain = {}
app_id = {}
app_domain = {}
device_model = {}
device_id = {}
device_ip = {}

for count, data in enumerate(train, start=1):
    C14_list = data['C14'].values
    for k,v in Counter(C14_list).items():
        if k in C14:
            C14[k] += v
        else:
            C14[k] = v

    C17_list = data['C17'].values
    for k,v in Counter(C17_list).items():
        if k in C17:
            C17[k] += v
        else:
            C17[k] = v

    C19_list = data['C19'].values
    for k,v in Counter(C19_list).items():
        if k in C19:
            C19[k] += v
        else:
            C19[k] = v

    C21_list = data['C21'].values
    for k,v in Counter(C21_list).items():
        if k in C21:
            C21[k] += v
        else:
            C21[k] = v

    site_id_list = data['site_id'].values
    for k,v in Counter(site_id_list).items():
        if k in site_id:
            site_id[k] += v
        else:
            site_id[k] = v

    site_domain_list = data['site_domain'].values
    for k,v in Counter(site_domain_list).items():
        if k in site_domain:
            site_domain[k] += v
        else:
            site_domain[k] = v

    app_id_list = data['app_id'].values
    for k,v in Counter(app_id_list).items():
        if k in app_id:
            app_id[k] += v
        else:
            app_id[k] = v

    app_domain_list = data['app_domain'].values
    for k,v in Counter(app_domain_list).items():
        if k in app_domain:
            app_domain[k] += v
        else:
            app_domain[k] = v

    device_model_list = data['device_model'].values
    for k,v in Counter(device_model_list).items():
        if k in device_model:
            device_model[k] += v
        else:
            device_model[k] = v

    device_id_list = data['device_id'].values
    for k,v in Counter(device_id_list).items():
        if k in device_id:
            device_id[k] += v
        else:
            device_id[k] = v

    device_ip_list = data['device_ip'].values
    for k,v in Counter(device_ip_list).items():
        if k in device_ip:
            device_ip[k] += v
        else:
            device_ip[k] = v

    if count % 100 == 0:
        print(f'{count} has finished')


for data in test:
    C14_list = data['C14'].values
    for k,v in Counter(C14_list).items():
        if k in C14:
            C14[k] += v
        else:
            C14[k] = v

    C17_list = data['C17'].values
    for k,v in Counter(C17_list).items():
        if k in C17:
            C17[k] += v
        else:
            C17[k] = v

    C19_list = data['C19'].values
    for k,v in Counter(C19_list).items():
        if k in C19:
            C19[k] += v
        else:
            C19[k] = v

    C21_list = data['C21'].values
    for k,v in Counter(C21_list).items():
        if k in C21:
            C21[k] += v
        else:
            C21[k] = v

    site_id_list = data['site_id'].values
    for k,v in Counter(site_id_list).items():
        if k in site_id:
            site_id[k] += v
        else:
            site_id[k] = v

    site_domain_list = data['site_domain'].values
    for k,v in Counter(site_domain_list).items():
        if k in site_domain:
            site_domain[k] += v
        else:
            site_domain[k] = v

    app_id_list = data['app_id'].values
    for k,v in Counter(app_id_list).items():
        if k in app_id:
            app_id[k] += v
        else:
            app_id[k] = v

    app_domain_list = data['app_domain'].values
    for k,v in Counter(app_domain_list).items():
        if k in app_domain:
            app_domain[k] += v
        else:
            app_domain[k] = v

    device_model_list = data['device_model'].values
    for k,v in Counter(device_model_list).items():
        if k in device_model:
            device_model[k] += v
        else:
            device_model[k] = v

    device_id_list = data['device_id'].values
    for k,v in Counter(device_id_list).items():
        if k in device_id:
            device_id[k] += v
        else:
            device_id[k] = v

    device_ip_list = data['device_ip'].values
    for k,v in Counter(device_ip_list).items():
        if k in device_ip:
            device_ip[k] += v
        else:
            device_ip[k] = v


# save dictionaries
with open('field2count/C14.pkl','wb') as f:
    pickle.dump(C14,f)

with open('field2count/C17.pkl','wb') as f:
    pickle.dump(C17,f)

with open('field2count/C19.pkl','wb') as f:
    pickle.dump(C19,f)

with open('field2count/C21.pkl','wb') as f:
    pickle.dump(C21,f)

with open('field2count/site_id.pkl','wb') as f:
    pickle.dump(site_id,f)

with open('field2count/site_domain.pkl','wb') as f:
    pickle.dump(site_domain,f)

with open('field2count/app_id.pkl','wb') as f:
    pickle.dump(app_id,f)

with open('field2count/app_domain.pkl','wb') as f:
    pickle.dump(app_domain,f)

with open('field2count/device_model.pkl','wb') as f:
    pickle.dump(device_model,f)

with open('field2count/device_id.pkl','wb') as f:
    pickle.dump(device_id,f)

with open('field2count/device_ip.pkl','wb') as f:
    pickle.dump(device_ip,f)