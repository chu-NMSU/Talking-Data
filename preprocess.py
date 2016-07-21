import pandas as pd
import numpy as np
import pinyin
import re
import gc
import time
import matplotlib
import matplotlib.pyplot as plt

# pd.set_option("display.max_rows",101) # set # of display rows
pattern = re.compile('[a-fA-Z0-9_ ]+')

def text_process_fun(x):
    # if re.match(pattern, x):
    #     return x
    # else:
    return pinyin.get(x, format='strip', delimiter=' ').replace(' ', '')

start_time = time.time()
# timestamp: 2016-05-01 00:55:25
events = pd.read_csv('data/events.csv', dtype={'device_id':str})
d_times = pd.DatetimeIndex(events['timestamp'].apply(\
        lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')))
events['dayofyear'] = d_times.dayofyear
events['hour'] = d_times.hour
events['weekday'] = d_times.weekday
print 'before pruning data, events.shape=', events.shape

# # only use phone China to train model
# lon_min, lon_max = 75, 135
# lat_min, lat_max = 15, 55
# idx_china = (events["longitude"]>lon_min) &\
#             (events["longitude"]<lon_max) &\
#             (events["latitude"]>lat_min) &\
#             (events["latitude"]<lat_max)
# events = events[idx_china]
# print 'after removing phones outside China, events.shape=', events.shape

app_events = pd.read_csv('data/app_events.csv', dtype={'app_id':str})
app_labels = pd.read_csv('data/app_labels.csv', dtype={'app_id':str})
# TODO we may merge similary category later
label_cate = pd.read_csv('data/label_categories.csv', dtype={'app_id':str})
phone = pd.read_csv('data/phone_brand_device_model.csv', \
        dtype={'device_id':str, 'phone_brand':str, 'device_model':str})
phone = phone.drop_duplicates(subset=['device_id']) # remove duplicates devices
df_train = pd.read_csv('data/gender_age_train.csv', dtype={'device_id':str})
df_test = pd.read_csv('data/gender_age_test.csv', dtype={'device_id':str})
phone['phone_brand_en'] = phone['phone_brand'].apply(text_process_fun)
phone['device_model_en'] = phone['device_model'].apply(text_process_fun)
print 'reading time=', time.time()-start_time

start_time = time.time()
'''aggregate app labels'''
app = app_labels.merge(label_cate, how='left', left_index=True, on='label_id')
app_group_cate = app.groupby(by=['app_id'])['category'].apply(lambda x : ' '.join(x))
app_group_cate.rename('text', inplace=True)
app_group_cate.to_csv('data/app_text.csv', index=True, header=True)
app_text = pd.read_csv('data/app_text.csv', dtype={'app_id':str, 'text':str})

'''aggregate device labels'''
'''some phone does not have app installed. so use inner join'''
# In [78]: len(set(pd.unique(events.event_id)).difference(set(pd.unique(app_events.event_id))))
# Out[78]: 1764854
events_join = events.merge(app_events, on='event_id') #, how='left'
# one_phone = events_join[events_join['device_id']=='-6401643145415154744'] # study one phone
'''each time different apps are detected'''
# one_phone.groupby('event_id')['app_id'].apply(lambda x:x.unique().shape)
# one_phone.groupby('event_id')['is_active'].sum()
events_join_app_text = events_join.merge(app_text, on='app_id')
events_device_app_text = events_join_app_text.groupby('event_id')['text'].\
        apply(lambda x : ' '.join(x))
events_device_app_text.to_csv('data/event_text.csv', index=True, header=True)
events_text = pd.read_csv('data/event_text.csv', dtype={'text':str})
events_join = one_phone = events_join_app_text = events_device_app_text = None
app = app_group_cate = None
gc.collect()

print 'joining app, event time=', time.time()-start_time
'''some device in train/test set is not record in events table, use phone_brand'''
# In [1]: len(set(pd.unique(df_test.device_id)).difference(set(pd.unique(events.device_id))))
# Out[1]: 76877
# In [33]: df_test.shape
# Out[33]: (112071, 1)
# In [2]: len(set(pd.unique(df_train.device_id)).difference(set(pd.unique(events.device_id))))
# Out[2]: 51336
# In [32]: df_train.shape
# Out[32]: (74645, 4)
'''some device are not installed any apps'''
# In [27]: len(set(pd.unique(events.event_id)).difference(set(pd.unique(app_events.event_id))))
# Out[27]: 1764854

# ## joint data
start_time = time.time()
'''events with phone app label text'''
events_join = events.merge(events_text, on=['event_id']) # , how='left'
#### events_join['text'].fillna('', inplace=True)
# '''some devices in 'events' do not have record in 'phone' '''
# # In [27]: len(set(pd.unique(events.device_id)).difference(set(pd.unique(phone.device_id))))
# # Out[27]: 2362
events_phone_join = events_join.merge(phone, on=['device_id']) # , how='left'

# '''some device do not have records, use brand and model to fill the text'''
df_train_join = df_train.merge(events_phone_join, on=['device_id']) # , how='left'
df_train_join['text'].fillna('', inplace=True)
# a = df_train_join[['device_id']].merge(phone[['device_id','phone_brand_en','device_model_en']], \
#         on='device_id')
# a.sort_values(by='device_id', inplace=True)
# df_train_join.sort_values(by='device_id', inplace=True)
# df_train_join['phone_brand_en'] = a['phone_brand_en'].values
# df_train_join['device_model_en'] = a['device_model_en'].values
# df_train_join['text_brand'] = df_train_join['text'] + ' ' + df_train_join['phone_brand_en']\
#         + ' ' + df_train_join['device_model_en']

'''9/11 test case have missing values...'''
# df_test_join.shape =             (112071, 12)
# df_test_join.text.isnull().sum() = 95226 # 
df_test_join = df_test.merge(events_phone_join, on=['device_id'], how='left') # 
df_test_join.drop_duplicates(subset=['device_id'],inplace=True) # keep only one unique device_id
df_test_join['text'].fillna('', inplace=True)
a = df_test_join[['device_id']].merge(phone[['device_id','phone_brand_en','device_model_en']], \
        on='device_id')
a.sort_values(by='device_id', inplace=True)
df_test_join.sort_values(by='device_id', inplace=True)
df_test_join['phone_brand_en'] = a['phone_brand_en'].values
df_test_join['device_model_en'] = a['device_model_en'].values
# df_test_join['text_brand'] = df_test_join['text'] + ' ' + df_test_join['phone_brand_en']+\
#      ' '+ df_test_join['device_model_en']

'''some phone brands in test data even do not exist in training data...'''
# In [1]: df_test_join[~df_test_join.phone_brand_en.isin(df_train_join.phone_brand_en.unique())].shape
# Out[1]: (1318, 14)

print 'join time=', time.time()-start_time

# df_train_join.drop(['phone_brand', 'device_model'], axis=1, inplace=True)
# df_test_join.drop(['phone_brand', 'device_model'], axis=1, inplace=True)
df_train_join.to_csv('data/train_text.csv', index=False)
df_test_join.to_csv('data/test_text.csv', index=False)
