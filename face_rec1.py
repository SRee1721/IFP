import numpy as np
import pandas as pd
import cv2

import redis
#insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

#connect to Redis Client

hostname='redis-11495.c330.asia-south1-1.gce.redns.redis-cloud.com'
port=11495
password='TQs2mVZtzxOihgYMHPVe3GkcK2IIrCwR'

r = redis.StrictRedis(host=hostname,
                      port=port,
                      password=password)

#configure face analysis
faceapp=FaceAnalysis(name='buffalo_sc',root='insightface_model',providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0,det_size=(640,640),det_thresh=0.5)

#ML Search Algorithm

def ml_search_algorithm(dataframe,feature_column,test_vector,name_role=['NAME','ROLE'],thresh=0.5):
   
   



#step 1 Take the dataframe (collection of data)
    dataframe1=dataframe.copy()
   




# step 2 Index face embedding from the dataframe and convey into array
    X_list=dataframe1[feature_column].tolist()
    x=np.asarray(X_list)


#step 3 Calculate cosine similarity
    similar=pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    similar_arr=np.array(similar).flatten()
    dataframe1['cosine']=similar_arr
   
#step4: Filter the data
    data_filter=dataframe1.query(f'cosine >= {thresh}')
   
    if len(data_filter)>0:
        data_filter.reset_index(drop=True,inplace=True)
        argmax=data_filter['cosine'].argmax()
        name,role=data_filter.loc[argmax][name_role]
    else:
        name,role='UNKNOWN','UNKNOWN'
    return name,role


#step 5  Get the person name

def face_prediction(test_image,dataframe,feature_column,name_role=['NAME','ROLE'],thresh=0.5):
    # step 1 tke the test image and apply to insight face
    results=faceapp.get(test_image)
    test_copy=test_image.copy()
   
    # steep 2 use for loop and extract the embedding and pass to ml algorithm
    for res in results:
        x1,y1,x2,y2=res['bbox'].astype(int)
        embeddings=res['embedding']
        person_name,person_role=ml_search_algorithm(dataframe,feature_column,test_vector=embeddings,
                                                   name_role=name_role,thresh=thresh)
        if(person_name=='UNKNOWN'):
            color=(0,0,255)
        else:
            color=(0,255,0)
        cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)
   
        text_gen=person_name
        cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,(0,255,0),1)
    return test_copy
