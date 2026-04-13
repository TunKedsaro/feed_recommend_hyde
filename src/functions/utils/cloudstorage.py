# import json
# import numpy as np
# import io
# from datetime import datetime, timedelta, timezone
# from google.cloud import storage

import json
import numpy as np
import io

from datetime import datetime, timedelta, timezone
from google.cloud import storage

from src.functions.utils.bigquery import DataQuery   # Deploy
# from functions.utils.bigquery import DataQuery         # jupyter

class GoogleCloudStorage:
    '''
    initial instance
    example 
    cgs = GoogleCloudStorage(bucket_name = "hyde-datalake")
    '''
    def __init__(self,bucket_name:str):
        self.bucket_name   = bucket_name
        self.client        = storage.Client()
        self.bucket_exists = False
        try:
            self.bucket = self.client.get_bucket(bucket_name)
            self.bucket_exists = True
            print(f"Bucket exists : {bucket_name}")
        except:
            self.bucket = None
            print(f"Bucket NOT exists : {bucket_name}")
        self.hyde_names      = ["hyde_text01.txt","hyde_text02.txt","hyde_text03.txt","hyde_text04.txt","hyde_text05.txt"]
        self.hyde_json       = ["hyde_text01.json","hyde_text02.json","hyde_text03.json","hyde_text04.json","hyde_text05.json"]
        self.embedding_names = ["embedding01.npy","embedding02.npy","embedding03.npy","embedding04.npy","embedding05.npy"]

    def blob_exists(self, blob_path:str) -> bool:
        '''
        check if object exists
        Ex. blob_exists("stu_p001/hyde/hyde_text01.txt")
        Rt. True/False
        '''
        return self.bucket.blob(blob_path).exists()

    ### ---------- Creation folder function ----------- ###
    def create_folder(self,folder_path):
        '''Creating folder and sub folder'''
        if not folder_path.endswith("/"):
            folder_path += "/"
        blob = self.bucket.blob(folder_path)
        blob.upload_from_string("")
        # print(f"Folder created : gs://{self.bucket}/{folder_path}")
        
    ### ---------- Remove function ----------- ###
    def delete_blob(self, blob_path):
        blob   = self.bucket.blob(blob_path)
        if blob.exists():
            blob.delete()
        print(f"Deleted: gs://{self.bucket_name}/{blob_path}")
    def delete_folder(self, folder_path):
        '''Remove nest blob(file) in folder'''
        if not folder_path.endswith("/"):
            folder_path += "/"
        blobs = self.bucket.list_blobs(prefix=folder_path)
        count = 0
        for blob in blobs:
            blob.delete()
            count += 1
        print(f"Deleted {count} objects under gs://{self.bucket_name}/{folder_path}")

    ### ---------- Upload folder function ----------- ###
    def upload_json(self,blob_path:str,json_data:str) -> None:
        '''upload json file to bucket'''
        blob   = self.bucket.blob(blob_path)
        blob.upload_from_string(
            json.dumps(json_data,ensure_ascii = False),
            content_type = "application/json"
        )
        # print(f"uploaded JSON -> gs://{self.bucket.name}/{blob_path}")
    def upload_text(self, blob_path, text_data):
        '''upload text file to bucket'''
        blob   = self.bucket.blob(blob_path)
        blob.upload_from_string(
            text_data,
            content_type = "text/plain"
        )
        # print(f"Uploaded text -> gs://{self.bucket.name}/{blob_path}")
    def upload_npy(self, blob_path, array):
        '''upload embedding vector'''
        buffer = io.BytesIO()
        np.save(buffer, array)
        buffer.seek(0)
        blob = self.bucket.blob(blob_path)
        blob.upload_from_file(
            buffer,
            content_type = "application/octet-stream"
        )
        # print(f"Uploaded NPY  -> gs://{self.bucket.name}/{blob_path}")
        
    ### ---------- Read file from GCS ---------- ###
    def read_json(self,blob_path:str) -> dict:
        '''
        read json file from gcs
        Ex. read_json("stu_p001/metadata/metadata.json")
        Rt. {}
        '''
        blob = self.bucket.blob(blob_path)
        return json.loads(blob.download_as_text())
    def read_text(self,blob_path:str) -> str:
        '''
        read text file from gcs
        '''
        blob = self.bucket.blob(blob_path)
        return blob.download_as_text()
    def read_npy(self,blob_path):
        '''
        read .npy (embedding vector) file
        '''
        blob   = self.bucket.blob(blob_path)
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        return np.load(buffer)
    
    ### ---------- Retriev any student data --------- ###
    def _prefix_exists(self,prefix:str) -> bool:
        '''
        - Did any blobs start with this prefix?
        - check are there items on that blob?
        '''
        if not prefix.endswith("/"):
            prefix = prefix + "/"
        blobs = list(self.bucket.list_blobs(prefix=prefix, max_results=1))
        return len(blobs) > 0 
    def _build_metadata_from_bigquery(self,student_id:str) -> dict:
        '''
        if there are no folder then Query data from BigQuery (student,interaction,feed)
        '''
        dq  = DataQuery()
        dqs = dq.get_students([student_id])
        if dqs.empty:
            return {}
        row = dqs.iloc[0].to_dict()
        return {
            "student_id"          : row["student_id"],
            "current_status"      : row["current_status"],
            "education_level"     : row["education_level"],
            "education_major"     : row["education_major"],
            "target_roles"        : row["target_roles"],
            "timezone"            : "UTC",                   # TODO : get value from yaml file
            "model_name"          : "gemini-2.5-flash",      # TODO : get value from yaml file
            "max_output_tokens"   : 1024,                    # TODO : get value from yaml file
            "feed_text_max_chars" : 872,                     # TODO : get value from yaml file
            "temperature"         : 0.1,                     # TODO : get value from yaml file
        }
    def retrieve_student_bundle(self, student_id:str) -> dict:
        results = {
            "metadata"  :{},
            "hyde"      :{},
            "embeddings":{},
            "status"    :""
        }
        ### ----------- bucket ---------- ###
        if self.bucket_exists:
            results["status"] += f"{self.bucket_name} /\n"
        else:
            results["status"] += f"{self.bucket_name} x\n"
        ### --------- student --------- ###
        student_prefix = f"{student_id}"
        if self._prefix_exists(student_prefix):
            results["status"] += f"|- {student_id} /\n"
        else:
            results["status"] += f"|- {student_id} x\n"
        ### ----------preparepart---------- ###
        metadata_prefix  = f"{student_id}/metadata"
        metadata_path    = f"{student_id}/metadata/metadata.json"
        embedding_prefix = f"{student_id}/embedding"
        hyde_prefix      = f"{student_id}/hyde"
        ### ----------metatada---------- ###       
        if self._prefix_exists(metadata_prefix): # if we have blob
            results["status"] += "  |- metadata folder /\n"
            if self.blob_exists(metadata_path):
                results["metadata"] =  self.read_json(metadata_path)
                results["status"] += "    |- metadata.json /\n"
            else:
                results["status"] += "    |- metadata.json x\n"
                # accivate query from BigQuery function
                print(f"activate query data from bigquery function ...")
                results["metadata"] = self._build_metadata_from_bigquery(student_id)
        else:
            results["status"] += "  |- metadata folder x\n"
            results["status"] += "    |- metadata.json x\n"
            print(f"activate query data from bigquery function ...")
            dq  = DataQuery()
            dqs = dq.get_students([student_id])
            dqs_dict = dqs.iloc[0].to_dict()
            results["metadata"] = self._build_metadata_from_bigquery(student_id)
        ### ----------hyde---------- ###
        if self._prefix_exists(hyde_prefix):
            results["status"] += "  |- hyde folder /\n"
            for name in self.hyde_names:
                path = f"{hyde_prefix}/{name}"
                if self.blob_exists(path):
                    results["status"] += f"    |- {path} /\n"
                    results["hyde"][name] = self.read_text(path)
                else:
                    results["hyde"][name] = ""
                    results["status"] += f"    |- {path} x\n"
        else:
            results["status"] += "  |- hyde folder x\n"
            for name in self.hyde_names:
                path = f"{hyde_prefix}/{name}"
                results["status"] += f"    |- {path} x\n"
                results["hyde"][name] = ""
                
        ### ----------embedding---------- ###
        print(f"- embedding_prefix -> {embedding_prefix}")
        if self._prefix_exists(embedding_prefix):
            results["status"] += "  |- embedding folder /\n"
            for name in self.embedding_names:
                path = f"{embedding_prefix}/{name}"
                if self.blob_exists(path):
                    results["status"] += f"    |- {path} /\n"
                    results["embeddings"][name] = self.read_npy(path)
                else:
                    results["embeddings"][name] = np.zeros((0,768),dtype=np.float32)
                    results["status"] += f"    |- {path} x\n"
        else:
            results["status"] += "  |- embedding folder x\n"
            for name in self.embedding_names:
                path = f"{embedding_prefix}/{name}"
                results["status"] += f"    |- {path} x\n"
                results["embeddings"][name] = np.zeros((0,768),dtype=np.float32)
                
        return results

    def retrieve_student_hyde_json(self, student_id:str) -> dict:
        results = {
            "hyde"      :{},
            "status"    :""
        }
        ### ----------- bucket ---------- ###
        if self.bucket_exists:
            results["status"] += f"{self.bucket_name} /\n"
        else:
            results["status"] += f"{self.bucket_name} x\n"
        ### --------- student --------- ###
        student_prefix = f"{student_id}"
        if self._prefix_exists(student_prefix):
            results["status"] += f"|- {student_id} /\n"
        else:
            results["status"] += f"|- {student_id} x\n"
        ### ----------preparepart---------- ###
        hyde_prefix      = f"{student_id}/hyde"
        ### ----------hyde---------- ###
        if self._prefix_exists(hyde_prefix):
            results["status"] += "  |- hyde folder /\n"
            for name in self.hyde_json:
                path = f"{hyde_prefix}/{name}"
                if self.blob_exists(path):
                    results["status"] += f"    |- {path} /\n"
                    results["hyde"][name] = self.read_json(path)
                else:
                    results["hyde"][name] = ""
                    results["status"] += f"    |- {path} x\n"
        else:
            results["status"] += "  |- hyde folder x\n"
            for name in self.hyde_names:
                path = f"{hyde_prefix}/{name}"
                results["status"] += f"    |- {path} x\n"
                results["hyde"][name] = ""
        return results
#     # def delete_by_ttl(self, prefix, ttl: timedelta):
#     #     '''Remove folder with setting time
#     #     timeformat support
#     #     timedelta(
#     #         days=...,
#     #         seconds=...,
#     #         microseconds=...,
#     #         milliseconds=...,
#     #         minutes=...,
#     #         hours=...,
#     #         weeks=...
#     #     )
#     #     '''
#     #     now    = datetime.now(timezone.utc)
#     #     blob   = self.bucket.blob(blob_path)
#     #     deleted = 0
#     #     for blob in blobs:
#     #         if blob.time_created and now - blob.time_created > ttl:
#     #             blob.delete()
#     #             deleted += 1
#     #     print(f"TTL cleanup deleted {deleted} objects under {prefix}")
        
# print("xxx")
# cgs = GoogleCloudStorage(bucket_name = "hyde-datalake")
# x  = cgs.retrieve_student_hyde_json("stu_p000")