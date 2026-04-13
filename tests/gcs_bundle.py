import json
import numpy as np
import io
from datetime import datetime, timedelta, timezone
from google.cloud import storage

from utils.cloudstorage import GoogleCloudStorage
from utils.bigquery import DataQuery

class GoogleCloudStorage:
    def __init__(self,bucket_name):
        self.bucket_name = bucket_name
        self.client = storage.Client()
        self.bucket_exists = False
        try:
            self.bucket = self.client.get_bucket(bucket_name)
            self.bucket_exists = True
            print(f"Bucket exists  : {bucket_name}")
        except:
            self.bucket = None
            print(f"Bucket NOT exists : {bucket_name}")
            
    def blob_exists(self, blob_path) -> bool:
        '''check if object exists'''
        return self.bucket.blob(blob_path).exists()

    ### ---------- Read file function ----------- ###
    def read_json(self, blob_path):
        '''read json file'''
        print("read_json...")
        blob   = self.bucket.blob(blob_path)
        return json.loads(blob.download_as_text())

    def read_text(self, blob_path):
        '''read text file'''
        blob   = self.bucket.blob(blob_path)
        return blob.download_as_text()

    def read_npy(self, blob_path):
        '''read .npy (embedding vector) file'''
        blob   = self.bucket.blob(blob_path)

        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        return np.load(buffer)

    def prefix_exists(self,prefix:str)->bool:
        '''Do any blobs start with this profex'''
        if not prefix.endswith("/"):
            prefix = prefix + "/"
        blobs = list(self.bucket.list_blobs(prefix=prefix, max_results=1))
        return len(blobs) > 0 

    def blob_exists(self, blob_path:str) -> bool:
        return self.bucket.blob(blob_path).exists()

    def _build_metadata_from_bigquery(self,student_id:str)->dict:
        print("activate query data from bigquery function ...")
        dq = DataQuery()
        dqs = dq.get_students([student_id])
        if dqs.empty:
            return {}
        row = dqs.iloc[0].to_dict()
        return {
            "student_id": row["student_id"],
            "current_status": row["current_status"],
            "education_level": row["education_level"],
            "education_major": row["education_major"],
            "target_roles": row["target_roles"],
            "timezone": "UTC",
            "model_name": "gemini-2.5-flash",
            "max_output_tokens": 1024,
            "feed_text_max_chars": 872,
            "temperature": 0.1,
        }

    def retrieve_student_bundle(self, student_id,embedding_names):
        result = {
            "metadata":{},
            "embeddings":{},
            "status":""
        }
        ### ---------bucket-------- ###
        if self.bucket_exists:
            result["status"] += f"{self.bucket_name} /\n"
        else:
            result["status"] += f"{self.bucket_name} x\n"
        ### --------- student --------- ###
        student_prefix = f"{student_id}"
        if self.prefix_exists(student_prefix):
            result["status"] += f"|- {student_id} /\n"
        else:
            result["status"] += f"|- {student_id} x\n"

        ### ----------preparepart---------- ###
        metadata_prefix = f"{student_id}/metadata"
        metadata_path   = f"{student_id}/metadata/metadata.json"
        embedding_prefix = f"{student_id}/embedding"
        
        ### ----------metatada---------- ###       
        if self.prefix_exists(metadata_prefix): # if we have blob
            result["status"] += "  |- metadata folder /\n"
            if self.blob_exists(metadata_path):
                result["metadata"] =  self.read_json(metadata_path)
                result["status"] += "    |- metadata.json /\n"
            else:
                result["status"] += "    |- metadata.json x\n"
                # accivate query from BigQuery function
                print(f"activate query data from bigquery function ...")
                result["metadata"] = self._build_metadata_from_bigquery(student_id)
        else:
            result["status"] += "  |- metadata folder x\n"
            result["status"] += "    |- metadata.json x\n"
            print(f"activate query data from bigquery function ...")
            dq = DataQuery()
            dqs = dq.get_students([student_id])
            dqs_dict = dqs.iloc[0].to_dict()
            result["metadata"] = self._build_metadata_from_bigquery(student_id)
            
        ### ----------embedding---------- ###
        print(f"- embedding_prefix -> {embedding_prefix}")
        if self.prefix_exists(embedding_prefix):
            result["status"] += "  |- embedding folder /\n"
            for name in embedding_names:
                path = f"{embedding_prefix}/{name}"
                if self.blob_exists(path):
                    result["status"] += f"    |- {path} /\n"
                    result["embeddings"][name] = self.read_npy(path)[0:3]
                else:
                    result["embeddings"][name] = np.array([])
                    result["status"] += f"    |- {path} x\n"
        else:
            result["status"] += "  |- embedding folder x\n"
            for name in embedding_names:
                path = f"{embedding_prefix}/{name}"
                result["status"] += f"    |- {path} x\n"
                result["embeddings"][name] = np.array([])
                
        return result
        
cgs = GoogleCloudStorage(bucket_name = "hyde-datalake")
student_id = "stu_p001"
x = cgs.retrieve_student_bundle(student_id,
                                ["embedding01.npy", "embedding02.npy", "embedding03.npy","embedding04.npy","embedding05.npy"]
                               )
print()
print(x['status'])