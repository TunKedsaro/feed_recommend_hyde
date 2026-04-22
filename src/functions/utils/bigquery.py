from google.cloud import bigquery
from typing import Any, Dict, List, Optional, Tuple
import yaml
from pathlib import Path
import pandas as pd
class DataQuery:
    def __init__(self, config_path: Path | None = None):
        self.client = bigquery.Client()
        config_path = config_path or (Path(__file__).resolve().parents[2] / "parameters" / "parameters.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.project = self.config["bigquery"]["project"]
        self.dataset = self.config["bigquery"]["dataset"]
        self.tables  = self.config["bigquery"]["tables"]

    def get_students(self, student_id: Optional[str] = None):
        print(f"Position : bigquery.py/class DataQuery/def get_students")
        print(f"- student_id : {student_id}")
        table_id = f"{self.project}.{self.dataset}.{self.tables['students']}"
        if student_id is None:
            query = f"""
            SELECT *
            FROM `{table_id}`
            """
            job = self.client.query(query)
        else:
            query = f"""
            SELECT *
            FROM `{table_id}`
            WHERE user_id = @student_id
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "student_id",
                        "STRING",
                        student_id
                    )
                ]
            )
            job = self.client.query(query, job_config=job_config)
        df = job.to_dataframe()
        print(f"student df ->\n{df}")
        df = df.rename(columns={"user_id": "student_id"})
        return df
    
    def get_l20_interaction(self,student_id):
        print(f"Position : bigquery.py/class DataQuery/def get_l20_interaction")
        table_id = f"{self.project}.{self.dataset}.{self.tables['l20_interaction']}"
        print(f"table_id : {table_id}")
        if student_id is None:
            query = f"""
            SELECT *
            FROM `{table_id}`
            """
            job = self.client.query(query)
        else:
            query = f"""
            SELECT *
            FROM `{table_id}`
            WHERE user_id = @student_id
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "student_id",
                        "STRING",
                        student_id
                    )
                ]
            )
            job = self.client.query(query, job_config=job_config)
        df = job.to_dataframe()
        df = df.rename(columns={"user_id": "student_id"})
        return df

        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.max_colwidth', None)
        # pd.set_option('display.width', None)
        # print(f"l20_interaction df ->\n{df}")


    # def get_students(self, student_id: Optional[str] = None):
    #     print(f"Position : bigquery.py/class DataQuery/def get_students")
    #     print(f"- student_id : {student_id}")
    #     table_id = f"{self.project}.{self.dataset}.{self.tables['students']}"

    #     query = f"""
    #     SELECT *
    #     FROM `{table_id}`
    #     WHERE user_id = @student_id
    #     """
    #     job_config = bigquery.QueryJobConfig(
    #         query_parameters=[
    #             bigquery.ScalarQueryParameter(
    #                 "student_id",
    #                 "STRING",
    #                 student_id
    #             )
    #         ]
    #     )

    #     job = self.client.query(query, job_config=job_config)
    #     df = job.to_dataframe()
    #     df = df.rename(columns={"user_id": "student_id"})
    #     return df
    
    # def get_interactions(self, student_id: Optional[str] = None):
    #     table_id = f"{self.project}.{self.dataset}.{self.tables['interactions']}"
    #     if student_id is None:
    #         query = f"""
    #         SELECT *
    #         FROM `{table_id}`
    #         """
    #         job = self.client.query(query)
    #     else:
    #         query = f"""
    #         SELECT *
    #         FROM `{table_id}`
    #         WHERE user_id = @student_id
    #         """
    #         job_config = bigquery.QueryJobConfig(
    #             query_parameters=[
    #                 bigquery.ScalarQueryParameter(
    #                     "student_id",
    #                     "STRING",
    #                     student_id
    #                 )
    #             ]
    #         )
    #         job = self.client.query(query, job_config=job_config)
    #     df= job.to_dataframe()
    #     print(f"df ->\n{df[df["user_id"]=="stu_p4198"]}")
    #     return df 
    
    def get_interactions(self, student_id: Optional[str] = None):
        print(f"Position : hydegenerator.py/class DataQuery/def get_interactions")
        table_id = f"{self.project}.{self.dataset}.{self.tables['interactions']}"
        query = f"""
        SELECT *
        FROM `{table_id}`
        WHERE user_id = @student_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "student_id",
                    "STRING",
                    student_id
                )
            ]
        )
        job = self.client.query(query, job_config=job_config)
        df  = job.to_dataframe()
        # print(f"df : \n{df}")
        return df
    
    # def get_user_events_json(self):
    #     print(f"Position : bigquery.py/class DataQuery/def get_user_events_json")
    #     table_id = f"{self.project}.{self.dataset}.{self.tables['feeds']}"
    #     query = f"""
    #     SELECT *
    #     FROM `{table_id}`
    #     """
    #     df = self.client.query(query).to_dataframe()
    #     print(f"df -> \n{df}")
    #     df["post_created_at"] = pd.to_datetime(df["post_created_at"], utc=True, errors="coerce")
    #     df["created_at"] = df["post_created_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    #     feeds_lookup = {}
    #     for _, row in df.iterrows():
    #         feed_id = row["post_id"]
    #         feeds_lookup[feed_id] = {
    #             "post_id": feed_id,
    #             "post_status": row["post_status"],
    #             "is_valid": row["is_valid"],
    #             "created_at": row["post_created_at"],
    #             "title": row["post_topic"],
    #             "feed_text": row["post_content_body"],
    #             "tags": row["post_tags"],
    #             "post_target_group":row["post_target_group"],
    #             "post_category":row["post_category"],
    #             "views": int(row["num_click"]),
    #             "like": int(row["num_like"]),
    #             "comment": int(row["num_comment"]),
    #             "share": int(row["num_share"]),
    #             "bookmark": int(row["num_bookmark"]),
    #         }
    #     return feeds_lookup
    def get_user_events_json(self, feed_ids: Optional[List[str]] = None):
        print("Position : bigquery.py/class DataQuery/def get_user_events_json")
        print(f"- feed_ids : {feed_ids}")
        table_id = f"{self.project}.{self.dataset}.{self.tables['feeds']}"
        if feed_ids is None or len(feed_ids) == 0:
            query = f"""
            SELECT *
            FROM `{table_id}`
            """
            job = self.client.query(query)
        else:
            query = f"""
            SELECT *
            FROM `{table_id}`
            WHERE post_id IN UNNEST(@feed_ids)
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("feed_ids", "STRING", feed_ids)
                ]
            )
            job = self.client.query(query, job_config=job_config)

        df = job.to_dataframe()
        # print(f"df -> \n{df}")

        df["post_created_at"] = pd.to_datetime(df["post_created_at"], utc=True, errors="coerce")
        df["created_at"] = df["post_created_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        feeds_lookup = {}
        for _, row in df.iterrows():
            feed_id = row["post_id"]
            feeds_lookup[feed_id] = {
                "post_id": feed_id,
                "post_status": row["post_status"],
                "is_valid": row["is_valid"],
                "created_at": row["post_created_at"],
                "title": row["post_topic"],
                "feed_text": row["post_content_body"],
                "tags": row["post_tags"],
                "post_target_group":row["post_target_group"],
                "post_category":row["post_category"],
                "views": int(row["num_click"] or 0),
                "like": int(row["num_like"] or 0),
                "comment": int(row["num_comment"] or 0),
                "share": int(row["num_share"] or 0),
                "bookmark": int(row["num_bookmark"] or 0),
            }

        return feeds_lookup
    
    ### ---------- Upload data ---------- ###
    def upload_data_to_student_table(self,student_json):
        # student_json = [
        #     {
        #         "student_id"         : "stu_p000",
        #         "preferred_language" : "en",
        #         "current_status"     : "student",
        #         "education_level"    : "bachelor",
        #         "education_major"    : "electrical engineering",
        #         "target_roles"       : "data science",
        #         "skills"             : "python;sql;statistics",
        #         "interests"          : "machine learning;career growth",
        #         "onboard_grp"        : "job_hunter",
        #         "onboard_grp_description": "looking to transition into data science role"
        #     }
        # ]
        students_table_id = "poc-piloturl-nonprod.gold_layer.students"
        errors = self.client.insert_rows_json(
            students_table_id,
            student_json
        )
        if errors:
            raise RuntimeError(errors)
        print("Students uploaded successfully")

    def upload_data_to_interactions_table(self,interactions_json):
        # interactions_rows = [
        #     {
        #         "user_id": "stu_p000",
        #         "feed_id": "TH_F001",
        #         "ts": "2026-01-06T13:12:10Z",
        #         "event_type": "view",
        #         "dwell_ms": 52000
        #     },
        #     {
        #         "user_id": "stu_p000",
        #         "feed_id": "TH_F001",
        #         "ts": "2026-01-06T13:13:05Z",
        #         "event_type": "like",
        #         "dwell_ms": 0
        #     }
        # ]
        interactions_table_id = "poc-piloturl-nonprod.gold_layer.interactions"
        errors = self.client.insert_rows_json(
            interactions_table_id,
            interactions_json
            )
        if errors:
            raise RuntimeError(errors)
        print("Interactions uploaded successfully")





# dq = DataQuery()
# dq.get_students()   