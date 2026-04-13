from google.cloud import bigquery
from typing import Any, Dict, List, Optional, Tuple
import yaml
from pathlib import Path

class DataQuery:
    def __init__(self, config_path: Path | None = None):
        self.client = bigquery.Client()
        config_path = config_path or (Path(__file__).resolve().parents[2] / "parameters" / "parameters.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.project = self.config["bigquery"]["project"]
        self.dataset = self.config["bigquery"]["dataset"]
        self.tables  = self.config["bigquery"]["tables"]

    # def get_students(self, student_id:Optional[str]=None):
    #     if student_id is None:
    #         query = """
    #         SELECT *
    #         FROM `poc-piloturl-nonprod.gold_layer.students`
    #         """
    #         job = self.client.query(query)
    #     else:
    #         query = """
    #         SELECT *
    #         FROM `poc-piloturl-nonprod.gold_layer.students`
    #         WHERE student_id = @student_id
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
    #     return job.to_dataframe()
    def get_students(self, student_id: Optional[str] = None):

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
            WHERE student_id = @student_id
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

        return job.to_dataframe()
    
    # def get_interactions(self,student_id:Optional[str]=None):
    #     if student_id is None:
    #         query = """
    #         SELECT *
    #         FROM `poc-piloturl-nonprod.gold_layer.interactions`
    #         """
    #         job = self.client.query(query)
    #     else:
    #         query = """
    #         SELECT *
    #         FROM `poc-piloturl-nonprod.gold_layer.interactions`
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
    #     return job.to_dataframe() 

    def get_interactions(self, student_id: Optional[str] = None):

        table_id = f"{self.project}.{self.dataset}.{self.tables['interactions']}"

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

        return job.to_dataframe()
    
    # def get_user_events_json(self):
    #     query = """
    #     SELECT *
    #     FROM `poc-piloturl-nonprod.gold_layer.feeds`
    #     """
    #     df = self.client.query(query).to_dataframe()
    #     # ensure created_at is ISO-8601 Z format
    #     df["created_at"] = df["created_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    #     feeds_lookup: Dict[str, Dict[str, Any]] = {}
    #     for _,row in df.iterrows():
    #         feed_id = row["feed_id"]
    #         feeds_lookup[feed_id] = {
    #             "feed_id"        : feed_id,
    #             "title"          : row["title"],
    #             "feed_text"      : row["feed_text"],
    #             "tags"           : row["tags"],                     
    #             "language"       : row["language"],
    #             "created_at"     : row["created_at"],
    #             "source"         : row["source"],
    #             "url"            : row["url"],
    #             "views"          : int(row["views"]),
    #             "embedding_input": row["embedding_input"]
    #         }
    #     return feeds_lookup
    def get_user_events_json(self):

        table_id = f"{self.project}.{self.dataset}.{self.tables['feeds']}"

        query = f"""
        SELECT *
        FROM `{table_id}`
        """

        df = self.client.query(query).to_dataframe()

        df["created_at"] = df["created_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        feeds_lookup = {}

        for _, row in df.iterrows():
            feed_id = row["feed_id"]
            feeds_lookup[feed_id] = {
                "feed_id": feed_id,
                "title": row["title"],
                "feed_text": row["feed_text"],
                "tags": row["tags"],
                "language": row["language"],
                "created_at": row["created_at"],
                "source": row["source"],
                "url": row["url"],
                "views": int(row["views"]),
                "embedding_input": row["embedding_input"],
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