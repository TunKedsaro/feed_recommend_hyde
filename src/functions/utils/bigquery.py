from google.cloud import bigquery
from typing import Any, Dict, List, Optional, Tuple
import yaml
from pathlib import Path
import pandas as pd
import json

verbose = 1

class SilverGoldDataQuery:
    def __init__(self, config_path: Path | None = None):
        """
        Read user and interaction data from Silver and Gold BigQuery tables.

        The BigQuery project, datasets, and table names are loaded from a YAML
        configuration file.
        """
        self.client = bigquery.Client()
        config_path = config_path or (Path(__file__).resolve().parents[2] / "parameters" / "parameters.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.project = self.config["silverbigquery"]["project"]
        self.silver_dataset = self.config["silverbigquery"]["dataset"]
        self.silver_tables  = self.config["silverbigquery"]["tables"]
        self.gold_dataset   = self.config["goldbigquery"]["dataset"]
        self.gold_tables    = self.config["goldbigquery"]["tables"]
    def display_configs(self) -> None:
        """Print the configured BigQuery project, datasets, and tables."""
        print(f"Position : bigquery.py/class SilverDataQuery/def display_configs") if verbose else None
        print(f"- self.project        : {json.dumps(self.project,indent=4)}")
        print(f"- self.silver_dataset : {json.dumps(self.silver_dataset,indent=4)}")
        print(f"- self.silver_tables  : {json.dumps(self.silver_tables,indent=4)}")
        print(f"- self.gold_tables    : {json.dumps(self.gold_dataset,indent=4)}")
        print(f"- self.gold_tables    : {json.dumps(self.gold_tables,indent=4)}")

    def profile_id_2_user_id(df, ip) -> Optional[str]:
        """
        Find a user ID from a DataFrame by using a profile ID.
        Args:
            dataframe:
                DataFrame containing profile_id and user_id columns.
            profile_id:
                Profile ID to search for.
        Returns:
            The matched user ID, or None when no match is found.
        """
        ip = str(ip).upper()
        user_id_matched = df[df["profile_id".fillna("").astype(str).str.upper().str.contains(ip, regex=False)]]
        if user_id_matched.empty:
            return None
        user_id = user_id_matched.iloc[0]["user_id"]
        if pd.isna(user_id):
            return None
        return user_id.upper()
    
    def get_user(self, profile_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get user data from the Silver user table.
        When profile_id is provided, the method returns the first matching
        user. The profile_id column is stored as a JSON array in BigQuery.
        Args:
            profile_id:
                Optional profile ID used to find a specific user.
        Returns:
            User data as a pandas DataFrame.
        """
        print(f"Position : bigquery.py/class SilverDataQuery/def get_user") if verbose else None
        print(f"- profile_id : {profile_id}") if verbose else None
        table_id = f"{self.project}.{self.silver_dataset}.{self.silver_tables['students']}"
        print(f"- table_id : {table_id}")
        if profile_id is None:
            query = f"""
            SELECT *
            FROM `{table_id}`
            """
            job = self.client.query(query)
        else:
            query = f"""
                SELECT student.*
                FROM `{table_id}` AS student
                WHERE EXISTS (
                    SELECT 1
                    FROM UNNEST(
                        IFNULL(
                            JSON_VALUE_ARRAY(student.profile_id),
                            ARRAY<STRING>[]
                        )
                    ) AS stored_profile_id
                    WHERE UPPER(stored_profile_id) = UPPER(@profile_id)
                )
                LIMIT 1
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "profile_id",
                        "STRING",
                        profile_id,
                    )
                ]
            )
            job = self.client.query(query, job_config=job_config)
        df = job.to_dataframe()
        if profile_id is not None and not df.empty:
            df["profile_id"] = profile_id
        # print(f"student df ->\n{df}") if verbose else None
        return df

    def get_interactions(self, profile_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get user interactions from the Silver interaction table.
        When profile_id is provided, only interactions for that profile are
        returned. Results are ordered from newest to oldest.
        Args:
            profile_id:
                Optional profile ID used to filter interactions.
        Returns:
            Interaction data as a pandas DataFrame.
        """
        print(f"Position : bigquery.py/class SilverDataQuery/def get_interactions") if verbose else None
        table_id = f"{self.project}.{self.silver_dataset}.{self.silver_tables['interactions']}"
        if profile_id is None:
            query = f"""
                SELECT *
                FROM `{table_id}`
                ORDER BY event_ts DESC
            """
            job = self.client.query(query)
        else:
            query = f"""
                SELECT *
                FROM `{table_id}`
                WHERE UPPER(profile_id) = UPPER(@profile_id)
                ORDER BY event_ts DESC
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "profile_id",
                        "STRING",
                        profile_id,
                    )
                ]
            )
            job = self.client.query(
                query,
                job_config=job_config,
            )
        df = job.to_dataframe()
        # print(f"student df ->\n{df}") if verbose else None
        return df
    
    def get_l20_interaction(self,profile_id: Optional[str] = None) -> pd.DataFrame:
        """
        Get L20 interaction data from the Gold table.
        When profile_id is provided, only records for that profile are
        returned.
        Args:
            profile_id:
                Optional profile ID used to filter L20 interactions.
        Returns:
            L20 interaction data as a pandas DataFrame.
        """
        print(f"Position : bigquery.py/class SilverDataQuery/def get_l20_interaction") if verbose else None
        print(f"- profile_id : {profile_id}") if verbose else None
        table_id = f"{self.project}.{self.gold_dataset}.{self.gold_tables['l20_interaction']}"
        print(f"- table_id : {table_id}") if verbose else None

        if profile_id is None:
            query = f"""
                SELECT *
                FROM `{table_id}`
            """
            job = self.client.query(query)
        else:
            query = f"""
                SELECT *
                FROM `{table_id}`
                WHERE UPPER(profile_id) = UPPER(@profile_id)
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "profile_id",
                        "STRING",
                        profile_id,
                    )
                ]
            )
            job = self.client.query(query,job_config=job_config)
        df = job.to_dataframe()
        # print(f"student df ->\n{df}") if verbose else None
        return df
    

# # Testing
# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_colwidth", None)   # Don't truncate long strings
# pd.set_option("display.width", None)          # Auto-detect terminal width
# pd.set_option("display.expand_frame_repr", False)

# profile_id_1 = "C602CB1F-AF6B-4783-972A-787F41EDC206"
# profile_id_2 = "5F37C2A2-9A9B-422F-B36F-E0D768274AB8"
# profile_id_3 = "BC428AF1-3AAE-4BE2-B57F-6B39A657905F"

# dq = SilverGoldDataQuery()
# dq.display_configs()
# dq.get_user(profile_id_1)
# print("#"*100)
# dq.get_interactions(profile_id_1)
# print("#"*100)
# dq.get_l20_interaction(profile_id_1)
# print("#"*100)

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
        print(f"Position : bigquery.py/class DataQuery/def get_students") if verbose else None
        print(f"- student_id : {student_id}") if verbose else None
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
        print(f"student df ->\n{df}") if verbose else None
        df = df.rename(columns={"user_id": "student_id"})
        return df
    
    def get_l20_interaction(self,student_id):
        print(f"Position : bigquery.py/class DataQuery/def get_l20_interaction") if verbose else None
        table_id = f"{self.project}.{self.dataset}.{self.tables['l20_interaction']}"
        print(f"table_id : {table_id}") if verbose else None
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
        print(f"student df ->\n{df}") if verbose else None
        df = df.rename(columns={"user_id": "student_id"})
        return df

    def get_interactions(self, student_id: Optional[str] = None):
        print(f"Position : bigquery.py/class DataQuery/def get_interactions") if verbose else None
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
        print(f"student df ->\n{df}") if verbose else None
        return df
    
    def get_user_events_json(self, feed_ids: Optional[List[str]] = None):
        print("Position : bigquery.py/class DataQuery/def get_user_events_json") if verbose else None
        print(f"- feed_ids : {feed_ids}") if verbose else None
        table_id = f"{self.project}.{self.dataset}.{self.tables['feeds']}"
        print(f"table_id : {table_id}")
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
        print(json.dumps(feeds_lookup,indent=4))
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
# dq.get_students("stu_p4198") 
# print("#"*100)
# dq.get_l20_interaction("stu_p4198")
# print("#"*100)
# dq.get_interactions("stu_p4198")
# print("#"*100)
# dq.get_user_events_json("FEED-229")
# print("#"*100)

def get_user_events(user_id: str) -> list[dict]:
    client = bigquery.Client()

    query = """
        SELECT *
        FROM `poc-piloturl-nonprod.gold_layer.student`
        WHERE user_id = @user_id
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "user_id",
                "STRING",
                user_id,
            )
        ]
    )

    query_job = client.query(query, job_config=job_config)
    rows = query_job.result()

    return [dict(row) for row in rows]
