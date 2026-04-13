from __future__ import annotations

import json
import os
import yaml
import numpy as np
import pandas as pd
import time
import io

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from google.cloud import bigquery
from google.cloud import storage

from src.functions.utils.logging import get_logger
from src.functions.utils.config  import PROJECT_ROOT, load_config
from src.functions.utils.llm_client import build_llm_client_from_yaml
from src.functions.utils.text_embeddings import GoogleEmbeddingModel
from src.functions.core.context_builder import build_user_context
from src.functions.core.history import build_history_summary

from src.functions.utils.cloudstorage import GoogleCloudStorage
from src.functions.utils.bigquery import DataQuery
from src.functions.utils.shin_embedder import embed_texts_gemini

from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback


import os
import pandas as pd
from typing import Dict

from concurrent.futures import ThreadPoolExecutor, as_completed


# ----------------------------------------------------------------------
# Helper function
# ----------------------------------------------------------------------
def prettyjson(txt:str) -> str:
    return str(json.dumps(txt,indent=4, ensure_ascii=False))


def save_timing_to_excel(
    *,
    student_id: str,
    timing_ms: Dict[str, float],
    file_path: str = "hyde_timing_report.xlsx",
):
    """
    Save timing report to Excel.
    If file exists → append new row.
    If not → create new file.
    """
    # 1️ Prepare single-row dataframe
    row_dict = {"student_id": student_id}
    row_dict.update(timing_ms)
    df_new = pd.DataFrame([row_dict])
    # 2️ If file exists → append
    if os.path.exists(file_path):
        df_old = pd.read_excel(file_path)
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new
    # 3 Save
    df_final.to_excel(file_path, index=False)
    # print(f"Timing report saved to {file_path}")


### ---------- initail value ---------- ###
class HydeGenerator(GoogleCloudStorage,DataQuery):
    def __init__(self,bucket_name:str,verbose:int=0):
        self.cgs     = GoogleCloudStorage(bucket_name=bucket_name)
        self.dq      = DataQuery()
        self.cfg     = load_config()
        self.verbose = verbose
    
    def _read_hyde_config(self,cfg: Dict[str, Any]) -> Tuple[int, int, int, bool, str]:
        """
        Read HyDE-related configuration with safe defaults.
        Returns
        -------
        history_threshold:
            Event count threshold for prompt selection
        recent_k:
            Max number of recent feeds used in HistorySummary
        feed_text_max_chars:
            Per-feed text truncation limit
        include_recent_feeds:
            Whether HistorySummary may include feed snippets
        query_embedding_model_name:
            Embedding model for HyDE queries
        """
        hyde_cfg = cfg.get("hyde", {}) if isinstance(cfg, dict) else {}

        history_threshold = int(hyde_cfg.get("history_threshold", 5))
        recent_k = int(hyde_cfg.get("recent_k", 5))
        feed_text_max_chars = int(hyde_cfg.get("feed_text_max_chars", 240))
        include_recent_feeds = bool(hyde_cfg.get("include_recent_feeds", True))

        # Default to same embedding family as feed embeddings
        query_embedding_model_name = str(
            hyde_cfg.get("query_embedding_model_name")
            or cfg.get("embeddings", {}).get("model_name", "")
            or "gemini-embedding-001"
        )

        # Hard safety guards
        history_threshold = max(1, history_threshold)
        recent_k = max(0, min(recent_k, 10))
        feed_text_max_chars = max(0, min(feed_text_max_chars, 2000))

        return (
            history_threshold,
            recent_k,
            feed_text_max_chars,
            include_recent_feeds,
            query_embedding_model_name,
        )
    def _load_prompts(self) -> Dict[str, str]:
        """
        Load HyDE prompt templates from parameters/prompts.yaml.
        Expected structure:
        hyde_prompts:
            hyde_a: "..."
            hyde_b: "..."
            hyde_c: "..."
        """
        prompts_path = PROJECT_ROOT / "parameters" / "prompts.yaml"
        with prompts_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("hyde_prompts", {}) or {}
    def _choose_hyde_prompt_key(self,num_events: int, history_threshold: int = 5) -> str:
        """
        Select HyDE prompt variant based on interaction volume.

        Rules
        -----
        - num_events >= history_threshold → history-heavy (hyde_b)
        - num_events <= 1               → onboarding / sparse (hyde_c)
        - otherwise                     → mixed (hyde_a)
        """
        if num_events >= history_threshold:
            return "hyde_b"
        if num_events <= 1:
            return "hyde_c"
        return "hyde_a"
    def _render_prompt(
        self,
        template: str,
        preferred_language: str,
        user_context_text: str,
        history_summary_text: Optional[str],
    ) -> str:
        """
        Render a prompt template using strict placeholder substitution.

        Supported placeholders:
        - {{preferred_language}}
        - {{UserContextText}}
        - {{HistorySummaryText}}

        No templating engine is used on purpose to keep behavior explicit.
        """
        s = template.replace("{{preferred_language}}", preferred_language or "th")
        s = s.replace("{{UserContextText}}", user_context_text or "")
        s = s.replace("{{HistorySummaryText}}", history_summary_text or "")
        return s
    
    # =============================================================================
    # HyDE output handling
    # =============================================================================
    def _extract_hyde_query_texts(self,hyde_json: Dict[str, Any]) -> List[str]:
        """
        Extract query_text values from HyDE JSON output.

        Expected structure:
        {
            "hyde_queries": [
            {"query_id": "...", "query_text": "...", ...},
            ...
            ]
        }

        Order is preserved and MUST match embedding row order.
        """
        if not isinstance(hyde_json, dict):
            raise ValueError("hyde_output must be a dict")
        items = hyde_json.get("hyde_queries") or []
        if not isinstance(items, list):
            raise ValueError("hyde_output.hyde_queries must be a list")

        out: List[str] = []
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                raise ValueError(f"hyde_output.hyde_queries[{i}] must be an object")
            out.append(str(it.get("query_text") or "").strip())
        return out
    
    def _upload_to_cgs(self,student_id,metadata,embedding,hyde_json):
        self.cgs.upload_json(
            blob_path   = f"{student_id}/metadata/metadata.json",
            json_data   = metadata
        )
        self.cgs.upload_npy(
            blob_path   = f"{student_id}/embedding/embedding01.npy",
            array       = embedding[0]
        )
        self.cgs.upload_npy(
            blob_path   = f"{student_id}/embedding/embedding02.npy",
            array       = embedding[1]
        )
        self.cgs.upload_npy(
            blob_path   = f"{student_id}/embedding/embedding03.npy",
            array       = embedding[2]
        )
        self.cgs.upload_npy(
            blob_path   = f"{student_id}/embedding/embedding04.npy",
            array       = embedding[3]
        )
        self.cgs.upload_npy(
            blob_path   = f"{student_id}/embedding/embedding05.npy",
            array       = embedding[4]
        )
        self.cgs.upload_json(
            blob_path = f"{student_id}/hyde/hyde.json",
            json_data = hyde_json
        )
        # self.cgs.upload_json(
        #     blob_path = f"{student_id}/hyde/hyde_text02.json",
        #     json_data = hyde_json['hyde_queries'][1]
        # )
        # self.cgs.upload_json(
        #     blob_path = f"{student_id}/hyde/hyde_text03.json",
        #     json_data = hyde_json['hyde_queries'][2]
        # )
        # self.cgs.upload_json(
        #     blob_path = f"{student_id}/hyde/hyde_text04.json",
        #     json_data = hyde_json['hyde_queries'][3]
        # )
        # self.cgs.upload_json(
        #     blob_path = f"{student_id}/hyde/hyde_text05.json",
        #     json_data = hyde_json['hyde_queries'][4]
        # )
        # self.cgs.upload_text(
        #     blob_path = f"{student_id}/hyde/hyde_text01.txt",
        #     text_data = hyde[0]
        # )
        # self.cgs.upload_text(
        #     blob_path = f"{student_id}/hyde/hyde_text02.txt",
        #     text_data = hyde[1]
        # )
        # self.cgs.upload_text(
        #     blob_path = f"{student_id}/hyde/hyde_text03.txt",
        #     text_data = hyde[2]
        # )
        # self.cgs.upload_text(
        #     blob_path = f"{student_id}/hyde/hyde_text04.txt",
        #     text_data = hyde[3]
        # )
        # self.cgs.upload_text(
        #     blob_path = f"{student_id}/hyde/hyde_text05.txt",
        #     text_data = hyde[4]
        # )
    
    #----------------------------------------------------------------------
    # main pipeline
    #----------------------------------------------------------------------
    def student_feed_id(self, user_id: str):
        client = bigquery.Client()
        query = """
            SELECT *
            FROM `poc-piloturl-nonprod.gold_layer.interactions`
            WHERE user_id = @user_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
            ]
        )
        rows = client.query(query, job_config=job_config).result()
        return [
            {k: str(v) for k, v in dict(row).items()}
            for row in rows
        ]
    

    def single_student_generator(self,student_id:str):

        t0_total = time.perf_counter()
        timing_ms : Dict[str,float] = {}

        status = "Complete"
        try:
            ### ----------- initail value ----------- ###
            t0 = time.perf_counter()
            students     = self.dq.get_students(student_id)       # TODO : change this method to overwrite for case () and identify student id to reduce time
            interactions = self.dq.get_interactions(student_id)   # TODO : change this method to overwrite for case () and identify student id to reduce time
            feeds_lookup = self.dq.get_user_events_json()         # TODO : change this method to overwrite for case () and identify student id to reduce time
            timing_ms["download_data_ms"] = (time.perf_counter() - t0) * 1000
            now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

            ### ----------- read HyDe-related configureation once ----------- ###
            t0 = time.perf_counter()
            history_threshold,recent_k,feed_text_max_chars,include_recent_feeds,query_embedding_model_name = self._read_hyde_config(self.cfg)
            expected_dim = int(self.cfg.get("embeddings", {}).get("dim", 0) or 0)
            timing_ms["read_config_ms"] = (time.perf_counter() - t0) * 1000

            ### ----------- Hyde prompt ----------- ###
            t0 = time.perf_counter()
            prompts = self._load_prompts()
            if not prompts:
                raise ValueError("hyde_prompts missing from parameters/prompts.yaml")
            client = build_llm_client_from_yaml(
                parameters_path=str(PROJECT_ROOT / "parameters" / "parameters.yaml")
                )
            timing_ms["load_prompt_and_client_ms"] = (time.perf_counter() - t0) * 1000

            ### ---------- 01 locate student row ---------- ###
            student_row_df = students[students["student_id"] == student_id]  # get student that we want from dataframe
            if len(student_row_df) == 0:                                     # check there are only one student
                raise ValueError(f"student_id {student_id} not found")
            student_row = student_row_df.iloc[0].to_dict()                   # ddataframe -> dict

            ### ---------- 02 build context ---------- ###
            t0 = time.perf_counter()
            user_ctx = build_user_context(student_row)                       # create user context class
            pref_lang = user_ctx.user_context_json.get("preferred_language", "th")
            user_events = interactions[interactions["user_id"] == student_id] # get user envent table
            num_events = int(len(user_events))
            if num_events > 0:
                history_summary_text = build_history_summary(                # build history summary
                    user_events,
                    preferred_language=pref_lang,
                    include_recent_feeds=include_recent_feeds,
                    recent_k=recent_k,
                    feeds_lookup=feeds_lookup or None,
                    feed_text_max_chars=feed_text_max_chars,
            )
            timing_ms["build_context_ms"] = (time.perf_counter() - t0) * 1000

            ### ---------- 03 build prompt ---------- ###
            t0 = time.perf_counter()
            prompt_key = self._choose_hyde_prompt_key(num_events, history_threshold)
            print(f"prompt_key -> {prompt_key}")
            template = prompts.get(prompt_key)
            if not template:
                raise ValueError(f"Missing prompt '{prompt_key}'")
            prompt = self._render_prompt(
                template=template,
                preferred_language=pref_lang,
                user_context_text=user_ctx.user_context_text,
                history_summary_text=history_summary_text,
            )
            timing_ms["build_prompt_ms"] = (time.perf_counter() - t0) * 1000

            ### ---------- 04 LLM call ---------- ###
            t0 = time.perf_counter()
            hyde_json = client.generate_json(prompt)
            # print(f"hyde_json -> {hyde_json}")
            timing_ms["llm_call_ms"] = (time.perf_counter() - t0) * 1000


            ### ---------- 05 Shin embedding ---------- ###
            t0 = time.perf_counter()
            hyde_query_texts = self._extract_hyde_query_texts(hyde_json)

            if hyde_query_texts:
                emb = embed_texts_gemini(
                    texts=hyde_query_texts,
                    output_dim=768,
                    task_type="RETRIEVAL_DOCUMENT",
                )
                if emb.ndim != 2:
                    raise ValueError(f"Invalid embedding shape {emb.shape}")
                dim = int(emb.shape[1])
            else:
                dim = expected_dim or 0
                emb = np.zeros((0, dim), dtype=np.float32)
            timing_ms["embedding_ms"] = (time.perf_counter() - t0) * 1000

            ### ---------- 06 save bundle locally ---------- ###
            bundle = {
                "bundle_version": "v2_hyde_embedded_queries",
                "student_id": student_id,
                "generated_at": now_iso,
                "prompt_key": prompt_key,
                "preferred_language": pref_lang,
                "num_events": num_events,
                "user_context_json": user_ctx.user_context_json,
                "user_context_text": user_ctx.user_context_text,
                "history_summary_text": history_summary_text,
                "hyde_output": hyde_json,
            }
            if self.verbose:
                print(bundle)
            
            ### ---------- 07 upload to GCS ---------- ###
            t0 = time.perf_counter()

            self.cgs.create_folder(f"{student_id}/metadata/")
            self.cgs.create_folder(f"{student_id}/hyde/")
            self.cgs.create_folder(f"{student_id}/embedding/")

            metadata = {
                "student_id"          :student_id, # 
                "current_status"      :student_row['current_status'], #
                "education_level"     :student_row['education_level'], #
                "education_major"     :student_row['education_major'], #
                "target_roles"        :student_row['target_roles'], #
                "timezone"            :self.cfg["app"]["timezone"], #
                "model_name"          :self.cfg["llm"]["model_name"], #
                "max_output_tokens"   :self.cfg["llm"]["max_output_tokens"], #
                "feed_text_max_chars" :self.cfg["hyde"]["feed_text_max_chars"], #
                "temperature"         :self.cfg["llm"]["temperature"], #
                "interaction"         :self.student_feed_id(student_id)
            }
            self._upload_to_cgs(
                student_id = student_id,
                metadata   = metadata,
                embedding  = emb,
                # hyde       = hyde_query_texts,
                hyde_json  = {"hq":hyde_json['hyde_queries']}
            )
            timing_ms["upload_gcs_ms"] = (time.perf_counter() - t0) * 1000

        except Exception as e:
            import traceback
            traceback.print_exc()
            status = "Fail"

        # start counting total process
        timing_ms["total_ms"] = (time.perf_counter() - t0_total) * 1000
        print("Timing summary (ms):", timing_ms)

        return {
            "status": status,
            "timing": timing_ms
        }
    

    def single_hyde_generator2(self, 
                                student_id:str,
                                students = None,
                                interactions = None,
                                feeds_lookup = None
                            ):
        status = "Complete"
        failed_students = []
        slow_students   = []
        t0_total        = time.perf_counter()
        try:
            # --------------------------------------------------
            # 1. Download data
            # อาจจะต้องมีการเปลี่ยนถ้าหากไปทำเป็น batch เราไม่ควรยิง Request ไปทุกๆครั้ง
            # --------------------------------------------------
            t0 = time.perf_counter()
            if students is None:
                students     = self.dq.get_students()
            if interactions is None:
                interactions = self.dq.get_interactions()
            if feeds_lookup is None:
                feeds_lookup = self.dq.get_user_events_json()
            download_ms  = (time.perf_counter()-t0)*1000
            now_iso      = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            print(f"Download time: {(download_ms/1000):.2f}s")
            # --------------------------------------------------
            # 2. Config
            # --------------------------------------------------
            t0 = time.perf_counter()
            history_threshold, recent_k, feed_text_max_chars, include_recent_feeds, query_embedding_model_name = self._read_hyde_config(self.cfg)
            read_config_ms = (time.perf_counter() - t0) * 1000
            expected_dim = int(self.cfg.get("embeddings",{}).get("dim",0) or 0)
            # --------------------------------------------------
            # 3. Load prompts + client
            # --------------------------------------------------
            t0 = time.perf_counter()
            prompts = self._load_prompts()
            client = build_llm_client_from_yaml(
                parameters_path = str(PROJECT_ROOT/"parameters"/"parameters.yaml")
            )
            load_prompt_client_ms = (time.perf_counter() - t0) * 1000
            # --------------------------------------------------
            # 4. Locate student
            # --------------------------------------------------
            student_row_df = students[students["student_id"] == student_id]
            if len(student_row_df) == 0:
                raise ValueError(f"{student_id} not found")
            student_row = student_row_df.iloc[0].to_dict()
            print(f"\nProcessing {student_id}")
            t0_student = time.perf_counter()
            timing = {
                "student_id" : student_id,
                "dowload_data_ms" : round(download_ms,2),
                "read_config_ms" : round(read_config_ms,2),
                "load_prompt_and_client_ms" : round(load_prompt_client_ms,2)
            }
            # ----------------------------
            # 5. Context
            # ----------------------------
            t0 = time.perf_counter()
            user_ctx = build_user_context(student_row)
            pref_lang = user_ctx.user_context_json.get("preferred_language", "th")
            pref_lang = "th"  # TODO : change it later but for now there are only th feeds
            user_events = interactions[interactions["user_id"] == student_id]
            num_events  = len(user_events)
            history_summary_text = ""
            if num_events > 0:
                history_summary_text = build_history_summary(
                        user_events,
                        preferred_language = pref_lang,
                        include_recent_feeds = include_recent_feeds,
                        recent_k = recent_k,
                        feeds_lookup = feeds_lookup or None,
                        feed_text_max_chars = feed_text_max_chars
                )
            timing["build_context_ms"] = round((time.perf_counter() - t0) * 1000,2)
            # ----------------------------
            # 6. Prompt
            # ----------------------------
            t0 = time.perf_counter()
            prompt_key = self._choose_hyde_prompt_key(num_events, history_threshold)
            template = prompts.get(prompt_key)
            if not template:
                raise ValueError(f"Missing prompt {prompt_key}")
            prompt = self._render_prompt(
                template=template,
                preferred_language=pref_lang,
                user_context_text=user_ctx.user_context_text,
                history_summary_text=history_summary_text,
            )
            timing["build_prompt_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            # ----------------------------
            # 7. LLM
            # ----------------------------
            t0 = time.perf_counter()
            hyde_json = client.generate_json(prompt)
            llm_time = time.perf_counter() - t0
            timing["llm_call_ms"] = round(llm_time * 1000,2)
            if llm_time > 20:
                print(f"⚠ Slow LLM ({llm_time:.2f}s)")
                slow_students.append(student_id)
            # ----------------------------
            # 8. Extract queries
            # ----------------------------       
            hyde_query_text = self._extract_hyde_query_texts(hyde_json)
            # ----------------------------
            # 9. Embedding
            # ----------------------------   
            t0 = time.perf_counter()
            if hyde_query_text:
                emb = embed_texts_gemini(
                    texts  = hyde_query_text,
                    output_dim = 768,
                    task_type="RETRIEVAL_DOCUMENT",
                )         
                if emb.ndim != 2:
                    raise ValueError(f"Invalid embedding shape {emb.shape}")
            else:
                emb = np.zeros((0, expected_dim), dtype=np.float32)
            # ----------------------------
            # 10. Upload
            # ----------------------------
            t0 = time.perf_counter()
            metadata = {
                "student_id" : student_id,
                "generated_at" : now_iso,
                "model" : self.cfg["llm"]["model_name"],
            }
            self._upload_to_cgs(
                student_id = student_id,
                metadata   = metadata,
                embedding  = emb,
                hyde_json  = {
                    "hq" : hyde_json.get("hyde_queries", [])
                    },
            )
            timing["upload_gcs_ms"] = round((time.perf_counter() - t0)*1000,2)
            timing["total_ms"]      = round((time.perf_counter() - t0_student)*1000,2)
            timing["status"]        = "done"
            print(f"✅ Done {student_id} in {(time.perf_counter()-t0_student):.2f}s")
        except Exception as e:
            print(f"❌ Failed {student_id} → {str(e)}")
            status = "Fail"
            failed_students.append({
                "student_id":student_id,
                "error":str(e)
            })
            timing = {
            "student_id": student_id,
            "status": str(e)
        }

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------
        total_time = round(time.perf_counter() - t0_total, 2)
        print("\nSingle Student Finished")
        print("Status:", status)
        print("Total Time (sec):", total_time)
        return {
            "student_id": student_id,
            "status": status,
            "timing": timing,
            "failed": failed_students,
            "slow": slow_students,
            "total_time_sec": total_time
        }

    def concurrent_of_single_student_generator(self):
        t0 = time.perf_counter()
        report_each_student = []
        students     = self.dq.get_students()
        interactions = self.dq.get_interactions()
        feeds_lookup = self.dq.get_user_events_json()
        for student_id in list(set(students["student_id"])):
            result = self.single_hyde_generator2(
                student_id,
                students = students,
                interactions = interactions,
                feeds_lookup = feeds_lookup
                )
            report_each_student.append({
                "student_id":student_id,
                "response"  :result
                })
        print(f"report_each_student -> {report_each_student}")
        return report_each_student


    # # legacy
    # def batch_student_generator(self):
    #     status = "Complete"
    #     student_id_updated = []
    #     failed_students    = []
    #     slow_students      = []
    #     timing_rows        = []
    #     t0_total = time.perf_counter()
    #     try:
    #         # --------------------------------------------------
    #         # 1. Download data once
    #         # --------------------------------------------------
    #         t0 = time.perf_counter()
    #         students     = self.dq.get_students()
    #         interactions = self.dq.get_interactions()
    #         feeds_lookup = self.dq.get_user_events_json()
    #         download_ms  = (time.perf_counter() - t0) * 1000
    #         now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    #         print(f"Download time: {(download_ms/1000):.2f}s")

    #         # --------------------------------------------------
    #         # 2. Config
    #         # --------------------------------------------------
    #         t0 = time.perf_counter()
    #         history_threshold, recent_k, feed_text_max_chars, include_recent_feeds, query_embedding_model_name = self._read_hyde_config(self.cfg)
    #         read_config_ms = (time.perf_counter() - t0) * 1000
    #         expected_dim = int(self.cfg.get("embeddings", {}).get("dim", 0) or 0)
    #         t0 = time.perf_counter()
    #         prompts = self._load_prompts()
    #         client = build_llm_client_from_yaml(
    #             parameters_path=str(PROJECT_ROOT / "parameters" / "parameters.yaml")
    #         )
    #         load_prompt_client_ms = (time.perf_counter() - t0) * 1000

    #         # --------------------------------------------------
    #         # 3. Loop students safely
    #         # --------------------------------------------------
    #         for idx, row in students.iterrows():
    #             t0_student = time.perf_counter()
    #             student_row = row.to_dict()
    #             student_id = str(student_row.get("student_id", "")).strip()
    #             print(f"\nProcessing {student_id} ({idx+1}/{len(students)})")
    #             timing = {
    #                 "student_id": student_id,
    #                 "download_data_ms": round(download_ms,2),
    #                 "read_config_ms": round(read_config_ms,2),
    #                 "load_prompt_and_client_ms": round(load_prompt_client_ms,2),
    #             }
    #             try:
    #                 # ----------------------------
    #                 # Context
    #                 # ----------------------------
    #                 t0 = time.perf_counter()
    #                 user_ctx = build_user_context(student_row)
    #                 pref_lang = user_ctx.user_context_json.get("preferred_language", "th")
    #                 user_events = interactions[interactions["user_id"] == student_id]
    #                 num_events = len(user_events)
    #                 history_summary_text = ""
    #                 if num_events > 0:
    #                     history_summary_text = build_history_summary(
    #                         user_events,
    #                         preferred_language=pref_lang,
    #                         include_recent_feeds=include_recent_feeds,
    #                         recent_k=recent_k,
    #                         feeds_lookup=feeds_lookup or None,
    #                         feed_text_max_chars=feed_text_max_chars,
    #                     )
    #                 timing["build_context_ms"] = round((time.perf_counter() - t0) * 1000,2)
    #                 # ----------------------------
    #                 # Prompt
    #                 # ----------------------------
    #                 t0 = time.perf_counter()
    #                 prompt_key = self._choose_hyde_prompt_key(num_events, history_threshold)
    #                 template = prompts.get(prompt_key)
    #                 if not template:
    #                     raise ValueError(f"Missing prompt {prompt_key}")
    #                 prompt = self._render_prompt(
    #                     template=template,
    #                     preferred_language=pref_lang,
    #                     user_context_text=user_ctx.user_context_text,
    #                     history_summary_text=history_summary_text,
    #                 )
    #                 timing["build_prompt_ms"] = round((time.perf_counter() - t0) * 1000,2)

    #                 # ----------------------------
    #                 # LLM CALL
    #                 # ----------------------------
    #                 t0 = time.perf_counter()
    #                 hyde_json = client.generate_json(prompt)
    #                 llm_time = time.perf_counter() - t0
    #                 timing["llm_call_ms"] = round(llm_time * 1000,2)
    #                 if llm_time > 20:
    #                     print(f"⚠ Slow LLM ({llm_time:.2f}s) → skip")
    #                     slow_students.append(student_id)
    #                 # ----------------------------
    #                 # Extract queries
    #                 # ----------------------------
    #                 print(f"hyde_json -> {hyde_json}")
    #                 hyde_query_texts = self._extract_hyde_query_texts(hyde_json)
    #                 # ----------------------------
    #                 # Embedding
    #                 # ----------------------------
    #                 t0 = time.perf_counter()
    #                 if hyde_query_texts:
    #                     emb = embed_texts_gemini(
    #                         texts=hyde_query_texts,
    #                         output_dim=768,
    #                         task_type="RETRIEVAL_DOCUMENT",
    #                     )
    #                     if emb.ndim != 2:
    #                         raise ValueError(f"Invalid embedding shape {emb.shape}")
    #                 else:
    #                     emb = np.zeros((0, expected_dim), dtype=np.float32)
    #                 timing["embedding_ms"] = round((time.perf_counter() - t0) * 1000,2)

    #                 # ----------------------------
    #                 # Upload
    #                 # ----------------------------
    #                 t0 = time.perf_counter()
    #                 metadata = {
    #                     "student_id": student_id,
    #                     "generated_at": now_iso,
    #                     "model": self.cfg["llm"]["model_name"],
    #                 }
    #                 self._upload_to_cgs(
    #                     student_id=student_id,
    #                     metadata=metadata,
    #                     embedding=emb,
    #                     hyde_json={"hq": hyde_json.get("hyde_queries", [])}
    #                 )
    #                 timing["upload_gcs_ms"] = round((time.perf_counter() - t0) * 1000,2)
    #                 timing["total_ms"] = round((time.perf_counter() - t0_student) * 1000,2)
    #                 timing["status"] = "done"
    #                 student_id_updated.append(student_id)
    #                 print(f"✅ Done {student_id} in {(time.perf_counter()-t0_student):.2f}s")
    #             except Exception as e:
    #                 print(f"❌ Failed {student_id} → {str(e)}")
    #                 failed_students.append({
    #                     "student_id": student_id,
    #                     "error": str(e)
    #                 })
    #                 timing["build_context_ms"] = "-"
    #                 timing["build_prompt_ms"] = "-"
    #                 timing["llm_call_ms"] = "-"
    #                 timing["embedding_ms"] = "-"
    #                 timing["upload_gcs_ms"] = "-"
    #                 timing["total_ms"] = "-"
    #                 timing["status"] = str(e)
    #             timing_rows.append(timing)
    #     except Exception as e:
    #         status = "Fail"
    #         print("Batch crashed:", e)

    #     # --------------------------------------------------
    #     # Save timing report
    #     # --------------------------------------------------
    #     timestamp = datetime.now().strftime("%y%m%d_%H%M")
    #     df_new = pd.DataFrame(timing_rows)
    #     file_path = f"hyde_timing_report_{timestamp}.xlsx"
    #     if os.path.exists(file_path):
    #         df_old = pd.read_excel(file_path)
    #         df_final = pd.concat([df_old, df_new], ignore_index=True)
    #     else:
    #         df_final = df_new
    #     df_final.to_excel(file_path, index=False)
    #     total_time = round(time.perf_counter() - t0_total, 2)
    #     print("\nBatch Finished")
    #     print("Updated:", len(student_id_updated))
    #     print("Failed:", len(failed_students))
    #     print("Slow:", len(slow_students))
    #     print("Total Time (sec):", total_time)
    #     return student_id_updated, status

    def batch_student_generator(self):

        status = "Complete"
        student_id_updated = []
        failed_students    = []
        slow_students      = []
        timing_rows        = []

        t0_total = time.perf_counter()
        try:
            # --------------------------------------------------
            # 1. Download data once
            # --------------------------------------------------
            t0 = time.perf_counter()
            students     = self.dq.get_students()
            interactions = self.dq.get_interactions()
            feeds_lookup = self.dq.get_user_events_json()
            download_ms  = (time.perf_counter() - t0) * 1000
            now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            print(f"Download time: {(download_ms/1000):.2f}s")
            # --------------------------------------------------
            # 2. Config
            # --------------------------------------------------
            t0 = time.perf_counter()
            history_threshold, recent_k, feed_text_max_chars, include_recent_feeds, query_embedding_model_name = self._read_hyde_config(self.cfg)
            read_config_ms = (time.perf_counter() - t0) * 1000
            expected_dim = int(self.cfg.get("embeddings", {}).get("dim", 0) or 0)
            t0 = time.perf_counter()
            prompts = self._load_prompts()
            client = build_llm_client_from_yaml(
                parameters_path=str(PROJECT_ROOT / "parameters" / "parameters.yaml")
            )
            load_prompt_client_ms = (time.perf_counter() - t0) * 1000
            # --------------------------------------------------
            # 3. Loop students safely
            # --------------------------------------------------
            total_students = len(students)
            for idx, row in students.iterrows():
                t0_student = time.perf_counter()
                student_row = row.to_dict()
                student_id = str(student_row.get("student_id", "")).strip()
                print(f"\nProcessing {student_id} ({idx+1}/{total_students})")
                timing = {
                    "student_id": student_id,
                    "download_data_ms": round(download_ms,2),
                    "read_config_ms": round(read_config_ms,2),
                    "load_prompt_and_client_ms": round(load_prompt_client_ms,2),
                }
                try:
                    # ----------------------------
                    # Context
                    # ----------------------------
                    t0 = time.perf_counter()
                    user_ctx = build_user_context(student_row)
                    pref_lang = user_ctx.user_context_json.get("preferred_language", "th")
                    user_events = interactions[interactions["user_id"] == student_id]
                    num_events = len(user_events)
                    history_summary_text = ""
                    if num_events > 0:
                        history_summary_text = build_history_summary(
                            user_events,
                            preferred_language=pref_lang,
                            include_recent_feeds=include_recent_feeds,
                            recent_k=recent_k,
                            feeds_lookup=feeds_lookup or None,
                            feed_text_max_chars=feed_text_max_chars,
                        )
                    timing["build_context_ms"] = round((time.perf_counter() - t0) * 1000,2)
                    # ----------------------------
                    # Prompt
                    # ----------------------------
                    t0 = time.perf_counter()
                    prompt_key = self._choose_hyde_prompt_key(num_events, history_threshold)
                    template = prompts.get(prompt_key)
                    if not template:
                        raise ValueError(f"Missing prompt {prompt_key}")
                    prompt = self._render_prompt(
                        template=template,
                        preferred_language="th",
                        # preferred_language=pref_lang,
                        user_context_text=user_ctx.user_context_text,
                        history_summary_text=history_summary_text,
                    )
                    timing["build_prompt_ms"] = round((time.perf_counter() - t0) * 1000,2)
                    # ----------------------------
                    # LLM CALL
                    # ----------------------------
                    t0 = time.perf_counter()
                    hyde_json = client.generate_json(prompt)
                    # print(f"hyde_json -> {prettyjson(hyde_json)}")
                    llm_time = time.perf_counter() - t0
                    timing["llm_call_ms"] = round(llm_time * 1000,2)
                    if llm_time > 20:
                        print(f"⚠ Slow LLM ({llm_time:.2f}s) → skip")
                        slow_students.append(student_id)
                    # ----------------------------
                    # Extract queries
                    # ----------------------------
                    hyde_query_texts = self._extract_hyde_query_texts(hyde_json)
                    # ----------------------------
                    # Embedding
                    # ----------------------------
                    t0 = time.perf_counter()
                    if hyde_query_texts:
                        emb = embed_texts_gemini(
                            texts=hyde_query_texts,
                            output_dim=768,
                            task_type="RETRIEVAL_DOCUMENT",
                        )
                        if emb.ndim != 2:
                            raise ValueError(f"Invalid embedding shape {emb.shape}")
                    else:
                        emb = np.zeros((0, expected_dim), dtype=np.float32)
                    timing["embedding_ms"] = round((time.perf_counter() - t0) * 1000,2)
                    # ----------------------------
                    # Upload
                    # ----------------------------
                    t0 = time.perf_counter()
                    metadata = {
                        "student_id": student_id,
                        "generated_at": now_iso,
                        "model": self.cfg["llm"]["model_name"],
                    }
                    self._upload_to_cgs(
                        student_id=student_id,
                        metadata=metadata,
                        embedding=emb,
                        hyde_json={"hq": hyde_json.get("hyde_queries", [])}
                    )
                    timing["upload_gcs_ms"] = round((time.perf_counter() - t0) * 1000,2)
                    timing["total_ms"] = round((time.perf_counter() - t0_student) * 1000,2)
                    timing["status"] = "done"
                    student_id_updated.append(student_id)
                    print(f"✅ Done {student_id} in {(time.perf_counter()-t0_student):.2f}s")
                except Exception as e:
                    print(f"❌ Failed {student_id} → {str(e)}")
                    failed_students.append({
                        "student_id": student_id,
                        "error": str(e)
                    })
                    timing["build_context_ms"] = "-"
                    timing["build_prompt_ms"] = "-"
                    timing["llm_call_ms"] = "-"
                    timing["embedding_ms"] = "-"
                    timing["upload_gcs_ms"] = "-"
                    timing["total_ms"] = "-"
                    timing["status"] = str(e)
                timing_rows.append(timing)
        except Exception as e:
            status = "Fail"
            print("Batch crashed:", e)
        # --------------------------------------------------
        # Save timing report
        # --------------------------------------------------
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        df_new = pd.DataFrame(timing_rows)
        file_path = f"hyde_timing_report_{timestamp}.xlsx"
        if os.path.exists(file_path):
            df_old = pd.read_excel(file_path)
            df_final = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_final = df_new
        df_final.to_excel(file_path, index=False)
        # --------------------------------------------------
        # JSON Report for GCS
        # --------------------------------------------------
        total_time = round(time.perf_counter() - t0_total, 2)
        report = {
            "run_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_students": total_students,
                "updated_count": len(student_id_updated),
                "failed_count": len(failed_students),
                "slow_count": len(slow_students),
                "total_time_sec": total_time,
            },
            "students": timing_rows,
            "failed": failed_students,
            "slow": slow_students
        }
        json_path = f"hyde_batch_report_{timestamp}.json"
        try:
            self.cgs.create_folder(f"report_{timestamp}/")
            self.cgs.upload_json(
                blob_path=f"report_{timestamp}/{json_path}",
                json_data=report
            )
            print(f"Report uploaded to GCS → report_{timestamp}/{json_path}")
        except Exception as e:
            print(f"GCS report upload failed: {e}")
        # --------------------------------------------------
        # Summary
        # --------------------------------------------------
        print("\nBatch Finished")
        print("Updated:", len(student_id_updated))
        print("Failed:", len(failed_students))
        print("Slow:", len(slow_students))
        print("Total Time (sec):", total_time)
        return student_id_updated, status

    def _safe_single_student_fast(
        self,
        student_id,
        students_df,
        interactions,
        feeds_lookup,
        prompts,
        client,
        now_iso,
        history_threshold,
        recent_k,
        feed_text_max_chars,
        include_recent_feeds,
    ):

        t0_total = time.perf_counter()
        timing = {}

        try:

            # ----------------------------
            # Locate student row
            # ----------------------------
            student_row_df = students_df[students_df["student_id"] == student_id]

            if len(student_row_df) == 0:
                raise ValueError(f"{student_id} not found")

            student_row = student_row_df.iloc[0].to_dict()

            # ----------------------------
            # Context
            # ----------------------------
            t0 = time.perf_counter()

            user_ctx = build_user_context(student_row)

            pref_lang = user_ctx.user_context_json.get("preferred_language", "th")

            user_events = interactions[interactions["user_id"] == student_id]

            num_events = len(user_events)

            history_summary_text = ""

            if num_events > 0:
                history_summary_text = build_history_summary(
                    user_events,
                    preferred_language=pref_lang,
                    include_recent_feeds=include_recent_feeds,
                    recent_k=recent_k,
                    feeds_lookup=feeds_lookup,
                    feed_text_max_chars=feed_text_max_chars,
                )

            timing["build_context_ms"] = (time.perf_counter() - t0) * 1000

            # ----------------------------
            # Prompt
            # ----------------------------
            t0 = time.perf_counter()

            prompt_key = self._choose_hyde_prompt_key(num_events, history_threshold)

            template = prompts[prompt_key]
            pref_lang = "th"
            prompt = self._render_prompt(
                template,
                pref_lang,
                user_ctx.user_context_text,
                history_summary_text
            )

            timing["build_prompt_ms"] = (time.perf_counter() - t0) * 1000

            # ----------------------------
            # LLM Call
            # ----------------------------
            t0 = time.perf_counter()

            hyde_json = client.generate_json(prompt)

            timing["llm_call_ms"] = (time.perf_counter() - t0) * 1000

            # ----------------------------
            # Extract queries
            # ----------------------------
            hyde_query_texts = self._extract_hyde_query_texts(hyde_json)

            # ----------------------------
            # Embedding
            # ----------------------------
            t0 = time.perf_counter()

            emb = embed_texts_gemini(
                texts=hyde_query_texts,
                output_dim=768,
                task_type="RETRIEVAL_DOCUMENT",
            )

            timing["embedding_ms"] = (time.perf_counter() - t0) * 1000

            # ----------------------------
            # Upload to GCS
            # ----------------------------
            t0 = time.perf_counter()

            metadata = {
                "student_id": student_id,
                "generated_at": now_iso,
                "model": self.cfg["llm"]["model_name"],
            }

            self._upload_to_cgs(
                student_id=student_id,
                metadata=metadata,
                embedding=emb,
                hyde_json={"hq": hyde_json["hyde_queries"]}
            )

            timing["upload_gcs_ms"] = (time.perf_counter() - t0) * 1000

            # ----------------------------
            # Total time
            # ----------------------------
            timing["total_ms"] = (time.perf_counter() - t0_total) * 1000

            return {
                "status": "Complete",
                "slow": timing["total_ms"] / 1000 > 20,
                "timing": timing,
                "student_id": student_id
            }

        except Exception as e:

            return {
                "status": "Fail",
                "error": str(e),
                "slow": False,
                "student_id": student_id
            }
        
#####################################################################################
    def single_student_generator(self, student_id: str):

        t0_total = time.perf_counter()

        status = "Complete"

        try:

            # ----------------------------
            # Download data
            # ----------------------------
            t0 = time.perf_counter()

            students = self.dq.get_students(student_id)
            interactions = self.dq.get_interactions(student_id)
            feeds_lookup = self.dq.get_user_events_json()

            download_ms = (time.perf_counter() - t0) * 1000

            now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

            # ----------------------------
            # Config
            # ----------------------------
            t0 = time.perf_counter()

            history_threshold, recent_k, feed_text_max_chars, include_recent_feeds, query_embedding_model_name = self._read_hyde_config(self.cfg)

            prompts = self._load_prompts()

            client = build_llm_client_from_yaml(
                parameters_path=str(PROJECT_ROOT / "parameters" / "parameters.yaml")
            )

            config_ms = (time.perf_counter() - t0) * 1000

            # ----------------------------
            # Run core pipeline
            # ----------------------------

            timing = self._single_student_core(
                student_id,
                students,
                interactions,
                feeds_lookup,
                prompts,
                client,
                now_iso,
                history_threshold,
                recent_k,
                feed_text_max_chars,
                include_recent_feeds,
            )

            timing["download_data_ms"] = download_ms
            timing["read_config_ms"] = config_ms

        except Exception as e:

            import traceback
            traceback.print_exc()

            status = "Fail"
            timing = {}

        timing["total_ms"] = (time.perf_counter() - t0_total) * 1000

        return {
            "status": status,
            "timing": timing,
        }
    
    def _single_student_core(
        self,
        student_id,
        students_df,
        interactions,
        feeds_lookup,
        prompts,
        client,
        now_iso,
        history_threshold,
        recent_k,
        feed_text_max_chars,
        include_recent_feeds,
    ):

        t0_total = time.perf_counter()
        timing = {}

        # ----------------------------
        # Locate student
        # ----------------------------
        student_row_df = students_df[students_df["student_id"] == student_id]

        if len(student_row_df) == 0:
            raise ValueError(f"{student_id} not found")

        student_row = student_row_df.iloc[0].to_dict()

        # ----------------------------
        # Context
        # ----------------------------
        t0 = time.perf_counter()

        user_ctx = build_user_context(student_row)

        pref_lang = user_ctx.user_context_json.get("preferred_language", "th")

        user_events = interactions[interactions["user_id"] == student_id]

        num_events = len(user_events)

        history_summary_text = ""

        if num_events > 0:
            history_summary_text = build_history_summary(
                user_events,
                preferred_language=pref_lang,
                include_recent_feeds=include_recent_feeds,
                recent_k=recent_k,
                feeds_lookup=feeds_lookup,
                feed_text_max_chars=feed_text_max_chars,
            )

        timing["build_context_ms"] = (time.perf_counter() - t0) * 1000

        # ----------------------------
        # Prompt
        # ----------------------------
        t0 = time.perf_counter()

        prompt_key = self._choose_hyde_prompt_key(num_events, history_threshold)

        template = prompts[prompt_key]

        prompt = self._render_prompt(
            template,
            pref_lang,
            user_ctx.user_context_text,
            history_summary_text,
        )

        timing["build_prompt_ms"] = (time.perf_counter() - t0) * 1000

        # ----------------------------
        # LLM
        # ----------------------------
        t0 = time.perf_counter()

        hyde_json = client.generate_json(prompt)

        timing["llm_call_ms"] = (time.perf_counter() - t0) * 1000

        # ----------------------------
        # Extract queries
        # ----------------------------
        hyde_query_texts = self._extract_hyde_query_texts(hyde_json)

        # ----------------------------
        # Embedding
        # ----------------------------
        t0 = time.perf_counter()

        emb = embed_texts_gemini(
            texts=hyde_query_texts,
            output_dim=768,
            task_type="RETRIEVAL_DOCUMENT",
        )

        timing["embedding_ms"] = (time.perf_counter() - t0) * 1000

        # ----------------------------
        # Upload
        # ----------------------------
        t0 = time.perf_counter()

        metadata = {
            "student_id": student_id,
            "generated_at": now_iso,
            "model": self.cfg["llm"]["model_name"],
        }

        self._upload_to_cgs(
            student_id=student_id,
            metadata=metadata,
            embedding=emb,
            hyde_json={"hq": hyde_json["hyde_queries"]},
        )

        timing["upload_gcs_ms"] = (time.perf_counter() - t0) * 1000

        timing["total_ms"] = (time.perf_counter() - t0_total) * 1000

        return timing
    
    def _safe_single_student_fast(
        self,
        student_id,
        students_df,
        interactions,
        feeds_lookup,
        prompts,
        client,
        now_iso,
        history_threshold,
        recent_k,
        feed_text_max_chars,
        include_recent_feeds,
    ):

        try:

            timing = self._single_student_core(
                student_id,
                students_df,
                interactions,
                feeds_lookup,
                prompts,
                client,
                now_iso,
                history_threshold,
                recent_k,
                feed_text_max_chars,
                include_recent_feeds,
            )

            return {
                "status": "Complete",
                "slow": timing["total_ms"] / 1000 > 20,
                "timing": timing,
                "student_id": student_id,
            }

        except Exception as e:

            return {
                "status": "Fail",
                "error": str(e),
                "slow": False,
                "student_id": student_id,
            }
        
    def batch_student_async(
        self,
        student_ids: Optional[List[str]] = None,
        max_workers: int = 5,
    ):

        status = "Complete"
        updated = []
        failed = []
        slow = []
        timing_rows = []

        t0_total = time.perf_counter()
        try:
            # --------------------------------------------------
            # 01 Download BigQuery data ONCE
            # --------------------------------------------------
            t0 = time.perf_counter()
            students_df  = self.dq.get_students()
            interactions = self.dq.get_interactions()
            feeds_lookup = self.dq.get_user_events_json()
            download_ms = (time.perf_counter() - t0) * 1000
            if student_ids is None:
                student_ids = students_df["student_id"].astype(str).tolist()
            student_ids = list(dict.fromkeys(student_ids))
            total_students = len(student_ids)
            print(f"Total students: {total_students}")
            print(f"Max workers: {max_workers}")
            print(f"Download time: {download_ms/1000:.2f}s")
            # --------------------------------------------------
            # 02 Load config + prompts + client ONCE
            # --------------------------------------------------
            history_threshold, recent_k, feed_text_max_chars, include_recent_feeds, query_embedding_model_name = self._read_hyde_config(self.cfg)
            prompts = self._load_prompts()
            client = build_llm_client_from_yaml(
                parameters_path=str(PROJECT_ROOT / "parameters" / "parameters.yaml")
            )
            now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            counter = 0
            # --------------------------------------------------
            # 03 Thread workers (BATCHED)
            # --------------------------------------------------
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for i in range(0, total_students, max_workers):
                    batch = student_ids[i:i+max_workers]
                    futures = {
                        executor.submit(
                            self._safe_single_student_fast,
                            sid,
                            students_df,
                            interactions,
                            feeds_lookup,
                            prompts,
                            client,
                            now_iso,
                            history_threshold,
                            recent_k,
                            feed_text_max_chars,
                            include_recent_feeds,
                        ): sid
                        for sid in batch
                    }
                    for future in as_completed(futures):
                        sid = futures[future]
                        counter += 1
                        print(f"[{counter}/{total_students}] Processing {sid}")
                        try:
                            result = future.result()
                            if result["status"] == "Complete":
                                updated.append(sid)
                                if result["slow"]:
                                    slow.append(sid)
                                timing = result.get("timing", {})
                                timing["student_id"] = sid
                                timing["status"] = "done"
                                timing_rows.append(timing)
                                print(f"✅ {sid} completed")
                            else:
                                failed.append({
                                    "student_id": sid,
                                    "error": result.get("error", "Unknown")
                                })
                                print(f"❌ {sid} failed")
                        except Exception as e:
                            failed.append({
                                "student_id": sid,
                                "error": str(e)
                            })
                            print(f"❌ {sid} crashed → {str(e)}")
        except Exception:
            status = "Fail"
            traceback.print_exc()
        # --------------------------------------------------
        # 04 Final report
        # --------------------------------------------------
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        self.cgs.create_folder(f"report_{timestamp}/")
        total_time = round(time.perf_counter() - t0_total, 2)
        report = {
            "run_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_students": total_students,
                "updated_count": len(updated),
                "failed_count": len(failed),
                "slow_count": len(slow),
                "total_time_sec": total_time,
                "max_workers": max_workers
            },
            "students": timing_rows,
            "failed": failed,
            "slow": slow
        }
        json_path = f"hyde_batch_async_report_{timestamp}.json"
        # with open(json_path, "w") as f:
        #     json.dump(report, f, indent=2)
        self.cgs.upload_json(
            blob_path=f"report_{timestamp}/{json_path}",
            json_data=report
        )
        print(f"JSON report saved → {json_path}")
        print(f"Uploaded to datalake → report/{json_path}")
        print("\nBatch Async Finished")
        print("Updated:", len(updated))
        print("Failed:", len(failed))
        print("Slow:", len(slow))
        print("Total Time (sec):", total_time)




hg = HydeGenerator(
    bucket_name = "hyde-datalake",
    verbose     = 0
)
# hg.concurrent_of_single_student_generator()
hg.single_hyde_generator2("stu_p000")