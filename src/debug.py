import time
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from src.functions.utils.cloudstorage import GoogleCloudStorage
from src.functions.utils.bigquery import DataQuery
from src.functions.utils.config  import PROJECT_ROOT, load_config
from src.functions.utils.llm_client import build_llm_client_from_yaml
from src.functions.core.context_builder import build_user_context
from src.functions.core.history import build_history_summary
from src.functions.utils.shin_embedder import embed_texts_gemini
from src.functions.utils.cost_logger import append_cost_log


class HydeGenerator(GoogleCloudStorage,DataQuery):
    def __init__(self,bucket_name:str, verbose:int=0):
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
    #----------------------------------------------------------------------
    # main pipeline
    #----------------------------------------------------------------------
    # def _interactions_to_json(self, interactions_df, student_id)->List[Dict]:
    #     """
    #     Convert interactions records for a specific student into JSON format
    #     Return
    #     ------
    #     List[Dict]
    #         [
    #             {
    #                 "user_id": "stu_p001",
    #                 "feed_id": "TH_F008",
    #                 "ts": "2026-01-04 14:05:02+00:00",
    #                 "event_type": "click",
    #                 "dwell_ms": 0
    #             },
    #             {},
    #             {}
    #         ]
    #     """
    #     df = interactions_df[interactions_df["user_id"] == student_id].copy()
    #     df["ts"] = df["ts"].astype(str)
    #     records = df.to_dict(orient="records")
    #     # print(f"recores -> \n{records}")
    #     return records
    
    def _interactions_to_json(self, interactions_df, student_id) -> List[Dict]:
        df = interactions_df[interactions_df["user_id"] == student_id].copy()
        if "ts" not in df.columns:
            if "event_ts" in df.columns:
                df["ts"] = df["event_ts"]
            else:
                raise ValueError("No timestamp column found (expected 'ts' or 'event_ts')")
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce") \
                    .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        records = df.to_dict(orient="records")
        return records
    
    def _upload_to_cgs(self,student_id,metadata,embedding,hyde_json):
        """
        Upload a HyDE bundle to Google Cloud Storage.

        Packages HyDE generation results for a student into a single JSON object
        containing metadata, generated queries, and embedding vectors, then uploads
        it to GCS.

        Parameters
        ----------
        student_id : str
            Student identifier.
        metadata : Dict[str, Any]
            Generation metadata.
        embedding : np.ndarray
            Query embedding vectors (expected shape: 5 x embedding_dim).
        hyde_json : Dict[str, Any]
            HyDE generator output containing synthetic queries under key "hq".
        """
        emb_list = embedding.tolist()
        bundle = {
            "student_id": student_id,
            "metadata": metadata,
            "hyde_queries": hyde_json.get("hq", []),
            "embeddings": {
                "embedding01": emb_list[0],
                "embedding02": emb_list[1],
                "embedding03": emb_list[2],
                "embedding04": emb_list[3],
                "embedding05": emb_list[4],
            }
        }
        # print(bundle)
        self.cgs.upload_json(
            blob_path=f"{student_id}/hyde_bundle.json",
            json_data=bundle
        )

    def _estimate_embedding_cost_usd(
        self,
        texts: List[str],
        model_name: str,
    ) -> float:
        """
        Temporary embedding cost estimator.
        Replace with real Gemini embedding pricing.
        """
        total_chars = sum(len(t or "") for t in texts)

        # TODO: replace with real pricing
        pricing = {
            "gemini-embedding-001": {
                "per_1k_chars": 0.00005,
            }
        }

        model_price = pricing.get(
            model_name,
            {"per_1k_chars": 0.00005},
        )

        return (total_chars / 1000) * model_price["per_1k_chars"]
    
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

        # --------------------------------------------------
        # 1. Download data
        # --------------------------------------------------
        print("01 Download data ...")
        t0 = time.perf_counter()
        if students is None:
            students     = self.dq.get_students()
        # print(students)
        if interactions is None:
            interactions = self.dq.get_interactions()
        # print(interactions)
        if feeds_lookup is None:
            feeds_lookup = self.dq.get_user_events_json()
        # print(feeds_lookup)
        download_ms  = (time.perf_counter()-t0)*1000
        print(f"Download time: {(download_ms/1000):.2f}s")
        # --------------------------------------------------
        # 2. Config
        # --------------------------------------------------
        print("02 Config ...")

        t0 = time.perf_counter()
        history_threshold, recent_k, feed_text_max_chars, include_recent_feeds, query_embedding_model_name = self._read_hyde_config(self.cfg)
        read_config_ms = (time.perf_counter() - t0) * 1000
        expected_dim = int(self.cfg.get("embeddings",{}).get("dim",0) or 0)
        # --------------------------------------------------
        # 3. Load prompts + client
        # --------------------------------------------------
        print("03 Load prompts + client ...")
        t0 = time.perf_counter()
        prompts = self._load_prompts()
        client = build_llm_client_from_yaml(
            parameters_path = str(PROJECT_ROOT/"parameters"/"parameters.yaml")
        )
        load_prompt_client_ms = (time.perf_counter() - t0) * 1000
        # --------------------------------------------------
        # 4. Locate student
        # --------------------------------------------------
        print("04 Locate student ...")
        print(student_id)
        student_row_df = students[students["student_id"] == student_id]
        if len(student_row_df) == 0:
            raise ValueError(f"{student_id} not found")
        student_row = student_row_df.iloc[0].to_dict()
        # print(f"\nProcessing {student_id}")
        t0_student = time.perf_counter()
        timing = {
            "student_id" : student_id,
            "download_data_ms" : round(download_ms,2),
            "read_config_ms" : round(read_config_ms,2),
            "load_prompt_and_client_ms" : round(load_prompt_client_ms,2)
        }
        # ----------------------------
        # 5. Context
        # ----------------------------
        print("05 Context ...")
        t0 = time.perf_counter()
        user_ctx = build_user_context(student_row)
        pref_lang = user_ctx.user_context_json.get("preferred_language", "th")
        # pref_lang = "th"  # TODO : change it later but for now there are only th feeds
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
        print("06 Prompt ...")
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
        print("07 LLM ...")

        t0 = time.perf_counter()

        hyde_json = client.generate_json(
            prompt,
            extra_log={
                "student_id": student_id,
                "pipeline": "hyde_generator",
            },
        )

        llm_time = time.perf_counter() - t0
        timing["llm_call_ms"] = round(llm_time * 1000,2)
        slow_time = self.cfg["llm"]["slow_time"]
        if llm_time > slow_time:
            print(f"⚠ Slow LLM ({llm_time:.2f}s)")
            slow_students.append(student_id)
        # ----------------------------
        # 8. Extract queries
        # ----------------------------   
        print("08 Extract queries ...")
        hyde_query_text = self._extract_hyde_query_texts(hyde_json)

        # ----------------------------
        # 9. Embedding
        # ----------------------------  
        print("09 Embeddding ...")

        t0 = time.perf_counter()
        output_dim = self.cfg["llm"]["output_dim"]
        embedding_model = self.cfg["llm"]["embedding_model"]
        if hyde_query_text:
            emb = embed_texts_gemini(
                texts  = hyde_query_text,
                output_dim = output_dim,
                task_type="RETRIEVAL_DOCUMENT",
                embedding_model = embedding_model
            )         
            if emb.ndim != 2:
                raise ValueError(f"Invalid embedding shape {emb.shape}")
        else:
            emb = np.zeros((0, expected_dim), dtype=np.float32)
        
        embedding_latency_s = time.perf_counter() - t0
        timing["embedding_ms"] = round(embedding_latency_s * 1000, 2)

        embedding_cost_usd = self._estimate_embedding_cost_usd(
            texts=hyde_query_text,
            model_name=embedding_model,
        )

        append_cost_log(
            {
                "event_type": "embedding",
                "student_id": student_id,
                "model_name": embedding_model,
                "num_texts": len(hyde_query_text),
                "total_chars": sum(len(t or "") for t in hyde_query_text),
                "latency_s": round(embedding_latency_s, 4),
                "estimated_cost_usd": embedding_cost_usd,
            }
        )
        # ----------------------------
        # 10. Upload
        # ----------------------------
        print("10 Upload ...")

        t0 = time.perf_counter()
        print(student_row)
        metadata = {
            "student_id"          :student_id, # 
            "current_status"      :student_row['curriculum_name'], #
            "education_level"     :student_row['student_year'], #
            "education_major"     :student_row['faculty_name'], #
            "target_roles"        :student_row['onboard_grp'], #
            "timezone"            :self.cfg["app"]["timezone"], #
            "model_name"          :self.cfg["llm"]["model_name"], #
            "max_output_tokens"   :self.cfg["llm"]["max_output_tokens"], #
            "feed_text_max_chars" :self.cfg["hyde"]["feed_text_max_chars"], #
            "temperature"         :self.cfg["llm"]["temperature"], #
            "interaction"         :self._interactions_to_json(interactions,student_id)
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
config_path = Path(__file__).resolve().parent / "parameters" / "parameters.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

bucket_name = config["bigquery"]["bucket"]
hg = HydeGenerator(
    bucket_name = bucket_name,
    verbose     = 0
)
hg.single_hyde_generator2(
    student_id = "stu_p7001"
)