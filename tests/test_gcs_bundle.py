import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from gcs_bundle import GoogleCloudStorage   # <-- change to your import


# -------------------------------------------------
# helpers
# -------------------------------------------------

def fake_blob(text=None, array=None, exists=True):
    """Create fake blob with configurable behavior"""
    blob = MagicMock()

    blob.exists.return_value = exists

    if text:
        blob.download_as_text.return_value = text

    if array is not None:
        def download_to_file(buffer):
            np.save(buffer, array)
        blob.download_to_file.side_effect = download_to_file

    return blob


# -------------------------------------------------
# 1️⃣ happy path (everything exists)
# -------------------------------------------------

@patch("your_module.storage.Client")
@patch("your_module.DataQuery")
def test_retrieve_bundle_all_exist(mock_dq, mock_client):

    # ---- mock bucket ----
    bucket = MagicMock()

    bucket.blob.side_effect = lambda path: {
        "stu_p001/metadata/metadata.json":
            fake_blob(text='{"student_id":"stu_p001"}'),

        "stu_p001/embedding/embedding01.npy":
            fake_blob(array=np.array([1, 2, 3])),

    }.get(path, fake_blob(exists=False))

    bucket.list_blobs.return_value = [MagicMock()]

    mock_client.return_value.get_bucket.return_value = bucket

    # ---- run ----
    gcs = GoogleCloudStorage("test-bucket")

    result = gcs.retrieve_student_bundle(
        "stu_p001",
        ["embedding01.npy"]
    )

    # ---- asserts ----
    assert result["metadata"]["student_id"] == "stu_p001"
    assert result["embeddings"]["embedding01.npy"].shape[0] == 3
    assert "embedding folder /" in result["status"]


# -------------------------------------------------
# 2️⃣ metadata missing → fallback to BigQuery
# -------------------------------------------------

@patch("your_module.storage.Client")
@patch("your_module.DataQuery")
def test_metadata_fallback_bigquery(mock_dq, mock_client):

    bucket = MagicMock()
    bucket.list_blobs.return_value = [MagicMock()]
    bucket.blob.return_value = fake_blob(exists=False)

    mock_client.return_value.get_bucket.return_value = bucket

    # ---- fake BigQuery ----
    dq_instance = MagicMock()
    dq_instance.get_students.return_value.iloc.return_value = [{
        "student_id": "stu_p001",
        "current_status": "active",
        "education_level": "bachelor",
        "education_major": "CS",
        "target_roles": "DS"
    }]

    mock_dq.return_value = dq_instance

    gcs = GoogleCloudStorage("test")

    result = gcs.retrieve_student_bundle("stu_p001", [])

    assert result["metadata"]["student_id"] == "stu_p001"


# -------------------------------------------------
# 3️⃣ embedding folder missing
# -------------------------------------------------

@patch("your_module.storage.Client")
def test_embedding_folder_missing(mock_client):

    bucket = MagicMock()

    # prefix not found
    bucket.list_blobs.return_value = []

    mock_client.return_value.get_bucket.return_value = bucket

    gcs = GoogleCloudStorage("test")

    result = gcs.retrieve_student_bundle(
        "stu_p001",
        ["embedding01.npy"]
    )

    assert result["embeddings"]["embedding01.npy"].size == 0
    assert "embedding folder x" in result["status"]


# -------------------------------------------------
# 4️⃣ student prefix missing
# -------------------------------------------------

@patch("your_module.storage.Client")
def test_student_prefix_missing(mock_client):

    bucket = MagicMock()

    bucket.list_blobs.return_value = []   # nothing exists

    mock_client.return_value.get_bucket.return_value = bucket

    gcs = GoogleCloudStorage("test")

    result = gcs.retrieve_student_bundle("stu_unknown", [])

    assert "|- stu_unknown x" in result["status"]
