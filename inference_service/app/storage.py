import os

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError


class ObjectStorage:
    def __init__(self):
        self.enabled = os.getenv("OBJECT_STORAGE_ENABLED", "true").lower() == "true"
        self.required = os.getenv("OBJECT_STORAGE_REQUIRED", "false").lower() == "true"
        self.bucket = os.getenv("S3_BUCKET", "synthetic-datasets")
        self.endpoint_url = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
        self.access_key_id = os.getenv("S3_ACCESS_KEY_ID", "minioadmin")
        self.secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY", "minioadmin")
        self.region_name = os.getenv("S3_REGION", "us-east-1")
        self.prefix = os.getenv("S3_PREFIX", "online-generations")
        self._client = None
        self._bucket_checked = False

    def _get_client(self):
        if self._client is None:
            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                config=Config(signature_version="s3v4"),
                region_name=self.region_name,
            )
        return self._client

    def _ensure_bucket(self):
        if self._bucket_checked or not self.enabled:
            return
        client = self._get_client()
        try:
            client.head_bucket(Bucket=self.bucket)
        except ClientError:
            client.create_bucket(Bucket=self.bucket)
        self._bucket_checked = True

    def upload_bytes(self, payload: bytes, object_key: str, content_type: str) -> str | None:
        if not self.enabled:
            return None
        try:
            self._ensure_bucket()
            self._get_client().put_object(
                Bucket=self.bucket,
                Key=object_key,
                Body=payload,
                ContentType=content_type,
            )
            return f"s3://{self.bucket}/{object_key}"
        except Exception:
            if self.required:
                raise
            return None

    def upload_file(self, local_path: str, object_key: str, content_type: str) -> str | None:
        if not self.enabled:
            return None
        try:
            self._ensure_bucket()
            self._get_client().upload_file(
                local_path,
                self.bucket,
                object_key,
                ExtraArgs={"ContentType": content_type},
            )
            return f"s3://{self.bucket}/{object_key}"
        except Exception:
            if self.required:
                raise
            return None
